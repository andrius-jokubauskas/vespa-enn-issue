# Exact nearestNeighbor brute force runs again in second phase and match-features, once per match thread

**Vespa version:** 8.763.13. The same code is on master today.

This repository contains a minimal application that reproduces the problem. See
[How to reproduce](#how-to-reproduce).

## Summary

When `nearestNeighbor` uses exact search ("exact fallback" because the filter is below
`approximateThreshold`, or `approximate:false`), the brute force over the filtered documents
should happen once, in first phase. Instead it happens again:

- in **second phase**, once **per match thread**, and
- in **match-features**, once more.

With 4 threads per search, a query computes up to **6×** the distances it needs. The only
setting that turns this on is a normal `rerank-count`.

## A simple example

Say a content node has 1,000,000 documents and the query filter keeps 200,000 of them. The filter
is below `approximateThreshold`, so the NN brute-forces those 200,000 documents.

1. **First phase:** 200,000 distances. This is expected.
2. **Second phase**, `rerank-count: 100`, 4 threads.
   - The 100 best hits are dealt to the threads round-robin, so each thread gets 25 docids spread over the whole index.
   - Each thread builds a **new** iterator tree and jumps to its docids.
   - When a docid is not within `distanceThreshold` (for example a document that only matched the text part), the strict exact iterator does not stop. It brute-forces forward to the next document within the threshold.
   - So each thread brute-forces almost the whole filtered set again: about 200,000 × 4 = **800,000** distances.
3. **Match-features** for the returned hits: one more pass, about **200,000** distances.

That is 1,200,000 distances in total where 200,000 plus a few hundred would do: **6×**.

## Where it happens (links to v8.763.13)

1. **Later phases build a new iterator tree.** Reuse is turned off on purpose
   ([match_tools.h#L122](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchcore/src/vespa/searchcore/proton/matching/match_tools.h#L122),
   [match_tools.cpp#L97-L102](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchcore/src/vespa/searchcore/proton/matching/match_tools.cpp#L97-L102),
   from [#25880](https://github.com/vespa-engine/vespa/pull/25880)). That is fine for most iterators.

2. **The new exact iterator keeps the first-phase strictness**
   ([nearest_neighbor_blueprint.cpp#L244-L247](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchlib/src/vespa/searchlib/queryeval/nearest_neighbor_blueprint.cpp#L244-L247)):

   ```cpp
   return ExactNearestNeighborIterator::create(
       _eval_stats, strict(), tfmd,               // strict() was decided for the first-phase scan
       std::make_unique<search::tensor::DistanceCalculator>(_attr_tensor, _query_tensor), _distance_heap,
       *_global_filter, _matching_phase != MatchingPhase::FIRST_PHASE);
   ```

3. **Later phases jump to known docids.** `DocumentScorer::score`
   ([document_scorer.cpp#L38](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchcore/src/vespa/searchcore/proton/matching/document_scorer.cpp#L38))
   and match-features
   ([extract_features.cpp#L47](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchcore/src/vespa/searchcore/proton/matching/extract_features.cpp#L47))
   set the range to the first..last docid and unpack each docid. `MultiSearch::doUnpack`
   ([multisearch.cpp#L25-L33](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchlib/src/vespa/searchlib/queryeval/multisearch.cpp#L25-L33))
   calls `doSeek(docid)` on every child behind that docid. `NearestNeighborBlueprint::always_needs_unpack()` returns `true`,
   so this happens even when the rank expression does not read any NN feature.

4. **A strict exact iterator that misses keeps scanning**
   ([exact_nearest_neighbor_iterator.cpp#L52-L69](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchlib/src/vespa/searchlib/queryeval/exact_nearest_neighbor_iterator.cpp#L52-L69)):

   ```cpp
   void doSeek(uint32_t docId) override {
       double distanceLimit = params().distanceHeap.distanceLimit();
       while (docId < getEndId()) {
           if (!has_filter || params().filter.check(docId)) {
               double d = computeDistance(docId, distanceLimit);   // brute force
               if (d <= distanceLimit) { ...; setDocId(docId); return; }
           }
           if (strict) { ++docId; } else { return; }               // strict: keep going
       }
       setAtEnd();
   }
   ```

   For a posting list or a bitvector, a strict seek is a cheap skip. For an exact NN iterator it
   is a linear brute force, which makes this iterator a special case.

5. **Second-phase hits are dealt round-robin**
   ([match_loop_communicator.cpp#L113](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchcore/src/vespa/searchcore/proton/matching/match_loop_communicator.cpp#L113),
   `out(picked % size())`). Each thread's docids span the whole index, so each thread repeats the
   full scan.

## How to reproduce

### What you need

- **Docker** with at least 8 GB of memory for the container. Docker Desktop, Rancher Desktop or Colima all work.
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, which installs Python 3.11+ and the dependencies (pyvespa, numpy) for you.
- About 10 GB of free disk for the Vespa image and data, and ~15 minutes.

### Steps

```sh
git clone https://github.com/andrius-jokubauskas/vespa-enn-issue.git
cd vespa-enn-issue

uv run repro.py deploy   # pulls vespaengine/vespa:8.763.13, starts container "annrepro", deploys ./app
uv run repro.py feed     # feeds 1,000,050 fake items (~7 min on a laptop)
uv run repro.py bench    # runs the query under each setting, prints the table, writes results.json
```

- **Other Docker engines.** Docker Desktop works as is. With Rancher Desktop or Colima, point the Docker client to their socket first, for example:
  - `export DOCKER_HOST=unix://$HOME/.rd/docker.sock` (Rancher Desktop)
  - `export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` (Colima)
- **Options** (environment variables):

  | Variable | Default | What it sets |
  |---|---|---|
  | `VESPA_PORT` | `8080` | Host port for the query and feed API |
  | `VESPA_IMAGE` | `vespaengine/vespa:8.763.13` | Vespa version to test |
  | `N_DOCS` | `1000000` | Number of items to feed |
  | `REPS` | `9` | Timed runs per setting |

- **Clean up:** `docker rm -f annrepro`.

To check that a patched or newer Vespa fixes the problem, run the same steps with
`VESPA_IMAGE=<your image>`. After a fix, the "÷ filter hits" column should stay near 1× for
every row.

### What the repro does

- `app/schemas/item.sd`: 1M items with `category`, `shade`, `title`, `popularity` and a 512-d bfloat16 `embedding` (HNSW). One content node, 4 threads per search (`app/services.xml`).
- Query: `(nearestNeighbor(embedding, q) or title contains "shirt") and category < 90 and shade < 90`.
  - **The filter.** `shade = 99 - category`, so the two filters are anti-correlated.
    - Vespa estimates the filter at 90% (≥ `approximateThreshold=0.85`), so it computes the global filter.
    - The real intersection is 80% (< 0.85), so the NN switches to **exact fallback**, and the iterator is strict.
    - With one plain filter, Vespa estimates correctly and skips the global filter. The NN is then non-strict and the problem does not show.
  - **Who gets reranked.** The `title` branch matches ~16k items spread over the whole index. They win first phase (`popularity`), so they are the documents that second phase reranks.
  - **Worst case.** Only 50 items lie within `distanceThreshold`. They are fed last, so they have the highest docids, and every later-phase seek brute-forces to the end of the index.
- Two rank profiles:
  - `rerank_uses_nn`: second phase is `firstPhase + closeness(label, nn)`.
  - `rerank_ignores_nn`: second phase is `firstPhase * 2`, with no NN feature anywhere.
- `bench` runs each setting once with `trace.explainLevel` to read the algorithm, the global-filter hits and `exact_nns_distances_computed`. It then runs the setting `REPS` times without trace for `querytime`.

### Results

Docker on an 8-core laptop, median of 9 runs:

| profile | rerankCount | threads | algorithm | filter hits | exact distances | ÷ filter hits | querytime p50 |
|---|---:|---:|---|---:|---:|---:|---:|
| rerank_uses_nn | 0 | 4 | exact fallback | 799,488 | 1,569,745 | 1.96× | 40 ms |
| rerank_uses_nn | 4 | 4 | exact fallback | 799,488 | 1,569,749 | 1.96× | 37 ms |
| rerank_uses_nn | 16 | 4 | exact fallback | 799,488 | 3,720,244 | 4.65× | 87 ms |
| rerank_uses_nn | 100 | 4 | exact fallback | 799,488 | 4,467,153 | 5.59× | 91 ms |
| rerank_uses_nn | 1000 | 4 | exact fallback | 799,488 | 4,735,936 | 5.92× | 121 ms |
| rerank_uses_nn | 100 | 2 | exact fallback | 799,488 | 3,089,974 | 3.86× | 112 ms |
| rerank_uses_nn | 100 | 1 | exact fallback | 799,488 | 2,369,380 | 2.96× | 153 ms |
| rerank_ignores_nn | 0 | 4 | exact fallback | 799,488 | 1,569,745 | 1.96× | 36 ms |
| rerank_ignores_nn | 100 | 4 | exact fallback | 799,488 | 4,467,153 | 5.59× | 89 ms |

- Only `rerankCount` changes between the first rows. Going from 0 to 100 makes the query 2.3×
  slower (40 → 91 ms), and 1000 makes it 3× slower (121 ms). The second-phase expression is
  trivial (`firstPhase + closeness(label, nn)`).
- `rerankCount` 0 → 1000 alone changes the exact distances from ~2× to ~6× the filter hits.
  - ~2× with `rerankCount=0`: first phase + match-features.
  - At most one docid per thread (`rerankCount` ≤ 4) adds nothing, because each thread's range is a single document.
  - A realistic `rerankCount` adds one full pass per thread.
- With `rerankCount=100`: 1 thread → 3×, 2 threads → 4×, 4 threads → 6×. That's 2 + threads.
- `rerank_ignores_nn` (no NN feature anywhere) pays the same.

## Possible fixes

**Expected cost after a fix:** 1 brute-force pass, plus one distance per reranked document and
per returned hit. Today it is (2 + threads) passes.

### Not safe: make only the exact iterator non-strict

```diff
     return ExactNearestNeighborIterator::create(
-        _eval_stats, strict(), tfmd,
+        _eval_stats, strict() && _matching_phase == MatchingPhase::FIRST_PHASE, tfmd,
```

This looks like a one-line fix, but strictness is a contract for the whole tree. A strict parent
assumes its children are strict
([orlikesearch.h#L43](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchlib/src/vespa/searchlib/queryeval/orlikesearch.h#L43)).
When a non-strict NN leaf misses, it stays behind the requested docid:

- `StrictHeapOrSearch::doSeek`, the default strict OR, keeps seeking the same child in
  `while (_child_docid[front] < docid)`
  ([orlikesearch.h#L115-L120](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchlib/src/vespa/searchlib/queryeval/orlikesearch.h#L115-L120)),
  which never ends. That is the OR in the repro query.
- `AndSearchStrict::advance` with the NN as first child doesn't move forward either
  ([andsearchstrict.h#L61-L62](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchlib/src/vespa/searchlib/queryeval/andsearchstrict.h#L61-L62)).

### Option 1: build the whole later-phase iterator tree non-strict

- Second phase and match-features only call `initRange` and then `unpack(docid)` on the root, for
  known docids in increasing order
  ([document_scorer.h#L33-L36](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchcore/src/vespa/searchcore/proton/matching/document_scorer.h#L33-L36),
  [extract_features.cpp#L47-L53](https://github.com/vespa-engine/vespa/blob/v8.763.13/searchcore/src/vespa/searchcore/proton/matching/extract_features.cpp#L47-L53)).
  They never use the root's "next hit", so they don't need strict iterators.
- A non-strict iterator gives the same match / no-match answer for a given docid, and every parent
  accepts non-strict children.
- Open questions: strictness is resolved once on the blueprint
  ([#30814](https://github.com/vespa-engine/vespa/pull/30814)), so later phases need their own
  non-strict pass or flag. Iterators that keep state between seeks (WAND, limiters, the NN heap)
  need the same review as in [#25880](https://github.com/vespa-engine/vespa/pull/25880).

### Option 2: keep the exact iterator strict, make its later-phase seeks cheap

- For example, first phase records in a bit vector which docids later phases would match, and
  later phases seek to the next set bit. This stays inside the NN code.
- The recorded set must equal what later phases match today (`distanceLimit()` of the read-only
  heap, [#31562](https://github.com/vespa-engine/vespa/pull/31562)), not the first-phase hits,
  because the first-phase limit tightens as the heap fills.
- First-phase threads would need to share or merge the bit vector.

### Checking a fix

The "÷ filter hits" column should drop to about 1× for every row, and hits, scores and
match-features must be identical to the stock image. We have not built a patched Vespa yet.
