"""Reproduce: exact nearestNeighbor brute force is repeated in second phase and match-features.

    uv run repro.py deploy   # start Vespa in Docker and deploy ./app
    uv run repro.py feed     # feed N_DOCS fake items
    uv run repro.py bench    # run one query under different rerankCount / thread settings

The query is a hybrid "nearestNeighbor OR text" search with a filter:

    (nearestNeighbor(embedding, q) or title contains "shirt") and category < 90 and shade < 90

`shade` is 99 - `category`, so the two filters are anti-correlated. Each alone keeps 90% of the
items, and Vespa estimates the filter at 90%. Together they keep only 80%. With
approximateThreshold=0.85, Vespa computes the global filter (estimate 0.9 >= 0.85), sees 0.8 < 0.85
and switches the nearestNeighbor to "exact fallback": brute force over the filtered items.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from vespa.application import Vespa
from vespa.deployment import VespaDocker

ROOT = Path(__file__).parent
IMAGE = os.environ.get("VESPA_IMAGE", "vespaengine/vespa:8.763.13")
URL = os.environ.get("VESPA_URL", "http://localhost")
PORT = int(os.environ.get("VESPA_PORT", "8080"))

N_DOCS = int(os.environ.get("N_DOCS", "1000000"))
N_NEAR = 50
DIM = 512
SEED = 42
APPROXIMATE_THRESHOLD = 0.85
HITS = 100

YQL = (
    "select * from item where "
    '({targetHits:100, distanceThreshold:0.5, label:"nn"}nearestNeighbor(embedding, q) '
    'or title contains "shirt") and category < 90 and shade < 90'
)


def query_vector() -> np.ndarray:
    v = np.random.default_rng(SEED).standard_normal(DIM)
    return v / np.linalg.norm(v)


def to_bf16_hex(v: np.ndarray) -> str:
    return (v.astype(np.float32).view(np.uint32) >> 16).astype(">u2").tobytes().hex()


def docs():
    """Random items: random embeddings (far from the query), 2% titled "shirt"."""
    rng = np.random.default_rng(SEED + 1)
    for i in range(N_DOCS):
        category = int(rng.integers(0, 100))
        yield {
            "id": f"item-{i}",
            "fields": {
                "category": category,
                "shade": 99 - category,
                "title": "shirt" if rng.random() < 0.02 else "item",
                "popularity": float(rng.random()),
                "embedding": {"values": to_bf16_hex(rng.standard_normal(DIM))},
            },
        }


def near_docs():
    """The only items within distanceThreshold of the query.

    They are fed last, so they get the highest local document ids. A strict brute-force iterator
    that misses a document therefore scans all the way to the end of the index.
    """
    rng = np.random.default_rng(SEED + 2)
    q = query_vector()
    for i in range(N_NEAR):
        yield {
            "id": f"near-{i}",
            "fields": {
                "category": 50,
                "shade": 49,
                "title": "item",
                "popularity": float(rng.random()) * 0.01,
                "embedding": {"values": to_bf16_hex(q + 0.05 * rng.standard_normal(DIM))},
            },
        }


def app() -> Vespa:
    return Vespa(url=URL, port=PORT)


def deploy():
    VespaDocker(port=PORT, container_image=IMAGE).deploy_from_disk(application_name="annrepro",
                                                                   application_root=ROOT / "app")


def feed():
    errors = []

    def callback(response, doc_id):
        if not response.is_successful():
            errors.append((doc_id, response.get_json()))

    t0 = time.time()
    a = app()
    a.feed_async_iterable(docs(), schema="item", namespace="repro", callback=callback, max_workers=128,
                          max_connections=4)
    a.feed_async_iterable(near_docs(), schema="item", namespace="repro", callback=callback)
    print(f"fed {N_DOCS + N_NEAR} docs in {time.time() - t0:.0f}s, {len(errors)} errors")
    for e in errors[:5]:
        print(e)


def walk(o, pred):
    if isinstance(o, dict):
        if pred(o):
            yield o
        for v in o.values():
            yield from walk(v, pred)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v, pred)


def run(profile: str, rerank_count: int, threads: int, trace: bool) -> dict:
    body = {
        "yql": YQL,
        "input.query(q)": {"values": to_bf16_hex(query_vector())},
        "ranking.profile": profile,
        "ranking.rerankCount": rerank_count,
        "ranking.matching.numThreadsPerSearch": threads,
        "ranking.matching.approximateThreshold": APPROXIMATE_THRESHOLD,
        "ranking.queryCache": False,
        "hits": HITS,
        "timeout": "20s",
        "presentation.timing": True,
    }
    if trace:
        body.update({"trace.level": 1, "trace.explainLevel": 1})
    d = app().query(body=body).json
    if "errors" in d.get("root", {}):
        raise RuntimeError(d["root"]["errors"])
    out = {"querytime_ms": d["timing"]["querytime"] * 1000, "total": d["root"]["fields"]["totalCount"]}
    if trace:
        nn = next(walk(d, lambda o: o.get("[type]") == "search::queryeval::NearestNeighborBlueprint"))
        stats = next(walk(d, lambda o: o.get("tag") == "query_execution_stats"))["stats"]
        out.update(algorithm=nn["algorithm"], strict=nn["strict"], filter_hits=nn["global_filter"]["hits"],
                   distances=stats.get("exact_nns_distances_computed", 0))
    return out


CASES = [
    ("rerank_uses_nn", 0, 4),
    ("rerank_uses_nn", 4, 4),
    ("rerank_uses_nn", 16, 4),
    ("rerank_uses_nn", 100, 4),
    ("rerank_uses_nn", 1000, 4),
    ("rerank_uses_nn", 100, 2),
    ("rerank_uses_nn", 100, 1),
    ("rerank_ignores_nn", 0, 4),
    ("rerank_ignores_nn", 100, 4),
]


def bench():
    reps = int(os.environ.get("REPS", "9"))
    for _ in range(3):
        run(*CASES[3], trace=False)
    rows = []
    print(f"| profile | rerankCount | threads | algorithm | filter hits | exact distances | ÷ filter hits | querytime p50 |")
    print("|---|---:|---:|---|---:|---:|---:|---:|")
    for profile, rc, th in CASES:
        traced = run(profile, rc, th, trace=True)
        times = [run(profile, rc, th, trace=False)["querytime_ms"] for _ in range(reps)]
        row = {"profile": profile, "rerank_count": rc, "threads": th, **traced,
               "ratio": traced["distances"] / traced["filter_hits"], "querytime_p50_ms": statistics.median(times)}
        rows.append(row)
        print(f"| {profile} | {rc} | {th} | {row['algorithm']} | {row['filter_hits']:,} | {row['distances']:,} | "
              f"{row['ratio']:.2f}× | {row['querytime_p50_ms']:.0f} ms |", flush=True)
    (ROOT / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    {"deploy": deploy, "feed": feed, "bench": bench}[sys.argv[1]]()
