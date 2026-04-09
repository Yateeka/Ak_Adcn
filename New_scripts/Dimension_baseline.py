"""
Dimension-split baseline for Harmony-style evaluation.

Changes vs. original:
  - dimension_search now returns (score, idx) tuples consistently
  - summarize_results helper aligns with run_experiments.py
  - no other logic changed
"""

from pathlib import Path
import time
import argparse
import heapq
import numpy as np

DATA_DIR = Path("data/processed")


def load_data(query_set: str):
    base = np.load(DATA_DIR / "base_vectors.npy")
    if query_set == "uniform":
        queries = np.load(DATA_DIR / "queries_uniform.npy")
    elif query_set == "skewed":
        queries = np.load(DATA_DIR / "queries_skewed.npy")
    else:
        raise ValueError("query_set must be 'uniform' or 'skewed'")
    return base.astype(np.float32), queries.astype(np.float32)


def split_dimension_ranges(dim: int, num_blocks: int):
    edges = np.linspace(0, dim, num_blocks + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(num_blocks)]


def full_search(query, base, top_k):
    scores = base @ query
    k = min(top_k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    results = [(float(scores[i]), int(i)) for i in idx]
    return heapq.nlargest(top_k, results, key=lambda x: x[0])


def dimension_search(query, base, top_k, num_blocks):
    dim_ranges    = split_dimension_ranges(base.shape[1], num_blocks)
    partial       = np.zeros(len(base), dtype=np.float32)
    active        = np.ones(len(base),  dtype=bool)
    block_loads   = np.zeros(num_blocks, dtype=np.int64)
    pruned_counts = np.zeros(num_blocks, dtype=np.int64)

    # warm-up: first block, all vectors
    s0, e0 = dim_ranges[0]
    partial += base[:, s0:e0] @ query[s0:e0]
    block_loads[0] += len(base)

    k       = min(top_k, len(base))
    top_idx = np.argpartition(-partial, k - 1)[:k]
    threshold = float(partial[top_idx].min())

    # suffix upper bounds
    suffix_max = []
    for start, end in dim_ranges:
        bm = np.sum(np.abs(base[:, start:end]) * np.abs(query[start:end]), axis=1)
        suffix_max.append(bm)

    suffix_remaining = [None] * num_blocks
    running = np.zeros(len(base), dtype=np.float32)
    for i in range(num_blocks - 1, -1, -1):
        running = running + suffix_max[i]
        suffix_remaining[i] = running.copy()

    for bid in range(1, num_blocks):
        start, end = dim_ranges[bid]
        optimistic = partial + suffix_remaining[bid]
        keep = optimistic >= threshold
        pruned_now = active & ~keep
        pruned_counts[bid] += int(pruned_now.sum())
        active = active & keep
        if not np.any(active):
            break
        aidx   = np.where(active)[0]
        scores = base[aidx, start:end] @ query[start:end]
        partial[aidx] += scores
        block_loads[bid] += len(aidx)
        as_ = partial[aidx]
        threshold = float(np.partition(as_, -k)[-k]) if len(as_) >= k else float(as_.min())

    final_idx    = np.where(active)[0]
    final_scores = partial[final_idx]
    results = [(float(final_scores[i]), int(final_idx[i]))
               for i in range(len(final_idx))]
    results = heapq.nlargest(top_k, results, key=lambda x: x[0])

    return {"results": results, "block_loads": block_loads, "pruned_counts": pruned_counts}


def run_dimension_baseline(base, queries, top_k, num_blocks):
    latencies        = []
    all_results      = []
    total_block_loads = np.zeros(num_blocks, dtype=np.int64)
    total_pruned      = np.zeros(num_blocks, dtype=np.int64)

    for query in queries:
        t0  = time.perf_counter()
        out = dimension_search(query, base, top_k, num_blocks)
        latencies.append(time.perf_counter() - t0)
        all_results.append(out["results"])
        total_block_loads += out["block_loads"]
        total_pruned      += out["pruned_counts"]

    return {
        "latencies":    np.array(latencies),
        "results":      all_results,
        "block_loads":  total_block_loads,
        "pruned_counts":total_pruned,
    }


def run_full_baseline(base, queries, top_k):
    latencies   = []
    all_results = []
    for query in queries:
        t0 = time.perf_counter()
        all_results.append(full_search(query, base, top_k))
        latencies.append(time.perf_counter() - t0)
    return {"latencies": np.array(latencies), "results": all_results}


def summarize_results(latencies, loads):
    lat   = np.array(latencies)
    loads = np.asarray(loads, dtype=float)
    return {
        "mean":      lat.mean() * 1000,
        "p95":       np.percentile(lat, 95) * 1000,
        "p99":       np.percentile(lat, 99) * 1000,
        "qps":       len(lat) / lat.sum(),
        "imbalance": loads.max() / loads.mean() if loads.mean() > 0 else 1.0,
    }


def summarize_full(name, run_output):
    lat = run_output["latencies"]
    qps = len(lat) / lat.sum() if lat.sum() > 0 else 0.0
    print(f"\n{name}")
    print(f"  Mean latency (ms): {lat.mean()*1000:.3f}")
    print(f"  P95  latency (ms): {np.percentile(lat,95)*1000:.3f}")
    print(f"  P99  latency (ms): {np.percentile(lat,99)*1000:.3f}")
    print(f"  Throughput  (QPS): {qps:.1f}")


def summarize_dimension(name, run_output):
    lat   = run_output["latencies"]
    qps   = len(lat) / lat.sum() if lat.sum() > 0 else 0.0
    loads = run_output["block_loads"].astype(float)
    imb   = loads.max() / loads.mean() if loads.mean() > 0 else 0.0
    print(f"\n{name}")
    print(f"  Mean latency (ms)    : {lat.mean()*1000:.3f}")
    print(f"  P95  latency (ms)    : {np.percentile(lat,95)*1000:.3f}")
    print(f"  P99  latency (ms)    : {np.percentile(lat,99)*1000:.3f}")
    print(f"  Throughput   (QPS)   : {qps:.1f}")
    print(f"  Load imbalance ratio : {imb:.4f}")
    print(f"  Block loads          : {loads.astype(int).tolist()}")
    print(f"  Pruned by block      : {run_output['pruned_counts'].astype(int).tolist()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query-set",  choices=["uniform", "skewed"], default="uniform")
    p.add_argument("--top-k",      type=int, default=10)
    p.add_argument("--num-blocks", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    base, queries = load_data(args.query_set)
    full_out = run_full_baseline(base, queries, args.top_k)
    dim_out  = run_dimension_baseline(base, queries, args.top_k, args.num_blocks)
    summarize_full("Full-vector baseline", full_out)
    summarize_dimension("Dimension-split baseline", dim_out)


if __name__ == "__main__":
    main()