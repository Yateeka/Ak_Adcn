"""
Vector-only baseline for Harmony-style evaluation.
No logic changes — summarize_results now returns a proper dict.
"""

from pathlib import Path
import time
import argparse
import heapq
import numpy as np
from sklearn.cluster import KMeans

DATA_DIR    = Path("data/processed")
RANDOM_STATE = 42


def load_data(query_set: str):
    base = np.load(DATA_DIR / "base_vectors.npy")
    if query_set == "uniform":
        queries = np.load(DATA_DIR / "queries_uniform.npy")
    elif query_set == "skewed":
        queries = np.load(DATA_DIR / "queries_skewed.npy")
    else:
        raise ValueError("query_set must be 'uniform' or 'skewed'")
    return base.astype(np.float32), queries.astype(np.float32)


def build_random_shards(base, num_shards):
    rng     = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(len(base))
    rng.shuffle(indices)
    split   = np.array_split(indices, num_shards)
    return [{"shard_id": i, "indices": idx, "vectors": base[idx]}
            for i, idx in enumerate(split)]


def build_cluster_shards(base, num_shards):
    km     = KMeans(n_clusters=num_shards, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(base)
    shards = []
    for sid in range(num_shards):
        idx = np.where(labels == sid)[0]
        shards.append({"shard_id": sid, "indices": idx, "vectors": base[idx]})
    return shards, km


def shard_search(query, shard, top_k):
    if len(shard["vectors"]) == 0:
        return []
    scores = shard["vectors"] @ query
    k      = min(top_k, len(scores))
    top    = np.argpartition(-scores, k - 1)[:k]
    return [(float(scores[i]), int(shard["indices"][i]), shard["shard_id"]) for i in top]


def merge_results(candidates, top_k):
    return heapq.nlargest(top_k, candidates, key=lambda x: x[0])


def run_random_partition(base, queries, num_shards, top_k):
    shards      = build_random_shards(base, num_shards)
    shard_loads = np.zeros(num_shards, dtype=np.int64)
    latencies, all_results = [], []

    for query in queries:
        t0, candidates = time.perf_counter(), []
        for shard in shards:
            candidates.extend(shard_search(query, shard, top_k))
            shard_loads[shard["shard_id"]] += len(shard["vectors"])
        latencies.append(time.perf_counter() - t0)
        all_results.append(merge_results(candidates, top_k))

    return {"latencies": np.array(latencies), "results": all_results, "shard_loads": shard_loads}


def run_cluster_partition(base, queries, num_shards, top_k, nprobe):
    shards, km  = build_cluster_shards(base, num_shards)
    shard_loads = np.zeros(num_shards, dtype=np.int64)
    latencies, all_results = [], []
    centroid_scores = km.cluster_centers_ @ queries.T

    for i, query in enumerate(queries):
        t0         = time.perf_counter()
        probe_ids  = np.argpartition(-centroid_scores[:, i], nprobe - 1)[:nprobe]
        candidates = []
        for sid in probe_ids:
            shard = shards[int(sid)]
            candidates.extend(shard_search(query, shard, top_k))
            shard_loads[int(sid)] += len(shard["vectors"])
        latencies.append(time.perf_counter() - t0)
        all_results.append(merge_results(candidates, top_k))

    return {"latencies": np.array(latencies), "results": all_results, "shard_loads": shard_loads}


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


def summarize(name, run_output):
    lat   = run_output["latencies"]
    qps   = len(lat) / lat.sum() if lat.sum() > 0 else 0.0
    loads = run_output["shard_loads"].astype(float)
    imb   = loads.max() / loads.mean() if loads.mean() > 0 else 0.0
    print(f"\n{name}")
    print(f"  Mean latency (ms)    : {lat.mean()*1000:.3f}")
    print(f"  P95  latency (ms)    : {np.percentile(lat,95)*1000:.3f}")
    print(f"  P99  latency (ms)    : {np.percentile(lat,99)*1000:.3f}")
    print(f"  Throughput   (QPS)   : {qps:.1f}")
    print(f"  Load imbalance ratio : {imb:.4f}")
    print(f"  Shard loads          : {loads.astype(int).tolist()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query-set",  choices=["uniform", "skewed"], default="uniform")
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--top-k",      type=int, default=10)
    p.add_argument("--nprobe",     type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    base, queries = load_data(args.query_set)
    rand_out    = run_random_partition(base, queries, args.num_shards, args.top_k)
    cluster_out = run_cluster_partition(base, queries, args.num_shards, args.top_k, args.nprobe)
    summarize("Random shard baseline",  rand_out)
    summarize("Cluster shard baseline", cluster_out)


if __name__ == "__main__":
    main()