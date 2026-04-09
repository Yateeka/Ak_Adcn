"""
Harmony-style pipeline with:
  1. Adaptive cost-model partitioning  (the key gap vs. the paper)
  2. Recall@K measurement against brute-force ground truth
  3. Per-partition load monitoring + live rebalance trigger
  4. All original pipeline logic preserved
"""

from pathlib import Path
import argparse
import heapq
import time
import numpy as np
from sklearn.cluster import KMeans

DATA_DIR = Path("data/processed")
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────
# Cost-model constants (paper §4.2.1)
# Tune these to match your hardware; defaults mimic the paper's
# example where dim-comm >> vec-comm.
# ──────────────────────────────────────────────────────────────
DEFAULT_C_DIM_COMP  = 20.0   # ms per dimension block per query
DEFAULT_C_DIM_COMM  = 30.0   # ms per dimension block per query
DEFAULT_C_VEC_COMP  = 15.0   # ms per vector shard per query
DEFAULT_C_VEC_COMM  =  1.0   # ms per vector shard per query
DEFAULT_ALPHA       =  0.5   # imbalance penalty weight
REBALANCE_THRESHOLD =  1.5   # trigger rebalance when max/avg load > this


# ══════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════

def load_data(query_set: str):
    base = np.load(DATA_DIR / "base_vectors.npy")
    if query_set == "uniform":
        queries = np.load(DATA_DIR / "queries_uniform.npy")
    elif query_set == "skewed":
        queries = np.load(DATA_DIR / "queries_skewed.npy")
    else:
        raise ValueError("query_set must be 'uniform' or 'skewed'")
    return base.astype(np.float32), queries.astype(np.float32)


# ══════════════════════════════════════════════════════════════
# Recall helper
# ══════════════════════════════════════════════════════════════

def compute_ground_truth(queries: np.ndarray, base: np.ndarray, top_k: int) -> list[list[int]]:
    """Brute-force exact top-K for recall evaluation."""
    ground_truth = []
    for q in queries:
        scores = base @ q
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        ground_truth.append(idx.tolist())
    return ground_truth


def recall_at_k(approx_results: list, ground_truth: list[list[int]], top_k: int) -> float:
    """Mean recall@K across all queries."""
    recalls = []
    for approx, exact in zip(approx_results, ground_truth):
        approx_ids = set(idx for _, idx, *_ in approx[:top_k])
        exact_ids  = set(exact[:top_k])
        recalls.append(len(approx_ids & exact_ids) / max(len(exact_ids), 1))
    return float(np.mean(recalls))


# ══════════════════════════════════════════════════════════════
# Cost model  (paper §4.2.1)
# ══════════════════════════════════════════════════════════════

def estimate_cost(
    num_vec_shards: int,
    num_dim_blocks: int,
    routing_loads: np.ndarray,
    n_queries: int,
    c_dim_comp: float = DEFAULT_C_DIM_COMP,
    c_dim_comm: float = DEFAULT_C_DIM_COMM,
    c_vec_comp: float = DEFAULT_C_VEC_COMP,
    c_vec_comm: float = DEFAULT_C_VEC_COMM,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """
    C(π, Q) = Σ_q C_q(π)  +  α · I(π)

    Per-query cost:
      C_q = num_dim_blocks*(c_dim_comp + c_dim_comm)
          + num_vec_shards*(c_vec_comp + c_vec_comm)

    Imbalance I(π) = std-dev of node loads (paper eq.).
    Node loads are approximated from the observed routing_loads.
    """
    per_query = (num_dim_blocks * (c_dim_comp + c_dim_comm) +
                 num_vec_shards * (c_vec_comp + c_vec_comm))
    total_query_cost = n_queries * per_query

    # imbalance from observed routing distribution
    loads = routing_loads.astype(np.float64)
    imbalance = float(np.std(loads)) if loads.sum() > 0 else 0.0

    return total_query_cost + alpha * imbalance


def adaptive_partition_params(
    current_vec_shards: int,
    current_dim_blocks: int,
    routing_loads: np.ndarray,
    n_queries: int,
    min_vec: int = 2,
    max_vec: int = 8,
    min_dim: int = 2,
    max_dim: int = 8,
) -> tuple[int, int]:
    """
    Grid search over (num_vec_shards, num_dim_blocks) and return
    the combination with the lowest estimated cost.
    Only called when load imbalance exceeds REBALANCE_THRESHOLD.
    """
    best_cost   = float("inf")
    best_vec    = current_vec_shards
    best_dim    = current_dim_blocks

    for nv in range(min_vec, max_vec + 1):
        for nd in range(min_dim, max_dim + 1):
            cost = estimate_cost(nv, nd, routing_loads, n_queries)
            if cost < best_cost:
                best_cost = cost
                best_vec  = nv
                best_dim  = nd

    return best_vec, best_dim


# ══════════════════════════════════════════════════════════════
# Core building blocks (unchanged from original)
# ══════════════════════════════════════════════════════════════

def split_dimension_ranges(dim: int, num_blocks: int):
    edges = np.linspace(0, dim, num_blocks + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(num_blocks)]


def build_vector_partitions(base: np.ndarray, num_partitions: int):
    kmeans = KMeans(n_clusters=num_partitions, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(base)
    partitions = []
    for pid in range(num_partitions):
        idx = np.where(labels == pid)[0]
        partitions.append({"partition_id": pid, "indices": idx, "vectors": base[idx]})
    return partitions, kmeans


def route_query_to_partitions(query: np.ndarray, kmeans: KMeans, nprobe: int):
    scores  = kmeans.cluster_centers_ @ query
    nprobe  = min(nprobe, len(scores))
    top_idx = np.argpartition(-scores, nprobe - 1)[:nprobe]
    return top_idx[np.argsort(-scores[top_idx])]


def merge_topk(results, top_k: int):
    if not results:
        return []
    return heapq.nlargest(top_k, results, key=lambda x: x[0])


def prewarm_heap(queries, base, top_k, warmup_queries, warmup_vectors):
    rng     = np.random.default_rng(RANDOM_STATE)
    q_idx   = rng.choice(len(queries), size=min(warmup_queries, len(queries)), replace=False)
    v_idx   = rng.choice(len(base),    size=min(warmup_vectors, len(base)),    replace=False)
    sq      = queries[q_idx]
    sv      = base[v_idx]
    heap    = []
    for q in sq:
        scores  = sv @ q
        local_k = min(top_k, len(scores))
        for i in np.argpartition(-scores, local_k - 1)[:local_k]:
            heapq.heappush(heap, float(scores[i]))
            if len(heap) > top_k:
                heapq.heappop(heap)
    return float(heap[0]) if len(heap) >= top_k else -np.inf


def make_suffix_upper_bounds(vectors, query, dim_ranges):
    num_blocks   = len(dim_ranges)
    block_bounds = []
    for start, end in dim_ranges:
        bound = np.sum(np.abs(vectors[:, start:end]) * np.abs(query[start:end]), axis=1)
        block_bounds.append(bound.astype(np.float32))
    suffix = [None] * num_blocks
    running = np.zeros(len(vectors), dtype=np.float32)
    for i in range(num_blocks - 1, -1, -1):
        running   = running + block_bounds[i]
        suffix[i] = running.copy()
    return suffix


def update_threshold(scores, current, top_k):
    if len(scores) == 0:
        return current
    if len(scores) >= top_k:
        return max(current, float(np.partition(scores, -top_k)[-top_k]))
    return max(current, float(scores.min()))


def dimension_pipeline(query, partition, dim_ranges, top_k, current_threshold):
    vectors = partition["vectors"]
    indices = partition["indices"]
    nb      = len(dim_ranges)
    block_loads    = np.zeros(nb, dtype=np.int64)
    pruned_counts  = np.zeros(nb, dtype=np.int64)

    if len(vectors) == 0:
        return {"results": [], "block_loads": block_loads,
                "pruned_counts": pruned_counts, "final_threshold": current_threshold}

    partial = np.zeros(len(vectors), dtype=np.float32)
    active  = np.ones(len(vectors),  dtype=bool)
    suffix  = make_suffix_upper_bounds(vectors, query, dim_ranges)
    thresh  = current_threshold

    for bid, (start, end) in enumerate(dim_ranges):
        optimistic = partial + suffix[bid]
        keep       = optimistic >= thresh
        pruned_counts[bid] += int((active & ~keep).sum())
        active = active & keep
        if not np.any(active):
            break
        aidx   = np.where(active)[0]
        scores = vectors[aidx, start:end] @ query[start:end]
        partial[aidx] += scores
        block_loads[bid] += len(aidx)
        thresh = update_threshold(partial[aidx], thresh, top_k)

    aidx   = np.where(active)[0]
    ascores = partial[aidx]
    if len(ascores) == 0:
        results = []
    else:
        lk   = min(top_k, len(ascores))
        best = np.argpartition(-ascores, lk - 1)[:lk]
        results = [(float(ascores[p]), int(indices[int(aidx[p])]),
                    partition["partition_id"]) for p in best]
        results = merge_topk(results, top_k)
        if len(results) >= top_k:
            thresh = max(thresh, results[-1][0])

    return {"results": results, "block_loads": block_loads,
            "pruned_counts": pruned_counts, "final_threshold": thresh}


def vector_pipeline(queries_in_partition, partition, dim_ranges, top_k, global_threshold):
    part_results  = {}
    total_loads   = np.zeros(len(dim_ranges), dtype=np.int64)
    total_pruned  = np.zeros(len(dim_ranges), dtype=np.int64)
    running_thresh = global_threshold

    for qid, q in queries_in_partition:
        out = dimension_pipeline(q, partition, dim_ranges, top_k, running_thresh)
        part_results[qid] = out["results"]
        total_loads  += out["block_loads"]
        total_pruned += out["pruned_counts"]
        running_thresh = max(running_thresh, out["final_threshold"])

    return {"partition_results": part_results, "block_loads": total_loads,
            "pruned_counts": total_pruned, "final_threshold": running_thresh}


# ══════════════════════════════════════════════════════════════
# Main query pipeline  — NOW with adaptive rebalancing
# ══════════════════════════════════════════════════════════════

def query_pipeline(
    queries, base,
    num_partitions, nprobe, num_blocks, top_k,
    warmup_queries, warmup_vectors,
    enable_adaptive=True,
    rebalance_every=200,        # check cost model every N queries
):
    # ── initial build ──────────────────────────────────────────
    cur_partitions = num_partitions
    cur_blocks     = num_blocks
    partitions, kmeans = build_vector_partitions(base, cur_partitions)
    dim_ranges         = split_dimension_ranges(base.shape[1], cur_blocks)

    global_threshold = prewarm_heap(queries, base, top_k, warmup_queries, warmup_vectors)

    latencies              = []
    all_results            = []
    routing_loads          = np.zeros(cur_partitions, dtype=np.int64)
    partition_compute_loads= np.zeros(cur_partitions, dtype=np.int64)
    block_compute_loads    = np.zeros(cur_blocks,     dtype=np.int64)
    block_pruned_counts    = np.zeros(cur_blocks,     dtype=np.int64)
    rebalance_events       = []

    for qid, query in enumerate(queries):

        # ── adaptive rebalance check ───────────────────────────
        if enable_adaptive and qid > 0 and qid % rebalance_every == 0:
            imbalance = (routing_loads.max() / routing_loads.mean()
                         if routing_loads.mean() > 0 else 1.0)
            if imbalance > REBALANCE_THRESHOLD:
                new_vec, new_dim = adaptive_partition_params(
                    cur_partitions, cur_blocks, routing_loads, rebalance_every)
                if new_vec != cur_partitions or new_dim != cur_blocks:
                    # rebuild index with new params
                    cur_partitions = new_vec
                    cur_blocks     = new_dim
                    partitions, kmeans = build_vector_partitions(base, cur_partitions)
                    dim_ranges         = split_dimension_ranges(base.shape[1], cur_blocks)
                    # reset load counters with new sizes
                    routing_loads           = np.zeros(cur_partitions, dtype=np.int64)
                    partition_compute_loads = np.zeros(cur_partitions, dtype=np.int64)
                    block_compute_loads     = np.zeros(cur_blocks,     dtype=np.int64)
                    block_pruned_counts     = np.zeros(cur_blocks,     dtype=np.int64)
                    rebalance_events.append({
                        "query_id": qid,
                        "imbalance": imbalance,
                        "new_vec_shards": cur_partitions,
                        "new_dim_blocks": cur_blocks,
                    })

        # ── route + search ─────────────────────────────────────
        t0 = time.perf_counter()
        routed = route_query_to_partitions(query, kmeans, nprobe)
        candidates      = []
        running_thresh  = global_threshold

        for pid in routed:
            pid = int(pid)
            routing_loads[pid] += 1
            part = partitions[pid]
            out  = vector_pipeline([(qid, query)], part, dim_ranges, top_k, running_thresh)
            candidates.extend(out["partition_results"].get(qid, []))
            partition_compute_loads[pid] += int(out["block_loads"].sum())
            # accumulate only if arrays match current size
            if len(out["block_loads"]) == len(block_compute_loads):
                block_compute_loads  += out["block_loads"]
                block_pruned_counts  += out["pruned_counts"]
            merged = merge_topk(candidates, top_k)
            if len(merged) >= top_k:
                running_thresh = max(running_thresh, float(merged[-1][0]))
            running_thresh = max(running_thresh, out["final_threshold"])

        final = merge_topk(candidates, top_k)
        latencies.append(time.perf_counter() - t0)
        all_results.append(final)
        global_threshold = max(global_threshold, running_thresh)

    return {
        "latencies":               np.array(latencies),
        "results":                 all_results,
        "routing_loads":           routing_loads,
        "partition_compute_loads": partition_compute_loads,
        "block_compute_loads":     block_compute_loads,
        "block_pruned_counts":     block_pruned_counts,
        "initial_threshold":       global_threshold,
        "rebalance_events":        rebalance_events,
        "final_num_partitions":    cur_partitions,
        "final_num_blocks":        cur_blocks,
    }


# ══════════════════════════════════════════════════════════════
# Summarise
# ══════════════════════════════════════════════════════════════

def summarize(results: dict, recall: float | None = None):
    lat   = results["latencies"]
    qps   = len(lat) / lat.sum() if lat.sum() > 0 else 0.0
    rl    = results["routing_loads"].astype(float)
    pc    = results["partition_compute_loads"].astype(float)
    bc    = results["block_compute_loads"].astype(float)

    print("\n── Harmony Pipeline Results ──────────────────────────")
    print(f"Queries              : {len(lat)}")
    print(f"Mean latency (ms)    : {lat.mean()*1000:.3f}")
    print(f"P95 latency  (ms)    : {np.percentile(lat,95)*1000:.3f}")
    print(f"P99 latency  (ms)    : {np.percentile(lat,99)*1000:.3f}")
    print(f"Throughput   (QPS)   : {qps:.1f}")
    if recall is not None:
        print(f"Recall@K             : {recall:.4f}")
    print(f"Routing imbalance    : {rl.max()/rl.mean():.4f}" if rl.mean() > 0 else "Routing imbalance: N/A")
    print(f"Partition imbalance  : {pc.max()/pc.mean():.4f}" if pc.mean() > 0 else "Partition imbalance: N/A")
    print(f"Block imbalance      : {bc.max()/bc.mean():.4f}" if bc.mean() > 0 else "Block imbalance: N/A")
    print(f"Final #partitions    : {results['final_num_partitions']}")
    print(f"Final #dim blocks    : {results['final_num_blocks']}")
    if results["rebalance_events"]:
        print(f"Rebalance events     : {len(results['rebalance_events'])}")
        for ev in results["rebalance_events"]:
            print(f"  query {ev['query_id']:5d}: imbalance={ev['imbalance']:.2f} "
                  f"→ {ev['new_vec_shards']}vec × {ev['new_dim_blocks']}dim")
    else:
        print("Rebalance events     : 0 (load stayed balanced)")


def summarize_results(latencies, loads):
    """Compact dict for run_experiments.py compatibility."""
    lat = np.array(latencies)
    loads = np.asarray(loads, dtype=float)
    return {
        "mean":      lat.mean() * 1000,
        "p95":       np.percentile(lat, 95) * 1000,
        "p99":       np.percentile(lat, 99) * 1000,
        "qps":       len(lat) / lat.sum(),
        "imbalance": loads.max() / loads.mean() if loads.mean() > 0 else 1.0,
    }


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query-set",       choices=["uniform", "skewed"], default="uniform")
    p.add_argument("--num-partitions",  type=int,   default=4)
    p.add_argument("--nprobe",          type=int,   default=2)
    p.add_argument("--num-blocks",      type=int,   default=4)
    p.add_argument("--top-k",           type=int,   default=10)
    p.add_argument("--warmup-queries",  type=int,   default=32)
    p.add_argument("--warmup-vectors",  type=int,   default=1000)
    p.add_argument("--no-adaptive",     action="store_true",
                   help="Disable adaptive repartitioning (fixed params)")
    p.add_argument("--rebalance-every", type=int,   default=200)
    p.add_argument("--compute-recall",  action="store_true",
                   help="Compute recall@K against brute-force ground truth")
    return p.parse_args()


def main():
    args  = parse_args()
    base, queries = load_data(args.query_set)

    results = query_pipeline(
        queries=queries, base=base,
        num_partitions=args.num_partitions,
        nprobe=args.nprobe,
        num_blocks=args.num_blocks,
        top_k=args.top_k,
        warmup_queries=args.warmup_queries,
        warmup_vectors=args.warmup_vectors,
        enable_adaptive=not args.no_adaptive,
        rebalance_every=args.rebalance_every,
    )

    recall = None
    if args.compute_recall:
        print("Computing ground truth for recall evaluation…")
        gt     = compute_ground_truth(queries, base, args.top_k)
        recall = recall_at_k(results["results"], gt, args.top_k)

    summarize(results, recall)


if __name__ == "__main__":
    main()