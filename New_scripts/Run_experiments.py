"""
run_experiments.py  — full evaluation suite

Compares:
  1. Vector baseline   (cluster-only partitioning)
  2. Dimension baseline (dimension-split only)
  3. Harmony fixed     (hybrid, no adaptive rebalancing)
  4. Harmony adaptive  (hybrid + cost-model rebalancing)  ← NEW

Reports latency, QPS, load imbalance, and Recall@K for each.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vector_baseline    import run_cluster_partition, load_data
from dimension_baseline import run_dimension_baseline
from harmony_pipeline   import (
    query_pipeline,
    compute_ground_truth,
    recall_at_k,
)


# ──────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────

def compute_metrics(latencies, loads, results=None, ground_truth=None, top_k=10):
    lat    = np.array(latencies)
    loads  = np.asarray(loads, dtype=float)
    metrics = {
        "mean":      lat.mean() * 1000,
        "p95":       np.percentile(lat, 95) * 1000,
        "p99":       np.percentile(lat, 99) * 1000,
        "qps":       len(lat) / lat.sum(),
        "imbalance": loads.max() / loads.mean() if loads.mean() > 0 else 1.0,
        "recall":    None,
    }
    if results is not None and ground_truth is not None:
        metrics["recall"] = recall_at_k(results, ground_truth, top_k)
    return metrics


# ──────────────────────────────────────────────────────────────
# Run everything
# ──────────────────────────────────────────────────────────────

def run_all(query_set="skewed", top_k=10):
    print(f"\nLoading data  (query_set={query_set}) …")
    base, queries = load_data(query_set)

    print("Computing brute-force ground truth for Recall@K …")
    ground_truth = compute_ground_truth(queries, base, top_k)

    # ── 1. Vector baseline ────────────────────────────────────
    print("\nRunning Vector baseline …")
    vec_out = run_cluster_partition(base, queries, num_shards=4, top_k=top_k, nprobe=2)
    # convert (score, global_idx, shard_id) → (score, global_idx, shard_id)
    vec_metrics = compute_metrics(
        vec_out["latencies"], vec_out["shard_loads"],
        vec_out["results"], ground_truth, top_k,
    )

    # ── 2. Dimension baseline ─────────────────────────────────
    print("Running Dimension baseline …")
    dim_out = run_dimension_baseline(base, queries, top_k=top_k, num_blocks=4)
    # dimension results are (score, idx) tuples — wrap to match recall helper
    dim_results_wrapped = [[(s, i, -1) for s, i in r] for r in dim_out["results"]]
    dim_metrics = compute_metrics(
        dim_out["latencies"], dim_out["block_loads"],
        dim_results_wrapped, ground_truth, top_k,
    )

    # ── 3. Harmony fixed (no adaptive) ────────────────────────
    print("Running Harmony fixed (no adaptive rebalancing) …")
    harm_fixed_out = query_pipeline(
        queries=queries, base=base,
        num_partitions=4, nprobe=2, num_blocks=4, top_k=top_k,
        warmup_queries=32, warmup_vectors=1000,
        enable_adaptive=False,
    )
    harm_fixed_metrics = compute_metrics(
        harm_fixed_out["latencies"],
        harm_fixed_out["partition_compute_loads"],
        harm_fixed_out["results"], ground_truth, top_k,
    )

    # ── 4. Harmony adaptive ───────────────────────────────────
    print("Running Harmony adaptive (cost-model rebalancing) …")
    harm_adapt_out = query_pipeline(
        queries=queries, base=base,
        num_partitions=4, nprobe=2, num_blocks=4, top_k=top_k,
        warmup_queries=32, warmup_vectors=1000,
        enable_adaptive=True,
        rebalance_every=200,
    )
    harm_adapt_metrics = compute_metrics(
        harm_adapt_out["latencies"],
        harm_adapt_out["partition_compute_loads"],
        harm_adapt_out["results"], ground_truth, top_k,
    )

    if harm_adapt_out["rebalance_events"]:
        print(f"  → {len(harm_adapt_out['rebalance_events'])} rebalance event(s) triggered")
        for ev in harm_adapt_out["rebalance_events"]:
            print(f"     query {ev['query_id']}: imbalance={ev['imbalance']:.2f} "
                  f"→ {ev['new_vec_shards']}vec × {ev['new_dim_blocks']}dim")
    else:
        print("  → No rebalancing needed (workload stayed balanced)")

    return {
        "Vector":          vec_metrics,
        "Dimension":       dim_metrics,
        "Harmony-fixed":   harm_fixed_metrics,
        "Harmony-adaptive":harm_adapt_metrics,
    }


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

COLORS = {
    "Vector":           "#888780",
    "Dimension":        "#378ADD",
    "Harmony-fixed":    "#1D9E75",
    "Harmony-adaptive": "#D85A30",
}


def _bar(ax, systems, values, title, ylabel, color_map):
    bars = ax.bar(systems, values, color=[color_map[s] for s in systems], width=0.5)
    ax.set_title(title, fontsize=11, fontweight="normal")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot(results):
    systems = list(results.keys())

    p99       = [results[s]["p99"]       for s in systems]
    qps       = [results[s]["qps"]       for s in systems]
    imbalance = [results[s]["imbalance"] for s in systems]
    recall    = [results[s]["recall"] or 0.0 for s in systems]

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Harmony Evaluation — Skewed Workload (CIC-IDS2017)", fontsize=13)

    _bar(axes[0, 0], systems, p99,       "P99 latency (ms)",    "ms",          COLORS)
    _bar(axes[0, 1], systems, qps,       "Throughput (QPS)",    "queries/sec", COLORS)
    _bar(axes[1, 0], systems, imbalance, "Load imbalance ratio","max / avg",   COLORS)
    _bar(axes[1, 1], systems, recall,    "Recall@K",            "recall",      COLORS)

    # imbalance vs latency scatter
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for s in systems:
        ax2.scatter(results[s]["imbalance"], results[s]["p99"],
                    color=COLORS[s], s=80, zorder=3)
        ax2.annotate(s, (results[s]["imbalance"], results[s]["p99"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax2.set_xlabel("Load imbalance (max/avg)")
    ax2.set_ylabel("P99 latency (ms)")
    ax2.set_title("Imbalance vs P99 latency")
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    fig2.savefig("imbalance_vs_latency.png", dpi=150)

    fig.tight_layout()
    fig.savefig("harmony_evaluation.png", dpi=150)

    # individual saves for backward compat
    for name, vals, fname in [
        ("P99 latency (ms)",   p99,       "p99_latency.png"),
        ("Throughput (QPS)",   qps,       "throughput.png"),
        ("Load imbalance",     imbalance, "imbalance.png"),
        ("Recall@K",           recall,    "recall.png"),
    ]:
        f, a = plt.subplots(figsize=(6, 4))
        _bar(a, systems, vals, name, name, COLORS)
        f.tight_layout()
        f.savefig(fname, dpi=150)
        plt.close(f)

    print("\nPlots saved: harmony_evaluation.png, p99_latency.png, "
          "throughput.png, imbalance.png, recall.png, imbalance_vs_latency.png")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all(query_set="skewed", top_k=10)

    print("\n═══ FINAL RESULTS ═══════════════════════════════════")
    header = f"{'System':<20} {'mean(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9} "
    header += f"{'QPS':>8} {'imbalance':>10} {'recall':>8}"
    print(header)
    print("─" * len(header))
    for name, m in results.items():
        rec = f"{m['recall']:.4f}" if m["recall"] is not None else "  N/A "
        print(f"{name:<20} {m['mean']:>9.2f} {m['p95']:>9.2f} {m['p99']:>9.2f} "
              f"{m['qps']:>8.1f} {m['imbalance']:>10.4f} {rec:>8}")

    plot(results)