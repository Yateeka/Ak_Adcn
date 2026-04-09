"""
run_all.py  —  one script to run the entire Harmony evaluation pipeline

Steps:
  1. Check data exists (prompt to run preprocess if not)
  2. Vector baseline
  3. Dimension baseline
  4. Harmony fixed (no adaptive)
  5. Harmony adaptive (cost-model rebalancing)
  6. Recall@K for all four
  7. Print comparison table
  8. Save all plots

Usage:
  python run_all.py                        # skewed workload, top-10
  python run_all.py --query-set uniform
  python run_all.py --top-k 20
  python run_all.py --no-recall            # skip brute-force ground truth (faster)
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── path setup ────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from vector_baseline    import run_cluster_partition, load_data
from dimension_baseline import run_dimension_baseline
from harmony_pipeline   import (
    query_pipeline,
    compute_ground_truth,
    recall_at_k,
)

DATA_DIR = Path("data/processed")

COLORS = {
    "Vector":            "#888780",
    "Dimension":         "#378ADD",
    "Harmony-fixed":     "#1D9E75",
    "Harmony-adaptive":  "#D85A30",
}


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def banner(msg: str):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


def check_data():
    required = [
        DATA_DIR / "base_vectors.npy",
        DATA_DIR / "queries_uniform.npy",
        DATA_DIR / "queries_skewed.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n[ERROR] Preprocessed data not found:")
        for p in missing:
            print(f"  {p}")
        print("\nRun this first:\n  python preprocess_cicids2017.py\n")
        sys.exit(1)


def compute_metrics(latencies, loads, results=None, ground_truth=None, top_k=10):
    lat   = np.array(latencies)
    loads = np.asarray(loads, dtype=float)
    m = {
        "mean":      lat.mean() * 1000,
        "p95":       np.percentile(lat, 95) * 1000,
        "p99":       np.percentile(lat, 99) * 1000,
        "qps":       len(lat) / lat.sum() if lat.sum() > 0 else 0.0,
        "imbalance": loads.max() / loads.mean() if loads.mean() > 0 else 1.0,
        "recall":    None,
    }
    if results is not None and ground_truth is not None:
        m["recall"] = recall_at_k(results, ground_truth, top_k)
    return m


# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════

def _bar(ax, systems, values, title, ylabel):
    bars = ax.bar(systems, values, color=[COLORS[s] for s in systems], width=0.5)
    for bar, v in zip(bars, values):
        label = f"{v:.3f}" if max(values) < 2 else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                label, ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def save_plots(results: dict, query_set: str):
    systems = list(results.keys())

    p99       = [results[s]["p99"]              for s in systems]
    qps       = [results[s]["qps"]              for s in systems]
    imbalance = [results[s]["imbalance"]        for s in systems]
    recall    = [results[s]["recall"] or 0.0    for s in systems]

    # ── 2×2 summary figure ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Harmony Evaluation  —  {query_set} workload  (CIC-IDS2017)",
                 fontsize=12, fontweight="normal")
    _bar(axes[0, 0], systems, p99,       "P99 latency",        "ms")
    _bar(axes[0, 1], systems, qps,       "Throughput",         "QPS")
    _bar(axes[1, 0], systems, imbalance, "Load imbalance",     "max / avg load")
    _bar(axes[1, 1], systems, recall,    "Recall@K",           "recall")
    fig.tight_layout()
    fig.savefig("harmony_evaluation.png", dpi=150)
    plt.close(fig)

    # ── individual PNGs for backward compat ───────────────────
    specs = [
        (p99,       "P99 latency (ms)",    "ms",        "p99_latency.png"),
        (qps,       "Throughput (QPS)",    "QPS",       "throughput.png"),
        (imbalance, "Load imbalance ratio","max / avg", "imbalance.png"),
        (recall,    "Recall@K",            "recall",    "recall.png"),
    ]
    for vals, title, ylabel, fname in specs:
        f, a = plt.subplots(figsize=(6, 4))
        _bar(a, systems, vals, title, ylabel)
        f.tight_layout()
        f.savefig(fname, dpi=150)
        plt.close(f)

    # ── imbalance vs latency scatter ──────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for s in systems:
        ax2.scatter(results[s]["imbalance"], results[s]["p99"],
                    color=COLORS[s], s=90, zorder=3)
        ax2.annotate(s, (results[s]["imbalance"], results[s]["p99"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax2.set_xlabel("Load imbalance (max / avg)")
    ax2.set_ylabel("P99 latency (ms)")
    ax2.set_title("Imbalance vs P99 latency")
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    fig2.savefig("imbalance_vs_latency.png", dpi=150)
    plt.close(fig2)

    saved = ["harmony_evaluation.png", "p99_latency.png", "throughput.png",
             "imbalance.png", "recall.png", "imbalance_vs_latency.png"]
    print("\nPlots saved:")
    for f in saved:
        print(f"  {f}")


# ══════════════════════════════════════════════════════════════
# Print results table
# ══════════════════════════════════════════════════════════════

def print_table(results: dict):
    banner("FINAL RESULTS")
    hdr = (f"{'System':<22} {'mean(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9}"
           f" {'QPS':>8} {'imbalance':>10} {'recall':>8}")
    print(hdr)
    print("─" * len(hdr))
    for name, m in results.items():
        rec = f"{m['recall']:.4f}" if m["recall"] is not None else "    N/A"
        print(f"{name:<22} {m['mean']:>9.2f} {m['p95']:>9.2f} {m['p99']:>9.2f}"
              f" {m['qps']:>8.1f} {m['imbalance']:>10.4f} {rec:>8}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full Harmony evaluation pipeline end-to-end.")
    p.add_argument("--query-set",      choices=["uniform", "skewed"], default="skewed")
    p.add_argument("--top-k",          type=int, default=10)
    p.add_argument("--num-shards",     type=int, default=4,
                   help="Vector shards for baselines")
    p.add_argument("--num-blocks",     type=int, default=4,
                   help="Dimension blocks for baselines and Harmony initial config")
    p.add_argument("--nprobe",         type=int, default=2)
    p.add_argument("--rebalance-every",type=int, default=200)
    p.add_argument("--no-recall",      action="store_true",
                   help="Skip Recall@K (faster, no brute-force ground truth)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 0. sanity check ───────────────────────────────────────
    check_data()

    banner(f"Loading data  (query_set={args.query_set})")
    base, queries = load_data(args.query_set)
    print(f"  Base vectors : {base.shape}")
    print(f"  Queries      : {queries.shape}")

    ground_truth = None
    if not args.no_recall:
        banner("Computing brute-force ground truth for Recall@K")
        t0 = time.perf_counter()
        ground_truth = compute_ground_truth(queries, base, args.top_k)
        print(f"  Done in {time.perf_counter()-t0:.1f}s")

    results = {}

    # ── 1. Vector baseline ────────────────────────────────────
    banner("Step 1 / 4  —  Vector baseline")
    t0 = time.perf_counter()
    vec_out = run_cluster_partition(
        base, queries, num_shards=args.num_shards,
        top_k=args.top_k, nprobe=args.nprobe)
    print(f"  Finished in {time.perf_counter()-t0:.1f}s")
    results["Vector"] = compute_metrics(
        vec_out["latencies"], vec_out["shard_loads"],
        vec_out["results"], ground_truth, args.top_k)

    # ── 2. Dimension baseline ─────────────────────────────────
    banner("Step 2 / 4  —  Dimension baseline")
    t0 = time.perf_counter()
    dim_out = run_dimension_baseline(
        base, queries, top_k=args.top_k, num_blocks=args.num_blocks)
    print(f"  Finished in {time.perf_counter()-t0:.1f}s")
    dim_results_wrapped = [[(s, i, -1) for s, i in r] for r in dim_out["results"]]
    results["Dimension"] = compute_metrics(
        dim_out["latencies"], dim_out["block_loads"],
        dim_results_wrapped, ground_truth, args.top_k)

    # ── 3. Harmony fixed ──────────────────────────────────────
    banner("Step 3 / 4  —  Harmony fixed  (no adaptive rebalancing)")
    t0 = time.perf_counter()
    harm_fixed_out = query_pipeline(
        queries=queries, base=base,
        num_partitions=args.num_shards, nprobe=args.nprobe,
        num_blocks=args.num_blocks, top_k=args.top_k,
        warmup_queries=32, warmup_vectors=1000,
        enable_adaptive=False,
    )
    print(f"  Finished in {time.perf_counter()-t0:.1f}s")
    results["Harmony-fixed"] = compute_metrics(
        harm_fixed_out["latencies"],
        harm_fixed_out["partition_compute_loads"],
        harm_fixed_out["results"], ground_truth, args.top_k)

    # ── 4. Harmony adaptive ───────────────────────────────────
    banner("Step 4 / 4  —  Harmony adaptive  (cost-model rebalancing)")
    t0 = time.perf_counter()
    harm_adapt_out = query_pipeline(
        queries=queries, base=base,
        num_partitions=args.num_shards, nprobe=args.nprobe,
        num_blocks=args.num_blocks, top_k=args.top_k,
        warmup_queries=32, warmup_vectors=1000,
        enable_adaptive=True,
        rebalance_every=args.rebalance_every,
    )
    print(f"  Finished in {time.perf_counter()-t0:.1f}s")
    if harm_adapt_out["rebalance_events"]:
        print(f"  Rebalance events: {len(harm_adapt_out['rebalance_events'])}")
        for ev in harm_adapt_out["rebalance_events"]:
            print(f"    query {ev['query_id']:5d}: imbalance={ev['imbalance']:.2f}"
                  f" → {ev['new_vec_shards']}vec × {ev['new_dim_blocks']}dim")
    else:
        print("  No rebalancing triggered (load stayed balanced)")
    results["Harmony-adaptive"] = compute_metrics(
        harm_adapt_out["latencies"],
        harm_adapt_out["partition_compute_loads"],
        harm_adapt_out["results"], ground_truth, args.top_k)

    # ── 5. Report + plots ─────────────────────────────────────
    print_table(results)
    save_plots(results, args.query_set)

    banner("Done")


if __name__ == "__main__":
    main()