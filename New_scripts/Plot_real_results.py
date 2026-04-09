"""
plot_real_results.py

Replace the placeholder dicts below with printed output from run_experiments.py,
then run this script to regenerate all charts.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paste your actual numbers here ───────────────────────────
results = {
    "Vector":           {"p99": 40,  "qps": 400, "imbalance": 2.50, "recall": 0.82},
    "Dimension":        {"p99": 20,  "qps": 520, "imbalance": 1.20, "recall": 0.89},
    "Harmony-fixed":    {"p99": 15,  "qps": 680, "imbalance": 1.05, "recall": 0.91},
    "Harmony-adaptive": {"p99": 12,  "qps": 730, "imbalance": 1.02, "recall": 0.91},
}

COLORS = {
    "Vector":           "#888780",
    "Dimension":        "#378ADD",
    "Harmony-fixed":    "#1D9E75",
    "Harmony-adaptive": "#D85A30",
}

systems = list(results.keys())
colors  = [COLORS[s] for s in systems]


def bar_chart(values, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(systems, values, color=colors, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


bar_chart([results[s]["p99"]       for s in systems], "P99 latency (skewed workload)", "ms",        "p99_latency.png")
bar_chart([results[s]["qps"]       for s in systems], "Throughput",                    "QPS",       "throughput.png")
bar_chart([results[s]["imbalance"] for s in systems], "Load imbalance ratio",          "max / avg", "imbalance.png")
bar_chart([results[s]["recall"]    for s in systems], "Recall@K",                      "recall",    "recall.png")

# scatter: imbalance vs latency
fig, ax = plt.subplots(figsize=(6, 5))
for s in systems:
    ax.scatter(results[s]["imbalance"], results[s]["p99"], color=COLORS[s], s=80, zorder=3)
    ax.annotate(s, (results[s]["imbalance"], results[s]["p99"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("Load imbalance (max / avg)")
ax.set_ylabel("P99 latency (ms)")
ax.set_title("Imbalance vs P99 latency")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("imbalance_vs_latency.png", dpi=150)
plt.close(fig)
print("Saved imbalance_vs_latency.png")