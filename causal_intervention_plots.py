"""
Plotting script for causal_intervention_results.json (RQ3).

Generates a suite of figures visualising layer-wise ablation impact, cumulative
window ablations, attention-vs-FFN specialisation, distance to the impossible
baseline, and convergence with probing divergence. All plots are saved as both
PNG and PDF in --output_dir (default: causal_plots/).

Example
-------
python causal_intervention_plots.py \
    --results causal_intervention_results.json \
    --output_dir causal_plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style (matches rq1_plot.py / prob_plots.py)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

KIND_COLORS = {
    "zero": "#1f77b4",
    "mean": "#ff7f0e",
    "random": "#2ca02c",
}
SCOPE_LINESTYLES = {
    "full": "-",
    "attn": "--",
    "ffn": ":",
}
SCOPE_LABELS = {
    "full": "Full layer",
    "attn": "Attention only",
    "ffn": "FFN only",
}
KIND_LABELS = {
    "zero": "Zero ablation",
    "mean": "Mean ablation",
    "random": "Random ablation",
}
MODEL_COLORS = {
    "translator": "#d62728",
    "impossible": "#1f77b4",
    "base": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_results(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def layer_array(single: dict, kind: str, scope: str, metric: str) -> np.ndarray:
    """Return per-layer values for a given (kind, scope, metric) combination."""
    block = single.get(f"{kind}_{scope}", {})
    if not block:
        return np.array([])
    layers = sorted(int(k) for k in block.keys())
    return np.array([block[str(l)].get(metric, 0.0) for l in layers], dtype=float)


def n_layers_from(single: dict) -> int:
    block = single.get("zero_full", {})
    return len(block) if block else 12


def save(fig: plt.Figure, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"))
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  saved {name}.png / {name}.pdf")


# ---------------------------------------------------------------------------
# Plots: single-layer ablations
# ---------------------------------------------------------------------------


def plot_layer_profile_metric(
        single: dict, metric: str, ylabel: str, title: str, output_dir: str, name: str
) -> None:
    """Line plot of a metric across layers, one line per (kind, scope)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    n = n_layers_from(single)
    xs = np.arange(n)
    for kind in ("zero", "mean", "random"):
        for scope in ("full", "attn", "ffn"):
            ys = layer_array(single, kind, scope, metric)
            if ys.size == 0:
                continue
            ax.plot(
                xs,
                ys,
                color=KIND_COLORS[kind],
                linestyle=SCOPE_LINESTYLES[scope],
                marker="o",
                markersize=4,
                linewidth=1.6,
                label=f"{KIND_LABELS[kind]} – {SCOPE_LABELS[scope]}",
            )
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8, loc="best", frameon=False)
    save(fig, output_dir, name)


def plot_delta_grid(single: dict, output_dir: str) -> None:
    """3x2 grid: rows = scope (full/attn/ffn), cols = ΔEM and ΔdepF1."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    n = n_layers_from(single)
    xs = np.arange(n)
    metrics = [("delta_em", "Δ Exact Match"), ("delta_dep_f1", "Δ Dep F1")]
    for row, scope in enumerate(("full", "attn", "ffn")):
        for col, (metric, ylab) in enumerate(metrics):
            ax = axes[row, col]
            for kind in ("zero", "mean", "random"):
                ys = layer_array(single, kind, scope, metric)
                if ys.size == 0:
                    continue
                ax.plot(
                    xs,
                    ys,
                    color=KIND_COLORS[kind],
                    marker="o",
                    markersize=4,
                    linewidth=1.6,
                    label=KIND_LABELS[kind],
                )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel(ylab if col == 0 else "")
            ax.set_title(f"{SCOPE_LABELS[scope]} – {ylab}")
            ax.grid(True, alpha=0.3)
            if row == 2:
                ax.set_xlabel("Layer")
                ax.set_xticks(xs)
            if row == 0 and col == 1:
                ax.legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("Single-layer ablation impact (Δ vs intact baseline)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, output_dir, "single_layer_delta_grid")


def plot_delta_heatmap(single: dict, scope: str, output_dir: str) -> None:
    """Heatmap of Δ metrics: rows = (kind, metric), cols = layer."""
    n = n_layers_from(single)
    rows = []
    row_labels = []
    for kind in ("zero", "mean", "random"):
        for metric, mlabel in (("delta_em", "ΔEM"), ("delta_dep_f1", "ΔdepF1")):
            arr = layer_array(single, kind, scope, metric)
            if arr.size == 0:
                continue
            rows.append(arr)
            row_labels.append(f"{KIND_LABELS[kind]} {mlabel}")
    if not rows:
        return
    matrix = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(9, 4 + 0.25 * len(rows)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([str(i) for i in range(n)])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Layer")
    ax.set_title(f"Ablation impact heatmap ({SCOPE_LABELS[scope]})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Δ from intact baseline")
    fig.tight_layout()
    save(fig, output_dir, f"delta_heatmap_{scope}")


def plot_attn_vs_ffn(single: dict, output_dir: str) -> None:
    """Bar chart per layer: attention-only vs FFN-only Δ for zero ablation."""
    n = n_layers_from(single)
    xs = np.arange(n)
    width = 0.4
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, metric, title in zip(
            axes,
            ("delta_em", "delta_dep_f1"),
            ("Δ Exact Match", "Δ Dep F1"),
    ):
        attn = layer_array(single, "zero", "attn", metric)
        ffn = layer_array(single, "zero", "ffn", metric)
        if attn.size == 0 or ffn.size == 0:
            continue
        ax.bar(xs - width / 2, attn, width=width, color="#9467bd", label="Attention only")
        ax.bar(xs + width / 2, ffn, width=width, color="#8c564b", label="FFN only")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.set_title(f"Attention vs FFN ablation – {title}")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, output_dir, "attention_vs_ffn")


def plot_critical_layers(single: dict, output_dir: str) -> None:
    """Bar chart of critical-layer indicator (zero_full)."""
    block = single.get("zero_full", {})
    if not block:
        return
    layers = sorted(int(k) for k in block.keys())
    crit = np.array([block[str(l)].get("critical", 0) for l in layers])
    delta_em = np.array([block[str(l)]["delta_em"] for l in layers])
    delta_f1 = np.array([block[str(l)]["delta_dep_f1"] for l in layers])
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = ["#d62728" if c else "#7f7f7f" for c in crit]
    ax.bar(layers, delta_em, color=bar_colors, label="ΔEM (red = critical)")
    ax.plot(layers, delta_f1, color="#1f77b4", marker="o", label="ΔdepF1")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ from intact baseline")
    ax.set_title("Critical-layer identification (zero, full)")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=False)
    save(fig, output_dir, "critical_layers")


# ---------------------------------------------------------------------------
# Plots: cumulative ablations
# ---------------------------------------------------------------------------


def plot_cumulative_window(cumul: dict, key: str, output_dir: str, name: str, title: str) -> None:
    entries = cumul.get(key, [])
    if not entries:
        return
    xs = np.arange(len(entries))
    delta_em = [e["delta_em"] for e in entries]
    delta_f1 = [e["delta_dep_f1"] for e in entries]
    em = [e["em"] for e in entries]
    f1 = [e["dep_f1"] for e in entries]

    if key == "early":
        labels = [f"0..{e['layers'][-1]}" for e in entries]
    elif key == "late":
        labels = [f"{e['layers'][0]}..{e['layers'][-1]}" for e in entries]
    else:
        labels = [
            f"{e['layers'][0]}..{e['layers'][-1]}" for e in entries
        ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(xs, em, marker="o", color="#d62728", label="EM")
    axes[0].plot(xs, f1, marker="s", color="#1f77b4", label="dep F1")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("Performance")
    axes[0].set_title(f"{title} – absolute")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(xs, delta_em, marker="o", color="#d62728", label="ΔEM")
    axes[1].plot(xs, delta_f1, marker="s", color="#1f77b4", label="ΔdepF1")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("Δ from intact baseline")
    axes[1].set_title(f"{title} – delta")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save(fig, output_dir, name)


def plot_middle_window_grid(cumul: dict, output_dir: str) -> None:
    entries = cumul.get("middle", [])
    if not entries:
        return
    widths = sorted({e.get("width", len(e["layers"])) for e in entries})
    fig, axes = plt.subplots(1, len(widths), figsize=(6 * len(widths), 4.5), sharey=True)
    if len(widths) == 1:
        axes = [axes]
    for ax, w in zip(axes, widths):
        sub = [e for e in entries if e.get("width", len(e["layers"])) == w]
        starts = [e["layers"][0] for e in sub]
        delta_em = [e["delta_em"] for e in sub]
        delta_f1 = [e["delta_dep_f1"] for e in sub]
        ax.plot(starts, delta_em, marker="o", color="#d62728", label="ΔEM")
        ax.plot(starts, delta_f1, marker="s", color="#1f77b4", label="ΔdepF1")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Window start layer")
        ax.set_title(f"Middle window width = {w}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Δ from intact baseline")
    fig.tight_layout()
    save(fig, output_dir, "middle_window_ablation")


# ---------------------------------------------------------------------------
# Plots: baselines and distance to impossible
# ---------------------------------------------------------------------------


def plot_baseline_comparison(baselines: dict, output_dir: str) -> None:
    metrics = ["em", "token_acc", "bleu1", "bleu4", "dep_f1", "uas", "las"]
    labels = ["EM", "TokAcc", "BLEU-1", "BLEU-4", "dep F1", "UAS", "LAS"]
    models = ["translator", "impossible", "base"]
    xs = np.arange(len(metrics))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(models):
        vals = [baselines.get(m, {}).get(k, 0.0) for k in metrics]
        ax.bar(xs + (i - 1) * width, vals, width=width, color=MODEL_COLORS[m], label=m.capitalize())
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Intact baselines: translator vs impossible vs GPT-2 base")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=False)
    save(fig, output_dir, "baseline_comparison")


def plot_distance_to_impossible(distance: dict, output_dir: str) -> None:
    em_entries = distance.get("em", [])
    f1_entries = distance.get("dep_f1", [])
    if not em_entries and not f1_entries:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if em_entries:
        layers = [e["layer"] for e in em_entries]
        ax.plot(
            layers,
            [e["abs_diff"] for e in em_entries],
            marker="o",
            color="#d62728",
            label="|EM_ablated − EM_impossible|",
        )
    if f1_entries:
        layers = [e["layer"] for e in f1_entries]
        ax.plot(
            layers,
            [e["abs_diff"] for e in f1_entries],
            marker="s",
            color="#1f77b4",
            label="|depF1_ablated − depF1_impossible|",
        )
    ax.set_xlabel("Ablated layer")
    ax.set_ylabel("Absolute distance")
    ax.set_title("Distance to impossible model after single-layer zero ablation")
    ax.set_xticks(sorted({e["layer"] for e in em_entries + f1_entries}))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save(fig, output_dir, "distance_to_impossible")


def plot_probing_correlation(probing: dict, output_dir: str) -> None:
    if not probing or not probing.get("available"):
        return
    props = ("pos", "dep_rel", "depth")
    em_rho = [probing.get(f"{p}_em_rho", float("nan")) for p in props]
    f1_rho = [probing.get(f"{p}_f1_rho", float("nan")) for p in props]
    em_p = [probing.get(f"{p}_em_p", float("nan")) for p in props]
    f1_p = [probing.get(f"{p}_f1_p", float("nan")) for p in props]

    xs = np.arange(len(props))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    em_clean = [0.0 if np.isnan(v) else v for v in em_rho]
    f1_clean = [0.0 if np.isnan(v) else v for v in f1_rho]
    bars_em = ax.bar(xs - width / 2, em_clean, width=width, color="#d62728", label="vs ΔEM")
    bars_f1 = ax.bar(xs + width / 2, f1_clean, width=width, color="#1f77b4", label="vs ΔdepF1")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([p.upper() for p in props])
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Causal-vs-probing correlation (layer-wise)")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, axis="y")

    def annotate(bars, ps):
        for bar, p in zip(bars, ps):
            if np.isnan(p):
                txt = "n/a"
            else:
                txt = f"p={p:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03 * np.sign(bar.get_height() or 1),
                txt,
                ha="center",
                va="bottom" if (bar.get_height() or 0) >= 0 else "top",
                fontsize=8,
            )

    annotate(bars_em, em_p)
    annotate(bars_f1, f1_p)
    ax.legend(frameon=False)
    save(fig, output_dir, "probing_correlation")


def plot_interaction_check(single: dict, cumul: dict, output_dir: str) -> None:
    """Compare additive predicted Δ vs measured Δ from cumulative width-2 windows."""
    middle = [e for e in cumul.get("middle", []) if e.get("width") == 2]
    base = single.get("zero_full", {})
    if not middle or not base:
        return
    starts, additive_em, measured_em = [], [], []
    additive_f1, measured_f1 = [], []
    for entry in middle:
        layers = entry["layers"]
        l, lp = layers[0], layers[1]
        a_em = base[str(l)]["delta_em"] + base[str(lp)]["delta_em"]
        a_f1 = base[str(l)]["delta_dep_f1"] + base[str(lp)]["delta_dep_f1"]
        additive_em.append(a_em)
        additive_f1.append(a_f1)
        measured_em.append(entry["delta_em"])
        measured_f1.append(entry["delta_dep_f1"])
        starts.append(l)
    xs = np.arange(len(starts))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, additive, measured, title in zip(
            axes,
            (additive_em, additive_f1),
            (measured_em, measured_f1),
            ("ΔEM", "ΔdepF1"),
    ):
        ax.plot(xs, additive, marker="o", linestyle="--", color="#7f7f7f", label="Additive prediction")
        ax.plot(xs, measured, marker="s", color="#d62728", label="Measured")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s},{s + 1}" for s in starts], rotation=45, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"Pairwise interaction check – {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, output_dir, "interaction_check")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Plot causal intervention results.")
    p.add_argument("--results", default="causal_intervention_results.json")
    p.add_argument("--output_dir", default="causal_plots")
    args = p.parse_args()

    print(f"Loading results from {args.results}")
    data = load_results(args.results)
    single = data.get("single_layer_ablations", {})
    cumul = data.get("cumulative_ablations", {})
    baselines = data.get("baselines", {})
    distance = data.get("distance_to_impossible", {})
    probing = data.get("probing_correlation", {})

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Writing plots to {args.output_dir}/")

    # Single-layer ablations
    plot_layer_profile_metric(
        single, "delta_em", "Δ Exact Match",
        "Per-layer Δ Exact Match across kinds and scopes",
        args.output_dir, "layer_profile_delta_em",
    )
    plot_layer_profile_metric(
        single, "delta_dep_f1", "Δ Dep F1",
        "Per-layer Δ Dependency F1 across kinds and scopes",
        args.output_dir, "layer_profile_delta_depf1",
    )
    plot_layer_profile_metric(
        single, "em", "Exact Match",
        "Absolute EM after single-layer ablation",
        args.output_dir, "layer_profile_em_absolute",
    )
    plot_layer_profile_metric(
        single, "dep_f1", "Dep F1",
        "Absolute Dep F1 after single-layer ablation",
        args.output_dir, "layer_profile_depf1_absolute",
    )
    plot_delta_grid(single, args.output_dir)
    for scope in ("full", "attn", "ffn"):
        plot_delta_heatmap(single, scope, args.output_dir)
    plot_attn_vs_ffn(single, args.output_dir)
    plot_critical_layers(single, args.output_dir)

    # Cumulative ablations
    plot_cumulative_window(
        cumul, "early", args.output_dir,
        "cumulative_early", "Early cumulative ablation (layers 0..k)",
    )
    plot_cumulative_window(
        cumul, "late", args.output_dir,
        "cumulative_late", "Late cumulative ablation (layers k..11)",
    )
    plot_middle_window_grid(cumul, args.output_dir)

    # Baselines and downstream analyses
    plot_baseline_comparison(baselines, args.output_dir)
    plot_distance_to_impossible(distance, args.output_dir)
    plot_probing_correlation(probing, args.output_dir)
    plot_interaction_check(single, cumul, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()