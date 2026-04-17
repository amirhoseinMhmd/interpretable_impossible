"""
Plotting script for causal_intervention_results.json (RQ3).

Produces six focused figures for the layer-wise causal intervention
experiment. Every plot is rendered as both PNG and PDF in --output_dir
(default: causal_plots/).

Figures
-------
1. critical_zone_callout   : EM and depF1 curves with critical-zone shading,
                             baseline reference lines, and annotated callout.
2. delta_heatmap_full      : Compact heatmap of ΔEM and ΔdepF1 across all
                             three ablation kinds for the full scope.
3. attention_vs_ffn        : Per-layer attention-only vs FFN-only impact
                             (zero ablation) for ΔEM and ΔdepF1.
4. cumulative_overlay      : Early [0..k] vs late [k..11] cumulative ablation
                             dep F1 on one axis.
5. middle_window_ablation  : Width-2 / width-3 sliding window ablation deltas.
6. baseline_comparison     : Translator vs impossible vs base bar chart across
                             all metrics.

Usage
-----
python causal_intervention_plots.py \\
    --results causal_intervention_results.json \\
    --output_dir causal_plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Style
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
    "zero":   "#1f77b4",
    "mean":   "#ff7f0e",
    "random": "#2ca02c",
}
KIND_LABELS = {
    "zero":   "Zero ablation",
    "mean":   "Mean ablation",
    "random": "Random ablation",
}
MODEL_COLORS = {
    "translator": "#d62728",
    "impossible": "#1f77b4",
    "base":       "#2ca02c",
}
MODEL_LABELS = {
    "translator": "Translator (intact)",
    "impossible": "Impossible LM",
    "base":       "GPT-2 base",
}
CRITICAL_COLOR = "#fde0dc"
CRITICAL_EDGE = "#d62728"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_results(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def n_layers_from(single: dict) -> int:
    block = single.get("zero_full", {})
    return len(block) if block else 12


def layer_array(single: dict, kind: str, scope: str, metric: str) -> np.ndarray:
    block = single.get(f"{kind}_{scope}", {})
    if not block:
        return np.array([])
    layers = sorted(int(k) for k in block.keys())
    return np.array([block[str(l)].get(metric, 0.0) for l in layers], dtype=float)


def critical_layers(single: dict, kind: str = "zero", scope: str = "full") -> List[int]:
    block = single.get(f"{kind}_{scope}", {})
    out = []
    for l in sorted(int(k) for k in block.keys()):
        if block[str(l)].get("critical", 0):
            out.append(l)
    return out


def baseline(baselines: dict, model: str, metric: str) -> float:
    return baselines.get(model, {}).get(metric, 0.0)


def shade_critical(ax: plt.Axes, layers: Sequence[int]) -> None:
    if not layers:
        return
    layers = sorted(layers)
    spans = []
    start = prev = layers[0]
    for l in layers[1:]:
        if l == prev + 1:
            prev = l
        else:
            spans.append((start, prev))
            start = prev = l
    spans.append((start, prev))
    for s, e in spans:
        ax.axvspan(s - 0.5, e + 0.5, facecolor=CRITICAL_COLOR, alpha=0.6,
                   edgecolor=CRITICAL_EDGE, linewidth=0.5, zorder=0)


def save(fig: plt.Figure, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"))
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  saved {name}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# 1. Critical zone callout
# ---------------------------------------------------------------------------


def plot_critical_zone_callout(single: dict, baselines: dict, output_dir: str) -> None:
    n = n_layers_from(single)
    xs = np.arange(n)
    crit = critical_layers(single)
    block = single.get("zero_full", {})
    if not block:
        return

    em = layer_array(single, "zero", "full", "em")
    f1 = layer_array(single, "zero", "full", "dep_f1")

    fig, ax = plt.subplots(figsize=(11, 6))
    shade_critical(ax, crit)
    ax.plot(xs, em, marker="o", color="#d62728", linewidth=2.2, markersize=7,
            label="EM after ablation")
    ax.plot(xs, f1, marker="s", color="#1f77b4", linewidth=2.2, markersize=7,
            label="Dep F1 after ablation")

    ref_translator_em = baseline(baselines, "translator", "em")
    ref_translator_f1 = baseline(baselines, "translator", "dep_f1")
    ref_imp_f1 = baseline(baselines, "impossible", "dep_f1")
    ref_base_f1 = baseline(baselines, "base", "dep_f1")

    ax.axhline(ref_translator_em, color="#d62728", linestyle="--", alpha=0.7,
               linewidth=1.2, label=f"Translator intact EM = {ref_translator_em:.2f}")
    ax.axhline(ref_translator_f1, color="#1f77b4", linestyle="--", alpha=0.7,
               linewidth=1.2, label=f"Translator intact depF1 = {ref_translator_f1:.2f}")
    ax.axhline(ref_imp_f1, color="#1f77b4", linestyle=":", alpha=0.7,
               linewidth=1.2, label=f"Impossible depF1 = {ref_imp_f1:.2f}")
    ax.axhline(ref_base_f1, color="#2ca02c", linestyle=":", alpha=0.7,
               linewidth=1.2, label=f"GPT-2 base depF1 = {ref_base_f1:.2f}")

    if crit:
        cmin, cmax = min(crit), max(crit)
        ax.annotate(
            f"Critical zone\nlayers {cmin}\u2013{cmax}",
            xy=((cmin + cmax) / 2, ax.get_ylim()[1] * 0.95),
            xytext=((cmin + cmax) / 2, ax.get_ylim()[1] * 0.78),
            ha="center",
            fontsize=11,
            color=CRITICAL_EDGE,
            arrowprops=dict(arrowstyle="->", color=CRITICAL_EDGE, lw=1.2),
        )

    ax.set_xticks(xs)
    ax.set_xlabel("Ablated layer")
    ax.set_ylabel("Score")
    ax.set_title("Critical-layer callout: where does the translator break?",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, output_dir, "critical_zone_callout")


# ---------------------------------------------------------------------------
# 2. Delta heatmap (full scope)
# ---------------------------------------------------------------------------


def plot_delta_heatmap(single: dict, output_dir: str) -> None:
    n = n_layers_from(single)
    rows, labels = [], []
    for kind in ("zero", "mean", "random"):
        for metric, mlabel in (("delta_em", "\u0394EM"), ("delta_dep_f1", "\u0394depF1")):
            arr = layer_array(single, kind, "full", metric)
            if arr.size == 0:
                continue
            rows.append(arr)
            labels.append(f"{KIND_LABELS[kind]}  {mlabel}")
    if not rows:
        return
    mat = np.vstack(rows)
    vmax = max(abs(mat).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([str(i) for i in range(n)])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ablated layer")
    ax.set_title("Ablation impact heatmap (full scope)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if abs(v) > vmax * 0.3:
                color = "white" if abs(v) > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("\u0394 from intact baseline")
    fig.tight_layout()
    save(fig, output_dir, "delta_heatmap_full")


# ---------------------------------------------------------------------------
# 3. Attention vs FFN
# ---------------------------------------------------------------------------


def plot_attention_vs_ffn(single: dict, output_dir: str) -> None:
    n = n_layers_from(single)
    xs = np.arange(n)
    width = 0.4
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, metric, title in zip(
        axes,
        ("delta_em", "delta_dep_f1"),
        ("\u0394 Exact Match", "\u0394 Dep F1"),
    ):
        attn = layer_array(single, "zero", "attn", metric)
        ffn = layer_array(single, "zero", "ffn", metric)
        if attn.size == 0 or ffn.size == 0:
            continue
        ax.bar(xs - width / 2, attn, width=width, color="#9467bd",
               edgecolor="black", linewidth=0.4, label="Attention only")
        ax.bar(xs + width / 2, ffn, width=width, color="#8c564b",
               edgecolor="black", linewidth=0.4, label="FFN only")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.set_title(f"Attention vs FFN \u2013 {title}")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(frameon=False)
    fig.suptitle("Attention vs feed-forward sublayer specialisation (zero ablation)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, output_dir, "attention_vs_ffn")


# ---------------------------------------------------------------------------
# 4. Cumulative overlay
# ---------------------------------------------------------------------------


def plot_cumulative_overlay(cumul: dict, baselines: dict, output_dir: str) -> None:
    early = cumul.get("early", [])
    late = cumul.get("late", [])
    if not early and not late:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    if early:
        xs = [e["layers"][-1] for e in early]
        ax.plot(xs, [e["dep_f1"] for e in early], marker="o", color="#d62728",
                label="Early ablation [0..k] dep F1", linewidth=1.8)
    if late:
        xs = [e["layers"][0] for e in late]
        ax.plot(xs, [e["dep_f1"] for e in late], marker="s", color="#1f77b4",
                label="Late ablation [k..11] dep F1", linewidth=1.8)
    for m in ("translator", "impossible", "base"):
        v = baseline(baselines, m, "dep_f1")
        ax.axhline(v, color=MODEL_COLORS[m], linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"{MODEL_LABELS[m]} ({v:.2f})")
    ax.set_xlabel("Cut-off layer k")
    ax.set_ylabel("Dependency F1")
    ax.set_title("Cumulative ablation overlay: how deep does structural recovery survive?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", frameon=False)
    save(fig, output_dir, "cumulative_overlay")


# ---------------------------------------------------------------------------
# 5. Middle window ablation
# ---------------------------------------------------------------------------


def plot_middle_window(cumul: dict, output_dir: str) -> None:
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
        ax.plot(starts, delta_em, marker="o", color="#d62728", label="\u0394EM", linewidth=1.8)
        ax.plot(starts, delta_f1, marker="s", color="#1f77b4", label="\u0394depF1", linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Window start layer")
        ax.set_title(f"Middle window width = {w}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    axes[0].set_ylabel("\u0394 from intact baseline")
    fig.suptitle("Sliding-window ablation: which contiguous block matters?",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, output_dir, "middle_window_ablation")


# ---------------------------------------------------------------------------
# 6. Baseline comparison
# ---------------------------------------------------------------------------


def plot_baseline_comparison(baselines: dict, output_dir: str) -> None:
    metrics = ["em", "token_acc", "bleu1", "bleu4", "dep_f1", "uas", "las"]
    labels = ["EM", "TokAcc", "BLEU-1", "BLEU-4", "dep F1", "UAS", "LAS"]
    models = ["translator", "impossible", "base"]
    xs = np.arange(len(metrics))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(models):
        vals = [baseline(baselines, m, k) for k in metrics]
        bars = ax.bar(xs + (i - 1) * width, vals, width=width,
                      color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if v > 0.02:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Intact baselines: translator vs impossible vs GPT-2 base")
    ax.set_ylim(0, max(0.05, max(baseline(baselines, m, k)
                                  for m in models for k in metrics) * 1.15))
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=False, loc="upper right")
    save(fig, output_dir, "baseline_comparison")


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

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Writing plots to {args.output_dir}/")

    plot_critical_zone_callout(single, baselines, args.output_dir)
    plot_delta_heatmap(single, args.output_dir)
    plot_attention_vs_ffn(single, args.output_dir)
    plot_cumulative_overlay(cumul, baselines, args.output_dir)
    plot_middle_window(cumul, args.output_dir)
    plot_baseline_comparison(baselines, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
