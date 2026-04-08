"""
Plotting script for causal_intervention_results.json (RQ3).

Produces a story-driven figure suite for the layer-wise causal intervention
experiment. Every plot is rendered as both PNG and PDF in --output_dir
(default: causal_plots/).

Figures
-------
1.  headline_summary           : 4-panel headline (EM, depF1, BLEU-4, LAS) with
                                 intact / impossible / base reference lines and
                                 critical-zone shading.
2.  critical_layers            : ΔEM and ΔdepF1 bar chart with critical layers
                                 highlighted plus annotated drops.
3.  delta_grid                 : 3x2 grid of Δ-EM and Δ-depF1 across all
                                 ablation kinds for full / attn / ffn scopes.
4.  delta_heatmap_{scope}      : Compact heatmap of Δ values across kinds.
5.  attention_vs_ffn           : Per-layer attention-only vs FFN-only impact.
6.  layer_profile_{metric}     : Multi-line per-layer absolute or Δ profiles.
7.  multi_metric_layer_profile : Six metrics (EM, BLEU-1, BLEU-4, depF1, UAS,
                                 LAS) on a small-multiples grid.
8.  cumulative_early           : Early ablation [0..k] absolute + delta panels.
9.  cumulative_late            : Late ablation [k..11] absolute + delta panels.
10. middle_window_ablation     : Width-2 / width-3 sliding window ablations.
11. cumulative_overlay         : Cumulative early vs late on one axis.
12. baseline_comparison        : Translator vs impossible vs base bar chart.
13. distance_to_impossible     : How close ablation drives translator to
                                 impossible / base baselines.
14. probing_correlation        : Spearman ρ between Δ-perf and probing
                                 divergence (if available).
15. interaction_check          : Predicted-additive vs measured Δ for width-2
                                 cumulative ablations.
16. critical_zone_callout      : Headline takeaway annotated figure.

Usage
-----
python causal_intervention_plots.py \
    --results causal_intervention_results.json \
    --output_dir causal_plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

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
SCOPE_LINESTYLES = {
    "full": "-",
    "attn": "--",
    "ffn":  ":",
}
SCOPE_LABELS = {
    "full": "Full layer",
    "attn": "Attention only",
    "ffn":  "FFN only",
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
    """Shade the critical layer indices on the x-axis of `ax`."""
    if not layers:
        return
    layers = sorted(layers)
    # Group contiguous layers into spans
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


def add_reference_lines(ax: plt.Axes, baselines: dict, metric: str,
                        models: Sequence[str] = ("translator", "impossible", "base")) -> None:
    """Add horizontal reference lines for each model's intact baseline."""
    for m in models:
        v = baseline(baselines, m, metric)
        ax.axhline(v, color=MODEL_COLORS[m], linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"{MODEL_LABELS[m]} = {v:.3f}")


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:+.2f}", fontsize: int = 8) -> None:
    for bar in bars:
        h = bar.get_height()
        if abs(h) < 1e-6:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=fontsize,
            color="black",
        )


def save(fig: plt.Figure, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"))
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  saved {name}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# 1. Headline summary
# ---------------------------------------------------------------------------


def plot_headline_summary(single: dict, baselines: dict, output_dir: str) -> None:
    n = n_layers_from(single)
    xs = np.arange(n)
    crit = critical_layers(single)

    metrics = [
        ("em",      "Exact Match"),
        ("dep_f1",  "Dependency F1"),
        ("bleu4",   "BLEU-4"),
        ("las",     "LAS"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        shade_critical(ax, crit)
        ys = layer_array(single, "zero", "full", metric)
        ax.plot(xs, ys, color="#d62728", marker="o", linewidth=2.0,
                markersize=6, label="Zero-full ablation")
        # Reference lines
        for m in ("translator", "impossible", "base"):
            v = baseline(baselines, m, metric)
            ax.axhline(v, color=MODEL_COLORS[m], linestyle="--",
                       linewidth=1.1, alpha=0.7,
                       label=f"{MODEL_LABELS[m]} ({v:.2f})")
        ax.set_title(f"{label} after single-layer zero ablation")
        ax.set_xticks(xs)
        ax.set_xlabel("Ablated layer")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if metric == "em":
            ax.legend(fontsize=8, loc="best", frameon=False)

    legend_handles = [
        Patch(facecolor=CRITICAL_COLOR, edgecolor=CRITICAL_EDGE,
              label="Critical layer (ΔEM > 5% or ΔdepF1 > 10%)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Causal intervention headline: which layers matter for translation?",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, output_dir, "headline_summary")


# ---------------------------------------------------------------------------
# 2. Critical layers bar chart
# ---------------------------------------------------------------------------


def plot_critical_layers(single: dict, baselines: dict, output_dir: str) -> None:
    block = single.get("zero_full", {})
    if not block:
        return
    layers = sorted(int(k) for k in block.keys())
    delta_em = np.array([block[str(l)]["delta_em"] for l in layers])
    delta_f1 = np.array([block[str(l)]["delta_dep_f1"] for l in layers])
    crit_mask = np.array([block[str(l)].get("critical", 0) for l in layers], bool)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    width = 0.7
    for ax, vals, ylabel, title in zip(
        axes,
        (delta_em, delta_f1),
        ("Δ Exact Match", "Δ Dependency F1"),
        ("Δ EM per ablated layer", "Δ Dep F1 per ablated layer"),
    ):
        colors = ["#d62728" if c else "#9ecae1" for c in crit_mask]
        bars = ax.bar(layers, vals, width=width, color=colors, edgecolor="black", linewidth=0.5)
        annotate_bars(ax, bars, fmt="{:+.2f}", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(layers)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    axes[1].set_xlabel("Ablated layer (zero, full)")

    legend_handles = [
        Patch(facecolor="#d62728", edgecolor="black", label="Critical layer"),
        Patch(facecolor="#9ecae1", edgecolor="black", label="Non-critical layer"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", frameon=False)
    fig.suptitle(
        f"Critical layers: ΔEM > {0.05} or ΔdepF1 > {0.10}",
        y=1.0, fontsize=13,
    )
    fig.tight_layout()
    save(fig, output_dir, "critical_layers")


# ---------------------------------------------------------------------------
# 3. Δ grid across kinds × scopes
# ---------------------------------------------------------------------------


def plot_delta_grid(single: dict, baselines: dict, output_dir: str) -> None:
    n = n_layers_from(single)
    xs = np.arange(n)
    crit = critical_layers(single)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    metrics = [("delta_em", "Δ Exact Match"), ("delta_dep_f1", "Δ Dep F1")]

    for row, scope in enumerate(("full", "attn", "ffn")):
        for col, (metric, ylab) in enumerate(metrics):
            ax = axes[row, col]
            shade_critical(ax, crit)
            for kind in ("zero", "mean", "random"):
                ys = layer_array(single, kind, scope, metric)
                if ys.size == 0:
                    continue
                ax.plot(xs, ys, color=KIND_COLORS[kind], marker="o",
                        markersize=4, linewidth=1.6, label=KIND_LABELS[kind])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{SCOPE_LABELS[scope]} – {ylab}")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylab)
            if row == 2:
                ax.set_xlabel("Layer")
                ax.set_xticks(xs)
            if row == 0 and col == 1:
                ax.legend(loc="best", frameon=False)
    fig.suptitle("Single-layer ablation impact (Δ vs intact baseline)", fontsize=13, y=1.0)
    fig.tight_layout()
    save(fig, output_dir, "delta_grid")


# ---------------------------------------------------------------------------
# 4. Δ heatmap per scope
# ---------------------------------------------------------------------------


def plot_delta_heatmap(single: dict, scope: str, output_dir: str) -> None:
    n = n_layers_from(single)
    rows, labels = [], []
    for kind in ("zero", "mean", "random"):
        for metric, mlabel in (("delta_em", "ΔEM"), ("delta_dep_f1", "ΔdepF1")):
            arr = layer_array(single, kind, scope, metric)
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
    ax.set_title(f"Ablation impact heatmap ({SCOPE_LABELS[scope]})")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if abs(v) > vmax * 0.3:
                color = "white" if abs(v) > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Δ from intact baseline")
    fig.tight_layout()
    save(fig, output_dir, f"delta_heatmap_{scope}")


# ---------------------------------------------------------------------------
# 5. Attention vs FFN
# ---------------------------------------------------------------------------


def plot_attention_vs_ffn(single: dict, output_dir: str) -> None:
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
        ax.bar(xs - width / 2, attn, width=width, color="#9467bd",
               edgecolor="black", linewidth=0.4, label="Attention only")
        ax.bar(xs + width / 2, ffn, width=width, color="#8c564b",
               edgecolor="black", linewidth=0.4, label="FFN only")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.set_title(f"Attention vs FFN – {title}")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(frameon=False)
    fig.suptitle("Attention vs feed-forward sublayer specialisation (zero ablation)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, output_dir, "attention_vs_ffn")


# ---------------------------------------------------------------------------
# 6. Per-layer profiles
# ---------------------------------------------------------------------------


def plot_layer_profile_metric(
    single: dict,
    baselines: dict,
    metric: str,
    ylabel: str,
    title: str,
    output_dir: str,
    name: str,
    show_baselines: bool = True,
) -> None:
    n = n_layers_from(single)
    xs = np.arange(n)
    crit = critical_layers(single)
    fig, ax = plt.subplots(figsize=(9, 5))
    shade_critical(ax, crit)
    for kind in ("zero", "mean", "random"):
        for scope in ("full", "attn", "ffn"):
            ys = layer_array(single, kind, scope, metric)
            if ys.size == 0:
                continue
            ax.plot(
                xs, ys,
                color=KIND_COLORS[kind],
                linestyle=SCOPE_LINESTYLES[scope],
                marker="o", markersize=3, linewidth=1.4,
                label=f"{KIND_LABELS[kind]} – {SCOPE_LABELS[scope]}",
            )
    if show_baselines and metric in ("em", "token_acc", "bleu1", "bleu2", "bleu3", "bleu4",
                                      "dep_f1", "uas", "las"):
        for m in ("translator", "impossible", "base"):
            v = baseline(baselines, m, metric)
            ax.axhline(v, color=MODEL_COLORS[m], linestyle=":", linewidth=1.2,
                       alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=7, loc="best", frameon=False)
    save(fig, output_dir, name)


# ---------------------------------------------------------------------------
# 7. Multi-metric small multiples
# ---------------------------------------------------------------------------


def plot_multi_metric_layer_profile(single: dict, baselines: dict, output_dir: str) -> None:
    metrics = [
        ("em",      "Exact Match"),
        ("token_acc", "Token Acc"),
        ("bleu1",   "BLEU-1"),
        ("bleu4",   "BLEU-4"),
        ("dep_f1",  "Dep F1"),
        ("uas",     "UAS"),
    ]
    n = n_layers_from(single)
    xs = np.arange(n)
    crit = critical_layers(single)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        shade_critical(ax, crit)
        ys = layer_array(single, "zero", "full", metric)
        ax.plot(xs, ys, color="#d62728", marker="o", linewidth=1.8, markersize=5,
                label="zero/full")
        for m in ("translator", "impossible", "base"):
            v = baseline(baselines, m, metric)
            ax.axhline(v, color=MODEL_COLORS[m], linestyle="--", linewidth=1.0,
                       alpha=0.7, label=MODEL_LABELS[m])
        ax.set_title(label)
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        if metric == "em":
            ax.legend(fontsize=7, loc="best", frameon=False)
    for ax in axes[1, :]:
        ax.set_xlabel("Ablated layer")
    fig.suptitle("Layer-wise causal impact across translation and structural metrics",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    save(fig, output_dir, "multi_metric_layer_profile")


# ---------------------------------------------------------------------------
# 8/9. Cumulative ablations
# ---------------------------------------------------------------------------


def plot_cumulative_window(
    cumul: dict, baselines: dict, key: str,
    output_dir: str, name: str, title: str,
) -> None:
    entries = cumul.get(key, [])
    if not entries:
        return
    xs = np.arange(len(entries))
    em = [e["em"] for e in entries]
    f1 = [e["dep_f1"] for e in entries]
    delta_em = [e["delta_em"] for e in entries]
    delta_f1 = [e["delta_dep_f1"] for e in entries]
    if key == "early":
        labels = [f"0..{e['layers'][-1]}" for e in entries]
    elif key == "late":
        labels = [f"{e['layers'][0]}..{e['layers'][-1]}" for e in entries]
    else:
        labels = [f"{e['layers'][0]}..{e['layers'][-1]}" for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(xs, em, marker="o", color="#d62728", label="EM", linewidth=1.8)
    ax.plot(xs, f1, marker="s", color="#1f77b4", label="dep F1", linewidth=1.8)
    for m in ("translator", "impossible"):
        ax.axhline(baseline(baselines, m, "em"), color=MODEL_COLORS[m],
                   linestyle="--", alpha=0.6, linewidth=1.0,
                   label=f"EM {MODEL_LABELS[m]}")
        ax.axhline(baseline(baselines, m, "dep_f1"), color=MODEL_COLORS[m],
                   linestyle=":", alpha=0.6, linewidth=1.0,
                   label=f"depF1 {MODEL_LABELS[m]}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Performance")
    ax.set_title(f"{title} – absolute")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", frameon=False, ncol=2)

    ax = axes[1]
    ax.plot(xs, delta_em, marker="o", color="#d62728", label="ΔEM", linewidth=1.8)
    ax.plot(xs, delta_f1, marker="s", color="#1f77b4", label="ΔdepF1", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Δ from intact baseline")
    ax.set_title(f"{title} – delta")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, output_dir, name)


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
        ax.plot(starts, delta_em, marker="o", color="#d62728", label="ΔEM", linewidth=1.8)
        ax.plot(starts, delta_f1, marker="s", color="#1f77b4", label="ΔdepF1", linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Window start layer")
        ax.set_title(f"Middle window width = {w}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Δ from intact baseline")
    fig.suptitle("Sliding-window ablation: which contiguous block matters?",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, output_dir, "middle_window_ablation")


# ---------------------------------------------------------------------------
# 12. Baseline comparison
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
# 13. Distance to impossible / base
# ---------------------------------------------------------------------------


def plot_distance_to_impossible(distance: dict, single: dict, baselines: dict,
                                output_dir: str) -> None:
    em_entries = distance.get("em", [])
    f1_entries = distance.get("dep_f1", [])
    if not em_entries and not f1_entries:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: distance to impossible
    ax = axes[0]
    layers = sorted({e["layer"] for e in em_entries + f1_entries})
    if em_entries:
        ax.plot([e["layer"] for e in em_entries],
                [e["abs_diff"] for e in em_entries],
                marker="o", color="#d62728",
                label="|EM_ablated − EM_impossible|", linewidth=1.8)
    if f1_entries:
        ax.plot([e["layer"] for e in f1_entries],
                [e["abs_diff"] for e in f1_entries],
                marker="s", color="#1f77b4",
                label="|depF1_ablated − depF1_impossible|", linewidth=1.8)
    ax.set_xlabel("Ablated layer")
    ax.set_ylabel("Absolute distance")
    ax.set_title("Distance to impossible model after single-layer ablation")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    # Right: absolute curves (translator vs ablated vs impossible vs base)
    ax = axes[1]
    n = n_layers_from(single)
    xs = np.arange(n)
    em_curve = layer_array(single, "zero", "full", "em")
    f1_curve = layer_array(single, "zero", "full", "dep_f1")
    ax.plot(xs, em_curve, marker="o", color="#d62728", label="Ablated EM", linewidth=1.6)
    ax.plot(xs, f1_curve, marker="s", color="#1f77b4", label="Ablated depF1", linewidth=1.6)
    ax.axhline(baseline(baselines, "translator", "em"),
               color="#d62728", linestyle="--", alpha=0.6, label="Translator EM")
    ax.axhline(baseline(baselines, "translator", "dep_f1"),
               color="#1f77b4", linestyle="--", alpha=0.6, label="Translator depF1")
    ax.axhline(baseline(baselines, "impossible", "dep_f1"),
               color="#1f77b4", linestyle=":", alpha=0.6, label="Impossible depF1")
    ax.axhline(baseline(baselines, "base", "dep_f1"),
               color="#2ca02c", linestyle=":", alpha=0.6, label="Base depF1")
    ax.set_xticks(xs)
    ax.set_xlabel("Ablated layer")
    ax.set_ylabel("Score")
    ax.set_title("Ablated translator vs reference baselines")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")

    fig.tight_layout()
    save(fig, output_dir, "distance_to_impossible")


# ---------------------------------------------------------------------------
# 14. Probing correlation
# ---------------------------------------------------------------------------


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
    fig, ax = plt.subplots(figsize=(8, 4.8))
    em_clean = [0.0 if (v is None or np.isnan(v)) else v for v in em_rho]
    f1_clean = [0.0 if (v is None or np.isnan(v)) else v for v in f1_rho]
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
            txt = "n/a" if (p is None or np.isnan(p)) else f"p={p:.2f}"
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.04 * (1 if h >= 0 else -1),
                    txt,
                    ha="center",
                    va="bottom" if h >= 0 else "top",
                    fontsize=8)
    annotate(bars_em, em_p)
    annotate(bars_f1, f1_p)
    ax.legend(frameon=False)
    save(fig, output_dir, "probing_correlation")


# ---------------------------------------------------------------------------
# 15. Interaction check
# ---------------------------------------------------------------------------


def plot_interaction_check(single: dict, cumul: dict, output_dir: str) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, additive, measured, title in zip(
        axes,
        (additive_em, additive_f1),
        (measured_em, measured_f1),
        ("ΔEM", "ΔdepF1"),
    ):
        width = 0.35
        ax.bar(xs - width / 2, additive, width=width, color="#7f7f7f",
               edgecolor="black", linewidth=0.4, label="Additive prediction")
        ax.bar(xs + width / 2, measured, width=width, color="#d62728",
               edgecolor="black", linewidth=0.4, label="Measured")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s},{s+1}" for s in starts], rotation=45, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"Pairwise interaction – {title}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(frameon=False)
    fig.suptitle("Two-layer ablation: additive prediction vs measured drop "
                 "(redundancy ↑ if measured < additive)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, output_dir, "interaction_check")


# ---------------------------------------------------------------------------
# 16. Critical zone callout
# ---------------------------------------------------------------------------


def plot_critical_zone_callout(single: dict, baselines: dict, output_dir: str) -> None:
    """Single high-impact figure summarising the critical-zone story."""
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
            f"Critical zone\nlayers {cmin}–{cmax}",
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

    # Headline / story
    plot_headline_summary(single, baselines, args.output_dir)
    plot_critical_zone_callout(single, baselines, args.output_dir)
    plot_critical_layers(single, baselines, args.output_dir)
    plot_multi_metric_layer_profile(single, baselines, args.output_dir)

    # Detailed single-layer ablation views
    plot_delta_grid(single, baselines, args.output_dir)
    for scope in ("full", "attn", "ffn"):
        plot_delta_heatmap(single, scope, args.output_dir)
    plot_attention_vs_ffn(single, args.output_dir)

    plot_layer_profile_metric(
        single, baselines, "delta_em", "Δ Exact Match",
        "Per-layer Δ Exact Match across kinds and scopes",
        args.output_dir, "layer_profile_delta_em",
    )
    plot_layer_profile_metric(
        single, baselines, "delta_dep_f1", "Δ Dep F1",
        "Per-layer Δ Dependency F1 across kinds and scopes",
        args.output_dir, "layer_profile_delta_depf1",
    )
    plot_layer_profile_metric(
        single, baselines, "em", "Exact Match",
        "Absolute EM after single-layer ablation",
        args.output_dir, "layer_profile_em_absolute",
    )
    plot_layer_profile_metric(
        single, baselines, "dep_f1", "Dep F1",
        "Absolute Dep F1 after single-layer ablation",
        args.output_dir, "layer_profile_depf1_absolute",
    )

    # Cumulative
    plot_cumulative_window(cumul, baselines, "early", args.output_dir,
                           "cumulative_early",
                           "Early cumulative ablation (layers 0..k)")
    plot_cumulative_window(cumul, baselines, "late", args.output_dir,
                           "cumulative_late",
                           "Late cumulative ablation (layers k..11)")
    plot_cumulative_overlay(cumul, baselines, args.output_dir)
    plot_middle_window(cumul, args.output_dir)

    # Baselines and convergence
    plot_baseline_comparison(baselines, args.output_dir)
    plot_distance_to_impossible(distance, single, baselines, args.output_dir)
    if probing.get("available"):
        plot_probing_correlation(probing, args.output_dir)
    else:
        print("  skipping probing_correlation (not available)")
    plot_interaction_check(single, cumul, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
