import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 2.0

MODEL_NAMES = {
    "translator": "Translator (shuffled → English)",
    "impossible": "Impossible LM (Kallini et al.)",
    "base": "GPT-2 Base (pretrained)",
}

MODEL_COLORS = {
    "translator": "#A32D2D",
    "impossible": "#534AB7",
    "base": "#1D9E75",
}

MODEL_MARKERS = {
    "translator": "o",
    "impossible": "s",
    "base": "D",
}

PROPERTY_LABELS = {
    "pos": "POS Tagging",
    "dep_rel": "Dependency Relations",
    "depth": "Syntactic Depth",
}

PROPERTY_METRICS = {
    "pos": ("accuracy", "Accuracy"),
    "dep_rel": ("accuracy", "Accuracy"),
    "depth": ("spearman_r", "Spearman ρ"),
}

PROPERTY_COLORS = {
    "pos": "#A32D2D",
    "dep_rel": "#534AB7",
    "depth": "#1D9E75",
}

PROPERTY_MARKERS = {
    "pos": "o",
    "dep_rel": "s",
    "depth": "D",
}

# Pairwise task constants
PAIRWISE_LABELS = {
    "arc": "Binary Arc Prediction",
    "relation": "Relation Classification",
}

PAIRWISE_METRICS = {
    "arc": ("f1", "F1 Score"),
    "relation": ("accuracy", "Accuracy"),
}

PAIRWISE_COLORS = {
    "arc": "#D4775D",
    "relation": "#2D7DA3",
}

PAIRWISE_MARKERS = {
    "arc": "^",
    "relation": "v",
}

PAIRWISE_DIVERGENCE_KEYS = {
    "arc": "pairwise_arc",
    "relation": "pairwise_relation",
}


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def _grid_size(per_head):
    """Get (n_layers, n_heads) from a per_head dict with 'L{l}_H{h}' keys."""
    layers, heads = [], []
    for key in per_head:
        parts = key.split("_")
        layers.append(int(parts[0][1:]))
        heads.append(int(parts[1][1:]))
    return max(layers) + 1, max(heads) + 1


def _parse_head_grid(per_head, metric_key, prop):
    """Extract a (n_layers, n_heads) grid of metric values from per_head dict."""
    n_layers, n_heads = _grid_size(per_head)

    grid = np.full((n_layers, n_heads), np.nan)
    for key, val in per_head.items():
        parts = key.split("_")
        l, h = int(parts[0][1:]), int(parts[1][1:])
        if val[prop] is not None:
            grid[l, h] = val[prop][metric_key]
    return grid, n_layers, n_heads


def _parse_pairwise_head_grid(per_head_pairwise, metric_key, task):
    """Extract a (n_layers, n_heads) grid from per_head_pairwise dict."""
    n_layers, n_heads = _grid_size(per_head_pairwise)

    grid = np.full((n_layers, n_heads), np.nan)
    for key, val in per_head_pairwise.items():
        parts = key.split("_")
        l, h = int(parts[0][1:]), int(parts[1][1:])
        if val.get(task) is not None and metric_key in val[task]:
            grid[l, h] = val[task][metric_key]
    return grid, n_layers, n_heads


def _has_pairwise(results):
    """Check if results contain pairwise probing data."""
    return (
        results.get("translator", {}).get("per_head_pairwise") is not None
        and len(results["translator"]["per_head_pairwise"]) > 0
    )


# ─────────────────────────────────────────────────────────────
# 1. Probing accuracy heatmaps: Translator vs Impossible vs Base
# ─────────────────────────────────────────────────────────────
def plot_probing_heatmaps(results, output_dir):
    """Side-by-side heatmaps of probing accuracy for each syntactic property."""
    model_keys = [k for k in ["translator", "impossible", "base"] if k in results]
    n_models = len(model_keys)

    for prop in ["pos", "dep_rel", "depth"]:
        metric_key, metric_label = PROPERTY_METRICS[prop]

        grids = {}
        for mk in model_keys:
            grids[mk], n_layers, n_heads = _parse_head_grid(
                results[mk]["per_head"], metric_key, prop
            )

        vmin = min(np.nanmin(g) for g in grids.values())
        vmax = max(np.nanmax(g) for g in grids.values())

        fig, axes = plt.subplots(1, n_models, figsize=(10 * n_models, 8))
        if n_models == 1:
            axes = [axes]

        for ax, mk in zip(axes, model_keys):
            grid = grids[mk]
            im = ax.imshow(
                grid, cmap="YlOrRd", aspect="auto",
                interpolation="nearest", vmin=vmin, vmax=vmax,
            )
            ax.set_xlabel("Attention Head Index", fontsize=12)
            ax.set_ylabel("Transformer Layer", fontsize=12)
            ax.set_title(MODEL_NAMES[mk], fontsize=12, fontweight="bold")
            ax.set_xticks(range(n_heads))
            ax.set_xticklabels([f"H{i}" for i in range(n_heads)], fontsize=8)
            ax.set_yticks(range(n_layers))
            ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)

            for i in range(n_layers):
                for j in range(n_heads):
                    val = grid[i, j]
                    if np.isnan(val):
                        continue
                    color = "white" if val > (vmin + vmax) / 2 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=5.5, color=color,
                    )

        fig.subplots_adjust(top=0.82, right=0.88, wspace=0.15)
        cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.65])
        cbar = fig.colorbar(im, cax=cbar_ax, label=metric_label)
        cbar.ax.tick_params(labelsize=10)

        fig.suptitle(
            f"Per-Head Probing {metric_label}: {PROPERTY_LABELS[prop]}\n"
            f"Higher = head encodes more syntactic information",
            fontsize=14, fontweight="bold", y=0.97, linespacing=1.5,
        )

        for ext in ("png", "pdf"):
            path = os.path.join(output_dir, f"probing_heatmap_{prop}.{ext}")
            plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: probing_heatmap_{prop}.png")


# ─────────────────────────────────────────────────────────────
# 2. Layer-wise probing accuracy profiles
# ─────────────────────────────────────────────────────────────
def plot_layer_profiles(results, output_dir):
    """Layer-wise mean probing performance for all three properties."""
    model_keys = [k for k in ["translator", "impossible", "base"] if k in results]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    for ax, prop in zip(axes, ["pos", "dep_rel", "depth"]):
        metric_key, metric_label = PROPERTY_METRICS[prop]

        for mk in model_keys:
            layer_vals = results[mk]["layer_summary"][prop]
            n_layers = len(layer_vals)
            ax.plot(
                range(n_layers), layer_vals,
                marker=MODEL_MARKERS[mk], linewidth=2.5,
                label=MODEL_NAMES[mk], color=MODEL_COLORS[mk],
                markersize=8,
            )

        # Word-embedding baselines
        for mk in model_keys:
            emb = results[mk]["word_embedding_baseline"].get(prop)
            if emb is not None:
                baseline_val = emb.get(metric_key, None)
                if baseline_val is not None:
                    ax.axhline(
                        y=baseline_val, color=MODEL_COLORS[mk],
                        linewidth=1.2, linestyle="--", alpha=0.5,
                        label=f"{MODEL_NAMES[mk].split('(')[0].strip()} word-emb",
                    )

        ax.set_xlabel("Transformer Layer", fontsize=12)
        ax.set_ylabel(f"Mean {metric_label}", fontsize=12)
        ax.set_title(PROPERTY_LABELS[prop], fontsize=13, fontweight="bold")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"{i}" for i in range(n_layers)], fontsize=9)
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Layer-wise Probing Performance: Translator vs Impossible LM vs GPT-2 Base\n"
        "Dashed lines = word-embedding baselines",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"probing_layer_profiles.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: probing_layer_profiles.png")


# ─────────────────────────────────────────────────────────────
# 3. Probing divergence heatmaps (ΔAcc / ΔSpearman)
# ─────────────────────────────────────────────────────────────
def plot_divergence_heatmaps(results, output_dir):
    """Three heatmaps of probing divergence (Translator - Impossible) per property."""
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    for idx, prop in enumerate(["pos", "dep_rel", "depth"]):
        ax = axes[idx]
        _, metric_label = PROPERTY_METRICS[prop]
        delta = np.array(results["divergence"][prop])
        n_layers, n_heads = delta.shape

        abs_max = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)))
        if abs_max == 0:
            abs_max = 1.0

        im = ax.imshow(
            delta,
            cmap="RdBu",
            aspect="auto",
            interpolation="nearest",
            vmin=-abs_max,
            vmax=abs_max,
        )

        ax.set_xlabel("Attention Head Index", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Transformer Layer", fontsize=12)
        ax.set_title(
            f"{PROPERTY_LABELS[prop]}\n"
            f"Blue = Translator higher  |  Red = Impossible higher",
            fontsize=11, fontweight="bold", linespacing=1.4,
        )
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{i}" for i in range(n_heads)], fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)

        for i in range(n_layers):
            for j in range(n_heads):
                val = delta[i, j]
                color = "white" if abs(val) > abs_max * 0.5 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=5.5, color=color,
                )

    fig.subplots_adjust(top=0.80, right=0.88, wspace=0.12)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax, label=f"Δ{metric_label} (Translator − Impossible)")
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        "Per-Head Probing Divergence  —  Translator vs Impossible LM\n"
        "Positive = translation fine-tuning induces syntactic encoding",
        fontsize=16, fontweight="bold", y=0.97, linespacing=1.5,
    )

    path = os.path.join(output_dir, "probing_divergence_heatmaps.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    path_pdf = os.path.join(output_dir, "probing_divergence_heatmaps.pdf")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 4. Top divergent heads (horizontal bar chart)
# ─────────────────────────────────────────────────────────────
def plot_top_divergent_heads(results, output_dir, top_n=10):
    """Top heads by probing divergence across all three properties."""
    all_heads = set()
    for prop in ["pos", "dep_rel", "depth"]:
        delta = np.array(results["divergence"][prop])
        n_layers, n_heads = delta.shape
        flat = [(l, h, delta[l, h]) for l in range(n_layers) for h in range(n_heads)]
        flat.sort(key=lambda x: abs(x[2]), reverse=True)
        for l, h, _ in flat[:top_n]:
            all_heads.add((l, h))

    if not all_heads:
        print("No divergent heads found, skipping top heads plot.")
        return

    all_heads = sorted(all_heads)
    head_labels = [f"Layer {l}, Head {h}" for l, h in all_heads]

    fig, ax = plt.subplots(figsize=(13, max(5, len(all_heads) * 0.4)))

    bar_height = 0.25
    y_positions = np.arange(len(all_heads))

    for i, prop in enumerate(["pos", "dep_rel", "depth"]):
        delta = np.array(results["divergence"][prop])
        values = [delta[l, h] for l, h in all_heads]
        ax.barh(
            y_positions + i * bar_height,
            values,
            height=bar_height,
            color=PROPERTY_COLORS[prop],
            alpha=0.85,
            label=PROPERTY_LABELS[prop],
        )

    ax.set_yticks(y_positions + bar_height)
    ax.set_yticklabels(head_labels, fontsize=10)
    ax.set_xlabel("Probing Divergence (Translator − Impossible)", fontsize=13)
    ax.set_title(
        "Most Divergent Heads by Probing Accuracy\n"
        "Positive = Translator encodes more syntax  |  Negative = Impossible encodes more",
        fontsize=13, fontweight="bold", linespacing=1.5,
    )
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)

    plt.tight_layout()
    path = os.path.join(output_dir, "probing_top_heads.png")
    plt.savefig(path, dpi=200)
    path_pdf = os.path.join(output_dir, "probing_top_heads.pdf")
    plt.savefig(path_pdf)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 5. Baseline comparison bar chart
# ─────────────────────────────────────────────────────────────
def plot_baseline_comparison(results, output_dir):
    """Bar chart comparing probe accuracy against all baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for ax, prop in zip(axes, ["pos", "dep_rel", "depth"]):
        metric_key, metric_label = PROPERTY_METRICS[prop]

        categories = []
        values = []
        colors = []

        # Head means for each model
        model_keys = [k for k in ["translator", "impossible", "base"] if k in results]
        short_names = {"translator": "Translator", "impossible": "Impossible", "base": "GPT-2 Base"}
        for mk in model_keys:
            layer_vals = results[mk]["layer_summary"][prop]
            categories.append(f"{short_names[mk]}\n(head mean)")
            values.append(np.mean(layer_vals))
            colors.append(MODEL_COLORS[mk])

        # Word-embedding baselines
        emb_colors = {"translator": "#D4775D", "impossible": "#8A7DC9", "base": "#5DBB9E"}
        for mk in model_keys:
            emb = results[mk]["word_embedding_baseline"].get(prop)
            if emb is not None and metric_key in emb:
                categories.append(f"Word-Emb\n({short_names[mk]})")
                values.append(emb[metric_key])
                colors.append(emb_colors[mk])

        # Random baseline (classification only) — use first available model's baselines
        if prop != "depth":
            ref_emb = None
            for mk in model_keys:
                ref_emb = results[mk]["word_embedding_baseline"].get(prop)
                if ref_emb is not None:
                    break
            if ref_emb is not None and "random_baseline" in ref_emb:
                categories.append("Random\nBaseline")
                values.append(ref_emb["random_baseline"])
                colors.append("#999999")
            if ref_emb is not None and "majority_baseline" in ref_emb:
                categories.append("Majority\nBaseline")
                values.append(ref_emb["majority_baseline"])
                colors.append("#BBBBBB")
            if ref_emb is not None and "random_label_baseline" in ref_emb:
                categories.append("Random-Label\nBaseline")
                values.append(ref_emb["random_label_baseline"])
                colors.append("#DDDDDD")

        bars = ax.bar(
            range(len(values)), values,
            color=colors, alpha=0.85, edgecolor="white", linewidth=0.8,
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(PROPERTY_LABELS[prop], fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.15)

    fig.suptitle(
        "Probing Performance vs Baselines\n"
        "Head mean = average across all attention heads",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "probing_baselines.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    path_pdf = os.path.join(output_dir, "probing_baselines.pdf")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 6. Entropy–probing correlation scatter
# ─────────────────────────────────────────────────────────────
def plot_entropy_probing_correlation(results, entropy_results_path, output_dir):
    """Scatter plot: entropy divergence vs probing divergence per head."""
    if entropy_results_path is None:
        print("No entropy results provided, skipping correlation plot.")
        return

    with open(entropy_results_path, "r") as f:
        entropy_data = json.load(f)

    # Get entropy divergence matrix
    if "comparisons" in entropy_data:
        delta_H = np.array(
            entropy_data["comparisons"]["translation_vs_impossible"]["delta_H"]
        )
    elif "raw_entropy" in entropy_data:
        H_t = np.array(entropy_data["raw_entropy"]["H_translation"])
        H_i = np.array(entropy_data["raw_entropy"]["H_impossible"])
        delta_H = H_i - H_t
    else:
        print("Cannot parse entropy results, skipping correlation plot.")
        return

    flat_entropy = delta_H.flatten()
    correlations = results.get("entropy_probing_correlation", {})

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    for ax, prop in zip(axes, ["pos", "dep_rel", "depth"]):
        delta_probe = np.array(results["divergence"][prop])
        flat_probe = delta_probe.flatten()

        _, metric_label = PROPERTY_METRICS[prop]

        # Color by layer
        n_layers, n_heads = delta_probe.shape
        layer_colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

        for layer in range(n_layers):
            idx_start = layer * n_heads
            idx_end = idx_start + n_heads
            ax.scatter(
                flat_entropy[idx_start:idx_end],
                flat_probe[idx_start:idx_end],
                color=layer_colors[layer],
                label=f"Layer {layer}",
                s=50, alpha=0.7,
                edgecolors="white", linewidth=0.3,
            )

        # Trend line
        mask = np.isfinite(flat_entropy) & np.isfinite(flat_probe)
        if mask.sum() > 2:
            z = np.polyfit(flat_entropy[mask], flat_probe[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(flat_entropy[mask].min(), flat_entropy[mask].max(), 100)
            ax.plot(x_line, p(x_line), "k--", alpha=0.4, linewidth=1.5)

        # Annotate correlation
        corr = correlations.get(prop, {})
        rho = corr.get("rho")
        p_val = corr.get("p_value")
        sig = corr.get("significant", False)
        if rho is not None:
            sig_str = "***" if sig else "n.s."
            ax.text(
                0.05, 0.95,
                f"ρ = {rho:.3f}\np = {p_val:.2e} {sig_str}",
                transform=ax.transAxes,
                fontsize=11, fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

        ax.set_xlabel("Entropy Divergence ΔH (bits)", fontsize=12)
        ax.set_ylabel(f"Probing Divergence Δ{metric_label}", fontsize=12)
        ax.set_title(PROPERTY_LABELS[prop], fontsize=13, fontweight="bold")
        ax.grid(alpha=0.15)

    # Shared layer legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center right",
        fontsize=8, ncol=1,
        bbox_to_anchor=(0.99, 0.5),
        title="Layer", title_fontsize=9,
    )

    fig.suptitle(
        "Entropy–Probing Correlation: Do Attention Changes Predict Syntactic Encoding?\n"
        "Each point = one attention head  |  ρ = Spearman correlation",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    path = os.path.join(output_dir, "entropy_probing_correlation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    path_pdf = os.path.join(output_dir, "entropy_probing_correlation.pdf")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 7. Layer-wise divergence profile (all 3 properties)
# ─────────────────────────────────────────────────────────────
def plot_divergence_layer_profile(results, output_dir):
    """Mean probing divergence per layer for all three properties."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for prop in ["pos", "dep_rel", "depth"]:
        delta = np.array(results["divergence"][prop])
        n_layers = delta.shape[0]
        layer_means = [np.mean(delta[l, :]) for l in range(n_layers)]

        _, metric_label = PROPERTY_METRICS[prop]
        ax.plot(
            range(n_layers), layer_means,
            marker=PROPERTY_MARKERS[prop],
            linewidth=2.5,
            label=f"{PROPERTY_LABELS[prop]} (Δ{metric_label})",
            color=PROPERTY_COLORS[prop],
            markersize=8,
        )

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.text(
        n_layers - 0.5, 0.002,
        "no divergence",
        fontsize=9, color="gray", ha="right", style="italic",
    )

    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel("Mean Probing Divergence (Translator − Impossible)", fontsize=14)
    ax.set_title(
        "Layer-wise Probing Divergence\n"
        "Positive = translation fine-tuning adds syntactic encoding in that layer",
        fontsize=13, fontweight="bold", linespacing=1.5,
    )
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {i}" for i in range(n_layers)], fontsize=9, rotation=45)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    path = os.path.join(output_dir, "probing_divergence_layer_profile.png")
    plt.savefig(path, dpi=200)
    path_pdf = os.path.join(output_dir, "probing_divergence_layer_profile.pdf")
    plt.savefig(path_pdf)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 7b. Three-way divergence layer profiles (token-level)
# ─────────────────────────────────────────────────────────────
DIVERGENCE_PAIR_LABELS = {
    "divergence": "Translator vs Impossible",
    "divergence_vs_base": "Translator vs GPT-2 Base",
    "divergence_base_vs_impossible": "GPT-2 Base vs Impossible",
}

DIVERGENCE_PAIR_COLORS = {
    "divergence": "#A32D2D",
    "divergence_vs_base": "#534AB7",
    "divergence_base_vs_impossible": "#1D9E75",
}

DIVERGENCE_PAIR_MARKERS = {
    "divergence": "o",
    "divergence_vs_base": "s",
    "divergence_base_vs_impossible": "D",
}

DIVERGENCE_PAIR_STYLES = {
    "divergence": "-",
    "divergence_vs_base": "--",
    "divergence_base_vs_impossible": ":",
}


def plot_threeway_divergence_layer_profile(results, output_dir):
    """Three-way model comparison: mean probing divergence per layer.

    Shows whether translator recovers to baseline or exceeds it.
    """
    div_keys = [k for k in ["divergence", "divergence_vs_base",
                             "divergence_base_vs_impossible"]
                if k in results]
    if len(div_keys) < 2:
        print("Need at least 2 divergence sets for three-way plot, skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for ax, prop in zip(axes, ["pos", "dep_rel", "depth"]):
        _, metric_label = PROPERTY_METRICS[prop]

        for dk in div_keys:
            if prop not in results[dk]:
                continue
            delta = np.array(results[dk][prop])
            n_layers = delta.shape[0]
            layer_means = [np.mean(delta[l, :]) for l in range(n_layers)]

            ax.plot(
                range(n_layers), layer_means,
                marker=DIVERGENCE_PAIR_MARKERS[dk],
                linestyle=DIVERGENCE_PAIR_STYLES[dk],
                linewidth=2.5,
                label=DIVERGENCE_PAIR_LABELS[dk],
                color=DIVERGENCE_PAIR_COLORS[dk],
                markersize=8,
            )

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xlabel("Transformer Layer", fontsize=12)
        ax.set_ylabel(f"Mean Δ{metric_label}", fontsize=12)
        ax.set_title(PROPERTY_LABELS[prop], fontsize=13, fontweight="bold")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"{i}" for i in range(n_layers)], fontsize=9)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Three-Way Probing Divergence: Does Translation Recover or Exceed Baseline?\n"
        "Translator vs Base near 0 = recovery  |  Positive = novel syntactic encoding",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"threeway_divergence_layer_profile.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: threeway_divergence_layer_profile.png")


def plot_threeway_pairwise_divergence_layer_profile(results, output_dir):
    """Three-way comparison for pairwise probing divergence."""
    div_keys = [k for k in ["divergence", "divergence_vs_base",
                             "divergence_base_vs_impossible"]
                if k in results]
    if len(div_keys) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, task in zip(axes, ["arc", "relation"]):
        div_data_key = PAIRWISE_DIVERGENCE_KEYS[task]
        _, metric_label = PAIRWISE_METRICS[task]

        for dk in div_keys:
            if div_data_key not in results[dk]:
                continue
            delta = np.array(results[dk][div_data_key])
            n_layers = delta.shape[0]
            layer_means = [np.mean(delta[l, :]) for l in range(n_layers)]

            ax.plot(
                range(n_layers), layer_means,
                marker=DIVERGENCE_PAIR_MARKERS[dk],
                linestyle=DIVERGENCE_PAIR_STYLES[dk],
                linewidth=2.5,
                label=DIVERGENCE_PAIR_LABELS[dk],
                color=DIVERGENCE_PAIR_COLORS[dk],
                markersize=8,
            )

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xlabel("Transformer Layer", fontsize=12)
        ax.set_ylabel(f"Mean Δ{metric_label}", fontsize=12)
        ax.set_title(PAIRWISE_LABELS[task], fontsize=13, fontweight="bold")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"{i}" for i in range(n_layers)], fontsize=9)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Three-Way Pairwise Probing Divergence\n"
        "Does translation recover relational structure or create novel encoding?",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"threeway_pairwise_divergence_layer_profile.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: threeway_pairwise_divergence_layer_profile.png")


# ─────────────────────────────────────────────────────────────
# 8. Pairwise: heatmaps (Translator vs Impossible per task)
# ─────────────────────────────────────────────────────────────
def plot_pairwise_heatmaps(results, output_dir):
    """Side-by-side heatmaps of pairwise probing per task."""
    model_keys = [k for k in ["translator", "impossible", "base"]
                  if k in results and results[k].get("per_head_pairwise")]
    n_models = len(model_keys)

    for task in ["arc", "relation"]:
        metric_key, metric_label = PAIRWISE_METRICS[task]

        grids = {}
        for mk in model_keys:
            grids[mk], n_layers, n_heads = _parse_pairwise_head_grid(
                results[mk]["per_head_pairwise"], metric_key, task
            )

        vmin = min(np.nanmin(g) for g in grids.values())
        vmax = max(np.nanmax(g) for g in grids.values())

        fig, axes = plt.subplots(1, n_models, figsize=(10 * n_models, 8))
        if n_models == 1:
            axes = [axes]

        for ax, mk in zip(axes, model_keys):
            grid = grids[mk]
            im = ax.imshow(
                grid, cmap="YlOrRd", aspect="auto",
                interpolation="nearest", vmin=vmin, vmax=vmax,
            )
            ax.set_xlabel("Attention Head Index", fontsize=12)
            ax.set_ylabel("Transformer Layer", fontsize=12)
            ax.set_title(MODEL_NAMES[mk], fontsize=12, fontweight="bold")
            ax.set_xticks(range(n_heads))
            ax.set_xticklabels([f"H{i}" for i in range(n_heads)], fontsize=8)
            ax.set_yticks(range(n_layers))
            ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)

            for i in range(n_layers):
                for j in range(n_heads):
                    val = grid[i, j]
                    if np.isnan(val):
                        continue
                    color = "white" if val > (vmin + vmax) / 2 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=5.5, color=color,
                    )

        fig.subplots_adjust(top=0.82, right=0.88, wspace=0.15)
        cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.65])
        fig.colorbar(im, cax=cbar_ax, label=metric_label)
        cbar_ax.tick_params(labelsize=10)

        fig.suptitle(
            f"Per-Head Pairwise Probing {metric_label}: {PAIRWISE_LABELS[task]}\n"
            f"Higher = head encodes more relational dependency structure",
            fontsize=14, fontweight="bold", y=0.97, linespacing=1.5,
        )

        for ext in ("png", "pdf"):
            path = os.path.join(output_dir, f"pairwise_heatmap_{task}.{ext}")
            plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: pairwise_heatmap_{task}.png")


# ─────────────────────────────────────────────────────────────
# 9. Pairwise: layer-wise profiles
# ─────────────────────────────────────────────────────────────
def plot_pairwise_layer_profiles(results, output_dir):
    """Layer-wise pairwise probing for arc / relation."""
    model_keys = [k for k in ["translator", "impossible", "base"]
                  if k in results and results[k].get("layer_summary_pairwise")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, task in zip(axes, ["arc", "relation"]):
        metric_key, metric_label = PAIRWISE_METRICS[task]

        for mk in model_keys:
            layer_vals = results[mk]["layer_summary_pairwise"][task]
            n_layers = len(layer_vals)
            ax.plot(
                range(n_layers), layer_vals,
                marker=MODEL_MARKERS[mk], linewidth=2.5,
                label=MODEL_NAMES[mk], color=MODEL_COLORS[mk],
                markersize=8,
            )

        # Pairwise baselines (from first available model)
        baseline_map = {
            "arc": ("word_emb_arc", "f1", "distance_arc", "f1"),
            "relation": ("word_emb_rel", "accuracy", None, None),
        }
        emb_key, emb_metric, dist_key, dist_metric = baseline_map[task]

        for mk in model_keys:
            pw_bl = results[mk].get("pairwise_baselines", {})
            if emb_key and pw_bl.get(emb_key):
                val = pw_bl[emb_key].get(emb_metric)
                if val is not None:
                    short = MODEL_NAMES[mk].split("(")[0].strip()
                    ax.axhline(
                        y=val, color=MODEL_COLORS[mk],
                        linewidth=1.2, linestyle="--", alpha=0.5,
                        label=f"{short} word-emb baseline",
                    )

        # Distance baseline (one line, model-independent)
        if dist_key:
            for mk in model_keys:
                pw_bl = results[mk].get("pairwise_baselines", {})
                if pw_bl.get(dist_key):
                    val = pw_bl[dist_key].get(dist_metric)
                    if val is not None:
                        ax.axhline(
                            y=val, color="#999999",
                            linewidth=1.2, linestyle=":", alpha=0.6,
                            label="Distance baseline",
                        )
                        break  # only one line needed

        ax.set_xlabel("Transformer Layer", fontsize=12)
        ax.set_ylabel(f"Mean {metric_label}", fontsize=12)
        ax.set_title(PAIRWISE_LABELS[task], fontsize=13, fontweight="bold")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"{i}" for i in range(n_layers)], fontsize=9)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Layer-wise Pairwise Dependency Probing\n"
        "Dashed = word-emb baseline  |  Dotted = distance baseline",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"pairwise_layer_profiles.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: pairwise_layer_profiles.png")


# ─────────────────────────────────────────────────────────────
# 10. Pairwise: divergence heatmaps
# ─────────────────────────────────────────────────────────────
def plot_pairwise_divergence_heatmaps(results, output_dir):
    """Heatmaps of pairwise probing divergence."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for idx, task in enumerate(["arc", "relation"]):
        ax = axes[idx]
        div_key = PAIRWISE_DIVERGENCE_KEYS[task]
        _, metric_label = PAIRWISE_METRICS[task]

        if div_key not in results["divergence"]:
            ax.set_visible(False)
            continue

        delta = np.array(results["divergence"][div_key])
        n_layers, n_heads = delta.shape

        abs_max = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)))
        if abs_max == 0:
            abs_max = 1.0

        im = ax.imshow(
            delta, cmap="RdBu", aspect="auto",
            interpolation="nearest", vmin=-abs_max, vmax=abs_max,
        )

        ax.set_xlabel("Attention Head Index", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Transformer Layer", fontsize=12)
        ax.set_title(
            f"{PAIRWISE_LABELS[task]}\n"
            f"Blue = Translator higher  |  Red = Impossible higher",
            fontsize=11, fontweight="bold", linespacing=1.4,
        )
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{i}" for i in range(n_heads)], fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)

        for i in range(n_layers):
            for j in range(n_heads):
                val = delta[i, j]
                color = "white" if abs(val) > abs_max * 0.5 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=5.5, color=color,
                )

    fig.subplots_adjust(top=0.80, right=0.88, wspace=0.12)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.65])
    fig.colorbar(im, cax=cbar_ax, label=f"Δ (Translator − Impossible)")
    cbar_ax.tick_params(labelsize=10)

    fig.suptitle(
        "Per-Head Pairwise Probing Divergence  —  Translator vs Impossible LM\n"
        "Positive = translation fine-tuning induces relational structure encoding",
        fontsize=16, fontweight="bold", y=0.97, linespacing=1.5,
    )

    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"pairwise_divergence_heatmaps.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: pairwise_divergence_heatmaps.png")


# ─────────────────────────────────────────────────────────────
# 11. Pairwise: divergence layer profile
# ─────────────────────────────────────────────────────────────
def plot_pairwise_divergence_layer_profile(results, output_dir):
    """Mean pairwise probing divergence per layer for all three tasks."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for task in ["arc", "relation"]:
        div_key = PAIRWISE_DIVERGENCE_KEYS[task]
        if div_key not in results["divergence"]:
            continue

        delta = np.array(results["divergence"][div_key])
        n_layers = delta.shape[0]
        layer_means = [np.mean(delta[l, :]) for l in range(n_layers)]

        _, metric_label = PAIRWISE_METRICS[task]
        ax.plot(
            range(n_layers), layer_means,
            marker=PAIRWISE_MARKERS[task], linewidth=2.5,
            label=f"{PAIRWISE_LABELS[task]} (Δ{metric_label})",
            color=PAIRWISE_COLORS[task], markersize=8,
        )

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel("Mean Pairwise Divergence (Translator − Impossible)", fontsize=14)
    ax.set_title(
        "Layer-wise Pairwise Probing Divergence\n"
        "Positive = translation fine-tuning adds relational encoding in that layer",
        fontsize=13, fontweight="bold", linespacing=1.5,
    )
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {i}" for i in range(n_layers)], fontsize=9, rotation=45)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"pairwise_divergence_layer_profile.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: pairwise_divergence_layer_profile.png")


# ─────────────────────────────────────────────────────────────
# 12. Pairwise: baseline comparison bar chart
# ─────────────────────────────────────────────────────────────
def plot_pairwise_baseline_comparison(results, output_dir):
    """Bar chart: pairwise probe accuracy vs all baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, task in zip(axes, ["arc", "relation"]):
        metric_key, metric_label = PAIRWISE_METRICS[task]
        categories, values, colors = [], [], []

        # Head means for each model
        model_keys = [k for k in ["translator", "impossible", "base"] if k in results]
        short_names = {"translator": "Translator", "impossible": "Impossible", "base": "GPT-2 Base"}
        for mk in model_keys:
            pw_summ = results[mk].get("layer_summary_pairwise", {})
            if task in pw_summ:
                categories.append(f"{short_names[mk]}\n(head mean)")
                values.append(np.mean(pw_summ[task]))
                colors.append(MODEL_COLORS[mk])

        # Pairwise baselines (from translator — baselines are model-independent input)
        pw_bl = results["translator"].get("pairwise_baselines", {})

        if task == "arc":
            if pw_bl.get("word_emb_arc") and "f1" in pw_bl["word_emb_arc"]:
                categories.append("Word-Emb\nPairwise")
                values.append(pw_bl["word_emb_arc"]["f1"])
                colors.append("#D4775D")
            if pw_bl.get("distance_arc") and "f1" in pw_bl["distance_arc"]:
                categories.append("Distance\nBaseline")
                values.append(pw_bl["distance_arc"]["f1"])
                colors.append("#8B8B8B")
            if pw_bl.get("word_emb_arc") and "random_baseline_acc" in pw_bl["word_emb_arc"]:
                categories.append("Random\nBaseline")
                values.append(pw_bl["word_emb_arc"]["random_baseline_acc"])
                colors.append("#999999")
            if pw_bl.get("word_emb_arc") and "majority_baseline_acc" in pw_bl["word_emb_arc"]:
                categories.append("Majority\nBaseline")
                values.append(pw_bl["word_emb_arc"]["majority_baseline_acc"])
                colors.append("#BBBBBB")
            if pw_bl.get("word_emb_arc") and "random_label_f1" in pw_bl["word_emb_arc"]:
                categories.append("Random-Label\nBaseline")
                values.append(pw_bl["word_emb_arc"]["random_label_f1"])
                colors.append("#DDDDDD")

        elif task == "relation":
            if pw_bl.get("word_emb_rel") and "accuracy" in pw_bl["word_emb_rel"]:
                categories.append("Word-Emb\nPairwise")
                values.append(pw_bl["word_emb_rel"]["accuracy"])
                colors.append("#D4775D")
            if pw_bl.get("word_emb_rel") and "random_baseline" in pw_bl["word_emb_rel"]:
                categories.append("Random\nBaseline")
                values.append(pw_bl["word_emb_rel"]["random_baseline"])
                colors.append("#999999")
            if pw_bl.get("word_emb_rel") and "majority_baseline" in pw_bl["word_emb_rel"]:
                categories.append("Majority\nBaseline")
                values.append(pw_bl["word_emb_rel"]["majority_baseline"])
                colors.append("#BBBBBB")

        if not values:
            ax.set_visible(False)
            continue

        bars = ax.bar(
            range(len(values)), values,
            color=colors, alpha=0.85, edgecolor="white", linewidth=0.8,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(PAIRWISE_LABELS[task], fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.15)

    fig.suptitle(
        "Pairwise Dependency Probing vs Baselines\n"
        "Head mean = average across all attention heads",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"pairwise_baselines.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: pairwise_baselines.png")


# ─────────────────────────────────────────────────────────────
# 13. Pairwise: top divergent heads (bar chart)
# ─────────────────────────────────────────────────────────────
def plot_pairwise_top_divergent_heads(results, output_dir, top_n=10):
    """Top heads by pairwise probing divergence across all three tasks."""
    all_heads = set()
    for task in ["arc", "relation"]:
        div_key = PAIRWISE_DIVERGENCE_KEYS[task]
        if div_key not in results["divergence"]:
            continue
        delta = np.array(results["divergence"][div_key])
        n_layers, n_heads = delta.shape
        flat = [(l, h, delta[l, h]) for l in range(n_layers) for h in range(n_heads)]
        flat.sort(key=lambda x: abs(x[2]), reverse=True)
        for l, h, _ in flat[:top_n]:
            all_heads.add((l, h))

    if not all_heads:
        print("No pairwise divergent heads found, skipping.")
        return

    all_heads = sorted(all_heads)
    head_labels = [f"Layer {l}, Head {h}" for l, h in all_heads]

    fig, ax = plt.subplots(figsize=(13, max(5, len(all_heads) * 0.4)))

    bar_height = 0.25
    y_positions = np.arange(len(all_heads))

    for i, task in enumerate(["arc", "relation"]):
        div_key = PAIRWISE_DIVERGENCE_KEYS[task]
        if div_key not in results["divergence"]:
            continue
        delta = np.array(results["divergence"][div_key])
        values = [delta[l, h] for l, h in all_heads]
        ax.barh(
            y_positions + i * bar_height, values,
            height=bar_height, color=PAIRWISE_COLORS[task],
            alpha=0.85, label=PAIRWISE_LABELS[task],
        )

    ax.set_yticks(y_positions + bar_height)
    ax.set_yticklabels(head_labels, fontsize=10)
    ax.set_xlabel("Pairwise Probing Divergence (Translator − Impossible)", fontsize=13)
    ax.set_title(
        "Most Divergent Heads by Pairwise Dependency Probing\n"
        "Positive = Translator encodes more relational structure",
        fontsize=13, fontweight="bold", linespacing=1.5,
    )
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"pairwise_top_heads.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: pairwise_top_heads.png")


# ─────────────────────────────────────────────────────────────
# 14. Combined: token-level + pairwise divergence overlay
# ─────────────────────────────────────────────────────────────
def plot_combined_divergence_profile(results, output_dir):
    """All 6 divergence curves (3 token-level + 3 pairwise) on one figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7))

    # Token-level
    for prop in ["pos", "dep_rel", "depth"]:
        delta = np.array(results["divergence"][prop])
        n_layers = delta.shape[0]
        layer_means = [np.mean(delta[l, :]) for l in range(n_layers)]
        _, metric_label = PROPERTY_METRICS[prop]
        ax1.plot(
            range(n_layers), layer_means,
            marker=PROPERTY_MARKERS[prop], linewidth=2.5,
            label=f"{PROPERTY_LABELS[prop]} (Δ{metric_label})",
            color=PROPERTY_COLORS[prop], markersize=8,
        )

    ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax1.set_xlabel("Transformer Layer", fontsize=14)
    ax1.set_ylabel("Mean Divergence (Translator − Impossible)", fontsize=12)
    ax1.set_title("Token-Level Probing Divergence", fontsize=13, fontweight="bold")
    ax1.set_xticks(range(n_layers))
    ax1.set_xticklabels([f"{i}" for i in range(n_layers)], fontsize=9)
    ax1.legend(fontsize=9, loc="best", framealpha=0.9)
    ax1.grid(alpha=0.15)

    # Pairwise
    for task in ["arc", "relation"]:
        div_key = PAIRWISE_DIVERGENCE_KEYS[task]
        if div_key not in results["divergence"]:
            continue
        delta = np.array(results["divergence"][div_key])
        n_layers = delta.shape[0]
        layer_means = [np.mean(delta[l, :]) for l in range(n_layers)]
        _, metric_label = PAIRWISE_METRICS[task]
        ax2.plot(
            range(n_layers), layer_means,
            marker=PAIRWISE_MARKERS[task], linewidth=2.5,
            label=f"{PAIRWISE_LABELS[task]} (Δ{metric_label})",
            color=PAIRWISE_COLORS[task], markersize=8,
        )

    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax2.set_xlabel("Transformer Layer", fontsize=14)
    ax2.set_ylabel("Mean Divergence (Translator − Impossible)", fontsize=12)
    ax2.set_title("Pairwise Dependency Probing Divergence", fontsize=13, fontweight="bold")
    ax2.set_xticks(range(n_layers))
    ax2.set_xticklabels([f"{i}" for i in range(n_layers)], fontsize=9)
    ax2.legend(fontsize=9, loc="best", framealpha=0.9)
    ax2.grid(alpha=0.15)

    fig.suptitle(
        "Token-Level vs Pairwise Probing Divergence by Layer\n"
        "Positive = translation fine-tuning induces syntactic encoding",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"combined_divergence_profile.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: combined_divergence_profile.png")


# ─────────────────────────────────────────────────────────────
# 15. Pairwise: entropy-probing correlation scatter
# ─────────────────────────────────────────────────────────────
def plot_pairwise_entropy_correlation(results, entropy_results_path, output_dir):
    """Scatter: entropy divergence vs pairwise probing divergence per head."""
    if entropy_results_path is None:
        print("No entropy results provided, skipping pairwise correlation plot.")
        return

    with open(entropy_results_path, "r") as f:
        entropy_data = json.load(f)

    if "comparisons" in entropy_data:
        delta_H = np.array(
            entropy_data["comparisons"]["translation_vs_impossible"]["delta_H"]
        )
    elif "raw_entropy" in entropy_data:
        H_t = np.array(entropy_data["raw_entropy"]["H_translation"])
        H_i = np.array(entropy_data["raw_entropy"]["H_impossible"])
        delta_H = H_i - H_t
    else:
        print("Cannot parse entropy results, skipping pairwise correlation plot.")
        return

    flat_entropy = delta_H.flatten()
    correlations = results.get("entropy_probing_correlation", {})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, task in zip(axes, ["arc", "relation"]):
        div_key = PAIRWISE_DIVERGENCE_KEYS[task]
        if div_key not in results["divergence"]:
            ax.set_visible(False)
            continue

        delta_probe = np.array(results["divergence"][div_key])
        flat_probe = delta_probe.flatten()
        _, metric_label = PAIRWISE_METRICS[task]

        n_layers, n_heads = delta_probe.shape
        layer_colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

        for layer in range(n_layers):
            idx_start = layer * n_heads
            idx_end = idx_start + n_heads
            ax.scatter(
                flat_entropy[idx_start:idx_end],
                flat_probe[idx_start:idx_end],
                color=layer_colors[layer], label=f"Layer {layer}",
                s=50, alpha=0.7, edgecolors="white", linewidth=0.3,
            )

        mask = np.isfinite(flat_entropy) & np.isfinite(flat_probe)
        if mask.sum() > 2:
            z = np.polyfit(flat_entropy[mask], flat_probe[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(flat_entropy[mask].min(), flat_entropy[mask].max(), 100)
            ax.plot(x_line, p(x_line), "k--", alpha=0.4, linewidth=1.5)

        corr = correlations.get(div_key, {})
        rho = corr.get("rho")
        p_val = corr.get("p_value")
        sig = corr.get("significant", False)
        if rho is not None:
            sig_str = "***" if sig else "n.s."
            ax.text(
                0.05, 0.95,
                f"ρ = {rho:.3f}\np = {p_val:.2e} {sig_str}",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

        ax.set_xlabel("Entropy Divergence ΔH (bits)", fontsize=12)
        ax.set_ylabel(f"Pairwise Divergence Δ{metric_label}", fontsize=12)
        ax.set_title(PAIRWISE_LABELS[task], fontsize=13, fontweight="bold")
        ax.grid(alpha=0.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center right",
        fontsize=8, ncol=1, bbox_to_anchor=(0.99, 0.5),
        title="Layer", title_fontsize=9,
    )

    fig.suptitle(
        "Entropy–Pairwise Probing Correlation\n"
        "Each point = one attention head  |  ρ = Spearman correlation",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"pairwise_entropy_correlation.{ext}")
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: pairwise_entropy_correlation.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot probing classifier results (RQ2)"
    )
    parser.add_argument(
        "--results", type=str, required=True,
        help="Path to probing_results.json",
    )
    parser.add_argument(
        "--entropy_results", type=str, default=None,
        help="Path to entropy results JSON (for correlation plot)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./plots",
        help="Directory to save plots (default: ./plots)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.results)

    print(f"Plotting probing results")
    print(f"Models:")
    print(f"  Translator: {results['models']['translation']}")
    print(f"  Impossible: {results['models']['impossible']}")
    if "base" in results["models"]:
        print(f"  Base:       {results['models']['base']}")
    print(f"  Sentences:  {results['n_sentences']}")
    print(f"Output directory: {args.output_dir}\n")

    plot_probing_heatmaps(results, args.output_dir)
    plot_layer_profiles(results, args.output_dir)
    plot_divergence_heatmaps(results, args.output_dir)
    plot_top_divergent_heads(results, args.output_dir)
    plot_baseline_comparison(results, args.output_dir)
    plot_divergence_layer_profile(results, args.output_dir)
    plot_entropy_probing_correlation(results, args.entropy_results, args.output_dir)

    # Three-way divergence (if base model data available)
    if "divergence_vs_base" in results:
        print("\nGenerating three-way divergence plots...")
        plot_threeway_divergence_layer_profile(results, args.output_dir)

    # Pairwise probing plots (if available)
    if _has_pairwise(results):
        print("\nGenerating pairwise probing plots...")
        plot_pairwise_heatmaps(results, args.output_dir)
        plot_pairwise_layer_profiles(results, args.output_dir)
        plot_pairwise_divergence_heatmaps(results, args.output_dir)
        plot_pairwise_divergence_layer_profile(results, args.output_dir)
        plot_pairwise_baseline_comparison(results, args.output_dir)
        plot_pairwise_top_divergent_heads(results, args.output_dir)
        plot_pairwise_entropy_correlation(results, args.entropy_results, args.output_dir)
        if "divergence_vs_base" in results:
            plot_threeway_pairwise_divergence_layer_profile(results, args.output_dir)
    else:
        print("\nNo pairwise probing data found, skipping pairwise plots.")

    print(f"\nAll plots saved to {args.output_dir}/")
