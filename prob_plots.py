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
}

MODEL_COLORS = {
    "translator": "#A32D2D",
    "impossible": "#534AB7",
}

MODEL_MARKERS = {
    "translator": "o",
    "impossible": "s",
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


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def _parse_head_grid(per_head, metric_key, prop):
    """Extract a (n_layers, n_heads) grid of metric values from per_head dict."""
    layers, heads = [], []
    for key in per_head:
        parts = key.split("_")
        layers.append(int(parts[0][1:]))
        heads.append(int(parts[1][1:]))
    n_layers = max(layers) + 1
    n_heads = max(heads) + 1

    grid = np.full((n_layers, n_heads), np.nan)
    for key, val in per_head.items():
        parts = key.split("_")
        l, h = int(parts[0][1:]), int(parts[1][1:])
        if val[prop] is not None:
            grid[l, h] = val[prop][metric_key]
    return grid, n_layers, n_heads


# ─────────────────────────────────────────────────────────────
# 1. Probing accuracy heatmaps: Translator vs Impossible
# ─────────────────────────────────────────────────────────────
def plot_probing_heatmaps(results, output_dir):
    """Side-by-side heatmaps of probing accuracy for each syntactic property."""
    for prop in ["pos", "dep_rel", "depth"]:
        metric_key, metric_label = PROPERTY_METRICS[prop]

        grid_t, n_layers, n_heads = _parse_head_grid(
            results["translator"]["per_head"], metric_key, prop
        )
        grid_i, _, _ = _parse_head_grid(
            results["impossible"]["per_head"], metric_key, prop
        )

        vmin = min(np.nanmin(grid_t), np.nanmin(grid_i))
        vmax = max(np.nanmax(grid_t), np.nanmax(grid_i))

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        for ax, grid, label in [
            (axes[0], grid_t, MODEL_NAMES["translator"]),
            (axes[1], grid_i, MODEL_NAMES["impossible"]),
        ]:
            im = ax.imshow(
                grid,
                cmap="YlOrRd",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlabel("Attention Head Index", fontsize=12)
            ax.set_ylabel("Transformer Layer", fontsize=12)
            ax.set_title(label, fontsize=12, fontweight="bold")
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
            fontsize=14,
            fontweight="bold",
            y=0.97,
            linespacing=1.5,
        )

        path = os.path.join(output_dir, f"probing_heatmap_{prop}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        path_pdf = os.path.join(output_dir, f"probing_heatmap_{prop}.pdf")
        plt.savefig(path_pdf, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 2. Layer-wise probing accuracy profiles
# ─────────────────────────────────────────────────────────────
def plot_layer_profiles(results, output_dir):
    """Layer-wise mean probing performance for all three properties."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    for ax, prop in zip(axes, ["pos", "dep_rel", "depth"]):
        metric_key, metric_label = PROPERTY_METRICS[prop]

        layer_t = results["translator"]["layer_summary"][prop]
        layer_i = results["impossible"]["layer_summary"][prop]
        n_layers = len(layer_t)

        ax.plot(
            range(n_layers), layer_t,
            marker=MODEL_MARKERS["translator"],
            linewidth=2.5,
            label=MODEL_NAMES["translator"],
            color=MODEL_COLORS["translator"],
            markersize=8,
        )
        ax.plot(
            range(n_layers), layer_i,
            marker=MODEL_MARKERS["impossible"],
            linewidth=2.5,
            label=MODEL_NAMES["impossible"],
            color=MODEL_COLORS["impossible"],
            markersize=8,
        )

        # Word-embedding baseline
        emb_t = results["translator"]["word_embedding_baseline"].get(prop)
        emb_i = results["impossible"]["word_embedding_baseline"].get(prop)
        if emb_t is not None:
            baseline_val = emb_t.get(metric_key, None)
            if baseline_val is not None:
                ax.axhline(
                    y=baseline_val, color=MODEL_COLORS["translator"],
                    linewidth=1.2, linestyle="--", alpha=0.5,
                    label="Translator word-emb baseline",
                )
        if emb_i is not None:
            baseline_val = emb_i.get(metric_key, None)
            if baseline_val is not None:
                ax.axhline(
                    y=baseline_val, color=MODEL_COLORS["impossible"],
                    linewidth=1.2, linestyle="--", alpha=0.5,
                    label="Impossible word-emb baseline",
                )

        ax.set_xlabel("Transformer Layer", fontsize=12)
        ax.set_ylabel(f"Mean {metric_label}", fontsize=12)
        ax.set_title(
            PROPERTY_LABELS[prop],
            fontsize=13, fontweight="bold",
        )
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"{i}" for i in range(n_layers)], fontsize=9)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Layer-wise Probing Performance: Translator vs Impossible LM\n"
        "Dashed lines = word-embedding baselines",
        fontsize=14, fontweight="bold", y=1.02, linespacing=1.5,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "probing_layer_profiles.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    path_pdf = os.path.join(output_dir, "probing_layer_profiles.pdf")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


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

        # Translator mean across heads
        layer_t = results["translator"]["layer_summary"][prop]
        mean_t = np.mean(layer_t)
        categories.append("Translator\n(head mean)")
        values.append(mean_t)
        colors.append(MODEL_COLORS["translator"])

        # Impossible mean across heads
        layer_i = results["impossible"]["layer_summary"][prop]
        mean_i = np.mean(layer_i)
        categories.append("Impossible\n(head mean)")
        values.append(mean_i)
        colors.append(MODEL_COLORS["impossible"])

        # Word-embedding baselines
        emb_t = results["translator"]["word_embedding_baseline"].get(prop)
        if emb_t is not None and metric_key in emb_t:
            categories.append("Word-Emb\n(Translator)")
            values.append(emb_t[metric_key])
            colors.append("#D4775D")

        emb_i = results["impossible"]["word_embedding_baseline"].get(prop)
        if emb_i is not None and metric_key in emb_i:
            categories.append("Word-Emb\n(Impossible)")
            values.append(emb_i[metric_key])
            colors.append("#8A7DC9")

        # Random baseline (classification only)
        if prop != "depth":
            if emb_t is not None and "random_baseline" in emb_t:
                categories.append("Random\nBaseline")
                values.append(emb_t["random_baseline"])
                colors.append("#999999")
            if emb_t is not None and "majority_baseline" in emb_t:
                categories.append("Majority\nBaseline")
                values.append(emb_t["majority_baseline"])
                colors.append("#BBBBBB")
            if emb_t is not None and "random_label_baseline" in emb_t:
                categories.append("Random-Label\nBaseline")
                values.append(emb_t["random_label_baseline"])
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
    print(f"  Sentences:  {results['n_sentences']}")
    print(f"Output directory: {args.output_dir}\n")

    plot_probing_heatmaps(results, args.output_dir)
    plot_layer_profiles(results, args.output_dir)
    plot_divergence_heatmaps(results, args.output_dir)
    plot_top_divergent_heads(results, args.output_dir)
    plot_baseline_comparison(results, args.output_dir)
    plot_divergence_layer_profile(results, args.output_dir)
    plot_entropy_probing_correlation(results, args.entropy_results, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")
