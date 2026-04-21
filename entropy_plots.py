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
    "translation": "Translator (shuffled → English)",
    "impossible": "Impossible LM (Kallini et al.)",
    "normal": "Standard GPT-2",
}

PAIR_LABELS = {
    "translation_vs_impossible": f"{MODEL_NAMES['translation']}  vs  {MODEL_NAMES['impossible']}",
    "translation_vs_normal": f"{MODEL_NAMES['translation']}  vs  {MODEL_NAMES['normal']}",
    "normal_vs_impossible": f"{MODEL_NAMES['normal']}  vs  {MODEL_NAMES['impossible']}",
}

PAIR_LABELS_SHORT = {
    "translation_vs_impossible": "Translator vs Impossible LM",
    "translation_vs_normal": "Translator vs Standard GPT-2",
    "normal_vs_impossible": "Standard GPT-2 vs Impossible LM",
}

PAIR_COLORS = {
    "translation_vs_impossible": "#A32D2D",
    "translation_vs_normal": "#534AB7",
    "normal_vs_impossible": "#1D9E75",
}

PAIR_MARKERS = {
    "translation_vs_impossible": "o",
    "translation_vs_normal": "s",
    "normal_vs_impossible": "D",
}

PAIR_ORDER = ["translation_vs_impossible", "translation_vs_normal", "normal_vs_impossible"]


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def get_perturbation_label(name):
    labels = {
        "LOCALSHUFFLE": "LocalShuffle (w=3)",
        "PARTIALREVERSE": "PartialReverse",
        "WORDHOP": "WordHop",
    }
    return labels.get(name, name)


# ─────────────────────────────────────────────────────────────
# 1. Three heatmaps side by side (ΔH for each pair)
# ─────────────────────────────────────────────────────────────
def plot_three_heatmaps(results, output_dir):
    pert_label = get_perturbation_label(results["perturbation"])

    subtitles = {
        "translation_vs_impossible": "Blue = Translator more focused\nRed = Impossible LM more focused",
        "translation_vs_normal": "Blue = Standard GPT-2 more focused\nRed = Translator more focused",
        "normal_vs_impossible": "Blue = Standard GPT-2 more focused\nRed = Impossible LM more focused",
    }

    # Find global color range
    all_vals = []
    for p in PAIR_ORDER:
        all_vals.extend(np.array(results["comparisons"][p]["delta_H"]).flatten())
    abs_max = max(abs(min(all_vals)), abs(max(all_vals)))

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    for idx, pair in enumerate(PAIR_ORDER):
        ax = axes[idx]
        delta_H = np.array(results["comparisons"][pair]["delta_H"])

        # Keep the Standard GPT-2 side blue in both panels where it appears.
        display_delta_H = -delta_H if pair == "translation_vs_normal" else delta_H
        n_layers, n_heads = display_delta_H.shape

        im = ax.imshow(
            display_delta_H,
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
            f"{PAIR_LABELS_SHORT[pair]}\n{subtitles[pair]}",
            fontsize=11,
            fontweight="bold",
            linespacing=1.4,
        )
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{i}" for i in range(n_heads)], fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)

        for i in range(n_layers):
            for j in range(n_heads):
                val = display_delta_H[i, j]
                color = "white" if abs(val) > abs_max * 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=color,
                )

    # cbar = fig.colorbar(
    #     im,
    #     ax=axes,
    #     label="ΔH (bits) — Entropy Divergence",
    #     shrink=0.8,
    #     pad=0.02,
    #     fraction=0.03,
    # )
    fig.subplots_adjust(top=0.80, right=0.88, wspace=0.12)

    cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.65])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, label="ΔH (bits) — Entropy Divergence")
    cbar.ax.tick_params(labelsize=10)


    fig.suptitle(
        f"Per-Head Attention Entropy Divergence  —  {pert_label}",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # fig.subplots_adjust(top=0.80, right=0.92, wspace=0.12)

    path = os.path.join(output_dir, "three_heatmaps.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    path_pdf = os.path.join(output_dir, "three_heatmaps.pdf")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 2. Layer profiles overlaid for all three comparisons
# ─────────────────────────────────────────────────────────────
def plot_layer_profiles(results, output_dir):
    pert_label = get_perturbation_label(results["perturbation"])

    fig, ax = plt.subplots(figsize=(11, 6))

    for pair in PAIR_ORDER:
        layer_means = results["comparisons"][pair]["layer_summary"]["layer_mean_delta_H"]
        n_layers = len(layer_means)
        ax.plot(
            range(n_layers),
            layer_means,
            marker=PAIR_MARKERS[pair],
            linewidth=2.5,
            label=PAIR_LABELS_SHORT[pair],
            color=PAIR_COLORS[pair],
            markersize=8,
        )

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    # Annotate zero line
    ax.text(
        11.5,
        0.02,
        "no divergence",
        fontsize=9,
        color="gray",
        ha="right",
        style="italic",
    )

    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel("Mean Entropy Divergence ΔH (bits)", fontsize=14)
    ax.set_title(
        f"Layer-wise Attention Entropy Divergence  —  {pert_label}\n"
        f"Positive = first model more focused  |  Negative = second model more focused",
        fontsize=13,
        fontweight="bold",
        linespacing=1.5,
    )
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {i}" for i in range(n_layers)], fontsize=9, rotation=45)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    path = os.path.join(output_dir, "layer_profiles.png")
    plt.savefig(path, dpi=200)
    path_pdf = os.path.join(output_dir, "layer_profiles.pdf")
    plt.savefig(path_pdf)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 3. Raw entropy comparison across three models (per layer)
# ─────────────────────────────────────────────────────────────
def plot_raw_entropy_by_layer(results, output_dir):
    pert_label = get_perturbation_label(results["perturbation"])
    input_label = "shuffled" if results["input_type"] == "impossible" else "well-formed"

    H_trans = np.array(results["raw_entropy"]["H_translation"])
    H_imp = np.array(results["raw_entropy"]["H_impossible"])
    H_norm = np.array(results["raw_entropy"]["H_normal"])
    n_layers = H_trans.shape[0]

    trans_means = [np.mean(H_trans[l, :]) for l in range(n_layers)]
    imp_means = [np.mean(H_imp[l, :]) for l in range(n_layers)]
    norm_means = [np.mean(H_norm[l, :]) for l in range(n_layers)]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        range(n_layers),
        trans_means,
        marker="o",
        linewidth=2.5,
        color="#A32D2D",
        label=MODEL_NAMES["translation"],
        markersize=8,
    )
    ax.plot(
        range(n_layers),
        imp_means,
        marker="s",
        linewidth=2.5,
        color="#534AB7",
        label=MODEL_NAMES["impossible"],
        markersize=8,
    )
    ax.plot(
        range(n_layers),
        norm_means,
        marker="D",
        linewidth=2.5,
        color="#1D9E75",
        label=MODEL_NAMES["normal"],
        markersize=8,
    )

    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel("Mean Attention Entropy H (bits)", fontsize=14)
    ax.set_title(
        f"Raw Attention Entropy per Layer  —  {pert_label}\n"
        f"Input: {input_label} sentences  |  Higher = more diffuse attention  |  Lower = more focused",
        fontsize=13,
        fontweight="bold",
        linespacing=1.5,
    )
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {i}" for i in range(n_layers)], fontsize=9, rotation=45)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    path = os.path.join(output_dir, "raw_entropy_by_layer.png")
    plt.savefig(path, dpi=200)
    path_pdf = os.path.join(output_dir, "raw_entropy_by_layer.pdf")
    plt.savefig(path_pdf)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 4. Scatter: translation vs impossible & translation vs normal
# ─────────────────────────────────────────────────────────────
def plot_entropy_scatter(results, output_dir):
    pert_label = get_perturbation_label(results["perturbation"])

    H_trans = np.array(results["raw_entropy"]["H_translation"])
    H_imp = np.array(results["raw_entropy"]["H_impossible"])
    H_norm = np.array(results["raw_entropy"]["H_normal"])
    n_layers, n_heads = H_trans.shape

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    layer_colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

    scatter_configs = [
        (H_imp, MODEL_NAMES["impossible"], "Translator vs Impossible LM"),
        (H_norm, MODEL_NAMES["normal"], "Translator vs Standard GPT-2"),
    ]

    for ax, (H_other, y_label, title) in zip(axes, scatter_configs):
        for layer in range(n_layers):
            ax.scatter(
                H_trans[layer, :],
                H_other[layer, :],
                color=layer_colors[layer],
                label=f"Layer {layer}",
                s=50,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.3,
            )

        lims = [
            min(H_trans.min(), H_other.min()) - 0.3,
            max(H_trans.max(), H_other.max()) + 0.3,
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

        # Annotate diagonal
        mid = (lims[0] + lims[1]) / 2
        ax.text(
            mid + 0.3,
            mid - 0.3,
            "equal entropy",
            fontsize=8,
            color="gray",
            rotation=38,
            style="italic",
        )

        ax.set_xlabel(f"H — {MODEL_NAMES['translation']} (bits)", fontsize=11)
        ax.set_ylabel(f"H — {y_label} (bits)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(alpha=0.1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        fontsize=9,
        ncol=1,
        bbox_to_anchor=(0.98, 0.5),
        title="Transformer Layer",
        title_fontsize=10,
    )

    fig.suptitle(
        f"Per-Head Entropy Comparison  —  {pert_label}\n"
        f"Above diagonal = other model has higher entropy (more diffuse)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.93])

    path = os.path.join(output_dir, "entropy_scatter.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    path_pdf = os.path.join(output_dir, "entropy_scatter.pdf")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 5. Triangulation: translation-specific heads
# ─────────────────────────────────────────────────────────────
def plot_triangulation(results, output_dir):
    pert_label = get_perturbation_label(results["perturbation"])

    tri = results.get("triangulation", {})
    trans_specific = tri.get("translation_specific_heads", [])
    imp_vs_norm = tri.get("impossible_vs_normal_only_heads", [])

    n_layers = len(results["raw_entropy"]["H_translation"])
    n_heads = len(results["raw_entropy"]["H_translation"][0])

    delta_H_ti = np.array(results["comparisons"]["translation_vs_impossible"]["delta_H"])
    delta_H_tn = np.array(results["comparisons"]["translation_vs_normal"]["delta_H"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    plot_configs = [
        (axes[0], delta_H_ti, "Translator vs Impossible LM", "Blue = Translator more focused"),
        (axes[1], delta_H_tn, "Translator vs Standard GPT-2", "Blue = Translator more focused"),
    ]

    for ax, delta_H, title, subtitle in plot_configs:
        abs_max = max(abs(delta_H.min()), abs(delta_H.max()))
        im = ax.imshow(
            delta_H,
            cmap="RdBu",
            aspect="auto",
            interpolation="nearest",
            vmin=-abs_max,
            vmax=abs_max,
        )

        for h in trans_specific:
            ax.plot(
                h["head"],
                h["layer"],
                marker="*",
                color="gold",
                markersize=22,
                markeredgecolor="black",
                markeredgewidth=1.2,
            )

        for h in imp_vs_norm:
            ax.plot(
                h["head"],
                h["layer"],
                marker="D",
                color="white",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.2,
            )

        ax.set_xlabel("Attention Head Index", fontsize=12)
        ax.set_ylabel("Transformer Layer", fontsize=12)
        ax.set_title(
            f"{title}\n{subtitle}",
            fontsize=12,
            fontweight="bold",
            linespacing=1.4,
        )
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{i}" for i in range(n_heads)], fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)
        fig.colorbar(im, ax=ax, label="ΔH (bits)", shrink=0.8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=16,
            label=f"Translation-specific heads ({len(trans_specific)} found)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=10,
            label=f"Impossible-vs-Standard only ({len(imp_vs_norm)} found)",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.02),
        framealpha=0.9,
    )

    fig.suptitle(
        f"Triangulation: Identifying Translation-Specific Heads  —  {pert_label}\n"
        f"★ = heads where Translator is more focused than BOTH other models",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    path = os.path.join(output_dir, "triangulation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    path_pdf = os.path.join(output_dir, "triangulation.pdf")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 6. Top divergent heads across all three comparisons
# ─────────────────────────────────────────────────────────────
def plot_top_heads_comparison(results, output_dir, top_n=15):
    pert_label = get_perturbation_label(results["perturbation"])

    all_heads = set()
    for pair in PAIR_ORDER:
        comp = results["comparisons"][pair]
        for h in comp["positive_heads"][:top_n]:
            all_heads.add((h["layer"], h["head"]))
        for h in comp["negative_heads"][:top_n]:
            all_heads.add((h["layer"], h["head"]))

    if not all_heads:
        print("No divergent heads found, skipping top heads plot.")
        return

    all_heads = sorted(all_heads)
    head_labels = [f"Layer {l}, Head {h}" for l, h in all_heads]

    fig, ax = plt.subplots(figsize=(13, max(5, len(all_heads) * 0.4)))

    bar_height = 0.25
    y_positions = np.arange(len(all_heads))

    for i, pair in enumerate(PAIR_ORDER):
        delta_H = np.array(results["comparisons"][pair]["delta_H"])
        values = [delta_H[l, h] for l, h in all_heads]
        ax.barh(
            y_positions + i * bar_height,
            values,
            height=bar_height,
            color=PAIR_COLORS[pair],
            alpha=0.85,
            label=PAIR_LABELS_SHORT[pair],
        )

    ax.set_yticks(y_positions + bar_height)
    ax.set_yticklabels(head_labels, fontsize=10)
    ax.set_xlabel("Entropy Divergence ΔH (bits)", fontsize=13)
    ax.set_title(
        f"Most Divergent Attention Heads Across All Comparisons  —  {pert_label}\n"
        f"Positive = first model more focused  |  Negative = second model more focused",
        fontsize=13,
        fontweight="bold",
        linespacing=1.5,
    )
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)

    plt.tight_layout()
    path = os.path.join(output_dir, "top_heads_comparison.png")
    plt.savefig(path, dpi=200)
    path_pdf = os.path.join(output_dir, "top_heads_comparison.pdf")
    plt.savefig(path_pdf)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot three-way attention entropy comparison results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to three-way results JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots",
        help="Directory to save plots (default: ./plots)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.results)

    pert = get_perturbation_label(results["perturbation"])
    print(f"Plotting results for {pert}")
    print(f"Input type: {results['input_type']}")
    print(f"Models:")
    print(f"  Translator:   {results['models']['translation']}")
    print(f"  Impossible:   {results['models']['impossible']}")
    print(f"  Standard:     {results['models']['normal']}")
    print(f"Output directory: {args.output_dir}\n")

    plot_three_heatmaps(results, args.output_dir)
    plot_layer_profiles(results, args.output_dir)
    plot_raw_entropy_by_layer(results, args.output_dir)
    plot_entropy_scatter(results, args.output_dir)
    plot_triangulation(results, args.output_dir)
    plot_top_heads_comparison(results, args.output_dir)

    print(f"\nAll 6 plots saved to {args.output_dir}/")
