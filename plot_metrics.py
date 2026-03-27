import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import os
import numpy as np
from compute_surprise import _render_token
import theme

theme.apply_matplotlib()


def plot_tokenwise_stats(stats, focal_idx=None, file_name="surprisal_plot.png"):
    surprisal = [stat["surprisal"] for stat in stats]
    tokens = [_render_token(stat["next_token"]) for stat in stats]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(surprisal, marker="o", linewidth=2, color=theme.PRIMARY)

    if focal_idx is not None:
        ax.plot(focal_idx, surprisal[focal_idx], marker="D", markersize=8,
                color=theme.DANGER, zorder=5)
        ax.annotate(
            f"${surprisal[focal_idx]:.1f}$ bits",
            xy=(focal_idx, surprisal[focal_idx]),
            xytext=(12, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10,
            color=theme.DANGER,
        )

    ax.set_xlabel("$\\mathit{next\\ token}$")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_ylabel("surprisal (bits)")
    ax.grid(True, axis="both", alpha=0.3)
    fig.tight_layout()
    plt.savefig(file_name, dpi=180)
    plt.close(fig)


def plot_lollipop_distribution(stats, focal_idx=0, file_name="lollipop.png"):
    """
    Creates a lollipop plot visualizing the probability distribution over top-k tokens for a given prefix.
    """
    topk = stats[focal_idx]["topk"]
    focal_prob = stats[focal_idx]["prob"]
    focal_rank = stats[focal_idx]["rank"]
    focal_token = _render_token(stats[focal_idx]["next_token"])
    prefix = stats[focal_idx]["prefix"]

    # Build plotting arrays
    labels = [_render_token(t) for t, _ in topk]  # top-k tokens
    values = [p for _, p in topk]  # top-k probabilities

    # Add an ellipsis to visually separate top-k tokens from the focal token, then add the focal token itself.
    labels += ["…", focal_token]
    values += [0.0, focal_prob]
    focal_x = len(values) - 1  # index of focal token in the plot

    # Lollipop plot
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    for xi, yi in zip(x, values):
        if labels[xi] == "…":  # skip drawing stem for spacer
            continue
        ax.hlines(xi, 0, yi, linewidth=2, color=theme.PRIMARY)
        ax.plot(yi, xi, marker="o", color=theme.PRIMARY)

    # Emphasize focal token
    ax.hlines(focal_x, 0, values[focal_x], linewidth=2, color=theme.DANGER)
    ax.plot(values[focal_x], focal_x, marker="D", markersize=6, color=theme.DANGER)
    surprisal_bits = round(-np.log2(values[focal_x]))
    ax.annotate(
        f"rank={focal_rank}, $P \\approx 2^{{-{surprisal_bits}}}$",
        xy=(values[focal_x], focal_x),
        xytext=(12, 0),
        textcoords="offset points",
        ha="left",
        va="center",
    )

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    prefix_tex = "\\ ".join(prefix.split())
    ax.set_xlabel(f"$P (\\,\\mathit{{next\\ token}} \\mid \\mathrm{{{prefix_tex}}}\\,)$")
    ax.set_xlim(0, max(values) * 1.1 if values else 1.0)
    ax.grid(axis="both", alpha=0.3)

    ax.invert_yaxis()

    fig.tight_layout()
    plt.savefig(file_name, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_path, "images")
    cache_path = os.path.join(base_path, "cache", "compute_surprise_cache.json")
    dataset_path = os.path.join(base_path, "data/data.json")

    # Lollipop for "The old man the boats."
    with open(cache_path) as f:
        cache = json.load(f)

    old_man_stats = cache.get("The old man the boats.", {}).get("layer=None_attn=False_mlp=False", {}).get("stats")
    plot_lollipop_distribution(
        old_man_stats,
        focal_idx=3,  # "the" after "The old man"
        file_name=os.path.join(image_path, "the_old_man_lollipop.png"),
    )

    plot_tokenwise_stats(
        old_man_stats,
        focal_idx=3,  # "the" after "The old man"
        file_name=os.path.join(image_path, "the_old_man_surprisal_plot.png")
    )

    print(f"\nPlots saved to {image_path}")
