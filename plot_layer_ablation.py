import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import numpy as np
import theme

theme.apply_matplotlib()

# Custom colormaps from theme palette
_DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "theme_diverging", [theme.PRIMARY, theme.BACKGROUND, theme.DANGER]
)
_SEQUENTIAL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "theme_sequential", [theme.DANGER, theme.WARNING, theme.BACKGROUND]
)

with open("cache/compute_layer_ablation_cache.json") as f:
    cache = json.load(f)
with open("data/data.json") as f:
    gps_data = json.load(f)

sentences = [
    item["sentence"]
    for item in gps_data
    if item.get("type") != "CENTER_EMBED" and item.get("control")
]
n = len(sentences)

N_LAYERS = 28
components = ["attn", "mlp"]


def get_surprisals(sent_type, condition):
    vals = []
    for sent in sentences:
        k = f"{sent_type}:{sent}"
        if k in cache and condition in cache[k]:
            vals.append(cache[k][condition]["surprisal"])
    return np.array(vals)


def get_topk_overlap(sent_type, condition):
    overlaps = []
    for sent in sentences:
        k = f"{sent_type}:{sent}"
        if k not in cache:
            continue
        base = cache[k].get("none")
        abl = cache[k].get(condition)
        if base and abl:
            base_toks = set(t for t, _ in base["topk"])
            abl_toks = set(t for t, _ in abl["topk"])
            overlaps.append(len(base_toks & abl_toks) / len(base_toks))
    return np.mean(overlaps) if overlaps else 0.0


# ------------------------------------------------------------------
# Figure 1: Heatmap (transposed — layers on x-axis)
# ------------------------------------------------------------------
# Two sub-heatmaps stacked: top = Δ GPS surprisal, bottom = top-5 overlap
# Each has 2 rows (attn, mlp) × 28 columns (layers)

delta_gps = np.zeros((2, N_LAYERS))
overlap = np.zeros((2, N_LAYERS))
baseline_gps = get_surprisals("gps", "none")

for li in range(N_LAYERS):
    for ci, comp in enumerate(components):
        cond = f"{comp}_L{li}"
        abl = get_surprisals("gps", cond)
        delta_gps[ci, li] = np.mean(abl) - np.mean(baseline_gps)
        overlap[ci, li] = get_topk_overlap("gps", cond)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(10, 3.0),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15},
)

# Diverging colormap for delta GPS (blue = decrease, red = increase)
vmax = max(abs(delta_gps.min()), abs(delta_gps.max()))
norm1 = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im1 = ax1.imshow(
    delta_gps, aspect="auto", cmap=_DIVERGING_CMAP, norm=norm1, interpolation="nearest"
)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["Attn", "MLP"])
cb1 = fig.colorbar(im1, ax=ax1, pad=0.01, aspect=12)
cb1.set_label("$\\Delta$ GPS surprisal (bits)", fontsize=10)

# Annotate extreme values
for ci in range(2):
    for li in range(N_LAYERS):
        val = delta_gps[ci, li]
        if abs(val) > 1.0:
            ax1.text(
                li,
                ci,
                f"{val:+.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(val) > 3 else theme.TEXT,
            )

# Top-5 overlap heatmap (sequential: dark = low overlap = big change)
im2 = ax2.imshow(
    overlap,
    aspect="auto",
    cmap=_SEQUENTIAL_CMAP,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Attn", "MLP"])
ax2.set_xticks(range(0, N_LAYERS, 2))
ax2.set_xticklabels([str(i) for i in range(0, N_LAYERS, 2)])
ax2.set_xlabel("Layer")
cb2 = fig.colorbar(im2, ax=ax2, pad=0.01, aspect=12)
cb2.set_label("Top-5 overlap", fontsize=10)

# Annotate low-overlap cells
for ci in range(2):
    for li in range(N_LAYERS):
        val = overlap[ci, li]
        if val < 0.6:
            ax2.text(
                li,
                ci,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

plt.savefig("images/layer_ablation_heatmap.png", dpi=200, bbox_inches="tight")
print("Saved images/layer_ablation_heatmap.png")
plt.close()

# ------------------------------------------------------------------
# Figure 2: Interaction at MLP L0 — GPS down, ALT up
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 4))

gps_base = get_surprisals("gps", "none")
gps_abl = get_surprisals("gps", "mlp_L0")
alt_base = get_surprisals("alt", "none")
alt_abl = get_surprisals("alt", "mlp_L0")

gps_delta = gps_abl - gps_base
alt_delta = alt_abl - alt_base

# Paired dot plot: one line per sentence connecting GPS Δ and ALT Δ
for i in range(n):
    ax.plot(
        [0, 1],
        [gps_delta[i], alt_delta[i]],
        color=theme.MUTED_TEXT,
        alpha=0.25,
        linewidth=0.7,
    )

# Means with error bars
gps_se = np.std(gps_delta) / np.sqrt(n)
alt_se = np.std(alt_delta) / np.sqrt(n)

ax.errorbar(
    0,
    np.mean(gps_delta),
    yerr=gps_se,
    fmt="o",
    color=theme.DANGER,
    markersize=10,
    capsize=5,
    linewidth=2,
    zorder=5,
)
ax.errorbar(
    1,
    np.mean(alt_delta),
    yerr=alt_se,
    fmt="o",
    color=theme.SUCCESS,
    markersize=10,
    capsize=5,
    linewidth=2,
    zorder=5,
)

ax.axhline(0, color=theme.MUTED_TEXT, linestyle="--", linewidth=0.8, alpha=0.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(["GPS\n(garden path)", "Natural\n(expected)"], fontsize=12)
ax.set_ylabel("$\\Delta$ surprisal at juke point (bits)", fontsize=12)
ax.set_xlim(-0.25, 1.25)

# Annotate means
ax.annotate(
    f"{np.mean(gps_delta):+.1f} bits\np = 0.006",
    xy=(0, np.mean(gps_delta)),
    xytext=(0.4, np.mean(gps_delta) - 5),
    fontsize=12,
    color=theme.DANGER,
    arrowprops=dict(arrowstyle="->", color=theme.DANGER, lw=1),
)
ax.annotate(
    f"{np.mean(alt_delta):+.1f} bits\np < 0.0001",
    xy=(1, np.mean(alt_delta)),
    xytext=(0.45, np.mean(alt_delta) + 5),
    fontsize=12,
    color=theme.SUCCESS,
    arrowprops=dict(arrowstyle="->", color=theme.SUCCESS, lw=1),
)

plt.tight_layout()
plt.savefig("images/mlp_l0_interaction.png", dpi=200, bbox_inches="tight")
print("Saved images/mlp_l0_interaction.png")
plt.close()
