"""
Generate parse tree diagrams for garden-path sentences using matplotlib.
Outputs PNG files with the project's theme styling.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import theme

theme.apply_matplotlib()


def draw_tree(ax, tree):
    """
    Draw a parse tree on the given axes.

    tree: nested tuple structure like:
        ("S", [
            ("NP", [("Det", "The"), ("N", "old")]),
            ("VP", [("V", "man"), ("NP", [("Det", "the"), ("N", "boats")])])
        ])
    """

    positions = {}
    leaf_x = [0]  # mutable counter for leaf positions

    def compute_positions(node, depth=0):
        label, children = node
        if isinstance(children, str):
            # POS tag node — position here, word goes one level below
            x = leaf_x[0]
            leaf_x[0] += 0.7
            positions[id(node)] = (x, -depth * 1.1)
            return x
        else:
            child_xs = []
            for child in children:
                cx = compute_positions(child, depth + 1)
                child_xs.append(cx)
            x = sum(child_xs) / len(child_xs)
            positions[id(node)] = (x, -depth * 1.1)
            return x

    compute_positions(tree)

    # Scale positions
    for nid, (px, py) in positions.items():
        positions[nid] = (px * 1.2, py * 0.4)

    def draw_node(node, depth=0):
        label, children = node
        x, y = positions[id(node)]

        if isinstance(children, str):
            # POS tag — tight boxed
            bbox = FancyBboxPatch(
                (x - 0.095, y - 0.065),
                0.19,
                0.13,
                boxstyle="round,pad=0.02",
                facecolor=theme.BACKGROUND,
                edgecolor=theme.PRIMARY,
                linewidth=1.0,
                zorder=3,
            )
            ax.add_patch(bbox)
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=theme.PRIMARY,
                zorder=4,
            )

            # Draw word below
            word = children
            wy = y - 0.5
            color = theme.TEXT
            ax.text(
                x,
                wy,
                word,
                ha="center",
                va="center",
                fontsize=13,
                fontstyle="italic",
                color=color,
            )

            # Edge from POS tag to word
            ax.plot(
                [x, x],
                [y - 0.065, wy + 0.15],
                color=theme.MUTED_TEXT,
                linewidth=1.2,
                solid_capstyle="round",
                zorder=1,
            )
        else:
            # Category node — tight boxed
            bbox = FancyBboxPatch(
                (x - 0.095, y - 0.065),
                0.19,
                0.13,
                boxstyle="round,pad=0.02",
                facecolor=theme.BACKGROUND,
                edgecolor=theme.PRIMARY,
                linewidth=1.2,
                zorder=3,
            )
            ax.add_patch(bbox)
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=theme.PRIMARY,
                zorder=4,
            )

            # Draw edges to children
            for child in children:
                cx, cy = positions[id(child)]
                # Connect to top of child box
                cy_top = cy + 0.065
                ax.plot(
                    [x, cx],
                    [y - 0.065, cy_top],
                    color=theme.MUTED_TEXT,
                    linewidth=1.2,
                    solid_capstyle="round",
                    zorder=1,
                )

            for child in children:
                draw_node(child, depth + 1)

    draw_node(tree)


def make_tree_figure(tree, title, filename, figsize=(9, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor(theme.BACKGROUND)
    fig.set_facecolor(theme.BACKGROUND)

    draw_tree(ax, tree)

    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.08)
    ax.axis("off")

    fig.tight_layout(pad=0.2)
    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor=theme.BACKGROUND,
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved {filename}")

gps_tree = (
    "S",
    [
        (
            "NP",
            [
                ("Det", "The"),
                ("N", "old"),
            ],
        ),
        (
            "VP",
            [
                ("V", "man"),
                (
                    "NP",
                    [
                        ("Det", "the"),
                        ("N", "boats"),
                    ],
                ),
            ],
        ),
    ],
)

alt_tree = (
    "S",
    [
        (
            "NP",
            [
                ("Det", "The"),
                ("Adj", "old"),
                ("N", "man"),
            ],
        ),
        (
            "VP",
            [
                ("V", "sat"),
                (
                    "PP",
                    [
                        ("P", "on"),
                        (
                            "NP",
                            [
                                ("Det", "the"),
                                ("N", "bench"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

if __name__ == "__main__":
    make_tree_figure(
        gps_tree,
        "GPS parse:  The old man the boats.",
        "images/the_old_man_parse_tree.png",
        figsize=(6, 6),
    )
