"""
Centralized font + color theme for all plots
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_EB12_PATH = "/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Regular.otf"
_EB12_IT_PATH = "/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Italic.otf"
_fe = fm.FontEntry(fname=_EB12_PATH, name="EB Garamond 12", style="normal", weight=400)
_fe_it = fm.FontEntry(fname=_EB12_IT_PATH, name="EB Garamond 12 Italic", style="italic", weight=400)
fm.fontManager.ttflist.insert(0, _fe)
fm.fontManager.ttflist.insert(0, _fe_it)

FONT_FAMILY = "EB Garamond 12"

BACKGROUND = "#faf7f0"  # cream
TEXT = "#4a3f30"  # walnut ink
MUTED_TEXT = "#7a6e5d"  # faded ink
BORDER = "#e6dfd3"  # warm border

PRIMARY = "#2b6a94"  # steel blue
SUCCESS = "#3d7a56"  # forest sage
WARNING = "#b5762a"  # amber ochre
DANGER = "#a04040"  # brick red
INFO = "#8a7422"  # aged gold

# Ordered accent palette for cycling through series
ACCENTS = [PRIMARY, SUCCESS, WARNING, DANGER, INFO]

# Experiment-specific mapping (indices 0, 1, 2)
EXP_COLORS = {
    0: SUCCESS,  # Experiment A
    1: PRIMARY,  # Experiment B
    2: WARNING,  # Experiment C
}


def apply_matplotlib():
    """Set matplotlib rcParams to match the project theme."""
    fm._load_fontmanager()  # pick up newly installed fonts
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY, "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": "EB Garamond 12 Italic",
            "mathtext.bf": FONT_FAMILY,
            "mathtext.sf": FONT_FAMILY,
            "mathtext.cal": FONT_FAMILY,
            "mathtext.tt": "DejaVu Sans Mono",
            "font.size": 12,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": BORDER,
            "axes.facecolor": BACKGROUND,
            "axes.prop_cycle": plt.cycler(color=ACCENTS),
            "figure.facecolor": BACKGROUND,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
            "grid.color": BORDER,
            "grid.alpha": 0.6,
            "legend.framealpha": 0.8,
            "legend.edgecolor": BORDER,
            "savefig.facecolor": BACKGROUND,
        }
    )
