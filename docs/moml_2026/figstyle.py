"""House style for the MoML figures: the repo palette (automl/figures.py) with
the paper's typography rules.  Helvetica-adjacent sans (Helvetica > Arial >
Nimbus Sans > DejaVu Sans), title 16 pt, everything else 12 pt, fixed dpi,
one vector PDF per panel plus a PNG preview.  Legends go outside the axes."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from automl.figures import C as PALETTE, INK, INK2  # fixed colour identities

FIG_DIR = HERE / "figures"
DPI = 300
BODY, TITLE = 12, 16
NEUTRAL = "#7a7a7a"          # scaffolding only: reference lines, 1:1 line

# one colour per entity, reused in every panel
COLOR = {
    "gfn2": PALETTE["blue"], "gxtb": PALETTE["orange"], "lnxtb": PALETTE["aqua"],
    "model": PALETTE["blue"], "hit": PALETTE["violet"],
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
        "font.size": BODY, "axes.labelsize": BODY, "axes.titlesize": TITLE,
        "xtick.labelsize": BODY, "ytick.labelsize": BODY, "legend.fontsize": BODY,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": DPI, "figure.dpi": DPI,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def figure(width: float, height: float):
    style()
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def legend_above(ax, handles=None, labels=None, ncol=None):
    """Single-row legend outside the axes, above the plot rectangle."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    ncol = ncol or len(handles)
    return ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                     ncol=ncol, handlelength=1.4, columnspacing=1.2,
                     borderaxespad=0.0)


def finish(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_DIR / f"{name}.pdf")
    return FIG_DIR / f"{name}.pdf"
