#!/usr/bin/env python3
"""Scheme figure: the predicted quantity, built from one measured block.

Three measured rows (MMTODGA, 3.0 M HCl, Isopar-L, 25 C -- only the metal
differs), the separation-factor subtraction written out with the displayed
rounding, and the lanthanide row with the adjacent pair marked and Pm absent.

Usage:  PYTHONPATH=$PWD python3 docs/talk_acs/make_scheme.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent

INK = "#111111"
GREY = "#7d838a"
ORANGE = "#d55e00"
TINT = "#fdeadd"
LINE = "#c9cdd2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": "white",
    "savefig.dpi": 400, "font.size": 9})

fig = plt.figure(figsize=(7.0, 2.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.axis("off")

# ------------------------------------------------- measured rows (table)
cols = [("metal", 5.5), ("extractant", 14.5), ("acid", 33.0),
        ("diluent", 46.5), ("T", 58.0), ("log D", 65.5)]
rows = [("Gd", "1.86", True),
        ("Tb", "2.25", True),
        ("Dy", "2.12", False)]
cond = {"extractant": "MMTODGA 0.1 M", "acid": "HCl 3.0 M",
        "diluent": "Isopar-L", "T": "25 °C"}

y0, rh = 86.0, 11.0
for name, x in cols:
    ax.text(x, y0 + 3.2, name, fontsize=8.5, color=GREY, va="bottom",
            style="italic")
ax.plot([4, 71], [y0 + 2.2, y0 + 2.2], color=INK, lw=0.9)
for i, (metal, ld, hl) in enumerate(rows):
    yy = y0 - i * rh
    if hl:
        ax.add_patch(Rectangle((4, yy - rh + 3.3), 67, rh - 0.8,
                               facecolor=TINT, edgecolor="none", zorder=0))
    ax.text(cols[0][1], yy - 2.0, metal, fontsize=10.5, color=INK,
            fontweight="bold" if hl else "normal", va="center")
    for key, x in cols[1:5]:
        ax.text(x, yy - 2.0, cond[key], fontsize=9, color=GREY, va="center")
    ax.text(cols[5][1], yy - 2.0, ld, fontsize=10.5, color=INK,
            fontweight="bold" if hl else "normal", va="center")
ax.plot([4, 71], [y0 - 2 * rh - 4.2, y0 - 2 * rh - 4.2], color=INK, lw=0.9)
ax.text(4, y0 - 2 * rh - 8.6, "same extractant, acid, diluent, temperature "
        "— only the metal differs", fontsize=8.5, color=GREY, va="center")

# ------------------------------------------------- the subtraction
bx, by, bw, bh = 74.5, 42.0, 23.5, 52.0
ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                            boxstyle="round,pad=1.0,rounding_size=1.5",
                            facecolor="white", edgecolor=INK, linewidth=0.9))
ax.text(bx + bw / 2, by + bh - 4.0, r"$\log SF =$",
        ha="center", va="top", fontsize=9.5, color=INK)
ax.text(bx + bw / 2, by + bh - 13.5,
        r"$\log D(\mathrm{A}) - \log D(\mathrm{B})$",
        ha="center", va="top", fontsize=9.5, color=INK)
ax.text(bx + bw / 2, by + bh - 28,
        r"$\log SF_{\,\mathrm{Tb/Gd}} = 2.25 - 1.86$",
        ha="center", va="top", fontsize=9.5, color=INK)
ax.text(bx + bw / 2, by + bh - 40, r"$= 0.39$",
        ha="center", va="top", fontsize=12, color=ORANGE,
        fontweight="bold")

# ------------------------------------------------- lanthanide row
series = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
          "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
x0, cw, yb, ch = 4.0, 6.25, 22.0, 13.0
for i, el in enumerate(series):
    xx = x0 + i * cw
    if el == "Pm":
        ax.add_patch(Rectangle((xx, yb), cw - 0.7, ch, facecolor="white",
                               edgecolor=LINE, linewidth=0.9,
                               linestyle=(0, (2, 2))))
        ax.text(xx + (cw - 0.7) / 2, yb + ch / 2, el, ha="center",
                va="center", fontsize=9, color=LINE)
    elif el in ("Gd", "Tb"):
        ax.add_patch(Rectangle((xx, yb), cw - 0.7, ch, facecolor=ORANGE,
                               edgecolor=ORANGE, linewidth=0.9))
        ax.text(xx + (cw - 0.7) / 2, yb + ch / 2, el, ha="center",
                va="center", fontsize=9.5, color="white",
                fontweight="bold")
    else:
        ax.add_patch(Rectangle((xx, yb), cw - 0.7, ch, facecolor="white",
                               edgecolor=GREY, linewidth=0.9))
        ax.text(xx + (cw - 0.7) / 2, yb + ch / 2, el, ha="center",
                va="center", fontsize=9, color=INK)

gx0 = x0 + series.index("Gd") * cw
gx1 = x0 + series.index("Tb") * cw + cw - 0.7
ax.plot([gx0, gx0, gx1, gx1], [yb - 2.0, yb - 4.2, yb - 4.2, yb - 2.0],
        color=ORANGE, lw=1.1)
ax.text((gx0 + gx1) / 2, yb - 6.0, "adjacent pair", ha="center", va="top",
        fontsize=8.5, color=ORANGE)
pmx = x0 + series.index("Pm") * cw + (cw - 0.7) / 2
ax.text(pmx, yb - 6.0, "absent", ha="center", va="top", fontsize=8.5,
        color=LINE)

ax.text(96.5, 4.0, r"$D = [\mathrm{Ln}]_{\mathrm{org}} \, / \, "
                   r"[\mathrm{Ln}]_{\mathrm{aq}}$",
        ha="right", va="bottom", fontsize=9, color=GREY)

fig.savefig(HERE / "scheme_target.png")
print("wrote scheme_target.png")
