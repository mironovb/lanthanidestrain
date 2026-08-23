#!/usr/bin/env python3
"""Scheme figure: the two quantities being predicted.

Panel A defines the distribution coefficient D for one lanthanide in a
two-phase extraction; panel B defines the adjacent-pair separation factor
on a measured 14-lanthanide series (one extractant, one condition set).
ACS-style: minimal colour, definitions written once, no annotations beyond
the labels.

Usage:  PYTHONPATH=$PWD python3 docs/talk_acs/make_scheme.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
D = json.loads((REPO / "docs/figures_arch/fig_data.json").read_text())

INK = "#111111"
GREY = "#8a8f94"
BLUE = "#2a6db0"
ORANGE = "#d55e00"
AQ = "#e8f1f8"
ORG = "#f5efe6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": "white",
    "savefig.dpi": 400, "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(7.0, 3.1))

# ------------------------------------------------------------- panel A
axA = fig.add_axes([0.045, 0.14, 0.30, 0.74])
axA.set_xlim(0, 10); axA.set_ylim(0, 10)
axA.axis("off")

axA.add_patch(Rectangle((1.2, 5.0), 7.6, 3.6, facecolor=ORG,
                        edgecolor=INK, linewidth=0.9))
axA.add_patch(Rectangle((1.2, 1.4), 7.6, 3.6, facecolor=AQ,
                        edgecolor=INK, linewidth=0.9))
axA.text(8.55, 8.15, "organic", ha="right", va="top", fontsize=8.5,
         color=INK, style="italic")
axA.text(8.55, 1.85, "aqueous", ha="right", va="bottom", fontsize=8.5,
         color=INK, style="italic")

org_pts = np.array([[2.1, 7.6], [3.5, 6.3], [4.9, 7.3],
                    [6.3, 6.0], [7.4, 7.0], [2.6, 5.6]])
aq_pts = np.array([[3.2, 3.0], [6.1, 2.4]])
axA.scatter(org_pts[:, 0], org_pts[:, 1], s=26, color=BLUE, zorder=3)
axA.scatter(aq_pts[:, 0], aq_pts[:, 1], s=26, color=BLUE, zorder=3)
axA.text(2.55, 7.62, r"Ln$^{3+}$", fontsize=8, color=BLUE, va="center")

axA.add_patch(FancyArrowPatch((4.6, 4.35), (4.6, 5.65),
                              arrowstyle="<|-|>", mutation_scale=10,
                              linewidth=0.9, color=INK))

axA.text(5.0, 0.15, r"$D \;=\; [\mathrm{Ln}]_{\mathrm{org}} \, / \, "
                    r"[\mathrm{Ln}]_{\mathrm{aq}}$",
         ha="center", va="bottom", fontsize=10.5, color=INK)
axA.set_title("A", fontsize=11, fontweight="bold", loc="left", x=-0.02)

# ------------------------------------------------------------- panel B
axB = fig.add_axes([0.435, 0.17, 0.545, 0.70])
ex = D["example"]
m = ex["metals"]
y = np.array(ex["y"])
xs = np.arange(len(m))

iA, iB = m.index("Gd"), m.index("Tb")
others = [i for i in range(len(m)) if i not in (iA, iB)]
axB.plot(xs[others], y[others], "o", ms=4.5, markerfacecolor="white",
         markeredgecolor=GREY, markeredgewidth=1.0, zorder=2)
axB.plot([xs[iA], xs[iB]], [y[iA], y[iB]], "o", ms=5.5, color=ORANGE,
         zorder=3)

xb = xs[iB] + 0.42
axB.plot([xs[iA], xb], [y[iA], y[iA]], lw=0.8, color=INK, ls=":")
axB.plot([xs[iB], xb], [y[iB], y[iB]], lw=0.8, color=INK, ls=":")
axB.add_patch(FancyArrowPatch((xb, y[iA]), (xb, y[iB]),
                              arrowstyle="<|-|>", mutation_scale=8,
                              linewidth=0.9, color=INK))
axB.text(xs[iA], y[iA] - 0.28, "B", ha="center", fontsize=9, color=ORANGE,
         fontweight="bold")
axB.text(xs[iB], y[iB] + 0.17, "A", ha="center", fontsize=9, color=ORANGE,
         fontweight="bold")

axB.text(0.03, 0.96,
         r"$\log SF \;=\; \log D(\mathrm{A}) \,-\, \log D(\mathrm{B})$",
         transform=axB.transAxes, fontsize=10.5, color=INK, va="top")
axB.text(0.03, 0.83, "A, B adjacent in the series",
         transform=axB.transAxes, fontsize=8.5, color=GREY, va="top")

axB.set_xticks(xs)
axB.set_xticklabels(m, fontsize=8)
axB.set_ylabel(r"$\log D$", fontsize=10)
axB.set_xlim(-0.7, len(m) - 0.2)
axB.set_ylim(y.min() - 0.55, y.max() + 0.45)
axB.set_title("B", fontsize=11, fontweight="bold", loc="left")

fig.savefig(HERE / "scheme_target.png")
print("wrote scheme_target.png")
