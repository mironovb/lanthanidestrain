#!/usr/bin/env python3
"""Data provenance figure: from the SAFE database to the scored pairs.

Top row: the five stages, each with its count and the reduction between
stages.  Bottom left: measurements per lanthanide in the modelled set (Pm
absent).  Bottom right: the effect of the collaborator's geometry repair
(August 2026) on the QC-passing rows and on the scored adjacent pairs.

Usage:  PYTHONPATH=$PWD python3 docs/talk_acs/make_data_fig.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

INK, GREY, LINE = "#111111", "#7d838a", "#c9cdd2"
ORANGE, BLUE = "#d55e00", "#2a6db0"
TINT = "#fdeadd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": "white",
    "savefig.dpi": 400, "font.size": 9, "axes.edgecolor": INK,
    "axes.linewidth": 0.8, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(7.0, 3.7))

# ------------------------------------------------------------ funnel
ax = fig.add_axes([0, 0.53, 1, 0.47])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

stages = [
    ("SAFE database", "48,138", "measurements",
     "31 elements\n181 publications", False),
    ("lanthanide(III)", "5,992", "measurements",
     "190 extractants\n14 lanthanides", False),
    ("3D structure", "956", "complexes",
     "Architector +\nGFN2-xTB", False),
    ("geometry QC", "4,746", "measurements",
     "162 extractants\n552 blocks", True),
    ("scored", "905", "adjacent pairs",
     "same extractant,\nsame conditions", True),
]
between = ["lanthanides only;\nduplicates removed",
           "one complex per\nmetal · extractant · anion",
           "rows whose complex\npasses QC",
           "neighbouring metals\nwithin a block"]

n = len(stages)
bw, gap = 16.6, 3.5
x0 = (100 - (n * bw + (n - 1) * gap)) / 2
yb, bh = 30, 60
for i, (name, num, unit, sub, edge) in enumerate(stages):
    x = x0 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, yb), bw, bh,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor="white",
                                edgecolor=ORANGE if edge else INK,
                                linewidth=1.1 if edge else 0.9))
    ax.text(x + bw / 2, yb + bh - 8, name, ha="center", va="center",
            fontsize=8.8, color=GREY, style="italic")
    ax.text(x + bw / 2, yb + bh - 25, num, ha="center", va="center",
            fontsize=15, color=INK, fontweight="bold")
    ax.text(x + bw / 2, yb + bh - 38, unit, ha="center", va="center",
            fontsize=8.5, color=INK)
    ax.text(x + bw / 2, yb + 9, sub, ha="center", va="center",
            fontsize=7.2, color=GREY, linespacing=1.2)
    if i < n - 1:
        xa, xb_ = x + bw + 0.4, x + bw + gap - 0.4
        ax.add_patch(FancyArrowPatch((xa, yb + bh / 2), (xb_, yb + bh / 2),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.9, color=INK))
        ax.text((xa + xb_) / 2, yb - 4, between[i], ha="center", va="top",
                fontsize=6.6, color=GREY, linespacing=1.25)

# ------------------------------------------------------------ rows per Ln
cov = pd.read_csv(REPO / "automl/reports/pair_coverage.csv")
order = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
         "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
rows = cov.set_index("metal")["rows"].reindex(order)
axL = fig.add_axes([0.075, 0.12, 0.50, 0.32])
xs = np.arange(len(order))
vals = rows.fillna(0).to_numpy()
axL.bar(xs, vals, color=BLUE, width=0.7, zorder=3)
pm = order.index("Pm")
axL.bar([pm], [vals.max() * 0.06], color="none", edgecolor=LINE,
        linestyle=(0, (2, 2)), width=0.7, zorder=3)
axL.text(pm, vals.max() * 0.09, "absent", ha="center", va="bottom",
         fontsize=6.5, color=LINE, rotation=90)
axL.set_xticks(xs); axL.set_xticklabels(order, fontsize=7.5)
axL.set_ylabel("measurements", fontsize=8.5)
axL.set_ylim(0, vals.max() * 1.15)
axL.tick_params(axis="y", labelsize=7.5)
axL.text(0.99, 0.95, "4,746 QC-passing measurements by lanthanide",
         transform=axL.transAxes, ha="right", va="top", fontsize=7.8,
         color=GREY)

# ------------------------------------------------------------ repair
axR = fig.add_axes([0.66, 0.12, 0.31, 0.32])
labels = ["QC-passing\nmeasurements", "adjacent pairs\nscored"]
before = [4746, 905]
after = [5479, 1230]
xs2 = np.arange(2)
axR.bar(xs2 - 0.19, before, width=0.36, color=LINE, zorder=3)
axR.bar(xs2 + 0.19, after, width=0.36, color=ORANGE, zorder=3)
for x, b_, a_ in zip(xs2, before, after):
    axR.text(x - 0.19, b_ * 1.03, f"{b_:,}", ha="center", va="bottom",
             fontsize=7.5, color=GREY)
    axR.text(x + 0.19, a_ * 1.03, f"{a_:,}", ha="center", va="bottom",
             fontsize=7.5, color=ORANGE, fontweight="bold")
axR.set_xticks(xs2); axR.set_xticklabels(labels, fontsize=7.5)
axR.set_yticks([])
axR.spines["left"].set_visible(False)
axR.set_ylim(0, 6600)
axR.text(0.5, 1.02, "geometry repair, Aug 2026", transform=axR.transAxes,
         ha="center", va="bottom", fontsize=7.8, color=GREY)
axR.text(1.0, 0.70, "before  →  after", transform=axR.transAxes,
         ha="right", va="center", fontsize=7, color=GREY)

fig.savefig(HERE / "data_provenance.png")
print("wrote data_provenance.png")
