#!/usr/bin/env python3
"""PRISMA-style flow of the data, from the SAFE database to the scored pairs.

Counts from data/processed/final_ml_dataset_no3d_info.json, the feature
matrix, and the collaborator's August 2026 geometry update.

Usage:  PYTHONPATH=$PWD python3 docs/talk_acs/make_prisma.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent

INK, GREY, FILL = "#111111", "#555a60", "#f4f6f8"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.dpi": 400})

fig = plt.figure(figsize=(6.2, 6.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

MX, MW = 5, 46      # main column
SX, SW = 57, 40     # exclusion column


def box(x, y, w, h, lines, bold_first=False, fill=FILL):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK,
                           linewidth=0.9))
    n = len(lines)
    for i, ln in enumerate(lines):
        yy = y + h - (i + 1) * h / (n + 1)
        ax.text(x + w / 2, yy, ln, ha="center", va="center",
                fontsize=8.6 if (i == 0 and bold_first) else 8.2,
                color=INK if i == 0 else GREY,
                fontweight="bold" if (i == 0 and bold_first) else "normal")


def down(x, y1, y2):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.9, color=INK,
                                 shrinkA=0, shrinkB=0))


def side(y, x1=MX + MW, x2=SX):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.9, color=INK,
                                 shrinkA=0, shrinkB=0))


# ------------------------------------------------------------ main column
H = 11
ys = [85, 65, 45, 25, 5]

box(MX, ys[0], MW, H, ["Records in the SAFE database",
                       "n = 48,138  (31 elements, 181 publications)"], True)
box(MX, ys[1], MW, H, ["Lanthanide(III) records with a valid D",
                       "n = 26,653"], True)
box(MX, ys[2], MW, H, ["Unique measurements",
                       "n = 5,992  (190 extractants, 14 lanthanides)"], True)
box(MX, ys[3], MW, H, ["With a QC-passing 3D structure",
                       "n = 5,479  (4,746 before the Aug 2026 repair)"], True)
box(MX, ys[4], MW, H, ["Adjacent-pair separations scored",
                       "n = 1,230  (905 before the repair)"], True)

for a, b in zip(ys[:-1], ys[1:]):
    down(MX + MW / 2, a, b + H)

# ------------------------------------------------------------ exclusions
def excl(y_arrow, lines):
    h = 3.1 * len(lines) + 2.0
    y = y_arrow - h / 2
    box(SX, y, SW, h, lines, fill="white")
    side(y_arrow)

gaps = [(ys[i] + ys[i + 1] + H) / 2 for i in range(4)]
excl(gaps[0], ["Excluded",
               "not lanthanide(III): 20,970",
               "D missing or ≤ 0: 515"])
excl(gaps[1], ["Excluded",
               "duplicates across element files: 18,763",
               "ambiguous extractant SMILES: 225",
               "repeat ingestions: 1,673"])
excl(gaps[2], ["Excluded",
               "no structure or failed geometry QC: 513"])
excl(gaps[3], ["Not scored",
               "single-metal blocks; non-adjacent pairs"])

fig.savefig(HERE / "prisma_flow.png")
print("wrote prisma_flow.png")
