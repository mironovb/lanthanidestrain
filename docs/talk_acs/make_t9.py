#!/usr/bin/env python3
"""T9_ranking.png -- decision quality: per-position ranking of held-out
extractants (Spearman and top-quartile hit rate) and the conformal
coverage/abstention summary. From automl/reports/decision_quality.json."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parents[1] / "automl/reports/decision_quality.json").read_text())
INK, SUB, GRID = "#101214", "#52514e", "#e3e5e8"
NAVY, BLUE, ORANGE, GREEN, GREY = "#1a2e4a", "#2a78d6", "#eb6834", "#1baf7a", "#9aa1a8"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
    "savefig.dpi": 300, "axes.edgecolor": SUB, "xtick.color": SUB,
    "ytick.color": SUB, "font.size": 12, "axes.spines.top": False,
    "axes.spines.right": False})

R = D["ranking"]["by_position"]
pos = list(R.keys()); rho = [R[p]["spearman"] for p in pos]
hit = [R[p]["top_quartile_hit_rate"] for p in pos]
fig = plt.figure(figsize=(12.4, 5.6))
ax = fig.add_axes([0.07, 0.17, 0.55, 0.68])
xs = np.arange(len(pos))
ax.bar(xs - 0.2, rho, width=0.38, color=BLUE, zorder=3, label="Spearman ρ (predicted vs measured ranking)")
ax.bar(xs + 0.2, hit, width=0.38, color=GREEN, zorder=3, label="top-quartile hit rate")
ax.axhline(0.25, color=GREEN, ls=":", lw=1.6, zorder=2, label="hit rate by chance (0.25)")
ax.set_xticks(xs); ax.set_xticklabels(pos, fontsize=10.5, rotation=0)
ax.set_ylim(0, 1.0); ax.set_ylabel("ranking quality across held-out extractants", fontsize=12)
ax.grid(color=GRID, axis="y", zorder=0)
ax.legend(fontsize=10.5, frameon=False, loc="upper left")
ax.set_title(f"A   for each adjacent pair, rank the extractants  (mean ρ = "
             f"{D['ranking']['mean_spearman_over_positions']:.2f}, mean hit rate = "
             f"{D['ranking']['mean_top_quartile_hit_rate']:.2f})", fontsize=12.5, color=NAVY, loc="left", pad=9)

axB = fig.add_axes([0.71, 0.17, 0.26, 0.68])
C = D["conformal"]
labels = ["80 %", "90 %"]
emp = [C["coverage"]["80%"]["empirical"], C["coverage"]["90%"]["empirical"]]
nom = [0.80, 0.90]
xs2 = np.arange(2)
axB.bar(xs2 - 0.2, nom, width=0.38, color=GREY, zorder=3, label="nominal")
axB.bar(xs2 + 0.2, emp, width=0.38, color=ORANGE, zorder=3, label="empirical")
for x, e in zip(xs2, emp):
    axB.text(x + 0.2, e + 0.012, f"{e:.3f}", ha="center", fontsize=11)
axB.set_xticks(xs2); axB.set_xticklabels(labels, fontsize=11.5)
axB.set_ylim(0, 1.05); axB.set_ylabel("coverage of conformal intervals", fontsize=12)
axB.grid(color=GRID, axis="y", zorder=0)
axB.legend(fontsize=10.5, frameon=False, loc="lower right")
hw = C["coverage"]["80%"]["median_half_width"], C["coverage"]["90%"]["median_half_width"]
axB.set_title(f"B   calibrated intervals\n     half-widths {hw[0]:.2f} / {hw[1]:.2f} log SF", fontsize=12.5, color=NAVY, loc="left", pad=9)
fig.savefig(HERE / "T9_ranking.png"); print("wrote T9_ranking.png")
