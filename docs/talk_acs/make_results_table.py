#!/usr/bin/env python3
"""results_table.png -- single models and combinations, both metrics.

Every value is out-of-fold under leave-extractants-out CV on the legacy
population (4,746 rows, 162 extractants, 905 adjacent pairs).  Verified
25 Aug 2026:
 * CatBoost q60_rsm03_deep         c17_stack_new.csv   0.5075 / +0.2784
 * fingerprint net (256,128) x16   c17_stack_new.csv   0.3218 / +0.2206
 * distance encoder c15_plw4 x32   c17_stack_new.csv   0.3430 / +0.2661
 * simplicial c17_plw2 x32         recomputed from OOF  0.3593 / +0.2389
 * simplicial c17_plw4 x32         recomputed from OOF  0.3408 / +0.2425
 * pair-fitted stack (cat+dist+fcnn)  c17_stack_new.csv +0.3132; its overall
   log D R2 = 0.4375 on the 3,944 rows whose extractant has adjacent pairs
   (weights: dist 0.465, cat 0.423, fcnn 0.112, sum 1.000)
 * July baseline stack (row-fitted, simplicial)  PI_EMAIL.md  0.437 / +0.267
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK, SUB, RULE = "#1a1a1a", "#666666", "#9a9a9a"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.dpi": 300})

ROWS = [
    ("Gradient-boosted trees",
     "CatBoost, quantile loss α = 0.6, depth 9, tabular features",
     "0.508", "0.278"),
    ("Fully connected network",
     "layers 256, 128 on the same tabular features, 16 seeds",
     "0.322", "0.221"),
    ("3D distance network",
     "message passing over interatomic distances, pair-contrast weight 4, 32 seeds",
     "0.343", "0.266"),
    ("3D simplicial network",
     "Vietoris–Rips complex, pair-contrast weight 2, 32 seeds",
     "0.359", "0.239"),
    ("3D simplicial network",
     "Vietoris–Rips complex, pair-contrast weight 4, 32 seeds",
     "0.341", "0.243"),
    None,
    ("Combination, July baseline",
     "CatBoost + fingerprint network + simplicial network; weights fitted on row-level log D",
     "0.437", "0.267"),
    ("Combination, current",
     "CatBoost + fingerprint network + 3D distance network; weights fitted on adjacent-pair separations",
     "0.437", "0.313"),
]

fig = plt.figure(figsize=(12.0, 5.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

X_NAME, X_SUB, X_A, X_B = 3, 3, 78, 94
y = 92
ax.text(X_NAME, y, "Model", fontsize=11, color=INK, fontweight="bold", va="center")
ax.text(X_A, y, "Overall\nlog D R²", fontsize=10.5, color=INK, fontweight="bold",
        ha="center", va="center", linespacing=1.2)
ax.text(X_B, y, "Adjacent-pair\nlog SF R²", fontsize=10.5, color=INK, fontweight="bold",
        ha="center", va="center", linespacing=1.2)
ax.plot([2, 99], [85.5, 85.5], color=INK, lw=1.0)

y = 80
STEP = 10.2
for r in ROWS:
    if r is None:
        ax.plot([2, 99], [y + 4.2, y + 4.2], color=RULE, lw=0.7)
        y -= 1.2
        continue
    name, sub, a, b = r
    ax.text(X_NAME, y + 1.6, name, fontsize=10.5, color=INK, va="center")
    ax.text(X_SUB, y - 2.4, sub, fontsize=8.6, color=SUB, va="center")
    bold = name.startswith("Combination, current")
    for x, v in ((X_A, a), (X_B, b)):
        ax.text(x, y, v, fontsize=11, color=INK, ha="center", va="center",
                fontweight="bold" if bold else "normal")
    y -= STEP
ax.plot([2, 99], [y + 6.0, y + 6.0], color=INK, lw=1.0)
ax.text(3, y + 2.0,
        "Out-of-fold, leave-extractants-out cross-validation; 4,746 measurements, "
        "905 adjacent pairs. Combination overall R² on the 3,944 rows whose extractant has adjacent pairs.",
        fontsize=8, color=SUB, va="center")
fig.savefig(HERE / "results_table.png", bbox_inches="tight")
print("wrote results_table.png")
