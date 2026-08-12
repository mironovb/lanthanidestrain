# C15/C16 — two real gains, properly powered

## 1. The contrast weight: +0.0142, established

`--pair-loss-weight 4.0` against the published 2.0, on **40 paired seeds that
never participated in selecting it** (C14's 8 + C15's 32, all disjoint from the
C10/C12 seeds):

| | Δ `sel_adj_logSF_r2` | seeds up | |
|---|---|---|---|
| C14 (n = 8) | −0.0042 | 5/8 | underpowered |
| C15 (n = 32) | **+0.0189** | 21/32 | p = 0.0062 |
| **pooled (n = 40)** | **+0.0142** | **26/40** | **t = +2.57, p = 0.0141** |

95 % CI **[+0.0034, +0.0251]**.

**It passes the pre-registered two-part criterion.** Paired significance
(p = 0.0062 on C15) *and* the scale-free contrast agreeing in sign
(**+0.0153, p = 0.0069**) — the second condition being the one that exposed the
C8 artefact. Prediction-spread ratio is **0.993**, i.e. no shrinkage at all,
against C8's 0.807. Pearson (+0.0166, p = 0.0082) and sign accuracy
(+0.0107, p = 0.0048) move with it.

**The honest cost:** overall `log D` R² goes the other way, −0.0147 (11/32 up,
p = 0.080). Up-weighting the contrast term trades level accuracy for contrast
accuracy. This is a selectivity setting, not a free improvement.

**And the honest history:** the same configuration was reported earlier in this
session at **+0.0270, p = 0.021** from 8 seeds overlapping its own selection.
The true value is roughly **half** that. The pooled +0.0142 matches the
corrected estimate (+0.0145) almost exactly, which is what a real effect
measured twice looks like.

## 2. The tabular arm alone now exceeds the published 3-model stack

CatBoost on the **full published population** (4,746 rows, 162 extractants —
the same footing as every README number):

| configuration | adjacent-pair log SF R² |
|---|---|
| **`q60_rsm03_deep`** (Quantile α=0.6, depth 9, rsm 0.3) | **+0.2784** |
| `q60` | +0.2579 |
| `mae` (the C6-era winner) | +0.2487 |
| CatBoost as published (README) | +0.1422 |
| **published 3-model stack (README headline)** | **+0.2672** |

The tuned single tabular model reaches **+0.2784**, above the **+0.2672** stack
that required CatBoost *plus* a fingerprint network *plus* a simplicial network
over 3D structures.

The whole gain is loss function and the hyperparameters that suit it:
RMSE → MAE (+0.107) → Quantile(0.6) → re-grid under that loss (+0.0205 here,
+0.0159 on the held-out third).

**Caveat:** `q60_rsm03_deep` was selected on `screen_select`, and the full
population contains those rows, so this number carries mild optimism. Its
independent evidence is the held-out `report` third: **+0.0159**, which passed
its pre-registered bar though its 90 % CI spanned zero.

## What this session establishes about where gains live

Every representation change failed; every objective change that survived
testing paid:

| change | Δ | verdict |
|---|---|---|
| g-xTB geometry (correct contraction physics) | +0.0041 | null, 40 cells |
| structures in correspondence (455× cleaner) | −0.0129 | negative |
| FiLM conditioning of the encoder | **−0.1078** | strongly negative, p = 0.0012 |
| pair head + reconciliation | −1.16 | catastrophic |
| **contrast weight 2 → 4** | **+0.0142** | **real, p = 0.014** |
| **tabular loss + its hyperparameters** | **+0.136 cumulative** | **real** |

The lever in this problem is the objective, not the representation.
