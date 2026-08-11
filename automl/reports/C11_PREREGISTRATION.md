# C11 — confirming the q60 hyperparameter re-grid on the held-out third

Written before the `report` partition is looked at for this contrast.

## Why the test exists

C6 established the tabular loss optimum at `Quantile(alpha=0.6)`. Its
hyperparameters (depth, lr, l2, rsm) were re-gridded around **MAE** — the
previous winner — and never around q60. Every hyperparameter in the shipped
tabular configuration was therefore chosen under a loss that has since been
beaten twice.

Closing that gap on the tuning population (`screen_select`, 106 extractants):

| config | adjacent-pair logSF R² |
|---|---|
| **q60_rsm03_deep** (depth 9, rsm 0.3) | **+0.2481** |
| q60_rsm03 | +0.2397 |
| q60 (previous winner) | +0.2304 |
| mae (the C6-era winner) | +0.2188 |

Apparent gain over q60: **+0.0178**.

## Why it cannot be believed yet

Finding E5 in `SCIENTIFIC_FINDINGS.md`: **the within-screening winner's curse is
~0.04 on the top cells** — larger than the effect being claimed. The number above
is the maximum over 8 new configurations on the partition they were selected on,
which is exactly the situation E5 describes.

## The test

Both `q60_rsm03_deep` and `q60` on the **`report`** partition (56 extractants,
disjoint from `screen_select`), 8 seeds, 5 folds × 3 repeats, identical rows.

**This is ONE look.** The report third has been chosen on before; it is spent
here for a single pre-specified two-config contrast and not re-entered.

## Declared in advance

- **Endpoint:** `sel_adj_logSF_r2` of q60_rsm03_deep minus q60 on `report`.
- **Real:** Δ ≥ +0.010 and the sign preserved. Half the screening estimate, which
  is roughly what E3 ("shrinkage from screening to held-out is worse than about
  half") predicts survives.
- **Null:** Δ < +0.010, including any negative value. Then the re-grid is
  winner's curse, the shipped q60 stands, and I report it as such.
- Quoted raw and Bonferroni-corrected for the 2 configs looked at.

A null here is a real result: it would say the tabular hyperparameters are
already at their optimum for this loss, and that E5's curse estimate applies at
full strength to a re-grid of this shape.
