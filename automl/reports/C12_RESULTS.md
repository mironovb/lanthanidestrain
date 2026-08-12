# C12 — the best result of the campaign is one number in the objective

## The result

**`--pair-loss-weight 4.0` instead of the published 2.0.** One flag. 8 paired
deterministic seeds, identical rows, full published population (4,746 rows,
162 extractants, 956 complexes).

| metric | published (2.0) | **4.0** | Δ | seeds up | p |
|---|---|---|---|---|---|
| **`sel_adj_logSF_r2`** | +0.1892 | **+0.2162** | **+0.0270** | **7/8** | **0.021** |
| `sel_adj_pearson` | +0.4472 | +0.4790 | +0.0318 | 7/8 | |
| `sel_adj_sign_accuracy` | +0.6612 | +0.6714 | +0.0102 | 6/8 | |
| `r2_overall` (log D) | +0.3008 | +0.3124 | +0.0115 | 5/8 | |

Wilcoxon p = 0.016. Per-seed Δ: +0.058, +0.022, +0.012, +0.008, −0.003, +0.073,
+0.026, +0.019.

For scale: the entire simplicial-network contribution — the project's headline
3D result, requiring a Vietoris–Rips complex and a message-passing network — is
**+0.038**. This is 70 % of that, from changing one float.

## It is not the calibration artefact

C8 produced a +0.0333 that dissolved under inspection: R² rose while Pearson
fell, and R² after optimal rescaling (= Pearson², which removes all calibration)
**reversed** to −0.0036. The same test applied here:

| | as scored | scale-free (r²) | predicted-separation sd |
|---|---|---|---|
| C8 "gain" | +0.0333 | **−0.0036** ✗ | 0.0889 vs 0.1102 — 19 % *more* shrunk |
| **plw4** | +0.0270 (p = 0.021) | **+0.0289 (p = 0.015)** ✓ | 0.1554 vs 0.1441 — 8 % *less* shrunk |

The scale-free gain is **larger** than the raw one, and the predictions are
**less** shrunk rather than more. Every diagnostic that exposed the artefact
points the other way. Pearson rising 7/8 alongside R² is the same story.

## The gains do not stack — they were one effect

C10 found three axes each worth ~+0.02 on 4 seeds. At 8 seeds:

| config | Δ | seeds up | p |
|---|---|---|---|
| combo3 (plw4 + q0.7 + adjw6) | +0.0348 | 5/7 | 0.055 |
| combo2 (plw4 + q0.7) | +0.0279 | 6/8 | 0.031 |
| **plw4 alone** | **+0.0270** | **7/8** | **0.021** |
| adjw6 alone | +0.0138 | 7/8 | 0.125 |
| **q0.7 alone** | **+0.0019** | 6/8 | 0.872 |

`combo2` ≈ `plw4`, because `q0.7` contributes nothing once measured properly.
`q0.7` looked like **+0.0168 on 4/4 seeds** in C10 and is **+0.0019** at eight —
a textbook instance of E5 (within-screening winner's curse, ~0.04 on top cells).
`combo3` is nominally highest but on fewer seeds, less consistent (5/7) and not
significant; `plw4` is the honest recommendation.

## Why it is plausible rather than lucky

The C10 sweep traces a smooth interior optimum through the published value:

| `--pair-loss-weight` | 1.0 | **2.0 (published)** | **4.0** | 8.0 |
|---|---|---|---|---|
| Δ vs published | −0.0132 | — | **+0.0252** | +0.0094 |

and the same shape appears on the adjacent-emphasis axis (1 → −0.021,
3 published → —, 6 → +0.014, 10 → +0.012). Both hardcoded defaults sit below
their optimum. Finding A1 already established that *training the contrast rather
than the absolute value* is the single largest lever in this project; this says
the published run under-weighted the very term A1 identified.

## Status and what would settle it

**Not held-out confirmed.** `plw4` was selected from a 19-configuration sweep
(C10, 4 seeds) and replicated on 8 seeds including 4 fresh ones, growing
slightly (+0.0252 → +0.0270) rather than shrinking as `q0.7` did. That
replication is real evidence but it is not a clean held-out look: the `report`
third was spent on C11 this session and should not be re-entered casually.

The clean confirmation is a single pre-registered look at an unspent partition,
or a fresh split. Until then this is **strong, internally consistent, and
selected** — reported as such.


---

## CORRECTION — the effect does not replicate on independent seeds

C14 re-ran the contrast on seeds disjoint from every seed that had touched it:

| seed set | Δ | up |
|---|---|---|
| C10 selection seeds | +0.0252 | 4/4 |
| C12 additional seeds | +0.0288 | 3/4 |
| **C14 fully independent** | **−0.0021** | 4/6 |
| **pooled, 14 seeds** | **+0.0145** | 11/14, p = 0.079 |

The headline above (+0.0270, p = 0.021) is **not supported**. It came from a
seed set overlapping the one that selected the configuration; on seeds that
never chose it, the effect is zero. Per-seed sd is 0.0285, so n = 8 could never
have separated +0.027 from +0.014 — a power calculation I should have run
before claiming p = 0.021.

What survives is weaker and worth stating exactly: 11/14 seeds up (binomial
p ≈ 0.03) plus a monotone interior optimum in the C10 grid make a small real
effect around **+0.015** plausible. It is not established, and the scale-free
check — which correctly distinguished this from the C8 artefact — tests the
*nature* of an effect, not its *existence*.
