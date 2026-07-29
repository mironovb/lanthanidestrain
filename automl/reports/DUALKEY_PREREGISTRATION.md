# Pre-registration: does the topology result survive a stricter definition of "identical conditions"?

**Written and committed before the contrasts are computed.** No new training —
every out-of-fold vector already exists, so nothing here can be tuned by
re-running.

---

## 1. The defect this tests

The headline metric, `adjacent_pair_metrics` in `automl/evaluation.py`, groups
measurements into blocks by **`composition_key`**:

```python
cond_key = df[GEOM_COND_COLS].astype(str).agg("|".join, axis=1)
df["composition_key"] = df[GROUP_COL].astype(str) + "||" + cond_key
```

`GEOM_COND_COLS` are **binned** condition columns. The same file, eight lines
later, builds a second key and says plainly why the first one is not good enough
(`automl/dataset.py:387-391`):

> Strict variant: every numeric condition included, so the *only* thing that
> varies inside a block is the lanthanide. Delta learning needs this — with the
> binned key two rows can share a block while differing in extractant
> concentration, **which turns a real log D difference into label noise** on a
> zero-difference feature vector.

`strict_composition_key` has existed in the codebase for the whole study. **The
headline metric has never been computed with it.**

This matters because the claim is specifically about *differences between two
lanthanides measured under identical conditions with the same extractant*. If
"identical conditions" is actually "conditions that fall in the same bin", then
part of the quantity being predicted is condition effect, not selectivity.

---

## 2. What is already known — disclosed in full

I re-scored **two single arms** while scoping this work, before writing this
document. Those numbers exist and are stated here rather than presented later as
though they were part of the confirmatory result:

| arm | binned key | strict key |
|---|---|---|
| S0 simplicial (16-seed ensemble) | **+0.2417** | **+0.1703** |
| repaired FCNN (`StandardScaler`, 16 seeds) | **+0.2206** | **+0.1741** |
| S0 − repaired | **+0.0211** | **−0.0038** |
| adjacent pairs *n* | 905 | 1417 |
| true Δ SD (log units) | 0.271 | 0.224 |

So it is already known that **every arm scores lower under the strict key**, and
that **S0's single-arm edge over the repaired baseline changes sign**. That
single-arm contrast was already null under the binned key
(+0.0261 [−0.005, +0.076], `fixed_baseline_test.csv`), so its reversal is a
change of sign within a null and is **not** by itself evidence against the claim.

What is **not** known, and is what this test decides, is the behaviour of the two
**stack contrasts** — the ones the paper's claim actually rests on. Those have
not been computed under the strict key in any form.

---

## 3. Endpoints, fixed now

Both published confirmatory contrasts, recomputed under **both** keys, on the
arms exactly as they exist on disk:

| # | contrast | published value (binned) |
|---|---|---|
| **1 (drop-in)** | full stack (CatBoost+repaired+S0) − best no-topology stack (CatBoost+repaired) | +0.0381 [+0.0191, +0.0495] |
| **2 (swap)** | full stack − same stack with S0 replaced by the matched control T0w | +0.0446 [+0.0298, +0.0544] |

**Fixed protocol, identical to the published one except for the block key:**
nested leave-one-extractant-out stack weights, simplex grid step 0.10 as in
`best_stack.nested_stack`; paired cluster bootstrap resampling **whole
extractants**, `n_boot=400`, seed 0, 90 % interval, in the
**multiplicity-respecting** form (each drawn copy of an extractant tagged so a
twice-drawn cluster counts twice — the correction from `bootstrap_check.py`).

**Multiplicity.** This is a re-analysis of an already-reported endpoint under a
second metric definition, so it is a **new look at the same question**. The look
count for the topology claim rises from 8 to **10**. Both the uncorrected 90 %
interval and the 10-look Bonferroni interval are reported for every contrast.

**Secondary and descriptive only** (no decision attaches to these): the same two
contrasts for the filtration replications (3.0 Å, 4.0 Å); the full
`control_factorial` cell table; `s2_test`, `s0x_test`, `pi_sweep_test`
endpoints; and single-arm values for every cell.

---

## 4. Decision rule

The rule is written here so it cannot be chosen after seeing the answer.

| outcome | consequence |
|---|---|
| contrasts 1 **and** 2 exclude zero under **both** keys, after 10-look Bonferroni | **The claim stands and is strengthened.** The result is a property of the chemistry, not of a binning choice. Both columns are reported side by side in every table from here on, and the strict key becomes the primary metric for all *new* work, since it is the honest one. |
| both exclude zero under the binned key only, and span zero under the strict key | **The claim is downgraded, publicly.** The published effect is then partly an artefact of averaging measurements taken under different conditions into one cell. The headline is rewritten as a binned-key result with the strict-key null stated in the abstract, not the appendix. Stage 3 targets the strict key as primary and the campaign becomes a correction plus a rebuild. |
| both exclude zero under the strict key only | The binned key is the defective one, as the source comment already argues. Re-baseline the whole study on the strict key; the published numbers are superseded, not merely supplemented. |
| either contrast spans zero under both keys | The published result does not reproduce under the corrected (multiplicity-respecting) bootstrap at all. That is a far larger problem than the key, and it is reported first. |

**A drop in absolute R² under the strict key is expected and is not by itself a
failure.** Finer blocks average fewer replicates, so each block mean is noisier
and the true Δ spread is smaller (0.271 → 0.224). The endpoints are **contrasts
between models scored on the same pairs**, and that comparison is unaffected by a
uniform loss of precision. Only the contrasts decide.

**No stopping on a favourable partial result.** Both keys are computed in one
run, from one command, and both are reported whatever they say.

---

## 5. Guards

- `control_guard.py --verify` must pass before and after. The change to
  `automl/evaluation.py` is an **added optional argument** whose default
  reproduces current behaviour exactly; a regression test asserts that
  `adjacent_pair_metrics(...)` with no `block_key` is bit-identical to the
  pre-change function on the published S0 vector, and the standing precondition
  (S0 re-ensembles to **+0.2382**) must still hold.
- `evaluation.py` is in the guard's `SOURCE_FILES`, which are *recorded, not
  frozen*. The manifest will be re-snapshotted **deliberately**, with the
  superseded manifest hash written into `DUALKEY_RESULTS.md`.
- No training. No writes to `data/`. Outputs to `automl/reports/dualkey_*` and
  `automl/reports/DUALKEY_RESULTS.md` only. Existing reports append-only.

---

**Bogdan Mironov · 29 July 2026**
