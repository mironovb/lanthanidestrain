# The noise floor of the adjacent-pair metric is not identifiable from this dataset

**Bogdan Mironov · 31 July 2026**
Data: `reliability.csv`, `ceiling_v2.csv`, `ceiling_test.csv` (withdrawn).
CPU only. Pre-registered as T1 of `CAMPAIGN3_PREREGISTRATION.md`.

---

## Why this was attempted a third time

A ceiling matters because the study's headline is **+0.2382** and nobody can say
whether that is a third of what is attainable or nearly all of it. Two attempts
failed:

- `ceiling_test` — **withdrawn** (`AUDIT_2026-07-30.md` E1). It measured the SD
  of pair differences across exact condition sets *inside* a binned block, a
  quantity the model can predict and the metric averages away on both sides.
- `ceiling_v2` — **NOT IDENTIFIABLE**. It estimated a noise SD on the 273 cells
  carrying duplicates and assumed it transferred to all 905 pairs.

T1 proposed split-half reliability, which measures the metric on itself and
needs no transfer assumption. It does not work either, and the reason is worth
recording precisely.

## Attempt 3a — split-half reliability

Split each replicated cell's rows into halves, average each, form adjacent pairs
from each half, correlate; Spearman–Brown to full size.

| grouping | cells ≥ 2 rows | pairs/split | half–half `r` |
|---|---|---|---|
| binned block (the metric's own key) | 653 | 268 | **−0.711** |
| binned block, split by DOI | 679 | 264 | **−0.848** |
| strict block, split by DOI | 259 | 104 | **−0.575** |

A reliability cannot be negative. **The estimator is not at fault** — on
simulated data with a true signal and independent noise it returns the right
answer, rising with replicate count:

```
n_replicates    2      4      8     16     32
split-half r  +0.51  +0.67  +0.80  +0.89  +0.94
pure noise    +0.02   --    +0.01   --    -0.00
```

So the negative values are a property of the data. Grouping *more finely* (by
DOI) made them **more** negative, which rules out the natural explanation that
random splitting breaks a matched experimental pairing.

## Attempt 3b — variance components

Estimate the pooled within-cell variance directly and subtract its contribution
from the observed pair variance.

| key | replicated cells | pooled within-cell SD | var(observed `dy`) | implied noise var |
|---|---|---|---|---|
| binned | 653 | 0.9492 | 0.0733 | **1.3979** |
| strict | 273 | 0.7194 | 0.0500 | **0.9845** |

The implied noise exceeds the observed variance **19-fold**. That is impossible
for a valid noise estimate, and it localises the failure exactly.

## Why every route fails: the replicated cells are not the modelled cells

| | |
|---|---|
| pairs whose cells carry **no replicate** | **70 %** (637 of 905) |
| var(`dy`) for those pairs | 0.0700 |
| var(`dy`), all pairs | 0.0733 |
| var(`dy`), pairs with min n ≥ 3 | 0.1019 |

If typical rows really carried the 0.95 SD measured on replicated cells, pairs
built from single-row cells would show var(`dy`) ≥ 1.8. They show **0.0700** —
26× smaller. The replicated cells therefore do not measure the noise of typical
rows. Two mechanisms, both visible in the data:

1. **Pooling.** 46 % of binned blocks contain more than one exact condition set,
   up to **117**. Within-cell scatter there is largely genuine condition
   variation, not measurement error.
2. **Selection.** A cell acquires a duplicate precisely when two sources report
   the same nominal experiment — which happens more often when they disagree.

Both push the replicate subset's variance above the population's, and neither
can be corrected without knowing the very quantity being estimated.

## What this means

- **No ceiling should be quoted for this metric from this dataset.** Three
  methods, three failures, each for a documented reason. This supersedes and
  explains `ceiling_v2`'s NOT IDENTIFIABLE verdict rather than merely repeating
  it.
- **The +0.2382 headline cannot be expressed as a fraction of attainable.** Any
  such statement in this study is unsupported; the withdrawn "39 % of
  attainable" was the first casualty and remains withdrawn.
- **It is answerable, but not here.** A designed replicate study — the same
  extractant/metal/conditions measured independently *by design* rather than
  opportunistically — would identify the noise floor. That is a data-collection
  recommendation, not an analysis one.

## Limits, stated

- The simulation validating the estimator uses Gaussian noise and a shared
  offset; real measurement error may be heavier-tailed. It establishes that the
  estimator works when its assumptions hold, not that they hold here.
- The DOI grouping relies on the 100 % join over 110 DOIs from `ceiling_v2`.
- "46 % of blocks pool >1 condition set" is a property of the *binned* key.
  The strict key does not pool, and still gives a negative split-half
  correlation — so pooling alone does not explain the whole failure, and the
  residual is unexplained. Recorded as unexplained rather than rationalised.
