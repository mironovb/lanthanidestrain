# The magnitude compression is mostly real, and mostly not fixable

**Bogdan Mironov · 29 July 2026**
Data: `calibration_test_binned.csv`, `calibration_test_strict.csv`.
Jobs 5278341, 5278342.

---

## The question

`PUBLICATION_ASSESSMENT.md` §3.3 and `PI_EMAIL.md` §7 both report that measured
adjacent-pair separations span roughly ±2 log units while predictions span about
±0.5 — every model "gets direction and ranking substantially better than
magnitude". Two readings, with opposite consequences, and nobody had separated
them:

- **Optimal shrinkage** — under squared error a genuinely uncertain model *should*
  shrink toward the mean. Then rescaling makes R² worse and there is nothing to
  fix; the compression is a property of the problem and the paper says so.
- **Miscalibration** — the models are trained on absolute log D and the difference
  is derived, so nothing forces the difference to be correctly scaled. Then a
  rescale is free R² and the calibrated number is the one to report.

## The measurement

A recalibration of the predicted difference, fitted **nested by extractant** —
the transform for extractant *g* comes from the other 161 only, the same
discipline as the stack weights — in three forms: `scale` (one parameter),
`affine`, and `isotonic` (any monotone map, an upper bound on what rank-preserving
calibration can do at all).

| model | raw | scale | affine | isotonic | span raw | span calibrated |
|---|---|---|---|---|---|---|
| **binned key** | | | | | | |
| CatBoost | +0.1422 | +0.1550 | +0.1654 | +0.1713 | 0.63 | 0.54 |
| repaired FCNN | +0.2206 | +0.2166 | +0.2126 | +0.1607 | 0.52 | 0.51 |
| S0 simplicial | +0.2382 | +0.2406 | +0.2385 | +0.1866 | 0.43 | 0.52 |
| **full stack** | **+0.2672** | **+0.2758** | +0.2745 | +0.2221 | **0.42** | **0.53** |
| **strict key** | | | | | | |
| **full stack** | **+0.1918** | +0.1914 | +0.1852 | +0.1874 | **0.38** | **0.46** |

Gain on the deployed stack, best of the three transforms, with a cluster
bootstrap over extractants (400 draws, 90 %):

| key | best transform | gain | interval | P(>0) |
|---|---|---|---|---|
| binned | `scale` | **+0.0087** | [+0.0003, +0.0200] | 0.95 |
| strict | `scale` | **−0.0004** | [−0.0137, +0.0119] | 0.54 |

## Reading

**Not established, and the honest word is "mostly shrinkage".**

The binned-key gain clears zero, but only just — the lower bound is **+0.0003**,
three ten-thousandths. And it is the **best of three transforms with no
multiplicity correction**; applying even a 3-look Bonferroni would sink it. It
does not reproduce under the strict key, where the same procedure returns
−0.0004 with P(>0) = 0.54, which is a coin flip.

This study has been caught by exactly this shape of claim before: a persistence-
image "tuning gain" of +0.0178 that replication reduced to +0.0003, and whose
post-mortem in `PI_SWEEP_PRECISION.md` says *"comparing a selected maximum
against a single baseline measurement is not conservative even when the noise
floor is known"*. A best-of-three maximum at P=0.95 that fails to replicate on a
second metric definition is that same object.

**So: report the raw numbers.** Do not report calibrated ones.

## The part that is a real result

Recalibration **cannot** repair the compression even when it is allowed to:

| key | span before | span after |
|---|---|---|
| binned | 0.42× the true spread | 0.53× |
| strict | 0.38× | 0.46× |

Even the isotonic transform — free to apply any monotone map — makes R² *worse*
on five of the six models. So the predictions are compressed because the models
are genuinely uncertain, not because their scale is wrong.

~~That combines with the ceiling measurement into one coherent statement for the
paper: the models recover about 39 % of the attainable variance and predict roughly
40–50 % of its true spread.~~

**WITHDRAWN 30 July 2026** — the ceiling half of that sentence is gone (erratum
below). What survives, and needs no ceiling:

> The models predict roughly **40–50 % of the true spread** of adjacent-lanthanide
> separation, and post-hoc rescaling does not repair it — even a free monotone map
> makes R² worse on five of six models. The compression is a property of the
> problem, not of the models' calibration.

That is a better sentence than the current "both versions compress toward zero",
because it says *why*, and it says what would and would not fix it.

## Caveats

- The transform is selected as the best of three and the reported interval is
  **not** corrected for that selection. Stated here rather than left implicit;
  it is the reason the binned result is called "not established" rather than
  "positive".
- `isotonic` overfits badly on this sample size (it costs the full stack −0.045
  under the binned key), which is informative in itself: with 905 adjacent pairs
  spread over 162 extractants there is not enough data to fit a free monotone
  map nested.
- CatBoost is the one model that does gain consistently (+0.1422 → +0.1713 under
  isotonic, binned). It is also the most compressed-in-the-wrong-direction arm
  (span 0.63). Worth a line in the paper, not a claim.

---

**Reproduce**

```bash
python3 -m automl.topo.calibration_test --key composition_key --n-boot 400
python3 -m automl.topo.calibration_test --key strict_composition_key --n-boot 400
```

---

## Erratum, 30 July 2026 — the ceiling is WITHDRAWN

**Every statement in this document about a ceiling, about "39 % of attainable", or
about "+0.412 of headroom" is withdrawn.** See
[`AUDIT_2026-07-30.md`](AUDIT_2026-07-30.md).

The estimator measured how much the adjacent-pair difference moves across distinct
*exact* condition sets inside one binned block, and treated that as irreducible. It
is not: the model holds **64 exact numeric `cond__` columns** and 46 % of binned
blocks vary internally, and `adjacent_pair_arrays` averages **y and p on the same
grouping**, so per-condition variation never reaches the metric at all.

The contradiction that settles it: inside the 203-pair subset the estimate came
from, the implied ceiling is **+0.173** — *below* the best model's +0.267. The
+0.679 divided that subset's noise variance (0.0235) by the **full** 905-pair set's
variance (0.0733), populations 2.6× apart in spread.

`ceiling_v2.py` rebuilt it properly, joining the **DOI** column that
`raw_data/*_SAFE.csv` carries and the ML table dropped (100 % join, 110 DOIs). The
verdict is **not identifiable**:

- **94 %** of the disagreement between rows with identical model features is
  *within a single paper*, not between papers — so it is not source conflict, which
  was the assumption the first attempt rested on.
- A quarter of it is a recorded covariate the pipeline drops, resolving to one
  cause: `cond__diluent__other` collapses **42 distinct solvents** over 5.5 % of
  rows.
- Three quarters has **nothing recorded varying at all**, median SD 0.231 log
  units — which propagated to two single-measurement cells exceeds the entire
  observed spread of the target.

**The four closure results are unaffected** — recombination, energetics, conformers
and the objective each failed on its own terms. What is withdrawn is any claim
about *how much room* they were failing to reach, and consequently the framing of
them as "routes to the headroom". Optimisation targets must not be expressed as a
fraction of an attainable maximum, because that maximum is unknown.
