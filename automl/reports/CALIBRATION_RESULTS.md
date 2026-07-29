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

That combines with the ceiling measurement ([`CEILING_RESULTS`](ceiling_test.csv),
adjacent-pair R² ≤ **+0.679**) into one coherent statement for the paper:

> The models recover about 39 % of the attainable variance in adjacent-lanthanide
> separation and predict roughly 40–50 % of its true spread. Both numbers are
> consequences of the same thing — the signal is small relative to the
> measurement noise in this dataset — and neither is repaired by post-hoc
> rescaling.

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
