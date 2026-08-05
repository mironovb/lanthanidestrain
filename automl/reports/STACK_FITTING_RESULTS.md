# A stack fitted on log D allocates its weights by the wrong criterion

**Bogdan Mironov · 5 August 2026**
Data: `pair_fit_replication.csv`, `arm_profile.csv`, `c5_test.csv`,
`pair_stack_probe.csv`. CPU only; no new model was trained.

---

## The finding

> **Stack weights fitted by minimising error on row-level log D are allocated by
> level accuracy. The metric scores differences between adjacent lanthanides, and
> level accuracy is the wrong criterion for that.**

The fitted weights show it directly. Over the four arms of the published stack:

| arm | level R² | pair R² | **row-fit weight** | **pair-fit weight** |
|---|---|---|---|---|
| **CatBoost** | **+0.4987** (best) | +0.1441 (worst) | **0.794** | 0.141 |
| G0 | +0.3733 | +0.2487 | 0.206 | 0.238 |
| repaired | +0.3217 | +0.2229 | 0.000 | 0.299 |
| **D0** | +0.3444 | **+0.2498** (best) | **0.000** | 0.323 |

Row-fitting gives **79 % of the weight to the arm that predicts log D best and
differences worst**, and **zero** to the arm that predicts differences best.

## The mechanism, measured rather than asserted

The explanation makes three falsifiable predictions. All three hold:

| prediction | outcome |
|---|---|
| the two fits disagree most about the most level-flattered arm | **CatBoost / CatBoost** ✓ |
| the heaviest row-fit weight goes to the arm best on **levels** | **CatBoost / CatBoost** ✓ |
| the heaviest pair-fit weight goes to the arm best on **pairs** | **D0 / D0** ✓ |

*A first version of this test reported a mismatch. It ranked "most
level-flattered" over all 15 arms, where S1 ties CatBoost at +0.3546 and sorts
first — but S1 carries no weight in this stack, so it cannot be what the two fits
disagree about. The test was mis-built, not the hypothesis; it is restricted to
the weighted arms above.*

## What it costs, and when

Fitting both stacks leave-extractants-out over all 162 extractants, changing only
the objective:

| arms | row-fitted | pair-fitted | delta | bootstrap draws +ve | random halves won |
|---|---|---|---|---|---|
| 2 (G0, repaired) | **+0.2743** | +0.2522 | **−0.0222** | 0.1 % | 0 % of 200 |
| 3 (+ CatBoost) | +0.2060 | +0.2671 | **+0.0610** | 96.8 % | 96.5 % |
| 4 (+ D0) | +0.2060 | +0.2645 | +0.0585 | 98.4 % | 98.0 % |
| all 15 | +0.2069 | +0.2658 | +0.0589 | 98.2 % | 99.5 % |

*(binned key; the strict key agrees throughout — +0.0714, +0.0670, +0.0505.)*

**Adding a third arm drops row-fitting from +0.2743 to +0.2060** — a stack made
*worse* by more information — while pair-fitting holds near +0.265 and stays
there through 15 arms.

**The exception is predicted by the mechanism.** With only G0 and repaired, no
level-flattered arm is present, so there is nothing for level-fitting to
mis-weight and row-fitting is the better of the two. An effect that says in
advance when it should not appear, and is then absent exactly there, is stronger
evidence than one that always wins.

## Relation to campaign 5's null

`CAMPAIGN5_RESULTS.md` reported this contrast as **not distinguishable** on a
single held-out look, because a 30-look Bonferroni budget accumulated across four
campaigns of unrelated hypotheses swallowed an interval that otherwise excluded
zero. **That verdict stands as recorded** — the look count was fixed before the
number existed, and revising it afterwards is the manipulation the accounting
exists to prevent.

This document does not revise it. It reports a different quantity: the effect
estimated on all 162 extractants with nested fitting, together with its
stability. Across 8 configurations the delta is positive in 6, median **+0.0587**,
with bootstrap support of 96.8–98.6 % and 96.5–100 % of 200 random extractant
halves agreeing in every multi-arm case. No single correction choice decides
that.

## What to do with it

For anyone deploying a stack on this metric: **fit the meta-learner on the
quantity it is scored on.** It is a four-parameter change, costs no new training,
and on this data recovers roughly +0.06 of adjacent-pair R² relative to
level-fitting whenever the pool contains an arm that is strong on levels and weak
on differences — which is the normal case, since gradient boosting is exactly
that arm.

The corollary is a warning: **adding a strong level predictor to a level-fitted
stack can make the selectivity metric worse.** That is not intuitive and it is
measured here at −0.068.

## Limits, stated

- No new model was trained; this re-weights existing out-of-fold predictions and
  inherits every limitation of the arms it combines.
- Stacking still does not beat the best single arm: pair-fitted minus D0 alone is
  −0.0033 on the tune half, and D0 wins all 41 folds under per-fold selection.
  The claim is about how to fit a stack, not that a stack wins.
- The estimate uses the full data, so it is not an independent confirmation in
  the pre-registered sense. Its warrant is stability across splits, arm sets and
  both block keys, and that is how it is reported.
- Training *on* pairs remains a failure by four independent routes; only the
  four-parameter meta-learner is fitted on pairs here.
