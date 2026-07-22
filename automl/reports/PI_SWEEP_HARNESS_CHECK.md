# Harness validation: does the selection statistic recover a known answer?

**Bogdan Mironov · 22 July 2026**
Run **while Stage A was still executing**, on the published arms only — no sweep
result informed it. Recorded because it nearly caused an unnecessary amendment
to the pre-registration, and because the check is worth keeping.

---

## The worry

The pre-registration fixes the selection rule as *maximise the stack gain on the
tune half*. The first partial read of Stage A, at **one seed per configuration**,
gave every configuration a gain of **exactly +0.0000**. A statistic that returns
the same value for every candidate cannot rank them, and the obvious reading was
that the tune half is too small (84 extractants, 452 adjacent pairs) for
per-extractant nested weights to be estimated at all.

## The check

Rather than amend the rule on that suspicion, run the statistic on three arms
whose answers are already known from the published full-data analysis:

* **S0** simplicial — **adds** (+0.0381, the headline result)
* **P0** persistence images, shipped settings — **does not add** (−0.0041)
* **T0w** matched 2D control — **does not add** (−0.0066)

If the tune-half statistic ranks these correctly, it is fit for selection.

## Result — it ranks them correctly

Tune half, weights refit on the tune half, no-topology stack there = **+0.2473**:

| arm | seeds | own adj R² | err corr | nested gain | **stack weight** | known truth |
|---|---|---|---|---|---|---|
| **S0** | 8 | +0.2429 | +0.928 | **+0.0039** | **0.41** | **adds** |
| **S0** | 16 | +0.2509 | +0.928 | **+0.0062** | **0.57** | **adds** |
| P0 | 8 | +0.1562 | +0.966 | −0.0027 | 0.00 | does not |
| P0 | 16 | +0.1546 | +0.968 | −0.0028 | 0.00 | does not |
| T0w | 8 | +0.1423 | +0.965 | −0.0078 | 0.00 | does not |
| T0w | 16 | +0.1357 | +0.965 | −0.0041 | 0.00 | does not |

**The separation is unambiguous.** The contributing arm receives weight 0.41–0.57
and a positive gain; both non-contributing arms receive weight **exactly 0.00**
and a negative gain. The ordering also holds on the mechanism's two axes: S0 is
both stronger (+0.243 vs +0.156) and less correlated (+0.928 vs +0.966).

**No amendment. The pre-registered rule stands unchanged.**

## Why the one-seed read was misleading

At one seed every arm is far too weak to earn any weight — the tune-half single
seed anchor scores **+0.0427** against a **+0.2473** baseline — so all
configurations tie at weight 0 and gain +0.0000. That is the statistic behaving
correctly on uninformative inputs, not failing. It resolves as soon as the seeds
accumulate, which is the reason the sweep was raised from 3 seeds to 8.

## What this changes about how results are reported

The **stack weight is recorded alongside the gain** for every configuration. A
gain of exactly +0.0000 is ambiguous on its own — it means the arm was given
weight 0 and ignored, not that it was given weight and failed to help — and those
are different findings. The weight disambiguates them.

## A note on two different restrictions, both correct

The tune-stage and confirm-stage numbers restrict to a half at different points,
and they are not comparable to each other:

* **Tune stage** — arms are *trained* on tune rows only, so their out-of-fold
  predictions exist only there and the stack weights are necessarily fitted on
  the tune half. S0 measured this way gains **+0.0062**.
* **Confirm stage** — the winner is retrained on all 162 extractants (Stage C),
  the stack is fitted as the published analysis fits it, and only then are rows
  restricted to the confirm half. S0 measured this way gains **+0.0415
  [+0.0185, +0.0519]** — the frozen positive control.

Each matches the procedure of its own stage. The gap between +0.0062 and +0.0415
is the cost of refitting per-extractant weights on 84 extractants, and it is a
reason to read tune-half gains as a *ranking* only, never as effect sizes.

---

*Reproduce: the arm values come from `control_factorial.load_cells` and
`best_stack.nested_stack` restricted to `pi_split.load()["tune"]`.*
