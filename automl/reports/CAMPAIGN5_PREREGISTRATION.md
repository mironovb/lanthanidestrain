# Pre-registration: fit the stack's weights on the quantity the stack is scored on

**Bogdan Mironov · 5 August 2026** — committed before the confirmatory look.
Follows `CAMPAIGN4_RESULTS.md`.

---

## 1. The hypothesis

Every stack in this study fits its weights by minimising error on **row-level
log D**, and the metric then differences block-averaged predictions. But
`sel_adj_logSF_r2` scores a **difference**, and the weights that best predict
levels are not the weights that best predict differences.

> **Fitting the same arms, with the same optimiser and the same
> leave-extractants-out folds, on the PAIR objective rather than the LEVEL
> objective improves the adjacent-pair metric.**

Nothing about the arms, the data or the split changes. Only the objective the
meta-learner is fitted against.

## 2. What the tune half already showed

The tune half exists for selection and it has selected. Reported here so the
confirmatory look is made against a stated expectation, not a fishing licence:

| arms | row-level | pair-level | gain |
|---|---|---|---|
| all 15 available | +0.2472 | +0.2669 | **+0.0197** |
| the published 4 (`G0`, `repaired`, `CatBoost`, `D0`) | +0.2522 | +0.2693 | **+0.0171** |

The effect holds at both weight counts, so it is not an artefact of fitting 15
weights over 41 groups. The row-level stack is **worse than its own best
component** (+0.2472 against D0's +0.2702), which is what fitting the wrong
objective looks like.

## 3. What this campaign does NOT claim

- **Not that stacking beats the best single arm.** It does not: pair-level minus
  D0 alone is −0.0033 on the tune half, and D0 is selected in all 41 folds when
  the choice is made per fold. The claim is strictly about *how a stack's
  weights should be fitted*, for anyone who wants a stack.
- **Not that training on pairs works.** It does not, and four experiments now
  say so: the auxiliary pair head, the pair head alone, inference-time
  reconciliation, and a standalone pair regressor (best +0.0344 against D0's
  +0.2702, with ridge collapsing to −231 on 990 features over 905 pairs). The
  pair target is too data-poor to learn from. This campaign changes only the
  *meta-learner*, which has 4 parameters, not the base models.

## 4. The confirmatory design, fixed now

- **Arms:** the four the published stack uses — `G0`, `repaired`, `CatBoost`,
  `D0`. Fixed in advance, not selected.
- **Optimiser:** non-negative least squares, weights summing to 1, identical for
  both arms of the contrast. Non-negative because a negative weight means the
  meta-learner is exploiting a sign flip it cannot justify chemically.
- **Folds:** leave-extractants-out, identical for both.
- **Contrast:** `pair-level − row-level`, computed on the **78 confirm
  extractants**, under **both** block keys, with the multiplicity-respecting
  cluster bootstrap over whole extractants.
- **Look count: 30.** 29 carried forward through campaign 4, plus this one.

| outcome | reported as |
|---|---|
| the contrast excludes zero after 30-look correction | **fitting a stack on the pair objective is a real improvement** — the study's first positive methodological result on the headline metric |
| it spans zero | the tune-half gain did not replicate; report the null |

**One look only.** The tune-half numbers in section 2 are already spent; there is
no second bite.

## 5. Pre-committed honesty clauses

- The row-level comparator must use the **identical** optimiser and folds.
  Comparing against `best_stack.nested_stack`, whose weight grid is a different
  algorithm, would confound "pair versus row" with "NNLS versus grid" — a trap
  the probe fell into once already and which was corrected before any number was
  believed.
- Both sides are scored on the identical pair set.
- If the gain is positive but the interval spans zero, that is a null and is
  reported as one. A +0.017 tune gain that does not replicate is exactly the
  pattern the two-stage design exists to catch, and it has caught it four times.
