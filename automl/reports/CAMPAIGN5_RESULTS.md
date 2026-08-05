# Fitting a stack on the pair objective: the effect replicated, the multiplicity budget did not

**Bogdan Mironov · 5 August 2026**
Pre-registered in `CAMPAIGN5_PREREGISTRATION.md`, committed before the look.
Data: `c5_test.csv`, `pair_stack_probe.csv`, `pair_regressor.csv`.
CPU only; no new model was trained.

---

## Verdict

**Null, by the pre-registered rule.** And it is the closest this study has come
to a positive result, in a way the previous four failures were not.

## 1. The confirmatory look — 78 held-out extractants, one look

| key | pairs | row-fitted | pair-fitted | delta | 90 % CI | 30-look CI | verdict |
|---|---|---|---|---|---|---|---|
| binned | 453 | +0.1673 | **+0.2231** | **+0.0559** | [+0.0076, +0.0845] | [−0.0127, +0.1244] | not distinguishable |
| strict | 766 | +0.0417 | **+0.1133** | **+0.0716** | [+0.0118, +0.1136] | [−0.0193, +0.1624] | not distinguishable |

The two sides differ in exactly one respect: the objective the meta-learner is
fitted against. Same four arms, fixed in advance; same non-negative least
squares; same leave-extractants-out folds; same pair set.

## 2. Why this null is not like the other four

| | previous screen winners | this one |
|---|---|---|
| direction on confirm | shrank | **grew**: +0.0171 → +0.0559 |
| magnitude | C1: +0.0176 → +0.0074, less than half | **3× larger on the held-out half** |
| agreement across block keys | typically one key only | **both**: +0.0559 binned, +0.0716 strict |
| uncorrected interval | usually spans zero | **excludes zero on both keys** |

What defeats it is the **30-look Bonferroni correction**, which carries every
look accumulated across four previous campaigns. This hypothesis was looked at
once.

**That is a reason for a future pre-registration with its own budget, not a
reason to reinterpret this one.** The look count was fixed at 30 before the
number existed; revising it afterwards is precisely the manipulation the
accounting exists to prevent. The null stands.

## 3. What is established regardless

**The row-level stack is badly mis-fitted, and that is not marginal.** On the
held-out half a stack fitted on levels reaches +0.1673 where the same arms
fitted on pairs reach +0.2231 — and under the strict key, +0.0417 against
+0.1133, nearly a threefold difference. Whatever the multiplicity accounting
says about the *contrast*, a practitioner choosing how to fit a stack now has a
measured reason to prefer the pair objective.

**Training on pairs does not work, by four independent routes.** The auxiliary
pair head (−0.0253), the pair head alone (−0.0832), inference-time
reconciliation (−1.2991), and a standalone pair regressor (best +0.0344 against
D0's +0.2702, with ridge collapsing to −231 on 990 features over 905 pairs). The
pair target is too data-poor to learn from: 905 pairs against 4,746 rows. The
level task is what makes the representation learnable, and the difference must be
taken afterwards.

**Stacking does not beat the best single arm.** Pair-level minus D0 alone is
−0.0033 on the tune half, and D0 is chosen in all 41 folds when the choice is
made per fold. The campaign's claim was never that a stack wins, only that a
stack should be fitted on the quantity it is scored on.

## 4. What would settle it

A fresh pre-registration whose look budget reflects its own family — the
meta-learner objective is a different question from encoder architecture, and
mixing them into one Bonferroni count is conservative to the point of being
uninformative. It needs a held-out half that has not been spent; the confirm
extractants have now been used four times.

The concrete design: fix the four arms, fix the optimiser, and test
`pair-fitted − row-fitted` on new extractants, or on a re-split of the data
under a fresh frozen seed, at a look count that starts from one.

## 5. Limits, stated

- No new model was trained. This re-weights existing out-of-fold predictions, so
  it inherits every limitation of the arms it combines.
- 453 binned pairs on the confirm half. The interval is wide because the data
  are few, not because the effect is unstable.
- The tune-half numbers were spent before this look and are reported in the
  pre-registration; there is no second bite.
