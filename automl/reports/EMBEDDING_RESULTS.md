# Cross-fitted stacking of learned representations is invalid — a design failure, not a result about topology

**Bogdan Mironov · 22 July 2026**

---

## Verdict

The embedding test **cannot answer the question it was built for**, and the
reason is a flaw in my design rather than a fact about the simplicial encoder.
It is written up because the flaw generalises well beyond this project.

Handing CatBoost the 864-dimensional out-of-fold SNN embedding alongside the
tabular block made it dramatically **worse**:

| arm | adjacent-pair R² | overall R² |
|---|---|---|
| B0 tabular (746 features) | +0.1422 | +0.4987 |
| B1 tabular + SNN embedding (1,610) | **−0.0092** | **+0.2856** |

That is not "the representation is useless". It is that the feature matrix is
**incoherent**.

---

## 1. The diagnosis, measured

Out-of-fold embeddings come from *different models* — one per fold, per repeat
(15 in total). Neural encoders have **no canonical basis**: two runs that learn
equally good representations can place them in arbitrarily rotated, reflected
and rescaled latent coordinates. So column *j* of the assembled matrix means one
thing for rows held out in fold 1 and something unrelated for rows held out in
fold 2.

Tested directly, on one seed's matrix:

| check | result |
|---|---|
| 3-fold accuracy predicting **which fold** a row came from, from its embedding alone | **1.000** (chance 0.200) |
| mean within-fold spread | 10.21 |
| mean between-fold centre distance | 16.54 (ratio 1.62) |

**The embedding encodes which model produced it, perfectly.** A downstream
learner can separate folds trivially, and the features it is given do not have
consistent meaning across rows. Averaging over repeats — which this run did —
makes it worse, by blending incompatible bases.

---

## 2. Why the scalar version works and this one cannot

This is the useful part.

Cross-fitted stacking is standard and sound **for scalar predictions**: a
prediction is basis-free. Whatever internal coordinates a fold's model used, its
output is a number on the target's own scale, directly comparable across folds.

It is **not** sound for high-dimensional learned representations, for exactly
the reason above. The fix requires either

* a **single** encoder shared across folds — which reintroduces leakage, since
  its representation would be fitted on rows it is later scored on; or
* explicit **basis alignment** across folds (Procrustes or CCA onto a common
  frame), which is doable but adds an estimation step of its own, itself fitted
  on data.

Neither is free, and both need their own validation.

**The stack test is already the basis-free version of this question.** It hands
the *scalar* out-of-fold prediction of the same encoder to the same kind of
downstream combination, and it gives the positive result:
+0.0381 [+0.0173, +0.0510] for dropping topology into the best no-topology
stack, surviving both the corrected resampling and five-test multiplicity
([`STACK_RESULTS.md`](STACK_RESULTS.md) §6, §9). So the question "does the
learned topological representation help a strong downstream learner" **has
been answered — affirmatively — by the scalar route**, and the 864-dimensional
route simply cannot answer it as constructed.

---

## 3. What this does not say

- It does **not** say the SNN representation is uninformative. The scalar
  stacking result says the opposite.
- It does **not** invalidate any other result in this study. Every other arm
  compares out-of-fold **predictions**, never raw embeddings.
- The random-encoder control was included in the run for exactly this reason: if
  it collapses the same way, the collapse is the design. That is the expected
  outcome given a 100 % fold-identifiability score, which is a property of the
  construction and not of what the encoder learned.

---

## 4. The transferable lesson

> **Stack predictions, not representations** — unless the representations are
> put in a common basis first. Out-of-fold neural embeddings from *k* folds are
> *k* different coordinate systems, and concatenating them yields a feature
> matrix whose columns change meaning by row. A quick diagnostic: try to predict
> fold identity from the embedding. If you can, the matrix is not usable as
> features.

This one is cheap to check and easy to get wrong, and I got it wrong here before
checking.

---

*Reproduce: `python3 -m automl.topo.embedding_test --n-boot 400`; the
fold-identifiability check is in this document's commit message.*

---

## 5. The random-encoder control confirms the diagnosis

The run included an untrained encoder (`--epochs 0`, so its weights keep their
initialisation) precisely to discriminate "the representation is useless" from
"the construction is invalid". Result:

| arm | adjacent-pair R² | overall R² |
|---|---|---|
| B0 tabular | +0.1422 | +0.4987 |
| B1 + **trained** SNN embedding | −0.0092 | +0.2856 |
| B2 + **random** SNN embedding | +0.0176 | +0.3337 |

**Both collapse.** An encoder that learned nothing damages the model just as
thoroughly as the trained one — slightly *less*, if anything. That is exactly
what the fold-identity diagnosis predicts: with `--epochs 0` each fold still gets
its own random initialisation, so the assembled matrix is just as
basis-incoherent, and the harm comes from the 864 incoherent columns rather than
from their content.

So the conclusion stands and is now controlled: **this test cannot speak to the
value of the representation in either direction.** The scalar route
([`STACK_RESULTS.md`](STACK_RESULTS.md)) is the one that can, and it says the
representation helps.
