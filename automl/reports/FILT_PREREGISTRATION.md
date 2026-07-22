# Pre-registration: is the result about the complex, or about one radius?

**Written and committed before any filtration run finished.**

---

## 1. The question this is meant to settle

The positive result ([`STACK_RESULTS.md`](STACK_RESULTS.md)) is currently
specific to **one encoder at one filtration radius**: the simplicial network on
the Vietoris–Rips complex at 3.5 Å. The persistence-image CNN **failed to
replicate** it, which leaves two live readings and no way to choose between
them:

* **(a) it is about message passing over a Vietoris–Rips complex** — in which
  case a *different radius* should also add, because the complex is still a
  complex and the mechanism (strong + decorrelated) should still be met; or
* **(b) it is about one specific complex construction** — in which case a
  different radius should not add, and the claim narrows from "the simplicial
  encoder" to "the simplicial encoder at 3.5 Å", which is close to a tuning
  artefact.

`SYNTHESIS.md` named this the single most informative remaining experiment, and
that judgement was recorded before the runs existed.

---

## 2. Design

Same architecture, same objective, same folds, same seeds — **only
`--filtration-max` changes**, so the comparison isolates the complex rather than
the model:

| arm | filtration | rationale | seeds |
|---|---|---|---|
| F30 | 3.0 Å | tighter: essentially the coordination sphere | 8 |
| S0 | 3.5 Å | the published arm | 16 (existing) |
| F40 | 4.0 Å | looser: the shipped asset's own max edge | 8 |

Seeds are the first 8 of the published matched set, so every contrast is
seed-paired with S0.

Thresholding a filtration is **exact**, not approximate: the VR complex is a
filtration, so keeping simplices born below a radius yields the VR complex at
that radius (noted in `simplicial_data.py`). F30 and F40 are therefore genuinely
different complexes of the same structures, not perturbations of one.

---

## 3. Endpoints, fixed now

For each of F30 and F40, on the identical rows and folds:

**Primary.** Δ = stack(CatBoost + repaired + F) − stack(CatBoost + repaired),
adjacent-pair R², nested per-extractant weights, paired cluster bootstrap over
extractants, 400 draws, seed 0, 90 % interval. *Does this radius also add to the
best no-topology stack?*

**Secondary, descriptive.** Each arm's own adjacent-pair R² and its error
correlation with the repaired baseline — the two quantities the mechanism says
must both be favourable, so a failure can be attributed to one or the other
rather than left unexplained (this is how the PI-CNN failure was diagnosed).

**Fixed:** 8 seeds per radius, all ensembled; no radius added after seeing a
result; no re-running.

| outcome | consequence |
|---|---|
| **both** F30 and F40 add | reading (a): the finding is about the complex, not a radius. The claim broadens to message passing over VR complexes, and 3.5 Å stops looking special. |
| **one** adds | partial: report which, and the mechanism numbers for the one that fails |
| **neither** adds | reading (b): the claim narrows to *the simplicial encoder at 3.5 Å*. That is much weaker and much closer to a tuning artefact, and must be reported as such. |

**Multiplicity.** These are two further tests of the same family. Intervals are
reported nominally and Bonferroni-corrected for the seven confirmatory looks
this project has now taken at "does topology add" (S0, S2, stack primary, stack
decisive, S0X, F30, F40).

**Honest prior:** unknown, genuinely. The mechanism predicts (a) if the encoder's
advantage comes from message passing over *any* reasonable complex, and (b) if
3.5 Å happens to sit where the coordination shell is captured but conformer
noise is not. I do not know which, which is why it is worth running.

---

## 4. Guards

`control_guard --verify` must pass; the published S0 must still re-ensemble to
+0.2382. Outputs to `automl/artifacts/topo_filt/` and `automl/reports/filt_*`
only.

---

*Signed off before any filtration run completed — B. Mironov, 22 July 2026.*
