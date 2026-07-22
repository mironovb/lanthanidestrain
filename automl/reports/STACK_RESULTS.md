# POSITIVE: topology adds to the strongest baseline, and it is topology-specific

**Bogdan Mironov · 22 July 2026**
Pre-registered in [`STACK_PREREGISTRATION.md`](STACK_PREREGISTRATION.md)
(commit `d105d24`) before the contrasts were computed. No new training — every
out-of-fold vector already existed.

---

## Verdict

**All three pre-registered contrasts behaved as the design required.**

| # | contrast | Δ adjacent-pair R² | 90 % CI | verdict |
|---|---|---|---|---|
| **1 primary** | blend(S0, repaired) − repaired | **+0.0351** | [+0.0167, +0.0646] | **ADDS** |
| 2 control | blend(T0w, repaired) − repaired | +0.0055 | [−0.0174, +0.0184] | nothing |
| **3 decisive** | blend(S0) − blend(T0w) | **+0.0296** | [+0.0050, +0.0654] | **ADDS** |

**`blend(repaired, S0) = +0.2511` — the highest adjacent-pair R² in the study**,
at a nested weight of 0.55 on the topological arm with a zero-width IQR.

**The control is what makes this a topology result.** Contrast 1 on its own could
be satisfied by any model with decorrelated errors — that is exactly the trap
that killed the earlier blend claim, where an "interior maximum" credited to
topology reproduced *larger* for a plain tabular MLP. Here the matched tabular
control (T0w: same harness, folds, seeds and objective, encoder removed) adds
**+0.0055, spanning zero**, while the topological arm adds +0.0351. The encoder
is doing the work.

### The caveat, stated in the verdict rather than a footnote

> **Contrast 3's four-test corrected interval is [−0.0095, +0.0687] — it spans
> zero.**

This is the fourth attempt at "topology beats/adds to the repaired baseline"
(S0, S2, this stack, and the extended-S0 run in progress). Nominally the
decisive contrast is significant; corrected for multiplicity it is
**suggestive, not established**. Contrast 1 survives correction
([+0.0042, +0.0660]); contrast 3 does not.

**The honest one-line summary:** *topology adds to the best available model, the
addition is specific to the topological encoder rather than to ensembling, and
after correcting for four attempts the specificity claim is suggestive rather
than proven.*

---

## 1. The blends

| blend | adjacent-pair R² | overall R² | nested weight on the partner |
|---|---|---|---|
| repaired alone | +0.2206 | +0.3218 | — |
| **repaired + S0** | **+0.2511** | +0.3927 | 0.55 (IQR 0) |
| repaired + S2 | +0.2447 | +0.3897 | 0.50 (IQR 0) |
| repaired + T0w | +0.2289 | +0.3304 | 0.40 (IQR 0) |
| repaired + CatBoost | +0.2210 | +0.4357 | 0.30 (IQR 0) |

Two things worth noting beyond the endpoint:

- The S0 blend improves **overall** R² as well (+0.3218 → +0.3927), so this is
  not a selectivity gain bought by wrecking accuracy — the trade-off that
  characterised every earlier topological arm.
- Every nested weight has a **zero-width IQR**: the same weight is chosen for
  every held-out extractant, so the blend is not being rescued by per-extractant
  tuning.

---

## 2. Why this worked when four other approaches did not

The failures were all attempts to make the *encoder itself* score higher —
better geometry (water↔octanol), more structures (S2 conformers), less variance
(S2 block-centring, pretraining). Every one failed, and the geometry audit
explains why: the adjacent-lanthanide contrast is redundant with the tabular
ionic radius and sits below the optimisation-noise floor.

This asks a different question: **not whether topology is better, but whether it
is different.** S0 (+0.2382) and the repaired baseline (+0.2206) are close in
accuracy and share no architecture — simplicial message passing over 3D
complexes versus an sklearn MLP on ECFP + RDKit. Their errors are decorrelated
enough that the combination beats both, and the matched tabular control shows
that decorrelation comes from the encoder, not from having two models.

That is a weaker claim than "topology wins" and a more useful one: **topology
earns a place in the best model**, which is the configuration anyone would
actually deploy.

---

## 3. Correction to an earlier overclaim

`WO_RESULTS.md` §3 said "the 3D geometry carries no recoverable signal that
improves either selectivity or overall accuracy beyond a strong tabular model."
**That was too strong**, in two ways:

1. The geometry audit tested the **89-column tabular 3D summary**, not the raw
   geometry. The SNN message-passes over the raw complex; the summary is lossy.
2. This result shows the SNN *does* contribute beyond the strongest tabular
   model, in a stack, specifically.

The defensible version: *no tabular 3D summary improves on a strong tabular
model, and the raw-geometry encoder does not win alone — but it is decorrelated
enough from tabular models to add in combination.*

---

## 4. Correctness

- The nested blend was **vectorised**, and the speedup is exact rather than
  approximate: the blend is linear in predictions and `adjacent_pair_arrays` is
  linear too, so `dp(w) = (1−w)·dp_a + w·dp_b` holds exactly and each
  extractant's pair vectors are computed once instead of once per
  (extractant, weight). Verified against the groupby path to 1e-9 before any
  result used it. The naive version did not finish in 40 minutes.
- **Nested weights**: chosen per extractant on the other 148 only, so no row
  influences the weight it is scored under.
- `control_guard --verify` passes: 324 artefacts byte-identical. S0 still
  re-ensembles to **+0.2382** and the published headline to **+0.2426**.

---

## 5. What this means for the paper

The claim to make is **not** "3D topology predicts lanthanide selectivity better
than 2D" — four experiments say it does not, alone. It is:

> A simplicial encoder over 3D complexes is **decorrelated** from fingerprint
> models, and blending it with the best tabular model raises adjacent-pair
> separation from +0.221 to **+0.251**, a gain a matched no-topology control
> does not reproduce.

with the multiplicity caveat attached. The supporting negative results — the
objective finding, the `QuantileTransformer` correction, and the geometry audit —
are what make this one credible rather than a lucky fourth try.

**Next, already running:** an extended-S0 ensemble (32 more seeds of the
unchanged config) and the SNN-embedding-into-CatBoost test, both pre-registered.
Either could sharpen or overturn this.

---

*Reproduce: `python3 -m automl.topo.stack_test --n-boot 400`.*

---

## 6. The best deployable model (descriptive, but with the decisive control)

**This section is descriptive, not a pre-registered endpoint** — the confirmatory
claim is section 1. It is reported because it answers the practical question and
because its controls are the strongest in the study.

A three-way nested stack of CatBoost (accuracy), the repaired FCNN (selectivity)
and S0 (3D):

| stack | adjacent-pair R² | overall R² | median weights |
|---|---|---|---|
| **CatBoost + repaired + S0** | **+0.2672** | **+0.4369** | 0.20 / 0.30 / **0.50** |
| no topology (CatBoost + repaired) | +0.2263 | +0.4328 | 0.30 / 0.70 |
| topology slot given to the control (T0w) | +0.2208 | +0.4288 | 0.30 / 0.30 / 0.40 |

| contrast | Δ | 90 % CI | 5-test corrected |
|---|---|---|---|
| drop-in: add S0 to the best no-topology stack | +0.0381 | [+0.0191, +0.0495] | **[+0.0166, +0.0595] adds** |
| **swap: S0 vs the matched control in the same slot** | **+0.0446** | [+0.0298, +0.0544] | **[+0.0272, +0.0621] adds** |

**Both survive five-test correction** — unlike the two-way stack, whose decisive
contrast did not. The swap is the strongest form of the control available: the
same stack, the same two partners, the same slot, and only the occupant changes.
Filling it with the matched tabular arm gives +0.2208; filling it with the
simplicial encoder gives +0.2672.

Two further points:

- **+0.2672 is the highest adjacent-pair R² in the study**, above the originally
  published +0.2641 blend, and it reaches it at **+0.4369 overall R²** — close to
  CatBoost’s +0.4987 and far above any pure topological arm (+0.37). The
  selectivity/accuracy trade-off that characterised every earlier topological arm
  is largely gone.
- The nested weights put the **largest share (0.50) on the topological arm**,
  chosen per held-out extractant on the others only.

