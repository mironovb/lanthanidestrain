# Water↔octanol reorganisation: the probe did not survive a strong baseline

**Bogdan Mironov · 21 July 2026**
Pre-registered in [`WO_PREREGISTRATION.md`](WO_PREREGISTRATION.md) (commit
`e28f4c9`), executed by
[`automl/topo/water_octanol_test.py`](../topo/water_octanol_test.py) — both
written and committed before any model was fit.

---

## Verdict

**Negative.** The full water↔octanol reorganisation block does not improve
overall `log D` against a strong learner; the primary endpoint is significantly
**worse**.

> **A2 − A0 = −0.0269, 90 % CI [−0.0385, −0.0112], P = 0.00.**
> A3 − A1 = −0.0024 [−0.0095, +0.0035], not distinguishable.

The rough probe's +0.015 was a **weak-baseline artefact**, and diagnosing that
is the useful part of this result.

---

## 1. The numbers

Four arms, CatBoost, leave-extractants-out (5×3, seed 42), on the identical
4,633-row / 149-extractant / 888-complex both-solvents subset:

| arm | features | overall R² | within | between |
|---|---|---|---|---|
| **A0** baseline_2d | 746 | **+0.5076** | +0.298 | +0.748 |
| A1 + feat3d | 835 | +0.4774 | +0.272 | +0.713 |
| A2 + water↔octanol | 768 | +0.4806 | +0.277 | +0.714 |
| A3 + feat3d + water↔octanol | 857 | +0.4750 | +0.274 | +0.706 |

Every arm that adds geometry is **below** the 2D baseline. The water↔octanol
block (A2) is a hair above the single-solvent 3D block (A1), but both sit under
A0.

---

## 2. Why the probe said +0.015 and the real test says −0.027

The pattern is the one that has recurred all through this project, and it is
worth stating as the lesson:

> **A feature that helps a weak baseline is noise a strong baseline is better
> off without.**

The probe used `HistGradientBoostingRegressor` at R² = 0.437 and a single scalar.
The pre-registered test uses CatBoost — the overall-`log D` champion, R² = 0.508
here — and the strong learner already extracts, from the 2D block, whatever the
geometric feature was standing in for. Adding 22 noisy columns then costs it a
little.

This is the same mechanism as three earlier findings this project produced by
insisting on the strong baseline:

- the FCNN scaler gain that shrank once the baseline was repaired and ensembled;
- the blend "interior maximum" that reproduced for a plain tabular MLP;
- the S2 variance reduction that made the ensemble worse.

Each time, a promising signal against a convenient baseline vanished or reversed
against the right one. **The probe should have run against CatBoost from the
start.** It is recorded here as a methodological error, not hidden.

---

## 3. What this closes

Combined with the four-test selectivity audit
([`WO_PREREGISTRATION.md`](WO_PREREGISTRATION.md) §1), the picture is now
complete and consistent:

| question | answer | evidence |
|---|---|---|
| Does geometry improve adjacent-lanthanide **selectivity**? | No | 4 tests, all null/negative; 0.013 Å signal is redundant with the tabular ionic radius and below the 0.04 Å noise floor |
| Does geometry improve **overall** `log D` over a strong tabular model? | No | A1, A2, A3 all below A0 (CatBoost) |
| Does topology beat a **matched** control on selectivity? | Yes, +0.049 | but not the repaired baseline, and it is inductive bias, not information |

**On this dataset, with these GFN2-xTB geometries, the 3D structure carries no
recoverable signal that improves either selectivity or overall accuracy beyond a
strong tabular model.** That is a clean, well-supported negative — the kind that
prevents a great deal of wasted effort, and answers "when does 3D help" with a
specific, mechanistic "not here, and here is why."

---

## 4. What would change the answer

Not architecture — that is now thoroughly excluded. Only the geometry itself:

1. **Better geometries.** The selectivity signal is below the optimisation-noise
   floor. Real conformer ensembles (CREST/metadynamics, energy-weighted) and/or
   higher theory (DFT coordination spheres) could lower that floor. Large
   compute; the only physically real lever left.
2. **A different target.** Where 3D demonstrably matters — reaction barriers,
   binding free energies from explicit-solvent simulation — geometry would not
   be redundant with a fingerprint. `log D` on this set is not that target.

---

## 5. Correctness

The block itself is sound; the null is trustworthy. Five correctness tests pass,
and two real bugs were caught building it: a Kabsch point-correspondence error (a
2.25 Å coordination-shell RMSD, larger than the whole complex) and a
sign-convention error (`rg_change` did not negate under solvent swap). The
featuriser reuses the shipped `_coordination_shell`; `charge_missing` is 0 on
both solvents; the block is not linearly recoverable from feat3d (tested).

`control_guard --verify`: every published OOF parquet and result CSV
byte-identical (0 moved); the only changed artefacts are the S2 figure I
re-captioned earlier. `data/` untouched.

---

*Reproduce: `python3 -m automl.qc.water_octanol_features` then
`python3 -m automl.topo.water_octanol_test --n-boot 400`.*

---

> **Correction (22 July 2026).** Section 3 below said the geometry carries no
> recoverable signal beyond a strong tabular model. That was too strong. The
> audit tested the **89-column tabular 3D summary**, not the raw geometry, and
> [`STACK_RESULTS.md`](STACK_RESULTS.md) subsequently showed the raw-geometry SNN
> encoder *does* add to the strongest baseline in a stack (+0.0351 [+0.017,
> +0.065]), with a matched no-topology control adding nothing. The defensible
> claim is: no tabular 3D summary improves on a strong tabular model, and the
> encoder does not win alone -- but it is decorrelated enough to add in
> combination.

---

> **Erratum (9 August 2026).** The "~0.04 Å optimisation-noise floor" cited above was
> never measured. It has now been measured directly — 390 perturbed-restart
> GFN2-xTB optimisations over 30 structures — and the true reproducibility floor
> is **≈ 0.0002 Å**, 200× smaller. The 0.013 Å adjacent-lanthanide step is
> therefore **~68× above** the floor, not below it. The empirical nulls in this
> report stand; the explanation offered for them does not.
> See [`NOISE_FLOOR.md`](NOISE_FLOOR.md) and `SCIENTIFIC_FINDINGS.md` §H1.
