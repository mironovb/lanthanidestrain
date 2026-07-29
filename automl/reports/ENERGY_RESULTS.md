# Reference xTB energetics: computed at last, and they make selectivity worse

**Bogdan Mironov · 29 July 2026**
Pre-registered in [`ENERGY_PREREGISTRATION.md`](ENERGY_PREREGISTRATION.md),
committed at `6abaf35` before any model had seen these features.
Data: `energy_test.csv`, `energy_diagnostic.csv`,
`automl/artifacts/xtb_reference/`. Jobs 5278069/5278074 (probe),
5278075 (campaign), 5278303 (test).

---

## What was done

`data/processed/feature_blocks/xtb_reference_calculation_queue.csv` has held 957
rows marked `not_run_requires_reference_xtb` for the whole study.
`binding_energy_eV`, `strain_energy_eV`, `homo_eV`, `lumo_eV` and
`homo_lumo_gap_eV` were null for all 5,992 rows, and `FINDINGS.md` §6 called them
"the most promising untested feature available".

They are now computed for **953 of 956** geometries, in ALPB water and ALPB
n-octanol, as `E_int = E(complex) − E(cage) − E(bare Ln³⁺)` with the metal simply
deleted at frozen geometry — avoiding the queue's requirement to fragment the
complex into chemically correct free ligands. That deviation from the queue's
own definition is recorded in the pre-registration §2.

The motivation was that **there is not one energetic descriptor in the entire
design matrix**, though a separation factor *is* a difference of complexation
free energies.

## The verdict

> **GFN2 energetics do not carry recoverable adjacent-pair information.**
> They make it substantially worse, and under the strict block key that is
> significant after Bonferroni for all 16 looks.

| variant | adj R² (binned) | adj R² (strict) | overall log D R² |
|---|---|---|---|
| **baseline CatBoost** | **+0.1422** | **+0.0819** | +0.4987 |
| + energy, all 57 columns | −0.0350 | −0.1994 | **+0.5068** |
| + energy, absolute only | −0.0057 | −0.2237 | +0.5018 |
| + energy, family-relative only | +0.0406 | −0.1713 | +0.4975 |

Contrasts against the baseline, multiplicity-respecting cluster bootstrap, 400
draws, 90 % interval, 16-look Bonferroni:

| variant | binned | strict |
|---|---|---|
| all | −0.2284 [−0.4760, −0.0482] · 16-look n.s. | **−0.2993** [−0.4566, −0.1792] · 16-look **[−0.5299, −0.0687] worse** |
| absolute only | −0.1929 [−0.4128, −0.0371] · 16-look n.s. | **−0.3166** [−0.4474, −0.2213] · 16-look **[−0.5046, −0.1287] worse** |
| family-relative only | −0.1477 [−0.3560, +0.0037] · n.s. | **−0.2638** [−0.3778, −0.1514] · 16-look **[−0.4520, −0.0756] worse** |

**Overall accuracy improves**: +0.4987 → **+0.5068**, the best overall log D R² in
the study. That is reported separately and must not be folded into a selectivity
claim — the pre-registration §5 fixed that in advance, precisely because this
combination was foreseeable.

## Why — measured, not narrated

The gate that ran *before* the campaign looked good. Substituting all 14
lanthanides into one **frozen cage** moved the interaction energy by **0.306 eV**
between adjacent members, **17.2×** the 0.0178 eV a separation factor of 2
corresponds to. The pre-registration said what that did and did not establish:

> it rules out a specific failure mode; it is not evidence for the hypothesis.

That caution was the right one, and `energy_diagnostic.csv` shows why. The probe
held the geometry fixed. **The dataset does not** — every complex is one
stochastic Architector/GFN2 conformer.

Within a ligand family (the same complex across the lanthanide series):

| feature | trend per series step | residual scatter | SNR | families with SNR < 1 |
|---|---|---|---|---|
| `e_int_water` | 0.1695 eV | 0.7313 eV | **0.25** | 99 % |
| `e_int_octanol` | 0.2009 eV | 0.7559 eV | **0.25** | 99 % |
| `dg_transfer` | 0.0136 eV | 0.0993 eV | **0.17** | 100 % |
| `gap_water` | 0.0249 eV | 0.3675 eV | **0.09** | 100 % |
| `q_metal_water` | 0.0019 e | 0.0145 e | **0.14** | 98 % |

The incumbent these have to displace is `Ionic Radius_metal`: a **lookup table**,
mean adjacent step 0.0141 Å, **zero scatter by construction**.

So every energy feature carries the same monotone series trend the ionic radius
carries, buried in conformer noise four times larger than the trend. A tree model
only ever compares values, so it splits on the noisy proxy in place of the exact
descriptor, and the selectivity metric collapses while overall accuracy — which
does not depend on resolving neighbours — improves slightly.

**This is not a new failure mode.** It is the one `dataset.py` already documents
for the *geometric* blocks, in the passage motivating g12/g13/g14:

> the baseline's clean, strictly monotone lanthanide descriptors (Z, index,
> Shannon radius) were being replaced by geometry proxies that carry the same
> trend *plus* conformer noise.

Energetics replicate it exactly, with numbers.

## What this means for the level of theory

The barrier is **not GFN2's element resolution** — that was measured at 17× the
required scale. The barrier is that **one conformer is a high-variance estimate**
of a quantity whose useful part is small.

Two remedies follow, and only one of them is cheap:

1. **Average the conformers.** The scatter is what a Boltzmann ensemble would
   reduce. To make the trend dominate on the most favourable feature, it has to
   fall **≥ 3.9×**, i.e. roughly **16 effectively independent conformers per
   complex** (scatter falls as √n). That is a quantitative target the
   metadynamics pilot did not have before — `S2_RESULTS.md`, `WO_RESULTS.md` and
   `PI_EMAIL.md` §9 all named multi-conformer sampling as the untested lever, and
   all three named it qualitatively.
2. **Raise the theory.** Not indicated by this result. Nothing here says GFN2's
   energies are wrong; it says one draw from a 0.73 eV distribution cannot
   resolve a 0.20 eV step. DFT on single conformers would inherit the same
   problem.

That distinction matters, because "use DFT" is the expensive conclusion and it is
**not** the one the data supports.

## Caveats, stated rather than buried

- **`E_int` is not a thermodynamic binding energy.** No relaxation of the cage,
  no proper reference states, ALPB applied to a highly charged fragment; the
  median is −55.2 eV, which is not a complexation free energy in any physical
  sense. It is an interaction descriptor, and the family-relative columns are
  what the model was expected to use.
- **Strain was not computed.** It needs a cage relaxation and was deferred to a
  second phase that this result makes unattractive.
- **3 of 956 geometries** produced no reference energies and are absent from the
  block rather than imputed.
- The binned-key contrasts are individually "worse" but **do not survive 16-look
  Bonferroni**; only the strict-key ones do. The conclusion rests on the strict
  column and on the diagnostic, not on the binned column.

## What was guarded

The A/B is exact. Adding 57 columns to the matrix cache must not change what
`baseline_2d` selects, or the control arm would be a different model; the guard
compares the column list before and after and aborts on any difference. It
passed: **746 columns, identical list**, and the baseline reproduces its
published values exactly (+0.1422 adjacent, +0.4987 overall).

Nothing was written to `data/` — including the queue CSV, whose
`calculation_status` column stays as shipped. The computed values live under
`automl/artifacts/xtb_reference/`.

---

**Reproduce**

```bash
MODE=probe sbatch automl/slurm/reference_xtb.sh
NSHARDS=16 MODE=run sbatch --array=0-15 automl/slurm/reference_xtb.sh
python3 -m automl.qc.energy_features
python3 -m automl.qc.energy_diagnostic
python3 -m automl.topo.energy_test --from-cache --n-boot 400
```
