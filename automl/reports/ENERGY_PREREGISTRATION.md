# Pre-registration: do reference xTB energetics carry adjacent-pair selectivity?

**Written and committed before any model is trained on these features.** The raw
energies are being computed as this is written; no model has seen them, and the
endpoints below are fixed now.

---

## 1. The gap this fills

`data/processed/feature_blocks/xtb_reference_calculation_queue.csv` holds 957
rows, every one marked `not_run_requires_reference_xtb`. `binding_energy_eV`,
`strain_energy_eV`, `homo_eV`, `lumo_eV` and `homo_lumo_gap_eV` are **null for
all 5,992 rows** of the dataset. `FINDINGS.md` §6 calls them "the most promising
untested feature available" and they have sat uncomputed for the whole study.

The omission matters for a specific reason: **there is not one energetic
descriptor in the entire design matrix.** Every feature is a fingerprint, an
RDKit scalar, a condition, or a geometric/topological summary. Yet a separation
factor *is* a difference of complexation free energies — the quantity being
predicted is thermodynamic, and nothing in the feature set expresses one.

## 2. What is computed, and one deviation from the queue

The queue defines binding as
`E_complex − (E_Ln_ion + n_ligs·E_free_ligand + n_fill·E_free_fill)`, which needs
the complex fragmented into chemically correct free ligands — a connectivity
problem, on structures with a documented coordination-QC history (258
`FAIL_LONG_BOND`, 215 borderline).

`automl/qc/reference_xtb.py` computes the same physics without fragmenting:

    E_int = E(complex) − E(cage) − E(bare Ln³⁺)

where **cage** is the complex with the metal atom deleted, at frozen geometry, at
charge `q − 3`. Deleting one atom needs no bond perception. For a fixed ligand
the cage term is near-common-mode between two adjacent lanthanides, so the
*difference* in `E_int` isolates the metal-dependent part — exactly what the
adjacent-pair metric scores. **This is a deviation from the queue's stated
definition and is recorded as one.** Strain, which needs a relaxation, is a
second phase and is not part of this endpoint.

Also computed: `dG_transfer = E(complex, ALPB octanol) − E(complex, ALPB water)`.
log D is a partition coefficient; nothing in the current matrix expresses one.
Plus HOMO/LUMO/gap and the Mulliken metal charge in both solvents.

## 3. The gate that has already passed, and what it does *not* establish

Before committing the campaign I ran a metal-substitution probe: all 14
lanthanides substituted into the same frozen cage, everything else held fixed
(job 5278074, `automl/artifacts/xtb_reference/metal_probe.csv`).

| quantity | median adjacent-pair |ΔE| | vs the SF=2 scale (0.0178 eV) |
|---|---|---|
| raw total energy | 0.2200 eV | 12.4× |
| **interaction energy** (bare ion removed) | **0.3063 eV** | **17.2×** |

The raw column is the misleading one — two different elements have different
total energies for the trivial reason that they are different atoms — which is
why the ion-subtracted column is the one that decides.

**What this establishes:** GFN2 is not blind to the identity of adjacent
lanthanides; its energies vary far above the scale a useful separation factor
corresponds to. A flat line here would have made the whole campaign a null.

**What it does not establish:** that the variation is *correct*. A method can
vary strongly and wrongly. The probe rules out a specific failure mode; it is not
evidence for the hypothesis. That is what the endpoints below are for.

## 4. Endpoints, fixed now

A new block-preset `baseline_2d_energy` = `baseline_2d` + the `g_energy` block.
The old preset is untouched, so the A/B is exact: same rows, same folds, same
seeds, same architecture, one block added.

The `g_energy` block carries both absolute values **and** their
within-`ligand_anion_family` relative forms, mirroring the existing
`add_within_ligand_relative` treatment — because the absolute binding energy is a
ligand property and the adjacent-pair signal lives in the *difference between
neighbouring Ln for the same ligand*.

| # | contrast | question |
|---|---|---|
| **1 (primary)** | CatBoost(`+energy`) − CatBoost | does the block help the strongest single model? |
| **2** | stack(CatBoost, repaired, S0)(`+energy`) − stack(CatBoost, repaired, S0) | does it help the deployed model? |
| **3 (specificity)** | contrast 2 restricted to `dG_transfer` alone vs `E_int` alone | which physics is doing the work, if any? |

Scored on adjacent-pair log-SF R² **and** overall log D R², under **both** block
keys, multiplicity-respecting cluster bootstrap, `n_boot=400`, seed 0, 90 %
interval, plus Bonferroni.

**Selection discipline.** Any choice among energy-feature variants is made on the
84 **tune** extractants of the frozen `pi_sweep/split.json` only. The 78 confirm
extractants are scored **once**, for whichever variant the tune half selects.

## 5. Decision rule

| outcome | consequence |
|---|---|
| contrast 1 **and** 2 exclude zero positive | Energetics are a real and missing feature class. This is the first non-geometric addition to the study and would likely be its largest single gain; report it as the headline alongside the topology result. |
| 1 positive, 2 spans zero | The block helps the tabular model but is redundant with what the stack already has. Report as a CatBoost improvement, not a model improvement. |
| 2 positive, 1 spans zero | Implausible on its face; treat as a multiplicity artefact unless it replicates on the confirm half, and say so. |
| both span zero | **GFN2 energetics do not carry recoverable adjacent-pair information at this level of theory**, despite resolving the metals 17× above the relevant scale. That is a strong, publishable negative — it would say the barrier is the *accuracy* of the semi-empirical energies, not their resolution, and it would point at DFT rather than at more features. |
| overall R² rises but adjacent-pair does not | Report exactly that. The study's standing weakness is that the stack (+0.4369) is worse on overall log D than plain CatBoost (+0.4987); a block that fixes only that is still worth having and must not be inflated into a selectivity claim. |

## 6. Guards

`control_guard --verify` before and after. Outputs to
`automl/artifacts/xtb_reference/` and `automl/reports/energy_*`. **Nothing is
written to `data/`** — including the queue CSV, whose `calculation_status`
column stays as shipped; the computed values live under `automl/artifacts/`
with their own provenance. Existing reports append-only. A unit test on a
synthetic fixture checks the energy bookkeeping (`E_int` sign, cage charge
`q − 3`) so a bookkeeping error cannot masquerade as chemistry.

---

**Bogdan Mironov · 29 July 2026**
