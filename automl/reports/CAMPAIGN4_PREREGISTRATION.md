# Pre-registration: build the neutral extracted species and test whether it matters

**Bogdan Mironov · 31 July 2026** — committed **before any structure is generated**.
Follows `SWEEP2_RESULTS.md`, `CAMPAIGN3_RESULTS.md`, `CEILING_CLOSED.md`.

---

## 1. Why

Three campaigns tested **15 cells** against the +0.2382 anchor; every one was null or
negative. All of them changed *how* the complex is represented. None changed **what
molecule is represented**.

> **953 of 956 modelled complexes are still cationic.** The species that partitions into
> kerosene is neutral. Charge neutralisation is the physics of solvent extraction and it
> is absent from every structure in this study.

## 2. What is measured, and what is not

Measured on the repo's own artefacts before designing anything:

| fact | consequence for the design |
|---|---|
| `charge == 3 − n_existing_nitrate` exactly (833 complexes at 0 nitrate/+2.99, 84 at 1/+1.98, 39 at 2/+1.00, **none at 3**) | `n_add = round(infer_charge)` per structure. Adding 3 everywhere would over-neutralise 123 complexes to −1/−2 |
| **100 % of complexes sit at CN ≥ 8** (63 % at CN 9) | the ligand already saturates the coordination sphere |
| Ln–O(nitrate) at GFN2/ALPB is **2.130 Å**, not the crystallographic 2.45–2.55 | every distance constant is taken from the corpus, not the literature |
| median molecular extent **10.7 Å**, 98 % of complexes have atoms beyond 6 Å | placement is surface-referenced, never at a fixed radius |

**The hypothesis being tested is therefore narrower than "nitrate competes with the
ligand for the inner sphere", and that is stated now rather than after the result.**
Because the sphere is saturated, nitrate can only occupy the second sphere. This is the
chemically correct species for a dataset dominated by diglycolamides — the 1:3 DGA
complex carries its nitrates as second-sphere counter-ions — but no inner-sphere
competition claim may be made from this campaign.

## 3. Replacement, not augmentation — declared in advance

Every neutral structure has total charge 0; every shipped structure is +1/+2/+3. Total
charge, dipole, Mulliken-derived descriptors and ALPB solvation energy therefore
separate the two sets **perfectly**.

- These assets are a **replacement** geometry set. Each arm is trained and scored on one
  asset only.
- Charged and neutral versions of the same complex **must never appear in the same
  fold**, and no run may mix assets.
- The output extxyz carries `charge:R:1` on every structure. Omitting it would make
  "charge missing" an augmentation marker — the exact confound
  `automl/qc/conformer_charges.py` exists to undo.

## 4. Arms and contrasts

Three assets over the **identical build_ids**, so all arms share a row set:

| asset | contents |
|---|---|
| `shipped` | the shipped geometries, subset to the HNO₃ complexes |
| `control` | re-optimised through the same ladder, **no anion added** |
| `neutral` | re-optimised through the same ladder, neutralised |

| contrast | question |
|---|---|
| control − shipped | how much is **re-optimisation alone**? |
| **neutral − control** | **the counter-ion effect — the pre-registered primary** |
| neutral − shipped | the end-to-end change |

The control is not a diagnostic and is not optional. The repo has already measured that
re-optimising the same molecule at a new level moves it by median 1.87 Å — "different
conformers, not refinements". Without the control, "the neutral structure differs from
the shipped one" carries no information at all.

## 5. Scope, and the coverage it costs

Nitrate only, on the 683 complexes whose single recorded acid is HNO₃ (98 % of complexes
map to exactly one acid). Placing nitrate on a complex measured in HCl would be
knowingly wrong chemistry.

**This retains 634 of 905 adjacent pairs (70 %).** The anchor is re-scored on the same
634 pairs; no number from this campaign may be compared to a 905-pair number.

## 6. Decision rules, fixed now

| situation | what is reported |
|---|---|
| `neutral − control` > **+0.005** on the 84 tune extractants | it becomes the single confirmatory candidate |
| its confirmatory contrast excludes zero after correction for **≥ 29 looks** | **the counter-ion matters.** This would be the study's first genuine improvement to the headline metric |
| it spans zero | screening noise; report the null |
| it does not clear +0.005 on tune | report the null and **do not** spend the confirmatory run |

Look count **29**: 26 carried forward from campaign 3, plus one per new arm. The confirm
extractants have already been spent twice (sweep2's C1). Carrying the count forward is
the point of counting looks; resetting it per campaign is the easiest way to manufacture
a winner.

Confirmatory stage: **16 seeds a side, both replicated**, scored **once** on the 78
confirm extractants, under **both** block keys, with the multiplicity-respecting cluster
bootstrap.

## 7. Honesty clauses, pre-committed

- **A pilot on whole ligand families runs first**, and the full generation is not
  submitted until its validation report passes clean. Whole families, because
  family-correlated failure is the failure mode that silently reshapes a dataset.
- **The run fails** if any extractant group (n ≥ 10) or metal (n ≥ 20) has a rejection
  rate above 3× the global rate, or if the accepted set's `(metal, extractant_group)`
  composition differs from the input at p < 0.01. Precedent: 26 of 86 octanol SCF
  failures were Eu, visible only because it was tabulated.
- **A rejected structure is recorded, never dropped silently**, with every gate value as
  a number and a `reject_code` from a frozen vocabulary.
- If the accepted fraction falls below **80 %**, the campaign reports the generation
  outcome and does **not** proceed to the ML comparison — a 30 % rejection would reshape
  the dataset more than the counter-ion changes it.
- `--verify-against-shipped` must reproduce the original geometries element-for-element
  through the new code path before any new asset is trusted.
- Nothing is written to `data/`.

---

## Amendment 1 — the outer-sphere premise is falsified by the pilot (31 July 2026)

Written **before** any amended structure was accepted or any model trained.

Section 2 asserted that "because the sphere is saturated, nitrate can only occupy
the second sphere", and forbade any inner-sphere competition claim. **The smoke run
falsifies that**, and the mistake is now clear: the analysis behind it tested **rigid**
insertion into the shipped geometry. Relaxation opens the coordination sphere.

Three dicyclohexano-18-crown-6 complexes, nitrate seeded at 5.75–6.50 Å:

| | seed Ln···N | final Ln···N | final Ln–O | O inside 3.10 Å |
|---|---|---|---|---|
| 57, nitrate 0 | 6.00 | **2.72** | 2.33 / 2.53 | 2 |
| 63, nitrate 1 | 6.50 | **2.57** | 2.13 / 2.77 | 2 |
| 68, nitrate 0 | 5.75 | **2.47** | 2.23 / 2.24 | 2 |

Ln–O of 2.13–2.53 Å against the corpus mean of **2.130 Å** is textbook bidentate
inner-sphere coordination, and Ln(crown)(NO₃)ₙ is a well-known structure. In every
case `cn_ligand` is **unchanged** (10→10, 10→10, 9→9): the ligand kept its own donors
and the nitrate coordinated in addition.

### What changes

1. **`MODE_MIGRATED` is no longer a rejection.** Inner-sphere coordination reached by
   relaxation is the correct outcome, not a failure. The binding mode is recorded per
   nitrate (`inner` / `outer`, with the Ln–O distances) and becomes a reported
   covariate rather than a filter.
2. **The G4 RMSD gate is mis-specified and is demoted to a diagnostic.** It was set
   assuming an outer-sphere ion that should not perturb the ligand. If the nitrate
   coordinates, the ligand *must* adjust — measured 1.19–1.64 Å RMSD against the
   control, with donor shifts of 0.33–0.48 Å. Rejecting those would discard exactly
   the physical response the campaign exists to capture.
3. **The structural gates that remain hard are the ones that cannot be explained by
   accommodation**: `cn_ligand` exactly unchanged (G5), and covalent connectivity over
   the original atoms bit-identical (G4c) — no bond may break or form anywhere in the
   ligand. G4c was specified in the plan and had not yet been implemented; it is now
   the load-bearing check, and it is added before any structure is accepted.
4. **The hypothesis is upgraded, not weakened.** The campaign can now test inner-sphere
   competition — the more interesting original question — because the anion reaches the
   inner sphere on its own. No claim is made that this happens for every family; the
   binding-mode split is reported.

### What does not change

The decision rule, the +0.005 gate, the 29-look count, the 16-seeds-a-side confirmatory
design, the replacement-not-augmentation declaration, and the 80 % accepted-fraction
floor all stand exactly as pre-registered.
