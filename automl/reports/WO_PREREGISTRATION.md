# Pre-registration: does the water↔octanol reorganisation block add to log D?

**Written and committed before any model is fit.** Nothing below may be revised
once the first fit runs.

Prior work: `CONTROL_RESULTS.md`, `S2_RESULTS.md`, and the geometry information
audit recorded in §Context of the plan. Baseline pinned by
`control_guard.py` (324 artefacts, `--verify` passing at commit time).

---

## 1. Why this target, and not selectivity

Four independent tests this session established that the **adjacent-lanthanide
selectivity** signal is not in the GFN2-xTB geometry:

| test | R² / corr |
|---|---|
| coordination-sphere within-block difference → logSF | −0.03 |
| full 89-column 3D block within-block difference → logSF | −0.04 (ridge) |
| coordination rigidity → per-extractant selectivity | +0.03 |
| water→octanol response → adjacent-pair R² | **hurts** (−0.13 → −0.22) |

The 0.013 Å adjacent-lanthanide radius difference is both already exact in the
tabular block and below the ~0.04 Å optimisation-noise floor. So topology is
capped for selectivity, which is why the control, S2 and its ablation all failed
there. This is accepted, not retried.

**But `log D` is a water/octanol partition coefficient**, and every complex was
re-optimised in both solvents. The complex's water→octanol geometric *response*
is a direct 3D probe of that exact partition, absent from 2D and from the
single-solvent 3D block. A single scalar already added +0.015 to overall `log D`
R² in a rough probe. This tests the full response block, honestly.

---

## 2. The block and the arms

`automl/qc/water_octanol_features.py` builds a 22-feature reorganisation block
per complex present in both solvents, from the existing charge npz — donor
distance shifts, coordination-shell Kabsch RMSD, polyhedron angle deformation,
metal and donor charge redistribution, and whole-complex compaction. Built and
validated (5 correctness tests pass, incl. sign-convention and
not-recoverable-from-feat3d).

Four arms, **identical row subset and folds**, CatBoost (overall-`log D`
champion), leave-extractants-out 5×3, seed 42:

| arm | features |
|---|---|
| A0 | baseline_2d (746) |
| A1 | baseline_2d + feat3d (835) |
| A2 | baseline_2d + water↔octanol (768) |
| A3 | baseline_2d + feat3d + water↔octanol (857) |

**The subset is 4,633 rows / 149 extractants / 888 complexes** — every complex
optimised in both solvents. (My earlier rough probe said 5,788; the proper cache
join drops more, and 4,633 is the honest number. All arms share exactly this
set — asserted in the harness, not assumed.)

---

## 3. Endpoints, fixed now

**Primary.** A2 − A0, overall `log D` R², paired cluster bootstrap over
extractants (`compare.paired_bootstrap`, the multiset version), 400 draws,
seed 0, 90 % interval. *Does the block add over 2D alone?* baseline_2d already
contains `Ionic Radius_metal` and `lanthanide_index`, so this controls for the
metal size — any gain is beyond it.

**Secondary.** A3 − A1. *Does it add beyond the single-solvent feat3d block?*
This is the novelty claim; the correctness suite already shows the block is not
linearly recoverable from feat3d.

**Descriptive.** Within- vs between-extractant R² decomposition (a per-complex
feature can only help the complex-driven part); reported for every contrast.

**Fixed:** the 4,633-row subset; CatBoost; 5×3 folds seed 42; the 22 features as
built; no feature added, removed or retuned after seeing a result.

| outcome | consequence |
|---|---|
| A2 − A0 excludes 0 positive **and** A3 − A1 excludes 0 | geometry carries new, non-redundant overall-`log D` signal — a genuine working geometric result |
| A2 − A0 positive, A3 − A1 spans 0 | the response helps but feat3d already captures it; report as such |
| A2 − A0 spans 0 | the probe's +0.015 did not survive the full protocol; report the null |

---

## 4. Scope and honesty, up front

- **This is overall accuracy on a 4,633-row subset, not the selectivity claim.**
  Both are reported together in `WO_RESULTS.md` so neither is oversold. The
  selectivity audit is the stated reason for changing target.
- CatBoost is the champion for overall `log D` (0.499 in the prior study); it is
  the hardest learner to add to, which is why it is the primary. A weaker
  learner would make the block look better for the wrong reason.
- The block only exists for both-solvent complexes; rows without are **excluded**,
  never imputed with a zero shift (which would read as "perfectly rigid").

---

## 5. Guards

1. `control_guard.py --verify` must pass after the run — every published OOF
   parquet byte-identical. New outputs in `automl/artifacts/water_octanol/` and
   `automl/reports/wo_*` only.
2. `data/` never written. No source edits while the analysis runs.
3. Reports append-only; `WO_RESULTS.md` carries the outcome.

---

*Signed off before any fit — B. Mironov, 21 July 2026.*
