# Operational report — 3D-feature AutoML study

## Outcome

Built and ran an AutoML study answering: *does information extracted from the
Architector / GFN2-xTB geometries improve the leave-extractants-out `log D`
model, and which 3D signal source carries it?*

**Answer: no, not measurably, as the geometries are currently generated.** The
improvements that do survive a paired significance test come from the learner
and the sample weighting, not from any feature block. Full narrative with
caveats: [`reports/FINDINGS.md`](reports/FINDINGS.md). Every number is
regenerated from artefacts by [`make_tables.py`](make_tables.py) into
[`reports/tables.md`](reports/tables.md).

Headline (protocol B, 5 repeats × 5-fold grouped CV, paired cluster bootstrap):

| | R² | vs LightGBM baseline |
|---|---|---|
| LightGBM 2D baseline | 0.490 | — |
| CatBoost + inverse-extractant weights, 2D | 0.528 | **+0.037 [+0.008, +0.066], P = 0.99** |
| full stack (CatBoost + weights + anchored + clip), **2D only** | **0.534** | +0.044 |
| full stack **+ G5 xTB electronics** | 0.534 | +0.0005 over 2D-only |

Paired against the CatBoost 2D baseline, the 2D-only stack has the *higher*
P(better) on both overall R² (0.69 vs 0.67) and within-extractant R²
(0.73 vs 0.53). On the full series the 3D block does not earn its place.

### The result that matters most

**That full-series null is an average of two significant, opposite effects.**
Split at the CN-9/CN-8 boundary and paired-test each half against its own 2D
baseline, with the geometry-QC flag held constant in both arms:

| | La–Gd (3980 rows) | Tb–Lu (1966 rows) |
|---|---|---|
| + curated 3D | −0.028 [−0.044, −0.004], **P = 0.034 worse** | +0.048 [+0.020, +0.076], **P = 0.998 better** |
| + G5 xTB electronics | −0.033 [−0.049, −0.015], **P = 0.002 worse** | +0.046 [+0.012, +0.077], **P = 0.990 better** |

Two confounds were tested and cleared:

* **Geometry-QC flag** (halves differ 88 % vs 64 % OK). Worth +0.0008 in the
  light half and +0.0125 in the heavy half — ≈ 20 % of the heavy gain. The
  remaining ≈ 80 % is descriptor signal and is significant on its own.
* **Data thinness** (82 extractants vs 187). Subsampling La–Gd to the heavy
  half's coverage — ending up *thinner*, baseline R² 0.29–0.38 vs 0.409 — leaves
  **11 of 12** matched comparisons negative across three draws, though only 3
  clear P < 0.05 individually. Corroborating, not decisive.

*Why* the halves differ is open. Candidates: CN-8 spheres better described by a
single conformer than CN-9, or heavy-Ln extraction being more geometry-driven.
Nothing measured here distinguishes them.

**Caveat that survives either answer:** even where 3D raises accuracy it
degrades the La→Lu ordering (log-SF R² −0.025 → −0.084/−0.096), trading
separation-factor prediction for pooled R² — the wrong trade for a separations
campaign.

## Scope

New directory `automl/` only. **No tracked file was modified; `data/`,
`raw_data/`, `reports/`, `src/`, `scripts/`, `slurm/` are untouched.**
Confirm with `git status --short` — the only entries are `?? automl/`, Python
bytecode caches, and `tests/_tmp/` (scratch written by the repo's own
`tests/test_stage_e.py` when pytest was run; left in place per AGENTS.md).

Generated artefacts (~93 MB) live under `automl/artifacts/` and are gitignored;
the rendered reports and figures are small and are the deliverable.

Python packages installed into `~/.local` (module environment unchanged):
`lightgbm catboost optuna ase rdkit ripser persim shap tabulate gudhi pytest`.

## Validation

* `python3 -m pytest tests/ -q` → **79 passed** (the 2 that failed initially
  were a missing GUDHI dependency in `src/geometry_features.py`, unrelated to
  this work; installing gudhi fixed them).
* `python3 -m pytest automl/tests -q` → **11 passed**. These check the geometry
  maths against analytic answers: SASA of an isolated atom equals
  `4π(r_vdw+r_probe)²` to 1e-9, the continuous shape measure is 0 for an ideal
  polyhedron and monotone under distortion, a fully enclosed metal gives
  %V_bur = 100 and solid-angle fraction 1.
* All 20 shell scripts pass `bash -n`; all 17 Python modules import cleanly.
* 3D extraction: 1235 geometries, **0 failures**.
* **Determinism check (free, from an accidental duplicate).** Two different
  SLURM jobs independently ran the same configuration (`delta-learning + G5`,
  5 repeats x 5 folds). Every metric agrees bit-for-bit: `r2_overall`
  0.528339 vs 0.528339, |diff| = 0.0e+00, and likewise for between, within,
  within-composition, MAE and the separation-factor R². The pipeline is fully
  reproducible given a spec.
* ~400 cross-validated experiments recorded to JSONL with full provenance.

## Cluster state

All work was submitted as SLURM jobs on `xeon-p8` and `debug-cpu` within the
account caps (40 submitted tasks, 2 concurrent nodes). Two self-feeding queue
drivers (`slurm/queue_driver.sh`, `queue_driver2.sh`) submitted later stages as
capacity freed.

Jobs I cancelled deliberately, to reprioritise toward the decisive experiments:
two Optuna workers (the LightGBM studies already had 100 trials), one CatBoost
greedy-selection task (superseded by the direct CatBoost ablation), and the
`champion_okonly` array (partial results kept and analysed — see FINDINGS §7).

**Nothing is left queued that needs supervision.** If jobs are still running,
`bash automl/refresh_reports.sh` regenerates every table and figure from
whatever has landed; it is login-node safe (reads results, fits no models).

## Known limits and risks

* Several comparisons sit at P(better) between 0.4 and 0.8 — genuinely
  undecided, not negative. They are labelled as such.
* `anchored` and `pairwise` are **batch** predictors: a row's prediction depends
  on which other rows are scored with it. No label leakage, but they are valid
  only for whole-extractant screening, not single-row inference.
* Protocol A results (`ablation`, `arch`, `models`, `select`, `cnfree`) used a
  splitter whose repeats were 81 % correlated. They are screening only and are
  never compared against protocol B numbers.
* Two shipped columns (`feat3d__polyhedron_scalars__coreCN_donor_gap`,
  `next_donor_dist`) contain `+inf` on 13 rows. Sanitised to NaN in this
  pipeline; **should be fixed upstream** in the dataset builder.

## Shortest next steps

```bash
module load anaconda/Python-ML-2025a
export PYTHONPATH=/home/gridsan/bmironov/lanthanidestrain
bash automl/refresh_reports.sh        # regenerate all tables + figures
less automl/reports/FINDINGS.md       # the report
```

The one experiment that would move the science: generate **5 conformers per
complex for 10–20 extractants spanning the series**, Boltzmann-average the
descriptors, and re-run the G1/G5 ablation against the single-conformer
versions. That measures directly how much of the 60–80 % descriptor scatter is
recoverable, on a few hundred extra geometries rather than a full regeneration.
