# AutoML on the 3D-enriched lanthanide extraction dataset

Goal: find the best way to improve the **leave-extractants-out** baseline for
`log D` using information extracted from the Architector / GFN2-xTB optimised
complex geometries, and identify *which* 3D signal source actually carries that
information.

Everything here is additive. `data/` is treated as read-only: no geometry is
regenerated, no shipped table is modified. All new artefacts live under
`automl/artifacts/` and `automl/reports/`.

---

# Headlines

**4,746 measurements · 162 extractants · 14 lanthanides · 953 unique GFN2-xTB
complexes · leave-extractants-out CV throughout.**

### 1. Topology improves adjacent-lanthanide separation — the hardest case

Adjacent-pair log-separation-factor R² rises from **+0.005** (FCNN on ECFP +
RDKit) to **+0.263**. Neighbouring lanthanides differ by ~0.013 Å in ionic
radius; that is the separation industry actually cares about.

| test | seeds | Δ adjacent-pair R² | 90 % CI | P(better) |
|---|---|---|---|---|
| SNN ensemble vs **FCNN** | 16 | **+0.2426** | [+0.181, +0.333] | 1.00 |
| PI-CNN ensemble vs **FCNN** | 15 | +0.1984 | [+0.107, +0.266] | 1.00 |
| SNN ensemble vs **CatBoost**, standalone | 16 | +0.0867 | [+0.025, +0.122] | 0.99 |
| SNN blend vs **CatBoost**, pre-registered w = 0.5 | 16 | +0.1004 | [+0.038, +0.140] | 1.00 |
| SNN blend vs **CatBoost**, nested leakage-free w | 16 | **+0.1074** | [+0.039, +0.150] | 1.00 |

Every interval is a paired cluster bootstrap resampling whole extractants.

### 2. The transferable finding: train the contrast, not the value

Selectivity is a within-extractant *difference* between two metals, but every
conventional model optimises *absolute* log D — the objective never touches the
quantity being scored. A pairwise-difference loss over composition blocks (3×
weight on adjacent lanthanides) plus adjacent-pair checkpoint selection is what
converts a null into a significant result. This should transfer to any
separation-factor problem, with or without topology.

### 3. Topology does *not* improve overall accuracy

Best topological arm R² = 0.375 against CatBoost's 0.499. The gain is
adjacent-pair-specific and costs ~0.06 overall R² at the operating point. The
models also under-predict separation *magnitude* (true ±2 log units, predicted
±0.5) — they rank and sign well, they do not size.

### 4. One control is still missing, and it decides what this is about

All 51 topological runs use a topological encoder; none tests the tabular
features with the same *objective* and no topology. If contrast-training alone
explains the gain, the finding is about the objective, not topology. ~2 h of
compute — see `reports/PUBLICATION_ASSESSMENT.md`.

### 5. Tighter geometries do not help — the limit is conformational

All 1,235 complexes were re-optimised at GFN2-xTB `tight` with ALPB solvation in
water and n-octanol; residual forces fell **83×** (0.185 → 0.0022 eV/Å).
Adjacent-pair R² did not move and descriptor smoothness improved by only +0.005
median across 19,127 cells. Median RMSD from input is 1.87 Å — these are
*different conformers*, not refinements. Multi-conformer sampling is the physics
lever; tightening the optimiser is not.

### 6. The earlier persistence-image null was a testing artefact

The prior ΔR² = +0.004 came from flattening 20×20 images into a tabular model,
which the asset manifest explicitly forbids
(`do_not_flatten_into_tabular_MLP`). With the CNN readout they were built for,
the same images reach +0.208.

## Read in this order

| file | what it is |
|---|---|
| `reports/PI_REPORT.md` | supervisor-facing: headline, caveat, figures, decision requested |
| `reports/PUBLICATION_ASSESSMENT.md` | is this publishable — verdict, strengths, eight blockers |
| `reports/TOPOLOGY_RESULTS.md` | full results, including superseded explanations and what refuted each |
| `reports/TOPOLOGY_METHODS.md` | methods, verification, and the bugs found along the way |
| `reports/FINDINGS.md` | the earlier tabular-descriptor study, partly superseded above |

Figures: `reports/figures/topo_*.png` (+ matched PDFs), from
`python3 -m automl.figures_topo --all`.

## Method notes that matter

- **Ensembling is mandatory, not decorative.** Single models are unstable (SNN
  seed SD 0.047; one seed scored +0.066 where another scored +0.197). Every
  headline number averages 15–16 seeds, and *all* seeds of a configuration go in
  — never a best-scoring subset.
- **Blend weights are pre-registered or nested.** 0.5 was fixed before the curve
  was computed; the nested variant chooses the weight per extractant using only
  the other 161, so no row influences the weight it is scored under.
- **Negative controls fire.** `pi_topoonly` (−1.74) and `snn_allatom` (−0.41)
  come back "worse" with intervals excluding zero — the tests can lose.
- **Colour is computed, not eyeballed.** `tests/test_palette.py` ports the CVD
  validator (Machado-Oliveira-Fernandes, OKLab ΔE) and rejected the house
  palette's orange/green pair, which collapses under protanopia at ΔE = 3.2.

---

## 1. Data

| item | value |
|---|---|
| source table | `data/processed/final_ml_dataset_3d.parquet` (read-only) |
| rows | 5992 unique measurements |
| target | `log_D` (mean 0.30, sd 1.67, min −12.5, max 4.2) |
| extractants | 190 unique canonical SMILES / 228 trade names |
| metals | 14 lanthanides, La–Lu (no Pm); Eu is 26 % of rows |
| geometries on disk | 1526 extended-XYZ files (`data/geometries/`, `data/processed/geometries_*`) |
| geometry key | `Z \| ligand_SMILES \| inner-sphere anion` — 1256 distinct complexes |

**Coverage.** The shipped `feat3d__*` columns only cover `qc_class == OK`
geometries: 4746 of 5992 rows (79 %). The extractor here resolves geometries by
file name against the local tree and computes descriptors for BORDERLINE and
FAIL_LONG_BOND structures as well, with the QC class carried as an explicit
feature (`qc` block). That raises 3D coverage to **5946 / 5992 rows (99.2 %)**
and 187 / 190 extractants, without pretending a long-bond artefact is a real
bond. Both row policies are evaluated (`--row-filters has3d,ok_only`).

---

## 2. Evaluation protocol

`automl/evaluation.py`.

Split: **grouped K-fold on `extractant_group`** (the canonical extractant
SMILES), 5 folds × 2–3 repeats with a seeded group permutation. An extractant is
never in train and test simultaneously. Every number reported is computed from
pooled out-of-fold predictions.

A single R² hides the thing that matters, so it is decomposed:

| metric | question it answers |
|---|---|
| `r2_overall` | overall fit on held-out extractants |
| `r2_between` | R² of the per-extractant mean (size-weighted): *can the model rank whole extractants?* |
| `r2_within` | R² after removing each extractant's own mean: *can it reproduce the spread inside one extractant?* |
| `r2_within_composition` | same, but also removing the condition set — pure lanthanide-series selectivity |
| `sel_spearman_mean` | Spearman of the predicted La→Lu order inside a fixed extractant + fixed conditions |
| `sel_logSF_r2` | R² of pairwise log separation factors log D(A) − log D(B) |
| `sel_sign_accuracy` | fraction of metal pairs whose separation direction is called correctly |

The `between`/`within` split is the crux: the 2D baseline is good at ranking
extractants and weak inside one, and it is the *inside* part that a separation
process buys.

---

## 3. 3D descriptor blocks

`automl/geom3d_features.py` — 293 descriptors per geometry, computed from atoms,
coordinates, xTB partial charges, xTB forces and the energies stored in the
extxyz comment line. 1235 geometries, 0 failures, ~1 s each.

| block | content | rationale |
|---|---|---|
| **G1 `first_shell`** | realised M–L distances within 3.10 Å, observed CN, donor element counts, donor electronegativity/hardness, inverse-cube and inverse-sixth distance sums, shell gap | the 2D graph lists donors that *could* bind; the optimised structure says which ones *do* |
| **G2 `contraction`** | the same shell after subtracting the Shannon ionic radius (`d − r_ion`, `d/r_ion`), per donor element | removes the trivial lanthanide-contraction trend, leaves the ligand-specific cavity fit |
| **G3 `polyhedron`** | continuous shape measures vs ideal CN-6…10 polyhedra (SAPR, TDD, TCTPR, CSAPR, muffin, …), donor angular statistics, shell anisotropy, convex-hull volume/area/sphericity | coordination geometry distortion |
| **G4 `steric`** | %V_bur at 3.5/5.0/7.0 Å, solid-angle shadowing, radial atom counts, per-shell carbon fraction, metal SASA | how completely the ligand wraps the cation |
| **G5 `electronic`** | metal partial charge, ligand→metal charge transfer, donor charge statistics, point-charge M–L Coulomb sum, dipole magnitude and orientation, energy per atom, **residual force norms** | GFN2-xTB electronic structure; forces at the optimum are a strain proxy |
| **G6 `rdf`** | element-resolved metal-centred radial distribution functions, 1.8–8.0 Å, Gaussian-smeared (32 bins × C/N/O/H) | smooth rotation-invariant fingerprint that transfers across coordination numbers |
| **G7 `global_shape`** | radius of gyration, principal moments, asphericity, NPR, convex hull, packing density, SASA split into apolar/polar and charge-weighted | log D is a *partition* coefficient — the exterior matters |
| **G8 `chelate`** | donor–donor graph distances → chelate ring sizes, bite angles, bound-fragment count, max denticity | chelate topology as realised in 3D |
| **G9 `topology`** | persistent-homology summaries (H0/H1 total persistence, entropy, max lifetime) of the heavy-atom cloud and of the 6 Å metal neighbourhood | tabular-friendly counterpart of the shipped persistence images |

Derived in `automl/dataset.py`:

| block | content |
|---|---|
| **G10 `rel`** | every scalar re-expressed relative to its ligand+anion family (Δ from family mean, z-score, rank across the series) |
| **G11 `pi`** | the shipped 20×20 GFN2-xTB persistence images, flattened |
| **G12 `smooth`** | per-family linear fit of each descriptor against the ionic radius → the **fitted** value (denoised size response) |
| **G13 `slope`** | the family slope — *how strongly this cavity reacts to cation size*; a ligand-level selectivity descriptor, identical for all metals in the family |
| **G14 `fmean`** | the family mean — a purely ligand-shaped 3D descriptor with the metal dependence integrated out |
| **G15 `cnfree`** | every descriptor with the coordination-number main effect regressed out, plus explicitly CN-invariant forms (per-donor hull volume, per-donor %V_bur, bond-valence sum) |
| **`g_core`** | 22-column curated block chosen by measured permutation importance (see §5) |
| **`qc`** | geometry QC class one-hot + `geometry_ok` |

G12–G15 exist because of two *measured* failures of the raw blocks — the
coordination-number staircase and single-conformer noise — both quantified in
§5 and in `reports/FINDINGS.md` §4. `c`-suffixed variants (`g12c`, `g13c`,
`g14c`, `g15c`) restrict the same transform to the physically interpretable
scalars, leaving out the 128 RDF bins so a small model can use them without
dilution.

---

## 4. Architectures

`automl/advanced.py`. A flat regressor spends its squared-error budget on the
between component, which carries most of the variance. Three alternatives
attack the within component directly.

* **`twostage`** — stage 1 predicts the extractant-level mean from ligand-level
  features (one row per extractant); stage 2 predicts `y − mean_extractant(y)`.
* **`anchored`** — level from the *flat* model, shape from the residual model:
  `ŷ = mean_block(flat) + shape_weight · centred_residual + (1 − shape_weight) · centred_flat`.
  This keeps the flat model's strong between skill and replaces only what it is
  bad at. `level ∈ {extractant, composition}`.
* **`pairwise`** — Δ-learning. Inside a *strict* composition block (same
  extractant, same acid, same diluent, same temperature, same concentrations)
  only the lanthanide changes, so all ligand and condition columns cancel
  exactly and the model learns separation factors directly. Values are
  reconstructed as `anchor + mean_j Δ̂(i,j)`.

---

## 5. What the sweeps found

The narrative version, with every number and its uncertainty, is
[`reports/FINDINGS.md`](reports/FINDINGS.md); the auto-generated tables are
[`reports/tables.md`](reports/tables.md). In brief:

1. **The learner beats every feature.** CatBoost with inverse-extractant
   weights gains +0.037 R² over LightGBM on identical 2D inputs and folds
   (paired bootstrap, P(better) = 0.99) — larger than any 3D block effect
   measured anywhere in the study.
2. **No 3D block beats that 2D model.** Paired against CatBoost + `group_inv`,
   every 3D configuration is neutral or worse; several significantly worse. The
   strongest 3D signal, block G1 (realised first shell), reaches ΔR² = +0.011 at
   P(better) = 0.78 in a separate CatBoost ablation — suggestive, interval
   crossing zero.
3. **Bulk concatenation is actively bad.** All 2263 3D columns at once leaves
   R² flat and halves the selectivity metrics.
4. **Where the 3D information does live** is the *realised donor set*: grouped-CV
   permutation importance ranks `g1__first_shell__donor_en_mean` (0.037) and
   `donor_hard_frac` (0.027) far above everything else, then the xTB charges
   `q_metal` (0.012) and `q_transfer` (0.009). The CatBoost ablation
   independently picks the same block. Which donors *actually* bind, and how
   hard they are, is not readable from the SMILES.
5. **Two measured artefacts in the generated geometries.** (a) `cn_for_Z`
   hard-codes CN 9 for La–Gd and CN 8 for Tb–Lu, injecting a Gd→Tb
   discontinuity 5–17× the typical adjacent-metal step into every geometric
   descriptor, where the measured log D has no step at all; the xTB charge is
   the one quantity free of it. (b) Only 20–37 % of a descriptor's within-family
   variation tracks the ionic radius — the rest is single-conformer scatter,
   against a 0.013 Å radius step between neighbouring lanthanides.
   Restricting to a single-CN subset, where artefact (a) cannot exist, makes
   the 3D blocks *worse* rather than better — so (b), conformer sampling, is
   the higher-priority fix.
6. **Do not use SMOTE/SMOGN-style target weighting.** `target_lds` is the worst
   scheme tested (R² 0.336 with the anchored model): the rare-target region is
   one extractant that a held-out fold cannot learn.

### Reading the metrics
`r2_between` is largely a ligand-identity question and is where the baseline is
already strong (~0.75). `r2_within` and the `sel_*` family are the separation
problem and are where the headroom is. A pooled R² of 0.53 coexists with a
median per-extractant R² of +0.03, with 45 % of held-out extractants predicted
worse than their own mean; only the decomposition shows that.

---

## 6. How to run

```bash
module load anaconda/Python-ML-2025a
export PYTHONPATH=/home/gridsan/bmironov/lanthanidestrain

# Stage A: 3D descriptors (SLURM array, ~2 min wall)
sbatch automl/slurm/extract_3d.sh

# cache the joined feature matrix
python3 -m automl.matrix_cache

# Stage B: sweeps
sbatch automl/slurm/sweep_ablation.sh     # feature-block ablation
sbatch automl/slurm/sweep_arch.sh         # architectures
sbatch automl/slurm/select.sh             # greedy block search + permutation importance
sbatch automl/slurm/sweep_models.sh       # model families x sample weighting

# Stage C: HPO (distributed Optuna, shared SQLite study)
sbatch automl/slurm/sweep_optuna.sh

# Stage D: stack + report
python3 -m automl.ensemble
python3 -m automl.analyze
```

`automl/slurm/queue_driver.sh` submits the later stages automatically as the
SuperCloud caps (40 submitted jobs, 2 concurrent nodes on `xeon-p8`) allow.

## 7. File map

| file | role |
|---|---|
| `geom3d_features.py` | 3D descriptor extraction from extended XYZ |
| `dataset.py` | feature matrix assembly, block definitions, derived blocks, presets |
| `matrix_cache.py` | build-once/load-everywhere cached matrix |
| `evaluation.py` | grouped CV splitter + decomposed metrics |
| `models.py` | model zoo, preprocessing, sample-weighting schemes, Optuna spaces |
| `advanced.py` | two-stage / anchored-residual / pairwise-Δ architectures |
| `experiment.py` | one CV experiment, JSONL result recording |
| `run_sweep.py` | ablation / models / arch / optuna sweep drivers |
| `select_features.py` | greedy block search + grouped-CV permutation importance |
| `champion.py` | shortlist re-run at 5 repeats with cluster-bootstrap CIs |
| `compare.py` | **paired** bootstrap between configurations (the decisive test) |
| `ensemble.py` | OOF stacking (NNLS, inverse-variance) + uncertainty calibration |
| `analyze.py` | results → ranked tables → markdown report |
| `make_tables.py` | regenerates every table quoted in `reports/FINDINGS.md` |
| `figures.py` | report figures |
| `split_variability.py` | how far a *single* leave-extractants-out split can move |
| `refresh_reports.sh` | re-runs analyze + tables + compare + ensemble + figures |
| `tests/` | analytic correctness tests for the geometric descriptors |

### Reproducing the reports

```bash
module load anaconda/Python-ML-2025a
export PYTHONPATH=/home/gridsan/bmironov/lanthanidestrain
bash automl/refresh_reports.sh     # login-node safe: reads results, fits nothing
python3 -m pytest automl/tests -q  # descriptor correctness (CShM, SASA, %V_bur)
```

The descriptor tests check the geometry maths against cases with an analytic
answer: SASA of an isolated atom equals `4*pi*(r_vdw+r_probe)^2` exactly, the
continuous shape measure is 0 for an ideal polyhedron and monotone under
distortion, and a fully enclosed metal gives %V_bur = 100.

Extra Python packages installed into `~/.local` for this work: `lightgbm`,
`catboost`, `optuna`, `ase`, `rdkit`, `ripser`, `persim`, `shap`, `tabulate`,
`gudhi`, `pytest`. Nothing in the module environment was changed.
