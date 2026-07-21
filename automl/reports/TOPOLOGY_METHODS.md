# Rigorous geometries + topological learning — methods and verification

Companion to `FINDINGS.md`. This file covers what was **built and verified**;
results live in `TOPOLOGY_RESULTS.md` once the runs complete. Everything here
is settled and will not change with the remaining arms.

---

## 0. Dataset, stated honestly

The draft abstract says *1,202 measurements / 109 extractants / 14 lanthanides*.
**No slice of this repo matches those numbers.** The geometry-backed set is:

| quantity | value |
|---|---|
| rows (measurements) | **4,746** |
| extractants (CV groups) | **162** |
| metals | **14** |
| distinct complexes behind those rows | **953** |
| Vietoris–Rips complexes shipped | 956 |
| persistence images shipped | 953 |

Any text produced from this work uses the actual counts.

The gap between 4,746 rows and 953 structures is the single most important
number for interpreting everything below: rows sharing a complex differ only in
experimental conditions, so the **effective structural sample size is ~953, not
4,746**. Joining on `geometry_feature_build_id` reaches all 4,746 rows;
`build_id` reaches only 4,402 and would silently drop 344.

---

## 1. Why the geometries were re-optimised

`force_max` in the shipped geometries is hard-capped at 0.19999 eV/Å with 94 %
between 0.15 and 0.20. Every optimisation stopped **on** an `fmax = 0.2 eV/Å`
criterion — 4× looser than the ASE default, ~10× looser than tight practice.
These are not relaxed structures; they are structures that ran out of criterion.
Part of what the prior study attributed to *conformer* scatter is
**optimisation** scatter.

Re-optimisation: GFN2-xTB, ANCopt `tight`, ALPB implicit solvent in **water and
n-octanol** (log D is a partition coefficient; a single phase is the wrong
reference state, and the water−octanol difference becomes a descriptor in its
own right).

### Charge is recovered per structure, never assumed

An early version assumed a uniform +3 charge based on a 4-structure sample.
Quantifying across all 1,523 files:

| inferred charge | structures |
|---|---|
| +3 | 1,214 |
| +2 | 241 |
| +1 | 68 |

A blanket +3 would have silently mis-specified **309 structures (20 %)**.
`infer_charge()` now recovers the charge from the rounded sum of stored xTB
Mulliken populations and **refuses to guess** when they are missing. A negative
control confirms this matters: forcing +3 on a true +1 structure changes the
energy by **+1.414 Eh**.

The backend reproduces the stored GFN2-xTB energies to ~6e-7 Eh across all three
charge states. Convergence is re-measured with an independent `--grad` single
point — nothing is called converged on the optimiser's own banner.

---

## 2. Two invalid prior tests, corrected

**Persistence images were never fairly tested.** The manifest ships them with an
explicit contract, `"readout": "CNN_or_ViT; do_not_flatten_into_tabular_MLP"`.
The earlier `g11` block flattened the 20×20 image into 279 tabular columns and
fed a gradient-boosted tree. That test was invalid by the asset's own terms:
flattening destroys the birth–death adjacency that makes a persistence image an
image. `automl/topo/pi_cnn.py` gives them the readout they were built for.

**The simplicial asset was never used at all.** `vietoris_rips_inputs.npz` ships
956 complexes as nodes + edges + triangles with filtration values and role
`simplicial_model_input`. Zero prior experiments touched it. It was also
missing the triangle→edge boundary map, which is now constructed and verified
(`∂₁∘∂₂ = 0` over GF(2)).

---

## 3. Bugs found and fixed (all would have produced wrong numbers silently)

These are recorded because each one *ran* and *reported a result* before being
caught — none of them crashed in a way that would have drawn attention.

### 3.1 MPSN permutation-invariance violation

The layer aggregated node←edge messages as
`scatter_mean(em, i) + scatter_mean(em, j)`. An edge is unordered, so whether a
node sits in row `i` or row `j` of `edge_index` is arbitrary; adding two
separately-normalised means makes the output depend on that arbitrary split.
The same flaw applied to the triangle→edge message, where assignment to slot
`e0/e1/e2` is equally arbitrary.

Confirmed as a real violation rather than rounding: the two orderings differ by
**2.628e-04 in float32 and 2.628e-04 in float64** — identical to 10 significant
figures, so it is not accumulation error. Fixed by aggregating once over
concatenated indices, which normalises by true degree.

### 3.2 NaN poisoning from the shipped asset

**292 of 219,583 partial charges** in `vietoris_rips_inputs.npz` are non-finite,
concentrated in 3 of 956 complexes (166 / 89 / 37 atoms). A single NaN node
feature does not stay local: message passing spreads it across the complex and
pooling turns the whole embedding into NaN, so every row referencing those
complexes would predict NaN. Fixed by imputing to zero **plus an explicit
missingness indicator**, so the model can distinguish an imputed charge from a
genuinely neutral atom. Pretraining additionally excludes imputed charges from
the reconstruction target — otherwise it learns to reproduce the imputation.

### 3.3 A smoke test that measured the wrong thing, three ways

The capacity check reported R² = 0.796 and looked like an architecture failure.
It was not:

1. Regularisation was left on. A capacity test asks whether the model *can*
   fit; dropout and weight decay make a model that can fit look like one that
   cannot.
2. The subset `df.head(60)` spanned only **21 distinct geometries**. Rows
   sharing a geometry differ only in conditions, so a topology-only model's
   ceiling there was R² = 0.884 — it scored 0.816, i.e. 92 % of the achievable
   maximum, against an unreachable 0.95 target.
3. The real cause: `run_fold` carved a 15 % inner-validation split out of the
   60 rows and then **scored all 60**, so ~15 % of scored rows were never
   trained on. This pins train-on-train R² near 0.89 regardless of epochs —
   exactly the flat curve observed (400 epochs → 0.897, 1500 → 0.889).

After all three fixes: **hybrid R² = 1.0000, topology-only R² = 0.9999.**

### 3.4 Geometry-source contamination hazard (closed before it bit)

`resolve_jobs` globs `*.xyz` repo-wide, first-match-wins over an unordered
`rglob`, and the re-optimised files deliberately reuse the original basenames.
Any future extraction could have silently featurised a **mixture** of loose and
tight geometries — descriptors corresponding to no single geometry set. An
explicit `--geom-root` was added and the default now excludes `geom_reopt`
entirely, so prior behaviour is exactly preserved (verified: 1,235 files
resolved, 0 contaminated).

### 3.5 An audit that failed silently

The first version of the chemistry audit hit an `ImportError` that was swallowed
by a `try/except` and printed an **empty** section — indistinguishable from a
pass. The report now states loudly when the audit did not run.

### 3.6 Infrastructure

- `ProcessPoolExecutor` could not pickle a local closure; the work is an
  external xtb subprocess, so `ThreadPoolExecutor` is both correct and simpler
  (the GIL is released during the subprocess).
- xtb runs single-threaded, so a sequential shard driver used 1 of 48 cores:
  0 structures in 18 min, versus 12 in under 5 min with a worker pool.
- Pretraining built the encoder with `tabular_dim=0`, so its head (768) could
  never match the fold model's (1514). `strict=False` forgives a missing key,
  **not** a shape mismatch; encoder weights are now transferred explicitly.

---

## 4. Correctness tests

23 tests, all passing (`automl/tests/test_simplicial.py`, `test_pi_cnn.py`):

- boundary structure: triangle edges are exactly the three vertex pairs;
  `∂₁∘∂₂ = 0` mod 2; filtration thresholding yields a genuine subcomplex;
  `heavy_only` drops exactly hydrogens; exactly one metal per complex
- batching: a complex alone equals the same complex in a group; offsets stay in
  range; no leakage across the batch
- invariances: rigid rotation/translation, node permutation
- the persistence CNN *uses* pixel layout (a shuffled image must change the
  prediction — the test the flattened version fails by construction)
- capacity sanity: the encoder stays small enough for 953 structures

Invariance here is **structural**, not learned: no raw coordinate enters the
network. Node inputs are the xTB partial charge, the missingness flag, metal and
donor flags, and distance to the metal; edge and triangle inputs are filtration
radii. All are invariant to rotation, translation and reflection.

---

## 5. Evaluation protocol (unchanged from the prior study, deliberately)

Folds come from `automl.evaluation.grouped_folds` (leave-extractants-out,
5 splits × 3 repeats, seed 42), metrics from `full_metrics`, and every arm
writes out-of-fold predictions in the tabular-sweep schema so
`automl.compare.paired_bootstrap` can pair any two arms on identical rows.
The bootstrap resamples **whole extractants** — rows within an extractant are
not independent, and a row-level interval would call noise significant.

Making the methods work means giving them the best achievable form, not moving
the yardstick.

### The adjacent-pair metric

Added to `evaluation.py` for the abstract's specific claim: separation between
lanthanides with `|Δ lanthanide_index| == 1`, the hardest case. Baseline values,
with a shuffled control to confirm the metric has teeth:

| | all pairs | adjacent pairs |
|---|---|---|
| logSF R² | 0.458 | **0.083** |
| sign accuracy | — | 0.651 |
| n pairs | — | 1,349 |
| true spread (sd) | — | 0.274 log units |
| shuffled control | — | R² = −28.5, sign acc 0.486 |

The 0.083 is the headroom the abstract's claim targets.

### Two baselines, both reported

| baseline | why it is here |
|---|---|
| FCNN on ECFP + RDKit | exactly what the abstract benchmarks against |
| CatBoost + `group_inv` | the strongest model in the prior study (R² ≈ 0.528) |

Reporting only the weak one would be a strawman; reporting only the strong one
would not test the abstract's actual claim. A win over one and not the other is
stated as exactly that.

Row sets were verified identical: the SNN arms and the `ok_only` baselines both
cover exactly 4,746 rows.

---

## 6. Reproducibility note

The first CV array was **discarded, not used**. Its tasks picked up source edits
mid-flight, so results came from a mixture of pre- and post-fix code. Those
artifacts are quarantined under
`automl/artifacts/topo_runs_invalid_prefix_bugs/` with a README recording
exactly which bugs were live, rather than deleted or silently reused. The rerun
followed a 5-stage GPU verification: 23 tests, capacity smoke R² = 1.0000, the
three NaN complexes returning finite predictions, pretrain→fold weight transfer,
and topology-only end-to-end.

---

## 7. A harness caveat found while running the baselines

`baseline::mlp::none` and `baseline::mlp::group_inv` returned **byte-identical**
out-of-fold predictions (max |difference| = 0.00e+00 over all 4,746 rows).

Cause: `automl/experiment.py` passes sample weights inside a
`try: model.fit(..., **fit_kwargs) except TypeError: model.fit(X, y)`. sklearn's
`MLPRegressor.fit()` takes no `sample_weight`, so the weights are **silently
dropped** while the run is still recorded with `weight_scheme=group_inv`.

This is pre-existing behaviour, not introduced here, and it does not affect the
conclusions below — the FCNN baseline is the same model either way. But it
means any "MLP + weight scheme" row, in this study or the prior one, is an
unweighted run wearing a weighted label, and should not be read as evidence
that weighting does nothing for neural models. Only the CatBoost weight
comparison is real (0.4987 unweighted vs 0.4917 weighted).

---

## 8. Stage 1 completion and a documented limitation

Final re-optimisation coverage (GFN2-xTB, ANCopt `tight`, ALPB):

| | water | octanol |
|---|---|---|
| succeeded | **1,232 / 1,235** (99.8 %) | **1,146 / 1,235** (92.8 %) |
| median fmax, input → output | 0.1853 → **0.00224** | 0.1852 → **0.00225** |
| meets target | 1,231 / 1,232 | 1,144 / 1,144 |
| formula preserved | **100 %** | **100 %** |
| median RMSD from input | 1.87 Å | 1.98 Å |
| CN changed (excluded downstream) | 8.7 % | 8.5 % |

Residual forces fall ~83×, and no structure in either phase gains or loses an
atom. The ~8.6 % CN-change rate reproducing independently across both solvents
is corroboration that Stage 2's negative answer is structural rather than an
artefact of one phase.

### The 86 octanol failures: diagnosed, not worked around

86 octanol optimisations abort. The cause was misdiagnosed twice before being
run down properly, and the sequence is worth recording:

1. **First hypothesis — wall-time.** The failures skew large (median 281 atoms
   vs 214) and all 86 succeed in *water* with identical charge and input
   geometry, so a resource limit looked obvious. A retry at **3× the timeout**
   (4.5 h) with half the workers recovered **2** of 86. The hypothesis was
   wrong.
2. **Actual cause — SCF non-convergence during optimisation.** The xtb log
   ends `scf: Self consistent charge iterator did not converge` →
   `optimizer_relax: SCF not converged, aborting`. `reason` is now recorded as
   `scf_not_converged` rather than the opaque `xtb_rc_128`.
3. **Not fixable by the standard remedy.** Electronic-temperature smearing —
   the usual xtb answer for hard SCF — was tested at 1,000 / 5,000 / 10,000 K
   and at a looser `normal` optimisation level. All still fail. `--etemp`
   support remains in `xtb_backend.optimize()` and records the value used, so
   any structure needing it is never silently mixed with the rest.
4. **Where the instability actually lives.** A **single point converges fine**
   at the input geometry in octanol *and* in water. Only the optimisation path
   fails: the structure relaxes into a region where ALPB-octanol SCF cannot
   converge. The failures concentrate on Eu (26 of 86), whose near-degenerate
   frontier orbitals make this the expected hard case.

**Consequence.** Octanol coverage is 92.8 %. The water−octanol difference
descriptors would be computed on the 1,146 complexes that have both phases.
**No result in `TOPOLOGY_RESULTS.md` depends on this** — every reported number
uses the shipped geometries, and the tight-geometry test used water only.
