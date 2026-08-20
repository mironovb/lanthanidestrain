# Collaborator update (2026-08-18): integrity, reproduction, and our best system on the expanded dataset

**Bogdan Mironov · 20 August 2026** · inputs: `collaborator_update/`
(dataset.parquet, 1,155 geometries, `metrics_reproduction_20260818.md`) ·
code: `automl/collab_repro.py`, `automl/collab_ours.py`,
`automl/topo/build_vr_collab.py` · outputs: `automl/reports/collab_repro/`,
`automl/reports/collab_ours/`.

## 1. Integrity and quality of the drop — PASS

- `dataset.parquet` SHA-256 = `fefbe…f5dd`, **byte-identical to the hash his
  document pins** (his gen3 runner enforces the same hash).
- Same 5,992 rows × 2,261 columns as our `final_ml_dataset_3d.parquet`, same
  `safe_exp_id` set; **`log_D` and all 64 `cond__*` columns are identical** —
  only 99 geometry-derived columns differ.
- What he changed: **re-optimised/repaired the geometry side.**
  `geometry_ok` 4,746 → **5,479** (+733 rows; every former
  BORDERLINE/FAIL class resolved to OK), 520 rows carry new `build_id`s,
  1,155 structures shipped. 179 of his OK rows (63 builds) have no structure
  in the drop or on this machine — flagged back to him; they are excluded
  from encoder arms only.
- His §2 cohort audit reproduces **exactly** from his parquet: 129
  quarantined, −2,851 incomplete-condition rows → 3,012; 2,520 cells,
  198 replicated; 8,195 candidate pairs, −688 geometry, −802 replicate.
  Our cohort: **6,705 pairs / 34 extractants / 91 pair labels** vs his 6,699
  (his exact "compact-invariant" 3D-completeness list isn't in the drop; our
  proxy keeps 6 extra pairs).

## 2. Reproduction of his headline metrics — PASS (within his §8 tolerance)

From-scratch reimplementation of his cohort, model and metrics (his repo is
not on this machine): antisymmetric ExtraTrees, group-balanced weights,
StratifiedGroupKFold with his split seeds, inner 3-fold grid, transitive
projection, PAIRMEAN baseline, extractant-resampled bootstrap.

| arm | his (5-seed mean ± sd) | ours (5-seed mean ± sd) |
|---|---|---|
| A2 macro MAE | 0.3192 ± 0.0078 | **0.3238 ± 0.0057** |
| A2+TP macro MAE | 0.3175 | **0.3205** |
| PAIRMEAN macro MAE | 0.4482 | **0.4500** |
| A2 pooled MAE / R² | 0.4189 / 0.323 | 0.4128 / 0.331 |
| A2 sign accuracy | 0.847 | 0.850 |
| TP bootstrap Δ (CI) | +0.0018 [+0.00003, +0.0036] | +0.0033 [+0.0014, +0.0053] |

His signature **pooled-vs-macro inversion** for PAIRMEAN (better pooled,
catastrophically worse macro, driven by TODGA's 43.6 % row share) reproduces
exactly. Residual differences are the ones his §8 predicts: fold-assignment
ordering across sklearn versions, forest thread nondeterminism, plus our
6-extra-pair cohort. Per-seed macro values interleave with his.

## 3. Our August-campaign system under HIS protocol — statistical tie, stricter CV

Our per-row out-of-fold predictions (leave-ONE-extractant-out over 187
extractants — strictly harder than his 5-fold grouping) mapped onto his cells
and scored with his metrics on the covered subset (~97 % of pairs; his arms
recomputed on the same subset):

| arm | macro MAE | pooled R² | adjacent-pair MAE |
|---|---|---|---|
| his A2 (repro) | 0.3096 | 0.369 | 0.1303 |
| his A2+TP | 0.3063 | 0.373 | 0.1204 |
| **ours: anchored** | 0.3146 | 0.354 | 0.1219 |
| **ours: anchored-3D blend** | 0.3199 | 0.399 | **0.1188** |

Every his-vs-ours bootstrap CI spans zero (overall macro AND
extractant-macro on adjacent pairs) — **two fully independent pipelines
(his: antisymmetric pair-feature forests; ours: anchored row-level CatBoost +
encoder shape) are statistically indistinguishable on his target.** The
pooled adjacent-MAE edge of our blend (0.119 vs 0.130) is pair-count
weighted, not a claimable win. Convergence of two independent stacks on
macro ≈ 0.31–0.32 suggests both are near the extractable signal for this
cohort under extractant-held-out evaluation.

## 4. Our best system retrained on his expanded dataset

His `geometry_ok` defines a new expanded population for our metric:
**5,479 rows → 1,230 adjacent pairs (+325 over the legacy 905).** We built a
combined VR edge asset (1,330 complexes = our 1,235 + his 95 new builds),
added `--population collab` to the trainer (inertness re-proven:
max|Δoof| = 0.000e+00 vs the published c15_plw4 s201 parquet), and retrained
both halves of the anchored-3D system (4 seeds each, deterministic,
leave-extractants-out 5×3):

| system (collab population) | adjacent R² |
|---|---|
| flat champion CatBoost (ens4) | ~+0.16 |
| anchored q60/q60 (ens4) | **+0.2682** |
| anchored + 3D shape blend (w = 0.35, all 1,220 covered pairs) | **+0.2878** |
| — legacy-905 slice (875 covered) | +0.3253 |
| — his-new-pairs slice (345) | +0.1666 |

Both August-campaign mechanisms **replicate on his population**: the anchored
architecture effect (+0.11 over flat) and the 3D shape contrast, which is
positive on every slice — all +0.0049, legacy +0.0047, **his new pairs
+0.0055** (same sign as the pre-declared fresh-444 confirmation, smaller
magnitude; 4-seed ensembles on both halves, exploratory on this population).
His new pairs score lower in absolute terms (+0.17) for the same reason our
fresh-444 did: sparser extractants and previously-QC-failed chemistry.

## 5. Messages back to the collaborator

1. Your headline metrics reproduce independently, from the spec alone.
2. Your A2 and our anchored system tie under your metric; ours carries a
   stricter CV. Worth combining: your TP post-processing on our arms, and
   our anchored decomposition with your antisymmetric pair features.
3. Your geometry repair is valuable to us: +325 adjacent pairs, and our best
   system holds +0.29 on the expanded population with the 3D contribution
   intact.
4. 179 of your geometry_ok rows (63 builds) shipped without structures —
   please include them next time.
5. Your `condition_id` cohort (complete-conditions only, 34 extractants)
   discards 2,851 rows we model; consider a NaN-tolerant variant — our
   models handle missing conditions natively.
