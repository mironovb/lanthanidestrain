# The August 2026 campaign: from +0.313 to a confirmed +0.33-class system

**Bogdan Mironov · 14-16 August 2026** · findings I14-I16 in
`SCIENTIFIC_FINDINGS.md` · every number below is out-of-fold under
leave-extractants-out CV on the canonical `automl/evaluation.py` pair
machinery · `control_guard --verify`: 324 pinned artefacts byte-identical.

## Headline

> **A single anchored model now beats every stack this project ever fitted,
> and the 3D encoder's contribution — confirmed on a never-touched
> population — flows through its shape channel.**
>
> Best system: `anchor + 0.65·shape_tabular + 0.35·shape_encoder` =
> **+0.326 adjacent-pair log SF R² on the 905 legacy pairs** (Pearson²
> +0.347), against +0.313 for the previous best stack.  On the frozen
> fresh-444 confirmation population the 3D-vs-tabular contrast is
> **+0.0156 (PASS of the single pre-declared look)** and is uniform across
> every population slice (~+0.015).

## The two mechanisms found

**1. The anchored decomposition (I14).**  Split the prediction into a
block-level anchor and a within-block shape, and train a *separate* model on
the block-centred target for the shape.  The 2026-06 architecture sweep had
this family and never scored it on the adjacent metric; a second
infrastructure gap (`models.make_model` silently drops `loss_function`) meant
it also never met the champion's quantile loss.  Combined:
flat champion +0.268 → anchored q60/q60 **+0.318** (8-seed ensemble;
two independent 4-seed ensembles +0.3188/+0.3113).  The architectural effect
is base-learner-general: anchored LightGBM at *untuned fast* parameters
reaches +0.304 (vs +0.157 flat CatBoost, +0.271 anchored CatBoost fast).

**2. The 3D shape channel (I15, CONFIRMED).**  With the level anchored away,
the distance encoder's block-centred predictions enter the shape mix at
weight ~0.35 (chosen nested, equal-extractant criterion) and add +0.008
legacy / **+0.0156 fresh**.  This is the first configuration in the project
where 3D information survives next to the strongest tabular model instead of
being absorbed by it: the flat-model routes all failed (I12: 3D tabular
columns collapse CatBoost; G1/G2: encoders interchangeable), because they let
3D perturb the level.  The anchor protects the level by construction.

## The confirmation discipline (replacing the spent look budgets)

- 444 adjacent pairs unlocked by relaxing `geometry_ok` were frozen
  (`fresh_eval.py`, commit 3d4fe50) before any new model ran; no selection
  ever touched them.
- The confirmation rule (`anchored_3d_confirm.py`: fixed w = 0.35, primary =
  sign of blend-vs-tabular contrast on the fresh 444) was committed while the
  C19 encoder retrain was still in the queue.
- Enabling infrastructure, each verified: expanded VR edge asset over the 279
  completed borderline builds (1,235 complexes, `build_vr_has3d.py`);
  `train.py --population` flag proven inert at **max|Δoof| = 0.000e+00**
  against the published c15_plw4 s201 parquet; C19 = 8 deterministic encoder
  seeds on the expanded 5,946-row population.

| population | blend | tabular-only | contrast |
|---|---|---|---|
| **fresh 444 (the look)** | +0.1207 | +0.1051 | **+0.0156 PASS** |
| legacy 905 | +0.3109 | +0.2964 | +0.0146 |
| all 1,349 | +0.2450 | +0.2301 | +0.0149 |

(The fresh population is intrinsically harder — borderline-QC rows, sparse
extractants; the pre-declared quantity is the contrast, not the absolute.)

## What was tried and killed, with the evidence

| candidate | result | source |
|---|---|---|
| stack recombination (profile arm, stratified weights, 17-arm pool, pair-model arms, anchored arms added or substituted) | all within noise of +0.3132 | `restack.csv` |
| all-pairs delta model (6,389 within-block pairs, 7× the old population) | +0.191 standalone (5× the 2026-07 attempt) but error-correlated with existing arms; zero stack marginal | `pair_model.csv` |
| within-block condition deltas (the largest measured correlate, \|r\| 0.30-0.36) | +0.004 over pair identity alone under LOEO — extractant-specific, does not transfer | `pair_model.csv` (identity ablations) |
| energy channel: relaxed-series g-xTB pilot spearman −0.124 (p=0.010), GFN2 null | **does not replicate** on the clean frozen-cage instrument at full scale: −0.064 (p=0.15), n=494 cells/71 extractants, matched GFN2 control | I16, `gxtb_cage_d2.csv` |
| frozen-cage GFN2 Δeint | position-dependent sign-flipping correlations (La-Ce +0.38, Gd-Tb −0.36) that cancel pooled; no model gain | pair-model energy runs |

## Standing scientific observations (label side)

- **Series shape (A2, robust):** a 12-value pair-identity lookup achieves
  LOEO R² +0.066; split-half r = 0.75; structure beyond the radius ramp at
  p = 2e-13; half-shell anomaly (Eu-Gd the only positive mean position,
  Gd-Tb the largest negative).  Already embodied in learned models (no stack
  marginal) — publishable as a data observation, not as a modelling lever.
- **Ceiling (A5):** no point ceiling is identifiable from this dataset
  (`CEILING_NOTE.md`); separations reproduce to ~0.16 on a 0.22 spread across
  independent condition sets.  The 0.53 quoted in README/deck is superseded.
- **Error anatomy (A4):** one DGA extractant carries 39% of the best system's
  squared error; the light half ~70%; 82% sits in the largest-|dy| quartile
  (dispersion ratio 0.47) — where any future gain must come from.

## Assets created

`fresh_eval.py` (+ frozen `fresh_pairs.json`), `series_shape.py`,
`adjacent_decomposition.py`, `restack.py`, `pair_model.py` (+ cached pair
tables), `anchored_champion.py` (+ 30 OOF parquets), `anchored_3d.py`,
`anchored_3d_confirm.py`, `build_vr_has3d.py` (+ 1,235-complex edge asset),
`energy_series.py`, `gxtb_cage_probe.py` (+ ~14k g-xTB single points with
gaps and Mulliken charges), full 853-cage GFN2 metal-swap surface
(`metal_probe.csv`), `CEILING_NOTE.md`, `arch_adj` sweep (54 cells scored on
the adjacent metric for the first time).

## Named next tests (not run; ordered by prior)

1. Anchored **LightGBM** with tuned parameters (+0.304 untuned at fast
   params suggests headroom over the CatBoost base).
2. Margin significance: the +0.326-vs-+0.313 system-level margin needs a
   designed split or paired-seed test before being quoted as a gain
   (the confirmed claims are the architecture contrast and the fresh-444
   3D contrast).
3. Yttrium training augmentation (C2; deferred — the conditions/pair-channel
   nulls lowered its prior).
4. snn-encoder shape channel (needs a triangulated expanded asset).
5. Update the Vogiatzis deck: ceiling slide per `CEILING_NOTE.md`; results
   table per I14/I15.
