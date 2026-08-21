# ACS oral presentation — plan

**Format.** 20 min talk + 5 min questions. Assertion–evidence slides: the
title states the finding as a sentence, the body is one visual. Depth lives
in the speaker notes, not on the slide. 16:9, ~24 slides + backup.

**Audience.** Separations chemists, computational chemists, ML-for-chemistry.
Assume they know solvent extraction and DFT but not our pipeline.

## Narrative

| # | slide | evidence |
|---|---|---|
| 1 | Title | — |
| 2 | Adjacent lanthanides are the industrially hard case | g1_pairs |
| 3 | Data: 4,746 measurements, 162 extractants, 14 lanthanides | g2_cv (top) |
| 4 | Evaluation: extractants held out; we score separations, not levels | g2_cv |
| 5 | Roadmap | — |
| 6 | **Part I** — do the structures encode the contraction? | — |
| 7 | We measure a per-ligand contraction slope at fixed ligand | g5_slope_demo |
| 8 | GFN2 under-responds 2.5×; g-xTB is within 8 % of experiment | f5_contraction |
| 9 | The cause is in GFN2's parameter file: linear in Z | g4_params |
| 10 | Consequence: eight 3D encoders are interchangeable | T5 (panel C) |
| 11 | **Part II** — the modelling problem | — |
| 12 | 87 % of the signal is level; the metric reads only the other 13 % | T1_scoring |
| 13 | So predict level and shape with separate models | T2_architecture |
| 14 | On one real block | T3_block |
| 15 | This beats every stack we ever fitted | T4_scoreboard |
| 16 | 3D enters only through the shape, at a fitted weight | T5_weight |
| 17 | The gain holds on pairs no decision ever touched | T6_confirm |
| 18 | It is small and uneven — this is what +0.008 looks like | a3_where |
| 19 | **Part III** — what does not work | — |
| 20 | Topology, persistence, conditions, energies: four negatives | T7_negatives |
| 21 | The persistence collapse localised by a block-mean control | T7 (annotated) |
| 22 | **Part IV** — independent check | — |
| 23 | A collaborator's model reproduces; his and ours tie | T8_collab |
| 24 | Limits: no ceiling is identifiable from this data | — |
| 25 | Summary + next | — |

**Backup:** metric definition; label-side series shape; per-extractant error;
g-xTB energy probe; expanded-population numbers; the withdrawn ceiling.

## New figures (12.4 × 5.6 in, 300 dpi, projection-sized fonts)

T1_scoring, T2_architecture, T3_block, T4_scoreboard, T5_weight,
T6_confirm, T7_negatives, T8_collab — all from out-of-fold predictions
via docs/figures_arch/fig_data.json and the reports.

**Reused:** g1_pairs, g2_cv, g4_params, g5_slope_demo, f5_contraction,
a3_where.

## Rules
Every number traceable to an artefact; no withdrawn claims (the 0.53
ceiling is out); negatives shown with the same weight as positives.
