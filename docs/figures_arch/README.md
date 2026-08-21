# Figures: the current best 3D system

Three figures, all built from out-of-fold predictions under
leave-extractants-out cross-validation. Regenerate with:

```bash
module load anaconda/Python-ML-2025a
PYTHONPATH=$PWD python3 docs/figures_arch/make_a1.py   # needs fig_data.json
PYTHONPATH=$PWD python3 docs/figures_arch/make_a2.py
PYTHONPATH=$PWD python3 docs/figures_arch/make_a3.py   # reads the parquets
```

| figure | what it shows |
|---|---|
| `a1_architecture.png` | The system: one model for the block level, a second for the within-block shape, and the 3D encoder mixed into the shape at weight 0.35. Panels B/C run the arithmetic on one real 14-lanthanide block; panel D shows why the split matters — 87 % of log D variance is level, and the scored metric reads none of it. |
| `a2_evidence.png` | How the weight is chosen (interior optimum, stable across folds), why the distance encoder takes it and the simplicial one does not (the two are 0.963-correlated), the gain across five evaluation sets including the held-out 444 pairs, seed-half robustness, and the representations that fail in the same slot. |
| `a3_where.png` | Where the 3D shape actually changes predictions: parity plot, per-position and per-extractant error change, and the honest size of the effect — fewer than half the extractants improve, which is what a +0.008 average looks like. |

`fig_data.json` holds the numbers behind a1/a2 (built once from the
out-of-fold parquets); a3 recomputes its own pair-level table.

Sources: `automl/artifacts/anchored_champ/` (level/shape models),
`automl/artifacts/topo_c15` and `topo_c17` (encoders),
`automl/reports/anchored_3d*.json`, `topo_shape.json`,
`anchored_champion.csv`.
