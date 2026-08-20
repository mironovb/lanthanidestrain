# Do topological representations add anything? Four tests, all negative

**Bogdan Mironov · 20 August 2026** · code: `automl/topo/topo_shape.py`,
`automl/topo/anchored_champion.py` (resid-only feature blocks) · results:
`automl/reports/topo_shape.json`, `anchored_champion.csv`.

Context: the current best system predicts a per-block level with one
CatBoost and the within-block shape with a second, then mixes in the
distance-encoder's shape at weight 0.35 (adjacent-pair R2 +0.326 on the
905-pair legacy set, +0.288 on the 1,220-pair expanded set). All the 3D
contribution so far comes from the *distance* encoder — edges only. These
tests ask whether the topological representations (simplicial message
passing over 2-simplices; persistent-homology descriptors) add anything
beyond that. They do not.

Every number below is out-of-fold, leave-extractants-out, on the legacy
905-pair evaluation set; blend weights are fitted nested per held-out
extractant.

## 1. Simplicial encoder in the shape channel — no contribution

| shape source | adjacent R2 | Pearson2 | fitted weight |
|---|---|---|---|
| tabular only | +0.3182 | +0.3306 | — |
| + distance encoder (c15_plw4, 32 seeds) | **+0.3258** | +0.3472 | 0.35 |
| + simplicial encoder (c17_plw4, 32 seeds) | +0.3172 | +0.3296 | **0.01** |
| + simplicial encoder (c17_plw2, 32 seeds) | +0.3184 | +0.3330 | 0.06 |
| + both encoders (weights fitted jointly) | +0.3258 | +0.3472 | dist 0.35, **snn 0.00** |

Offered the same slot the distance encoder uses, the simplicial network is
assigned essentially zero weight, and when both are available the fit puts
nothing on it.

**Why: redundancy, not weakness.** Pair-level correlations between the
three shape predictions:

| | tabular | distance | simplicial |
|---|---|---|---|
| tabular | 1.000 | 0.743 | 0.749 |
| distance | 0.743 | 1.000 | **0.963** |
| simplicial | 0.749 | 0.963 | 1.000 |

The two encoders make nearly the same predictions. Both correlate with the
tabular model's residual — the only thing a blend can exploit — at a similar
level (distance 0.196, simplicial 0.176), and the distance encoder is
slightly better, so it takes the whole weight. This is the pair-level
version of the earlier finding that eight 3D encoders span an effective
rank of 1.05.

## 2. Triangles vs edges at matched seeds — no advantage for triangles

Two configuration-matched simplicial cells that differ only in
`--no-triangles`, restricted to their 8 shared seeds:

| cutoff | with triangles | without triangles | difference |
|---|---|---|---|
| 4.0 A, encoder alone | +0.2319 | +0.2518 | −0.0199 |
| 4.0 A, in the blend | +0.3180 | +0.3246 | −0.0066 |
| 3.5 A, encoder alone | +0.2316 | +0.2473 | −0.0157 |
| 3.5 A, in the blend | +0.3182 | +0.3188 | −0.0006 |

2-simplices do not beat edges alone; both differences favour the edge-only
model, though the blend differences are within seed noise. On this evidence
the planned work — building triangulated versions of the expanded assets and
retraining the simplicial network on them (a GPU job plus a trainer change) —
was not run.

## 3. Persistent-homology features in the shape model — collapse

The 22 persistence statistics (`g9__topology__*`: H0/H1 totals, maxima,
entropies, counts, birth/death means over heavy atoms and the metal
neighbourhood) and the 279 persistence-image pixels (`g11`), given **only**
to the residual/shape model (the level model is untouched), 4 seeds each:

| shape model features | adjacent R2 | vs reference |
|---|---|---|
| baseline (746 columns) | **+0.3188** | — |
| + 22 persistence statistics | +0.0897 | −0.229 |
| + 279 persistence-image pixels | +0.0022 | −0.317 |
| + both | negative per seed | — |
| + 22 persistence statistics, **block-mean control** | +0.2687 | −0.050 |

The block-mean control replaces each added column by its mean within the
composition block: identical columns, identical between-block information,
zero within-block variation. It recovers **78 %** of the collapse, so the
damage comes overwhelmingly from how these descriptors vary *across metals
inside a block* — which is the only variation the shape model can use, and
for these features it is noise. Adding columns per se costs little (−0.05).

This was worth running because it tested a prediction that turned out wrong:
3D-derived tabular columns were known to collapse a flat model, and the
expectation was that the anchored decomposition would protect against it,
since the level is produced separately. It does not — the shape model is if
anything the more vulnerable place to put them, because within-block noise
is exactly what it is trying to fit.

## Conclusion

Topology, in both forms available here, contributes nothing beyond the
distance encoder: the simplicial network because its predictions are 96 %
the same, persistence descriptors because their within-block variation is
noise. The distance-encoder shape channel (+0.008 legacy, confirmed
+0.0156 on the held-out 444-pair set) remains the only 3D contribution that
survives testing.
