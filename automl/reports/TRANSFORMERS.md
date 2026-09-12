# Transformers for adjacent-lanthanide selectivity

12 September 2026. Three transformer architectures were trained and scored
under the same held-out protocol as every other model in this repository.
The short version: a transformer over the atoms of each complex is as good a
3D shape source as the message-passing distance encoder, and no better; the
one pre-registered held-out test passes. Transformers over the tabular
features, and set transformers over the metals of a block, lose to gradient
boosting by a wide margin and lower the score at every blend weight. The best
system is unchanged.

Figures: `figures/transformers_legacy.png` (per-seed scores and blend weight
curves, legacy population) and `figures/transformers_confirm.png` (the
held-out look). Numbers: `transformer_eval.json`, `transformer_rule.json`,
`transformer_confirm.json`, `anchored_ft.csv`, `block_transformer.csv`,
`cell_transformer.csv`.

## 1. What was tested

The current best system (findings I14 and I15) is

    prediction = anchor + (1 - w) * shape_tabular + w * shape_3D,   w = 0.35

where the anchor is the block mean of a CatBoost level model, the tabular
shape is a CatBoost model of the block-centred target, and the 3D shape comes
from the distance encoder (SchNet-style continuous-filter message passing over
Vietoris-Rips edges, 4.0 A, heavy atoms). The anchor cancels in every scored
pair, so the adjacent-pair score depends on the two shape terms only.

A transformer can take three places in that system:

| place | model | tokens | what it replaces |
|---|---|---|---|
| A. 3D shape source | attention encoder (`automl/topo/attn_gnn.py`, `--arch attn`) | atoms of one complex; self-attention with a learned per-head bias over the interatomic distance (Graphormer-style); dense over all pairs, or restricted to the <= 4.0 A neighbourhood as a control | the distance encoder body only; node inputs, readout, head, losses, folds and seeds identical |
| B. level and shape models | FT-Transformer (`automl/topo/anchored_ft.py`) | one token per numeric column (81) plus 8 tokens for the 665 ECFP bits, a CLS token; pinball loss at 0.6 like the CatBoost champion; also a flat variant | CatBoost level and shape models |
| C. shape model | set transformer over a composition block (`automl/topo/block_transformer.py` over rows; `automl/topo/cell_transformer.py` over the (block, metal) cells the metric scores) | the metals of one block attend to each other; output centred within the block by construction; L1 loss on the block-centred target | the CatBoost shape model |

Protocol, unchanged from every other result here: leave-extractants-out 5
folds x 3 repeats, out-of-fold predictions, adjacent-pair log SF R^2, inner
extractant-grouped early stopping, fixed seeds. The legacy 905 pairs are the
iteration set. The 444 pairs frozen before any model in the August work was
trained are the held-out set; they were looked at once, under a rule
committed to git before the look (commit `fed0f6a`).

Two gates were passed first. The `train.py` edit that adds `--arch attn` was
proven inert by rerunning a published distance-encoder seed under
`--deterministic`: max |delta OOF| = 0 over 4,746 rows. The capacity gate
(`--smoke`: fit 60 rows with regularisation off, R^2 must exceed 0.95) failed
for the attention encoder at 0.49, and a bisection on the debug GPU showed
why: exact distances 0.49, zero attention layers 0.52, deterministic mode off
0.49, lower learning rate 0.49, pair loss off 1.00, and the distance encoder
under the identical arguments 0.53 with the pair loss and 1.00 without. The
contrast objective never batches rows from single-member blocks, and the
60-row smoke subset is mostly singletons, so the gate had never been run with
the pair loss before. With the pair loss off the attention encoder fits to
1.00 like every other encoder; the smoke script now runs the gate that way.

## 2. Legacy population, 905 pairs

Standalone = the source's own pair differences. Blend = anchor + tabular
shape + source shape with the weight chosen nested per held-out extractant
(equal-extractant criterion, as in I15). Tabular only = +0.3182.

| model | seeds | per-seed R^2, mean +- sd | ensemble standalone | nested blend weight | blend R^2 |
|---|---|---|---|---|---|
| distance encoder (reference, I15) | 32 | +0.205 +- 0.031 | +0.266 | 0.35 | +0.3258 |
| distance encoder, 8 lowest-numbered seeds | 8 | | +0.261 | 0.30 | +0.3239 |
| attention encoder, dense | 8 | +0.203 +- 0.021 | +0.236 | 0.00 | +0.3173 |
| attention encoder, <= 4 A | 4 | +0.213 +- 0.003 | +0.241 | 0.00 | +0.3180 |
| FT-Transformer, level and shape | 4 | +0.100 +- 0.049 | +0.146 | 0.00 | +0.3182 |
| FT-Transformer, flat | 4 | +0.114 +- 0.109 | +0.171 | | |
| block transformer over rows | 8 | -0.051 +- 0.151 | +0.127 | 0.00 | +0.3174 |
| block transformer over cells | 8 | +0.024 +- 0.082 | +0.091 | 0.00 | +0.3182 |

Two-source blends (distance encoder plus one transformer source, nested
weights): the second source gets weight 0.00 in every case and the score
stays at +0.3258.

The nested weight of 0.00 for the attention encoders is not a missing
signal. The pair-level correlation of the attention encoder's differences
with the residual of the tabular shape is +0.196, the same as the distance
encoder's (+0.196), and the two encoders correlate at 0.936 with each other.
The blend score as a function of a fixed weight (descriptive, not nested):

| w | 0.0 | 0.1 | 0.2 | 0.3 | 0.35 | 0.4 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| distance encoder, 32 seeds | .3182 | .3238 | .3270 | .3277 | .3272 | .3261 | .3221 | .3069 | .2661 |
| distance encoder, 8 seeds | .3182 | .3231 | .3256 | .3258 | .3250 | .3236 | .3191 | .3030 | .2611 |
| attention encoder, dense | .3182 | .3244 | .3273 | .3271 | .3258 | .3237 | .3171 | .2943 | .2360 |
| attention encoder, <= 4 A | .3182 | .3226 | .3244 | .3234 | .3219 | .3197 | .3133 | .2924 | .2408 |
| FT-Transformer, level and shape | .3182 | .3141 | .3072 | .2973 | .2912 | .2845 | .2687 | .2285 | .1462 |
| block transformer over rows | .3182 | .3115 | .3021 | .2899 | .2828 | .2750 | .2573 | .2136 | .1273 |
| block transformer over cells | .3182 | .3104 | .2993 | .2849 | .2765 | .2672 | .2462 | .1942 | .0914 |

On the pair-weighted score the dense attention encoder peaks at +0.3273 (w =
0.2) against +0.3277 for the distance encoder (w = 0.3) and +0.3258 for the
distance encoder at the matched 8 seeds. The nested procedure minimises the
mean over extractants of the per-extractant squared error, and on that
criterion the attention blend gets worse with weight (0.0492 at w = 0 to
0.0497 at w = 0.3) while the distance blend improves (to 0.0486): the
attention encoder's gain sits in the pair-rich extractants and costs the
small ones. The three tabular transformers lower both criteria at every
weight.

## 3. Held-out confirmation, the one look

Rule frozen before the look: for the attention encoders, encoder
substitution at the fixed weight 0.35 of the current best system (a
legacy-fitted weight of zero would make the contrast identically zero and
test nothing); for the tabular sources, the legacy-fitted weight, which is
zero, so their held-out contrast is zero by construction and no claim is
made for them. Models for this look were trained on the expanded population
(5,946 rows), four attention seeds against the fifteen distance-encoder seeds
of I15.

| pairs | tabular level/shape | + distance encoder shape, w 0.35 | + attention encoder shape, w 0.35 | attention vs tabular | attention vs distance |
|---|---|---|---|---|---|
| frozen 444, held out | +0.1051 | +0.1207 | +0.1309 | +0.0258 | +0.0102 |
| legacy 905 | +0.2964 | +0.3109 | +0.3050 | +0.0087 | -0.0059 |
| union 1,349 | +0.2301 | +0.2450 | +0.2447 | +0.0146 | -0.0003 |

Primary contrast on the 444: +0.0258, PASS. The distance encoder's own
confirmation in I15 was +0.0156 on the same pairs. Against the distance
encoder the attention encoder is ahead on the 444, behind on the 905 and
level on the union: a tie between the two 3D encoders, now established on
held-out data rather than by inspection of the legacy population alone. Per
seed on the expanded population the attention encoder scores +0.152 +- 0.024
(4 seeds) against +0.144 +- 0.018 for the distance encoder (15 seeds).

## 4. Why the tabular and set transformers lose

FT-Transformer. Out-of-fold log D R^2 is 0.27 to 0.39 per seed against 0.49
to 0.51 for the CatBoost level/shape model; the model is under-fitting the
level, and the adjacent-pair score follows (ensemble +0.146 against +0.318).
A one-seed sweep on the legacy population (seed 42, reference +0.078): longer
patience +0.099, learning rate 3e-4 +0.079, dim 128 with four layers +0.099
(remaining variants of the sweep in `anchored_ft.csv`). Unlike CatBoost, the
transformer does better flat (+0.171) than as a level/shape pair (+0.146):
the block-centred residual is a small-signal target it cannot fit from 3,700
rows. Both variants are unstable across seeds (flat: +0.158, +0.119, +0.218,
-0.040).

Set transformers. Only blocks with two or more metals carry a signal for a
block-centred model: 205 of the 552 composition blocks in the legacy
population, 143 per training fold. Training and validation curves on one
fold (cell version): training L1 falls from 0.74 to 0.16 in 120 epochs while
the validation L1 never improves on its epoch-0 value (0.575 to 0.60), and a
metal-only variant with all other inputs zeroed also gets worse on held-out
extractants as it trains (validation L1 0.52 to 0.70). The model memorises
blocks. The row version, which spreads attention over condition replicates
of the same metal, behaves the same (per-seed -0.36 to +0.10). The CatBoost
shape model does not face this limit because its target is centred by
extractant, so every one of the 4,746 rows trains it.

## 5. Conclusions

1. A transformer over the atoms of a complex is a valid 3D shape source and
   reproduces the distance encoder's held-out gain (+0.0258 against +0.0156
   on the frozen 444 pairs; tie on the union of 1,349). It does not improve
   the best system: added on top of the distance encoder it earns weight
   0.00, and the two encoders' pair predictions correlate at 0.936. This is
   a fourth architecture (after the simplicial, distance and persistence
   encoders) landing on the same predictions, in line with the effective
   rank of 1.05 measured in I17: the 3D channel is saturated by what the
   geometries carry, not by the encoder.
2. The dense attention encoder sees every atom pair, the <= 4 A control only
   the message-passing neighbourhood; they score the same (+0.203 +- 0.021
   and +0.213 +- 0.003 per seed). Long-range geometry adds nothing here.
3. Transformers over the tabular features lose to gradient boosting by 0.15
   to 0.23 in R^2 and reduce the score at every blend weight; set
   transformers over the metals of a block are data-limited at 205 blocks.
   The result the level/shape decomposition and CatBoost give (+0.318 tabular,
   +0.326 with 3D) stands.

Cost: one attention seed is 14 minutes on one V100 for the full 5 x 3
protocol, about half the distance encoder's wall time per seed on the same
data.

## 6. Files

- `automl/topo/attn_gnn.py`, `automl/topo/train.py` (`--arch attn`,
  `--attn-heads`, `--attn-sparse`, `--attn-bias-max`; inert for every other
  arch, proof above)
- `automl/topo/anchored_ft.py`, `automl/topo/block_transformer.py`,
  `automl/topo/cell_transformer.py`
- `automl/topo/transformer_eval.py` (legacy analysis, `--freeze`,
  `--confirm`), `automl/topo/transformer_figs.py`
- `automl/slurm/tf_smoke.sh`, `tf_attn.sh`, `tf_tab.sh`, `tf_cell.sh`,
  `tf_ft_sweep.sh`
- artefacts: `automl/artifacts/topo_tf_attn/` (per-seed OOF parquets and
  run configs), `anchored_ft/`, `block_tf/`, `cell_tf/`
