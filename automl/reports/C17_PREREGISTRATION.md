# C17 — give the published 3D model the tuning it never received

Written and committed before any C17 cell runs.

## The gap

The loss-side parameters live in the **shared** `run_fold`
(`automl/topo/train.py` ~694–702, 996–1128), not inside the dist encoder. Every
loss finding proven on `--arch dist` therefore applies mechanically to
`--arch snn` — the **published headline 3D model** — and none has ever been run
there.

| lever | dist runs | **snn runs** | evidence on dist |
|---|---|---|---|
| `pair_loss_weight = 4` | 78 | **0** | **+0.0142**, 95 % CI [+0.0034, +0.0251], p = 0.014, 40 independent seeds |
| `pair_adj_weight = 10` | 57 | **0** | its best adjacent-emphasis value |
| `preset = baseline_2d_mphys` | 154 | **0** | ~+0.03 |
| `level_loss` / `level_quantile` / `pair_subsample` / `pair_loss_kind` | many | **0** | mixed |

snn sits at `plw = 2.0` in **371 of 380** runs; the sole exception is one seed
at 5.0 (n = 1, inside noise). snn and dist are statistically indistinguishable
head-to-head (+0.0036 ± 0.0348, 16 shared seeds, `automl/artifacts/topo_encoder`)
— yet dist's best tuned arm reaches **+0.2356** against snn's best family at
**+0.2056**. The published 3D model is not worse; it is **untuned**.

## Why this and not another representation

Representation changes have now failed repeatedly and with good power:
g-xTB geometry **+0.0041 n.s.** (40 cells), correspondence **−0.0129**,
FiLM **−0.1078** (p = 0.0012), pair-head reconciliation **−1.16**. Objective
changes that survived testing paid: plw4 **+0.0142**, and the tabular arm
+0.1422 → **+0.2784** on loss alone.

This campaign moves an established, independently replicated objective result
onto the architecture that never received it. It is the highest-prior 3D
experiment available, and it is not a new asset, geometry or encoder.

## Design

Base for every cell:
`--arch snn --select-on adjacent --filtration-max 3.5 --folds 5 --repeats 3
--deterministic` — 3.5 Å is snn's published radius, **triangles kept** (this is
the simplicial model, not a graph ablation).

| arm | added flags | seeds |
|---|---|---|
| `ctrl` | — (published snn) | 32 |
| **`plw4`** | `--pair-loss-weight 4.0` | 32 |
| `adjw10` | `--pair-adj-weight 10.0` | 24 |
| `mphys` | `--preset baseline_2d_mphys` | 24 |
| `combo` | `plw4 + adjw10 + mphys` | 32 |

144 cells. Seeds **501+**, disjoint from every prior campaign (7–83, 101–191,
201–418), so nothing here is contaminated by a run that selected these values.

**Power.** Per-seed sd of a paired Δ is **0.0285**. 32 seeds resolve
**+0.0141**; 24 resolve +0.0163; 8 resolve only +0.028 — which is why three
positives dissolved last session. Seeds are spent instead of configurations.

## Declared in advance

- **Primary endpoint:** paired Δ on `sel_adj_logSF_r2`, `plw4 − ctrl`, 32 seeds.
- **Real** requires **both**:
  1. paired p < 0.05, **and**
  2. the scale-free contrast (R² after optimal rescaling = Pearson², which
     removes every calibration effect) **agreeing in sign**.
  Both, because the C8 artefact passed the first and failed the second: its
  +0.0333 reversed to −0.0036 once the scale was free.
- **Null:** anything else, including a positive point estimate that misses
  significance. It will be reported as a null.
- `adjw10` and `mphys` at 24 seeds are **secondary**. `combo` is one cell of a
  2³ design and is reported descriptively — if it wins, the factorial is owed.
- Every result quoted as mean Δ, **seeds up / total**, paired t, Wilcoxon, and
  the scale-free contrast. Never a bare mean.

## What is deliberately not done

**The `report` third stays unspent.** `automl/artifacts/c6_split/split.json`
documents it as "ONE pre-declared configuration"; it is the only clean
confirmation partition left, and it was spent this week on a *CatBoost*
contrast, not a 3D one. It is reserved for a single look once a winner exists.

## Then: does topology's contribution to the STACK grow?

The 3D arm's published value is its **marginal** contribution to the 3-model
stack (+0.0381), not its standalone score. CatBoost has moved +0.1422 →
**+0.2784** and now beats the 3D arm outright, so under the project's own
mechanism rule a stronger partner can **shrink** topology's marginal
contribution. That is measured, not assumed, via
`c6_final.nested_pair_stack` (NNLS on pair vectors), reported under **both**
`composition_key` and `strict_composition_key` — under the strict key the
published contribution is already *not distinguishable*
(+0.0177 [−0.0023, +0.0367]), which is the honest bound.

## Honest statement of expectation

I cannot promise a positive result, and this session has shown what happens
when a design is bent toward producing one: three separate positives at n = 4–8
that did not survive. What this design guarantees is that the answer will be
trustworthy at the effect size actually in play. A null is publishable here: it
would say the loss findings are dist-specific rather than objective-level, which
is itself new information about where the 3D arm's ceiling comes from.
