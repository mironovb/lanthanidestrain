# Pre-registration: stop spending the gradient on the block mean

**Written and committed before any run of the decomposed objective exists.**

---

## 1. The defect, measured

The adjacent-pair metric scores a *difference* between two lanthanides inside one
composition block. The block mean is nuisance — it says which ligand and which
conditions, both of which the ECFP and `cond__` blocks already state, and which
CatBoost already predicts better than any network here (overall R² **+0.4987**
against the deployed stack's **+0.4369**).

Decomposing the target on this dataset:

| blocking | Var(block mean) | Var(within-block contrast) | contrast share of a plain MSE gradient |
|---|---|---|---|
| `composition_key` | 1.8171 | 0.8420 | **31.7 %** |
| `strict_composition_key` | 2.4089 | 0.2502 | **9.4 %** |

So the published Huber objective spends **68–91 % of its gradient on a quantity
the metric never reads.**

`--pair-loss-weight 2.0` was the existing response, and it is a partial one by
construction: it *adds* a contrast term on top of the full MSE. There is no
setting of it that *removes* the level term. Lowering the whole Huber weight is
not a substitute either, because that term contains level and contrast together.

## 2. The change

`--level-weight` replaces the Huber-on-raw-target with a **per-block level
term**, leaving the existing pairwise term to carry the contrast:

```
loss = level_weight · huber(mean_b(pred), mean_b(target))
     + pair_loss_weight · Σ_pairs w · ((pred_i − pred_j) − (t_i − t_j))²
```

The level term is **one value per block, not one per row**: a composition block
is a single nuisance parameter however many measurements sit in it, and
row-weighting would hand the largest blocks the same dominance the plain MSE
already gives them. `automl/tests/test_decomposed_loss.py` pins that, and pins
that leaving the flag unset reproduces the published objective exactly.

`--block-key` selects which blocking the contrast loss, the block-centred
embedding and the checkpoint-selection metric all use, so the model can be
trained against the strict definition of "identical conditions" that
`DUALKEY_PREREGISTRATION.md` tests.

## 3. Design

6 cells × 8 seeds = **48 runs**.

| axis | levels |
|---|---|
| `--level-weight` | 0.1, 0.3, 1.0 |
| `--block-key` | `composition_key`, `strict_composition_key` |

Everything else is the published S0 configuration: `--arch snn
--pair-loss-weight 2.0 --select-on adjacent --folds 5 --repeats 3`, seeds
7/11/23/37/42/51/67/83.

**Sized against the real cluster limit, not an assumed one.** This account is
capped by `GrpTRES` at **one node** on `xeon-g6-volta` — two concurrent jobs,
about 17 runs per hour. 48 runs is ~3 hours. The factorial I first sketched
(dim × layers × lr × dropout × filtration × heavy-only = 648 cells) would have
been a fortnight of wall clock and is abandoned.

`--deterministic` is **off**, for the same reason as Amendment 1 to
`ENCODER_PREREGISTRATION.md`: the arms this is compared against were trained
without it.

## 4. Selection protocol

Every run trains on **all 162 extractants**; the frozen 84/78 tune/confirm split
(`automl/artifacts/pi_sweep/split.json`) is applied **at scoring time only**.

This is not a preference. `--restrict-groups` was tried in the persistence-image
sweep, removed 57 % of the training rows (4,742 → 2,030), collapsed the arm from
+0.1562 to +0.0362 and left the selection rule unable to rank anything. It cost
66 GPU runs and produced Amendment 2a. It is not repeated.

So: **the winning cell is chosen on the 84 tune extractants**, and the 78 confirm
extractants are scored **once**, for that cell only.

## 5. Endpoints, fixed now

Primary, on the **confirm** half, against the published S0 ensemble in the same
slot:

| # | contrast |
|---|---|
| **1 (primary)** | stack(CatBoost, repaired, **best decomposed cell**) − stack(CatBoost, repaired, **S0**) |
| **2** | best decomposed cell alone − S0 alone |
| **3 (descriptive)** | the level-weight main effect and the block-key main effect, averaged over the other axis |

Contrast 3 is **main effects, not cell ranking**. With 8 seeds per cell the
per-cell standard error is around 0.017, so individual cells cannot be ranked;
16 runs per level gives about 0.012. That is the design lesson
`PI_SWEEP_PRECISION.md` paid 25 runs to learn, applied in advance this time.

Multiplicity-respecting cluster bootstrap, `n_boot=400`, seed 0, 90 % interval,
under **both** block keys. Look count rises from 16 to **19**; corrected
intervals reported beside uncorrected ones.

## 6. Decision rule

| outcome | consequence |
|---|---|
| contrast 1 excludes zero positive on the confirm half | **The objective was the binding constraint, not the representation.** This would be the study's first genuine improvement to the headline metric rather than another control, and it would say the +0.04 topology effect was measured through a loss that was mostly looking elsewhere. Report as the headline. |
| contrast 2 positive, contrast 1 spans zero | The decomposed arm is a better single model but not a better stack member — most likely because it has become more correlated with the repaired baseline. Report with the error correlation, which is the other axis of the mechanism rule. |
| both span zero | The level term was not the constraint. That is worth stating plainly: it would mean an objective spending 91 % of its gradient on nuisance is *not* what limits this problem, which points at the representation or the data rather than the loss. |
| the best cell is `level_weight = 1.0` + binned key | The published objective was already near-optimal on this axis and the diagnosis in §1, though arithmetically correct, does not bind. Say so. |

**A result in any of those four rows is reportable.** Writing them down now is
the point.

## 7. Guards

`control_guard --verify` before and after. Runs write to
`automl/artifacts/topo_objective/`, which no published test reads.
`control_factorial._matches` rejects `level_weight` and a non-default
`block_key`, so a decomposed run can never be swept into a published cell — with
a test. No writes to `data/`. Existing reports append-only.

---

**Bogdan Mironov · 29 July 2026**
