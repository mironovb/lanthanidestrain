# Pre-registration: is the adjacent-pair gain topology, or is it the objective?

**Written and committed before any run of this experiment is submitted.**
Nothing below may be revised once the first `sbatch` lands. The point of fixing
it in advance is that *every* outcome already has a written meaning, so no
result can be reframed after the fact — in either direction.

Prior work being tested: commit `4769e76`, reported in `PI_REPORT.md` and
`PUBLICATION_ASSESSMENT.md`.

---

## 1. The question

The study reports adjacent-lanthanide separation R² rising from +0.005 (FCNN on
ECFP + RDKit) to +0.263, with five paired cluster-bootstrap intervals excluding
zero. All 51 of its runs use a topological encoder — 27 `snn`, 24 `picnn`, none
without.

The mechanism the study identified is **"train the contrast, not the absolute
value"**: a loss on within-composition pairwise differences plus checkpoint
selection on adjacent-pair R². Both are properties of the *objective*, not of
the representation. So a tabular model given the same objective might capture
most of the gain, and until that is measured the gain is not attributable.

**This experiment measures the attribution. It does not re-measure anything
already reported.**

---

## 2. Design

A 2×2 factorial — {no topology, PI-CNN, SNN} × {plain objective, contrast
objective} — at the **same 16 seeds** the published SNN ensemble uses:

`{7, 11, 23, 37, 42, 51, 67, 83, 211, 223, 233, 241, 251, 263, 271, 281}`

|  | plain MSE + MSE selection | contrast loss + adjacent selection |
|---|---|---|
| **no topology** (`--arch tabular`) | **T1** (16 new) | **T0** (16 new) ← the control |
| **PI-CNN** | **P1** (16 new) | **P0** (15 exist + 1 new) |
| **SNN** | **S1** (16 new) | **S0** (16 exist, unchanged) |

Plus **T0w** (16 new): T0 with `head_hidden=512`. See §5.

**Every cell except S0 is run fresh from current source.** The existing
plain-objective runs (`snn_hybrid` −0.1469 written 16:19, `pi_hybrid` +0.1563
written 16:53) predate `snn.py` 17:19 and `train.py` 18:00 — that is, they
predate the MPSN permutation-invariance fix, `node_feat_dim` 4→5, and the radial
readout. Reusing them would confound the objective with source drift. They are
reported in the study as they stand and are not used here.

**The pairing.** In `run_fold`, `np.random.default_rng(seed)` is created before
the model and drives both the inner-validation extractant split and the
composition-block batch order. Neither depends on architecture, so at a given
seed every cell sees the *same* held-out validation extractants and the *same*
batch ordering. Asserted by
`automl/tests/test_topo_control.py::test_the_two_arms_draw_the_same_folds_and_batch_order`.

---

## 3. Endpoints, fixed now

| | contrast | statistic |
|---|---|---|
| **Primary** | S0 − T0 | does topology add *on top of* the objective? |
| **Secondary** | P1 − T1 | does topology help with **no** contrast objective anywhere? |
| **Tertiary** | (S0−T0) − (S1−T1) | interaction: is topology only useful once the contrast is trained? |
| Descriptive | T0 − T1, S0 − S1, P0 − P1 | what the objective buys, per architecture |

**Statistic.** 16-seed mean out-of-fold prediction per cell; adjacent-pair
log-separation-factor R² via `automl.evaluation.adjacent_pair_metrics`; paired
cluster bootstrap resampling whole extractants, **400 draws, seed 0**, via
`automl.topo.adjacent_test.paired_adjacent` — the identical function every
published interval in this study came from. "Excludes zero" means the 90 %
interval (5th–95th percentile of the paired difference) does not contain 0.

**Frozen analysis choices.** 16 seeds per cell; *all* seeds enter the ensemble
(no subset selection, ever); 400 draws; seed 0; no cell re-run after its result
is seen; no additional configurations added to any cell.

---

## 4. What each outcome means — decided now

| outcome | meaning | consequence for the paper |
|---|---|---|
| **S0 − T0 excludes 0, positive** | topology adds beyond the objective | topology paper; every current claim stands as written |
| **S0 − T0 spans 0, P1 − T1 excludes 0** | topology carries the signal; the objective is what exposes it | topology paper, with the objective as the enabling method; §3.1 answered |
| **S0 − T0 and P1 − T1 both span 0** | the gain is attributable to the objective | title becomes *train the contrast*; **topology reported as a null**, explicitly and in the abstract |
| **S0 − T0 excludes 0, negative** | topology hurts once the objective is fixed | reported as such; contrast-trained tabular becomes the recommendation |

---

## 5. The capacity objection, handled in advance

T0 has fewer parameters than S0 by construction — it lacks the encoder and the
864 embedding columns into its head. A reviewer will say the control was
under-powered.

So **T0w** (`head_hidden=512`, double the head width) is run alongside, and:

> **The control's reported value is the *better* of T0 and T0w on the primary
> metric.**

Taking the max is selection on the test metric and therefore slightly inflates
the control — *in the direction that makes topology's job harder*. That is the
direction a control should err in, which is why it is chosen here and fixed
before any number is seen.

---

## 6. Declared invariant under every outcome

This experiment adds an arm. It re-runs nothing that has been reported. The
following cannot move, and `automl/topo/control_guard.py --verify` proves it by
hashing all 125 pinned artefacts — including every one of the 51 out-of-fold
parquets the published tests were computed from.

The manifest itself is pinned here, so the baseline state is fixed by this
commit and cannot be quietly re-snapshotted later:

```
sha256(automl/artifacts/topo_control/_baseline_snapshot/manifest.json)
  = 786fae08596666c8d2f2b3bfd7889124f2d2a3b1d1b4f57e1cd4ccd5ed8ec4f1
```

What cannot move:

1. the five reported deltas (arm vs named baseline, unchanged inputs);
2. the negative controls `pi_topoonly` (−1.74) and `snn_allatom` (−0.41);
3. the blend curve's interior maximum at w ≈ 0.7 (descriptive, not inferential);
4. the Stage-2 geometry null and the conformational-limit conclusion;
5. every number in `TOPOLOGY_RESULTS.md`.

**Documentation policy.** `PI_REPORT.md`, `PUBLICATION_ASSESSMENT.md`,
`TOPOLOGY_RESULTS.md` and `TOPOLOGY_METHODS.md` keep every number and every
word. The finding lands in a new `CONTROL_RESULTS.md`, and each of those four
gets exactly **one** pointer line at the top. Nothing published is contradicted
in place; the correction is impossible to miss.

`automl/refresh_reports.sh` is **not** run: it invokes `ensemble_adjacent` with
`--baseline baseline::mlp::none`, while `adjacent_ensemble.csv` on disk holds
CatBoost-baseline numbers (`baseline_obs` 0.1422 / 0.1441) — the ones quoted in
the committed reports. A refresh would silently rewrite that table against a
different baseline.

---

## 7. What this experiment cannot settle

Stated now so it is not discovered as a limitation later.

- **It is one dataset.** 162 extractants, no external validation. Attribution
  established here is attribution on this dataset.
- **It does not fix the magnitude compression.** Both arms under-predict how
  large a separation is (`topo_adjacent_parity.png`); the control does not
  address that and will not.
- **It does not make CatBoost contrast-trained.** T0 is the contrast-trained
  *neural* comparator. A pairwise-ranking GBDT predicts order rather than
  log D, which makes overall R² incomparable and needs its own validation — out
  of scope here, and named as remaining future work rather than quietly dropped.
- **`--select-on adjacent` still selects checkpoints on the reported metric**
  for T0, P0 and S0 alike. Legal (inner validation, folds grouped by extractant)
  and now symmetric across the factorial, but it remains a disclosure item.

---

*Signed off before submission — B. Mironov, 21 July 2026.*
*Runs land in `automl/artifacts/topo_control/` only. Analysis:*
*`automl/topo/control_factorial.py`. Results: `automl/reports/CONTROL_RESULTS.md`.*
