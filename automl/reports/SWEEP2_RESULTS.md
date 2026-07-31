# Angular information does not reach the adjacent-pair metric, and one route destroys it

**Bogdan Mironov · 31 July 2026**
Pre-registered in `SWEEP2_PREREGISTRATION.md`, committed before the first run.
Data: `sweep2_cells.csv`, `sweep2_test.csv`, `sweep2_coverage.csv`,
`within_block_signal.csv`, `metric_tension.csv`.
72 GPU runs: 44 screen (11 cells × 4 seeds), 24 confirmatory (16 a side), 4 post-hoc.

---

## 1. What was asked

`AUDIT_2026-07-30.md` found, from a full inventory of all 662 encoder runs on
disk, that **no angular or polyhedral quantity had ever reached a neural
encoder** in this study. Every run used `preset=baseline_2d`; no angle, CShM,
%V_bur or three-body term had ever entered one. A coordination polyhedron is an
angular object, so this was the largest unexplored region in the project.

Three axes, eleven cells, one anchor:

| axis | cells | what changed |
|---|---|---|
| **A** angular information | A1, A2, A3 | 119 angular/polyhedral columns into the tabular head; per-node cosine histograms into message passing; a donor–M–donor angular readout |
| **B** auxiliary target | B1, B2, B3 | a second head predicting CShM, xTB `E_int`/`dg_transfer`, or metal charge transfer |
| **C** readout / optimisation | C1–C4 | radial basis 32→64 bins and 8→10 Å; attention pooling; lr; weight decay |

Capacity was deliberately **not** swept: `snn_wide_pair` had already collapsed
to +0.0021 when capacity met the contrast objective.

## 2. The screen — tune half only, 84 extractants

Anchor **A0 = +0.2362**.

| cell | axis | tune adj-R² | vs A0 | tune strict | overall R² |
|---|---|---|---|---|---|
| A0 | (anchor) | +0.2362 | — | +0.1369 | +0.4743 |
| **A1** | angular | **−0.0805** | **−0.3167** | −0.1058 | +0.3722 |
| A2 | angular | +0.2129 | −0.0233 | +0.1223 | +0.4285 |
| A3 | angular | +0.2101 | −0.0261 | +0.1121 | +0.5099 |
| B1 | auxiliary | +0.2283 | −0.0079 | +0.1436 | +0.4708 |
| B2 | auxiliary | +0.2360 | −0.0002 | +0.1417 | +0.5035 |
| B3 | auxiliary | +0.2411 | +0.0048 | +0.1974 | +0.5082 |
| **C1** | readout | **+0.2539** | **+0.0176** | +0.1523 | +0.4382 |
| C2 | readout | +0.2107 | −0.0255 | +0.0993 | +0.4928 |
| C3 | readout | +0.1973 | −0.0389 | +0.1339 | +0.4936 |
| C4 | readout | +0.2059 | −0.0303 | +0.0976 | +0.5259 |

**Main effects per axis:** angular **−0.1220**, auxiliary **−0.0011**, readout
**−0.0193**.

Only C1 cleared the pre-registered +0.005 screening gate.

## 3. The confirmatory look — 78 held-out extractants, 16 seeds a side

One look, both block keys, multiplicity-respecting cluster bootstrap,
Bonferroni over ≥ 21 looks.

| key | delta | 95 % CI | 21-look CI | verdict |
|---|---|---|---|---|
| binned | +0.0074 | [+0.0019, +0.0134] | **[−0.0025, +0.0172]** | not distinguishable |
| strict | −0.0010 | [−0.0087, +0.0104] | [−0.0173, +0.0154] | not distinguishable |

**C1 did not replicate.** The effect fell from +0.0176 on tune to +0.0074 on
confirm — less than half — and the corrected interval spans zero. Under the
strict key it is essentially zero.

> **Verdict: the sweep is a null on all three axes.**

This is the fourth time in this study that a screen-selected winner has failed
its own confirmation, and the two-stage design exists for precisely that.

## 4. The result that is not a null — A1, and why it collapses

A1 is the one large effect, and it is in the wrong direction: **−0.3167**,
turning a +0.24 metric negative. Overall R² fell only 0.10. That asymmetry is
the whole finding.

**The inputs were present.** `sweep2_coverage.csv`: of the 119 added columns,
median coverage is **100 %**, mean 88.8 %, 88 fully populated, 4 entirely empty.
So this is "119 well-measured angular descriptors actively hurt", not "the
features were missing". Those are different claims and only measurement
separates them.

**Where the damage comes from.** `sel_adj_logSF_r2` scores the *difference*
between adjacent lanthanides inside a composition block, so a feature helps only
if its own within-block difference tracks `dy`:

| | geometry (119) | published `cond` (64) |
|---|---|---|
| varying **within** a block | 87 (73 %) | 29 (45 %) |
| block-constant | **5** | 31 |
| median \|corr(Δfeature, `dy`)\| | **0.0495** | 0.0804 |
| max \|corr\| | 0.183 | 0.357 |

Geometry is almost entirely within-block-varying, and that variation is nearly
orthogonal to what the metric scores.

**The controlled test.** `--extra-block-mean` replaces every added column by its
per-block mean: same 119 columns, same between-block content, within-block
variation removed by construction. Leak-free because 0 of 552 blocks span more
than one extractant group, so the mean never crosses a CV fold.

| | tune adj-R² | vs A0 | vs A1 |
|---|---|---|---|
| A0 anchor | +0.2362 | — | — |
| A1 raw | −0.0805 | −0.3167 | — |
| **A1BM** block means | **+0.2148** | −0.0215 | **+0.2952** |

Removing only the within-block variation recovers **93 %** of the damage. The
head was fitting within-block geometry variation the metric cannot use.

**Why that variation is noise is already known.** `CONFORMER_RESULTS.md`
measured it: the shipped geometries are arbitrary Architector local minima,
82 % are not the global minimum, and the within-family SD of the gap is
0.503 eV. Geometric descriptors of arbitrary minima differ between adjacent
lanthanides for reasons unrelated to chemistry. Two campaigns, run for different
purposes, converge on the same conclusion.

## 5. Two secondary findings

**The radial basis saturates for a quarter of every ligand.** Over 30,140 atoms:
median distance-to-metal 6.44 Å, p90 9.34 Å, and **24.1 %** lie beyond the
hardcoded 8.0 Å cutoff used in all 662 prior runs; 18.4 % sit in the 8–10 Å
shell. These are extractant ligands and they are larger than the basis was built
for. **This does not establish that fixing it helps** — C1 changed the cutoff
*and* the bin count, and C1 did not replicate. Testing the cutoff alone needs its
own pre-registration and a fresh held-out half; the confirm extractants are
spent. The decomposition cells were built and deliberately **not run**, because
decomposing screening noise is chasing noise.

**The two obvious selection criteria disagree** (`metric_tension.csv`, post-hoc):

```
selecting on ADJACENT-PAIR picks C1: adjacent +0.0176, overall -0.0361
selecting on OVERALL R2    picks C4: adjacent -0.0303, overall +0.0516
```

Each winner is negative on the other metric. 5 of 10 cells improved overall R²
*while losing* adjacent-pair R²; only 2 improved adjacent-pair R² at all.
Selecting this sweep on overall R² — the default choice — would have picked C4
and given up 0.0303 of the quantity the study exists to predict.

A correlation between the two gains is **not** claimable: Pearson +0.701 against
Spearman −0.006 is the signature of one high-leverage point (A1), and excluding
it gives −0.323 with an interval spanning zero. Reported as an argmax comparison
instead, which is robust to that.

## 6. What this says

- **The encoder is not limited by its blindness to angles.** Three independent
  routes — tabular, message-passing, readout — all fail, with an axis mean of
  −0.1220. For a coordination polyhedron that is genuinely surprising, and it
  points away from the representation and toward the data.
- **Auxiliary multi-task is a flat null** (−0.0011). Notably `E_int` as a
  *target* neither helps nor hurts, where as an *input* it destroyed the metric
  (`ENERGY_RESULTS.md`).
- **The one hyperparameter that had never been exposed produced the only screen
  winner, and it did not survive.** Two-stage discipline held.

## 7. Limits, stated

- 4 seeds per screening cell. Legitimate only because `--deterministic` makes
  runs bit-identical; there is no run-to-run spread left to average away. The
  confirmatory stage used 16 a side, and the anchor moved 0.0117 on the confirm
  half between 4 and 16 seeds — comparable to C1's entire effect, which is why
  the pre-registration demanded 16.
- The screen ranks on the tune half only; no inferential claim is made from it.
- Sections 4–5 are **post-hoc**. A1BM was designed after seeing the screen and
  is kept out of `CELLS` so it cannot compete for the pre-registered gate. It
  explains a −0.3167 effect far too large to be noise; it does not establish a
  new one.
- Bit-identical determinism was verified on this cluster's GPUs only.

---

**Reproduce**

```bash
automl/slurm/campaign_driver.sh automl/slurm/sweep2.sh 44 8 34
python3 -m automl.topo.sweep2_test --n-boot 400
automl/slurm/campaign_driver.sh automl/slurm/sweep2.sh 24 8 30 MODE=confirm CELL=C1
python3 -m automl.topo.sweep2_test --confirm C1 --n-boot 400
sbatch --array=0-3 automl/slurm/sweep2_posthoc.sh
python3 -m automl.topo.sweep2_test --posthoc
python3 -m automl.topo.within_block_signal
python3 -m automl.topo.metric_tension
python3 -m automl.topo.sweep2_coverage
```

**Bugs this campaign caught before they could become results**

| what | how it would have surfaced |
|---|---|
| an all-NaN feature column poisoned every prediction | a NaN contrast reads as "no effect", i.e. as a null |
| `--angular-readout` silently also widened `node_feat` | A3 would have carried two axis-A changes at once, unattributable |
| the confirmatory contrast was coded at 4 seeds, not 16 | a normal-looking delta, CI and verdict on an under-replicated comparison |
| post-hoc cells could match a pre-registered cell | A1BM runs averaged into cell A1's published contrast |
| my own "32 block-constant columns" | an ablation with nothing to train on; the real count is 5 |
| my own proposed anticorrelation | Pearson +0.816 driven by one point; killed by its own robustness check |
