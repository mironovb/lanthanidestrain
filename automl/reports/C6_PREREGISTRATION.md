# Campaign 6: align the contrast term with the metric it is scored by

**Committed before any screening result was read.** The screening jobs
(5329849, 5329850) were submitted at 09:2x on 6 August 2026 and this file was
written while they were still queued. Its purpose is to fix the decision rule
while the numbers are still unknown, so the endpoint is one pre-declared look
rather than the maximum of 39 configurations.

---

## 1. The motivating measurement

`train.py` builds the contrast term from **raw row pairs**, weights
`|Δ lanthanide_index| ≤ 1` by 3.0, and squares the error. `dl == 0` is a
*same-metal* pair — two replicate measurements of one metal in one block.
Censused on the modelled rows:

| pair class | n | % of pairs | RMS Δy | weight | % of weighted squared-error mass |
|---|---|---|---|---|---|
| **same metal (dl = 0)** | 19,482 | 12.5 % | 1.483 | **3.0** | **26.0 %** |
| adjacent (dl = 1) | 18,065 | 11.6 % | 1.216 | 3.0 | 16.2 % |
| non-adjacent | 118,119 | 75.9 % | 1.558 | 1.0 | 57.9 % |

- **61.6 % of the squared mass inside the 3×-weighted "adjacent emphasis" term
  is same-metal pairs** — precisely the population `adjacent_pair_arrays`
  averages away before scoring anything.
- The metric sees 1,349 adjacent pairs; the loss sees 18,065 (×13.4).
- Pair count is quadratic in block size, the metric linear in distinct metals,
  so the **top 10 blocks take 59.6 %** of the pair-loss mass.

`evaluation.py:181-192` already records that enumerating raw row pairs once
produced a figure that *inverted* the published result. The evaluator was
fixed; the loss was not.

**Why this is worth a campaign.** Every deliberate *tuning* campaign in this
study returned ≈ 0 (sweep2, campaign 3, campaign 4, 57 PI constructions). Every
large win was the repair of a train/eval mismatch: the rank transform
(+0.005 → +0.221), adding the contrast term at all (+0.186), fitting the stack
on pairs (+0.0559). This is one more of the latter kind, not the former.

## 2. Arms

Screening: 39 configurations × 4 seeds = 156 runs, `--arch dist`,
`--repeats 1`, trained and scored on the **screen+select** 106 extractants
(604 pairs). Five axes:

| axis | question |
|---|---|
| **w1** (12 cells) | the contrast term's shape — alignment, adjacent-only, Huber, emphasis weight, term weight |
| **w6** (11 cells) | receptive field. The radial readout resolves 0.258 Å against a 0.013 Å contraction step; the two prior wide-field runs (`snn_filt5` −0.0686, `snn_allatom` −0.3273) both predate the contrast objective and bound nothing current |
| **w3** (7 cells) | the rank-K radius-interaction head, `pred = f(u) + Σ g_k(u)·φ_k` |
| **w4** (5 cells) | aqueous-phase and f-shell metal constants |
| **w7** (4 cells) | arms built to be decorrelated rather than strong |

## 3. What was gated *before* it could spend GPU time

- **Byte-identity.** One deterministic configuration run from a pristine HEAD
  worktree and from the working tree: `max|Δoof| = 0.000e+00` over 4,746 rows.
  Every new flag is genuinely default-off, so published arms remain comparable.
  (The first attempt at this gate compared against a *published* run and
  returned 1.9 — because that arm was trained without `--deterministic` and
  carries a ~0.009 run-to-run floor. It measured GPU reduction order, not code.)
- **Metal descriptors.** `automl/reports/c6_prescreen.csv` correlates each
  candidate's within-block difference with `dy` before it is trained on.
  `mphys__dG_hyd` reaches **0.215**, above the incumbent `Ionic Radius_metal`
  (0.171) and above the best of A1's 119 geometry columns (0.183); the block
  median is 0.139 against A1's 0.0495. This is the gate A1 lacked.
- **Smoke.** Two epochs of every new code path. It caught a real defect: the
  first three-way split was drawn on 187 extractants from the matrix cache
  while the modelled set has 162, so 10 screen names did not exist in the data.

## 4. Decision rule — fixed here, before the numbers

**Screening gate: +0.02** over the `a0_published` control, on the 4-seed
ensemble, scored on screen+select. Not +0.005: the 8-seed per-cell SE is ~0.017
and an identical config re-runs 0.0092 apart, so a smaller gate screens noise.
Cells below the gate are **discarded, not ranked**.

**Shortlist:** at most 6 cells clearing the gate, plus `a0_published` as the
matched control, re-run at **16 seeds × 5 folds × 3 repeats on the full 162
extractants**.

**The endpoint, one look:** the adjacent-pair R² of the best shortlisted single
arm, and of the pair-fitted nested stack over the shortlist plus the repaired
fingerprint network and the re-tuned CatBoost, **restricted to the `report`
third of extractants** (56 extractants, 301 pairs). Those extractants take part
in no ranking decision at any stage, so this is a single pre-declared look and
**no multiplicity correction applies**. Block key: binned `composition_key`, the
published one. `strict_composition_key` is reported alongside as a secondary,
not as an endpoint.

Under leave-extractants-out every extractant is held out in some fold, so the
final model may still *train* on the report extractants — what is protected is
the *choosing*, not the fitting.

**Expected shrinkage, stated in advance.** This study's own rate is roughly
half (C1 +0.0176 → +0.0074; PI sweep 2.9 σ → 0.1 σ; campaign 5 the sole
exception, growing). A cell screening at +0.03 that reports +0.015 is a
**success**, not a disappointment, and will be described as one.

## 4b. Addendum — the bar on the report third, fixed before the endpoint

Committed while screening was still running and before any endpoint was
computed. **The report third is an easier subset than the full set**, and
comparing a report-third endpoint against the full-set incumbent (+0.2474)
would manufacture an improvement out of subset difficulty alone:

| incumbent arm (16 seeds, published) | full (905 pairs) | screen+select (604) | **report (301)** |
|---|---|---|---|
| D0 distance GNN | +0.2474 | +0.2311 | **+0.3030** |
| G0 graph encoder | +0.2459 | +0.2386 | **+0.2670** |
| repaired fingerprint net | +0.2206 | +0.2235 | **+0.2026** |

**The number to beat on the report third is therefore +0.3030, not +0.2474.**
Any endpoint below +0.3030 there is a regression against the incumbent however
favourably it compares to the published headline, and will be reported as one.
The full-set number will be quoted alongside it, labelled as including
extractants that screening touched.


## 4c. The endpoint configuration, declared before the look

Selected on **screen+select only** (106 extractants, 604 pairs), at 16 seeds on
the full protocol. The full-set confirm table was deliberately *not* used to
choose, because it includes the report third and choosing on it would be the
contamination this whole design exists to avoid.

| confirm cell, scored on screen+select | adj R2 | delta vs control |
|---|---|---|
| **z1_mphys_f40_w10** | **+0.2742** | **+0.0444** |
| e0_mphys | +0.2550 | +0.0252 |
| a7_w10_only | +0.2539 | +0.0241 |
| z0_mphys_f40 | +0.2522 | +0.0223 |
| b7_f40_fb64 | +0.2460 | +0.0162 |
| a0_published (control) | +0.2298 | 0 |

**The pre-declared endpoint arm is `z1_mphys_f40_w10`:**

```
--arch dist --pair-loss-weight 2.0 --select-on adjacent \
    --preset baseline_2d_mphys --filtration-max 4.0 --rbf-bins 64 \
    --pair-adj-weight 10.0
```

It composes the three axes that independently cleared the screening gate: the
aqueous/f-shell metal block, the full 4.0 A graph with a matched radial basis,
and an adjacent-pair emphasis of 10. It carries **no** `--pair-metric-align`.

**The pre-declared endpoint stack** is the nested pair-fitted NNLS over that arm,
the repaired fingerprint network, and the re-tuned CatBoost (`mae`) and
fingerprint (`narrow`) partners, all on full data.

Both are reported on the report third, once. Observed shrinkage from screening
to confirmation was **heavier than the "about half" predicted in section 4**:
e0_mphys +0.0853 -> +0.0252 (30%), b7_f40_fb64 +0.0711 -> +0.0162 (23%),
a7_w10_only +0.0519 -> +0.0241 (46%). Recorded here because the prediction was
made in advance and was too optimistic.

## 5. What each outcome means

| outcome on the report third | reading |
|---|---|
| best single arm ≥ **+0.27** | the alignment repair is real; the incumbent D0 (+0.2474) is superseded and the loss/metric mismatch is the campaign's portable finding |
| best single arm +0.25 – +0.27 | a genuine but small improvement; report it as such and do not dress the stack number up to compensate |
| best single arm ≤ **+0.2474** | the mismatch is measurable but does not bind — the same shape as `OBJECTIVE_RESULTS.md`, where a correct diagnosis of where the gradient goes did not translate into the metric. Report the null; the census stands as a finding about the loss regardless |
| stack < best single arm | say so. The pair-fitted stack already beats its own best arm by only −0.003 on the tune half, and a stack that adds nothing is a result, not something to hide behind |

**The honest prior.** Four of the last five campaigns returned nothing. w6, w3
and w4 are tuning-shaped and should be expected to return ≈ 0 on that record;
only w1 is defect-shaped. If w1 fails, the campaign fails, and the census in §1
is what survives.

## 6. Protected state

`data/` is read-only throughout. Every run passes
`--out-dir automl/artifacts/topo_c6*` — never the default, because `train.py`
*appends* to `<out-dir>/results.jsonl` and `topo_runs/results.jsonl` is
SHA-pinned. `control_guard --verify` passed before the campaign (324 artefacts
byte-identical) and must pass after. `--snapshot` is never run.
