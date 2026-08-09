# Campaign 6 — running log

Chronological record of what was run, what it cost, what it returned, and every
claim that was later **corrected or withdrawn**. Kept separate from
[`C6_RESULTS.md`](C6_RESULTS.md) (the conclusions) and
[`SCIENTIFIC_FINDINGS.md`](SCIENTIFIC_FINDINGS.md) (the standing register).

Its purpose is auditability: anyone should be able to see which number came from
which job, at how many seeds, and which of my statements did not survive.

---

## Job ledger

| wave | jobs | runs | purpose | outcome |
|---|---|---|---|---|
| guard | 5329757 | 1 | re-run a published arm with new code | superseded by the deterministic A/B |
| identity | 5329786 | 2 | old vs new code, `--deterministic` | **max\|Δoof\| = 0.000e+00** over 4,746 rows |
| smoke ×2 | 5329815, 5329819 | 24 | 2-epoch probe of every new flag | caught the wrong-population split (187 vs 162 extractants) |
| screening | 5329849/50 | 155 | 39 configs × 4 seeds, 5 axes | 0 failures; `--pair-metric-align` lost in 16/16 |
| confirm | 5331136/7 | 96 | 6 cells × 16 seeds, full data | endpoint chosen on selection half only |
| partners | 5329823 | 18 | CatBoost/FCNN re-tune on the adjacent metric | **CatBoost MAE +0.0594** |
| partners-full | 5342868 | 4 | winners at 16 seeds, full data | **MAE +0.1066 full, log D +0.0115** |
| nbr-graph | 5342863→5342871 | – | rebuild graphs past 4.0 Å | verify **0 disagreements / 2,301,232 edges**; first run died on an atomic-write bug |
| w8a/w8b | 5342876, 5343265/6 | 32 | emphasis 20/40; graphs 5/6/8 Å, kNN | all below 4.0 Å at matched seeds |
| w9 | 5343377 | 32 | robust level loss on the neural arm | `mae_f40` best at 4 seeds — later shrank |
| regrid | 5343376 | 7 | CatBoost grid *around* MAE | shipped hyperparameters already right |
| quantile | 5343869, 5344381, 5344666 | 11 | α sweep, then full-data confirm | α=0.7 peak on screen half; **did not hold** on full |
| starvation | 5343868 | 16 | `--pair-subsample` control | **falsified** the data-poverty hypothesis |
| securing | 5343531/2 | 48 | `g0` 3.5 vs 4.0 Å; dist 4.0 Å at 32 bins | widening does not transfer to the SNN |
| 3D push | 5344875/6 | 36 | topology-only, no-ECFP, capacity, solvent | topology-only −0.18 to −0.21; rest ≈ 0 |
| night | 5345465+ (driver) | 140 | extra seeds, composition grid, solvent | the extra seeds are what mattered |
| top-up | 5348025/6 | 48 | top 4-seed cells → 12 seeds | **every large delta shrank** |
| solvent-matched | 5348397/8 | 24 | 3 geometries on identical 96 extractants | gas-phase best; solvent does not help |
| δ sweep | 5348591/2 | 72 | Huber δ ∈ {0.05…0.5} at 12 seeds | **falsified** my interior-maximum prediction |

Roughly **760 GPU runs and 40 CPU fits**, 0 unexplained failures. Every run
wrote to `automl/artifacts/topo_c6*` — never the SHA-pinned `topo_runs`.

---

## Claims I made and then corrected

This section exists because five of them did not survive, and the pattern is
itself a result.

| # | claim as first stated | what killed it | status |
|---|---|---|---|
| 1 | the loss/metric mismatch repair (`--pair-metric-align`) would be the campaign's win | its own 16 cells, all ≤ control | **falsified** |
| 2 | MAE helps because it is *robust* | Huber is robust and does not help | **falsified** |
| 3 | MAE helps because it targets the *median* | α = 0.7 beats α = 0.5 | **falsified** |
| 4 | the upper quantile works by down-weighting a left tail | the gain is *larger* where the tail is absent | **falsified** |
| 5 | collapsing replicates fails through *data poverty* | thinning to the same count **helps** (+0.044 vs −0.064) | **falsified** |
| 6 | "nothing past 4 Å helps" | 8 Å arrived at 4 seeds looking best → **withdrawn**; then 12 seeds put 4 Å back on top → **reinstated** | reinstated |
| 7 | α = 0.7 is the quantile optimum | on full data at 16 seeds, MAE (α = 0.5) wins | **withdrawn** |
| 8 | `t1_d02_f40` (δ = 0.2) is the best single arm in the study | matched contrast vs δ = 1.0 spans zero on every partition | **withdrawn** |
| 9 | the 7.5 % subsampling lever is worth +0.079 | +0.044 at 12 seeds | halved |
| 10 | the tabular family is more internally diverse than the 3D family | error correlation tracks model strength at r = +0.696; only one tabular arm is strong enough to match on | **withdrawn** (§G1b) |

**The common failure mode in 7, 8 and 9:** a difference of ~0.02 taken at face
value on an axis whose seed noise is ~0.03. That is E5, and it is why the
operational rule is now *no 4-seed cell may be promoted, quoted, or used to close
an axis*.

---

## What the campaign actually established

| result | size | evidence |
|---|---|---|
| CatBoost RMSE → MAE | **+0.1066** full, **+0.0552** held-out, log D **+0.0158** | intervals exclude zero |
| 3D: use the whole 4.0 Å graph + 64-bin basis | **+0.0139** full, **+0.0106** held-out | consistent across 3 partitions; held-out interval touches zero |
| deployable stack (2 arms, was 3) | **+0.3099** vs published **+0.2901** | held-out |
| **the 3D family has effective rank 1.05** | 97.4 % of variance in PC1 across 8 encoder variants | §G1 |
| **architecture ≈ random seed** for this metric | within-config 0.9900 vs across-config 0.9864, matched 8-seed ensembles | §G2 |
| repairing a real train/eval mismatch *hurts* | −0.064 | 16/16 cells |
| selection over 39 cells **inverts** the ranking | endpoint last of six; control second | pre-registered holdout |
| within-screening winner's curse | ~0.04 on the top cells | 4 → 12 seed comparison |

---

## Protected state

`control_guard --verify`: 324 frozen artefacts byte-identical, before and after.
`data/` untouched throughout (`git status --short data/` empty). No `--snapshot`.
