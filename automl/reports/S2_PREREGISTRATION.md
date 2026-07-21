# Pre-registration: does a variance-reduced SNN beat the repaired baseline?

**Written and committed before any S2 run is submitted.** Nothing below may be
revised once the first `sbatch` lands.

> ### Amendment 1 — 21 July 2026, before any S2 result existed
>
> **What changed:** the self-supervised encoder is pretrained **once and shared
> across all 32 seeds**, instead of being recomputed inside every run.
>
> **Why:** every seed was re-solving an identical self-supervised problem
> (masked charges and edge radii, no log D). Pretraining once removes 31 of 32
> repetitions of the same computation.
>
> **Correction to the figure first quoted here.** I originally justified this
> with "~3 min per epoch → ~1 h per run → ~16 h across 32 seeds", extrapolated
> from the *first* epoch. That was wrong by about 4×: the first epoch is slow
> because it populates the complex cache, and later epochs hit it. Measured end
> to end, all 20 epochs take **~10 min** and a fold takes **25 s**, so a full
> pretraining run is ~16 min and the true saving is roughly **5 h, not 16 h**.
> The decision stands — 31 redundant repetitions of an identical computation is
> reason enough — but the number that motivated it was overstated and is
> corrected here rather than left in place.
>
> **Why it does not weaken the arm:** S0 uses **no** pretraining at all and
> still shows a +0.060 ensemble gain with per-seed SD 0.047 — all of its seed
> diversity already comes from the supervised phase (fold splits, batch order,
> dropout, head init), none of which a shared encoder touches. The pretraining
> lever itself is unchanged: still 20 epochs over all 2,797 structures.
>
> **Standing of this amendment:** the array was cancelled and **zero S2 runs had
> written any output**, so no outcome influenced it. It is a compute decision,
> not a result-driven one. The endpoints, seed count, arm definition and
> decision rules in §3–§5 are untouched. Recorded here rather than made quietly.

Prior work: `CONTROL_RESULTS.md` (commits `4f39b91` → `89e77b3`). Baseline state
pinned at
`sha256(automl/artifacts/topo_control/_baseline_snapshot/manifest.json) =`
`1f9b47acb264927761ddbe00c3f89a458843a210bc1f189701dd40298d0a2e6d`
(324 artefacts, 141 out-of-fold parquets).

---

## 1. Where this starts

| | adjacent-pair R² |
|---|---|
| SNN + contrast (S0, 16 seeds) | +0.2382 |
| tabular + contrast (T0w, the matched control) | +0.2006 |
| **repaired FCNN baseline** (`StandardScaler`, 16 seeds) | **+0.2206** |

S0 beats the *matched* control by +0.0485 [+0.009, +0.106], and the *repaired*
baseline by **+0.0261 [−0.005, +0.076] — not distinguishable from zero.**

**This is a second attempt at the same claim.** It is not a fresh question, and
§5 states what that costs.

---

## 2. The diagnosis this arm is built from

The SNN is **variance-limited, not representation-limited**. From
`control_cells.csv`, all 16-seed ensembles on identical rows:

| arm | per-seed SD | ensemble gain | parameters |
|---|---|---|---|
| tabular + contrast | 0.027 | +0.033 | 516 k |
| PI-CNN + contrast | 0.026 | +0.036 | — |
| **SNN + contrast** | **0.047** | **+0.060** | **1,110 k** |
| SNN + plain | 0.144 | +0.116 | 1,110 k |

Noisiest arm by 2×, largest ensemble gain, 1.11 M parameters on **953 distinct
geometries** — `train.py`'s own docstring calls that "the number that actually
governs overfitting here."

Two competing explanations were tested first and **both failed**, which is why
the arm targets variance and not architecture:

* *The joint `LayerNorm` over `[embedding, tabular]` squashes topology.*
  Measured: both streams at std ≈ 1.0. No imbalance.
* *The control is unfairly strong.* It is fair; the pre-registered max rule made
  it **harder** (T0w head 516 k vs S0 head 449 k).

---

## 3. The arm — S2, fixed now

**S2 = SNN + contrast objective + conformer augmentation + conformer-pretrained
encoder + block-centred relative embedding, 32 seeds.**

One arm, four levers, bundled because the chosen protocol trades
attribution-of-cause for power. Per-lever ablation is reported descriptively and
is **not** a confirmatory test.

| lever | built | evidence it is real |
|---|---|---|
| conformer augmentation | 2,797 structures over 956 complexes (**2.93×**) | mean |Δ M–L| 0.306 Å between conformers; adjacent-Ln signal is 0.013 Å |
| pretraining on all conformers | masked-charge + edge-radius reconstruction, no log D | targets free on every structure |
| 32 seeds | from 16 | largest ensemble gain of any arm; also narrows the interval |
| block-centred embedding | `[emb, emb − block mean]` | 347 of 552 blocks are single-metal, so concatenated not substituted |

**Asset integrity, already verified and not to be revisited:**
`build_vr_conformers --verify-against-shipped` rebuilds the *original*
geometries through the new path and reproduces the shipped asset on **20/20**
complexes (atomic numbers, `is_metal`, `is_coord_donor`, edge and triangle
filtrations). Two bugs were caught by that check and fixed — a donor rule
written as a 3.10 Å cutoff when the shipped rule is nearest-`core_cn`, and a
`coreCN` read from a dataset column that disagrees with what was built. Either
would have marked every augmented structure.
`charge_missing` is 0.0000 on originals **and** conformers, so nothing
distinguishes augmented from original except coordinates.

---

## 4. Endpoints, fixed now

**Primary.** S2 (32-seed ensemble) − repaired FCNN baseline
(`oof_fcnn_std_scaler_ens16.parquet`, +0.2206). Paired cluster bootstrap over
extractants, 400 draws, seed 0, 90 % interval, via
`control_factorial.paired_adjacent_fast` — verified to reproduce the published
headline `+0.2426 [+0.181, +0.333]` exactly.

**Secondary, descriptive only.** S2 − T0w; S2 − S0; per-lever ablation.

**Fixed:** 32 seeds, all ensembled, no subset selection; no cell re-run after
its result is seen; no lever added, removed or retuned after launch. The first
16 seeds are the existing matched set, so S2 − S0 stays seed-paired.

| outcome | consequence |
|---|---|
| S2 − repaired baseline excludes 0, positive | **a significant topology result against the strongest baseline available** |
| spans 0 but S2 > S0 | the levers worked and topology still does not clear the repaired baseline — report both |
| spans 0 and S2 ≈ S0 | the variance diagnosis was wrong; report as a second failed attempt |
| excludes 0, negative | report it |

---

## 5. Multiplicity and power — stated before, not after

**This is the second confirmatory test of "topology beats the repaired
baseline."** S0 was the first (+0.0261, spans zero). The 90 % interval is the
pre-registered headline per the chosen protocol, but **the two-test corrected
(95 % one-sided) interval is reported beside it every time the headline is
quoted.** A reader must not have to reconstruct that.

From the S0 test, SE ≈ 0.0246. So the point estimate must reach:

* **≈ +0.041** to clear zero uncorrected;
* **≈ +0.045** corrected for two tests.

The current gap is +0.026, so **the effect must grow by ~60 %.** Variance
reduction helps twice — raising the ensemble mean and narrowing the interval —
but this is not a comfortable margin, and the honest prior is that it may not
clear. Saying so now is what makes the result meaningful if it does.

---

## 6. Guards

1. `control_guard.py` pins **324 artefacts** including every published OOF
   parquet and `oof_fcnn_std_scaler_ens16.parquet` — the file the primary
   endpoint is scored against. `--verify` must pass after the run.
2. New directories only: `automl/artifacts/topo_s2/`, `vr_conformers/`,
   `conformer_charges/`. Nothing writes to a published directory.
3. `data/` is never written.
4. No source edits while the array is in flight.
5. Existing reports stay append-only; `S2_RESULTS.md` carries the outcome.
6. S0 must still re-ensemble to **+0.2382** and the headline to
   **+0.2426 [+0.181, +0.333]**. If either drifts, the harness moved and nothing
   in the run is trustworthy.

---

*Signed off before submission — B. Mironov, 21 July 2026.*
