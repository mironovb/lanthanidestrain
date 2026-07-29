# Re-analysis, 29 July 2026: five measurements that change what this study is about

**Bogdan Mironov**
Interim synthesis. Two campaigns (encoder generality, decomposed objective) are
still on the cluster; this covers what is finished. Every result below has a
pre-registration committed before its data existed.

---

## The short version

The study's headline — *message passing over a Vietoris–Rips complex adds
**+0.0409** adjacent-pair R² to the best no-topology stack* — **survives its own
bootstrap correction and does not survive a stricter definition of "identical
conditions"**. Alongside that, four other measurements were made that nobody had
made, and together they reframe the problem:

| # | measurement | value | what it changes |
|---|---|---|---|
| 1 | **Ceiling** on adjacent-pair R² | **+0.679** | the best model is at **39 %** of attainable; +0.412 of headroom exists |
| 2 | **Block-key robustness** | +0.0375 **adds** / +0.0177 **n.s.** | the effect is a *binned-key* effect; claim downgraded |
| 3 | **Reference energetics** (957 complexes, finally computed) | −0.2993, significant | energetics *hurt* selectivity; the barrier is conformer scatter, **not** theory level |
| 4 | **Magnitude compression** | recalibration gain +0.0087 [+0.0003, +0.0200], not replicated | compression is **shrinkage**, not miscalibration, and is not fixable post hoc |
| 5 | **Run-to-run noise floor** | **0.0000** (bit-identical) | the "irreducible noise" premise was false; sweeps can now select |

The through-line: **this dataset's adjacent-pair signal is small relative to its
measurement noise, and most apparent gains are artefacts of how the metric,
the features or the training loop were measured.** The contribution of the paper
is as much the method for telling those apart as any single effect.

---

## 1. How much of this metric is attainable at all

Nobody had asked. Against a ceiling of 1.0 the best model (+0.2672) is poor;
against 0.35 it is near-optimal. Those imply opposite next steps.

A **model-free** estimate: take the same adjacent pair, same extractant, same
binned block, measured at two or more genuinely different *exact* condition sets.
Under the binned key those rows collapse into one cell sharing one feature
vector, so their disagreement is irreducible.

> **Ceiling: adjacent-pair R² ≤ +0.679.** The best published model reaches
> **39 %** of it. Headroom: **+0.412**.

Under the strict key the same rows are separate blocks with different feature
vectors, so part of that disagreement is condition dependence a model can in
principle predict — making **+0.530 a floor rather than a cap** there.

**A data-quality finding fell out of it.** Two other estimators returned negative
"ceilings" and are reported as failures with their cause: a (block, metal) cell
acquires a duplicate row *precisely when two sources report the same system and
disagree*, so replicated cells are the worst case rather than a sample — their
pooled scatter (Var 0.15 on the difference) exceeds the entire observed spread of
the quantity being predicted (Var 0.073). They measure source conflict, not
measurement precision. [`ceiling_test.csv`]

## 2. The headline is a binned-key result

`adjacent_pair_metrics` blocks by `composition_key`, built from **binned**
condition columns. `strict_composition_key` — every numeric condition matched —
has been in `dataset.py` since the beginning, with a comment saying the binned
key *"turns a real log D difference into label noise"*. The metric had never been
computed with it.

Pre-registered decision rule, committed before the contrasts existed:

| contrast | binned | strict |
|---|---|---|
| drop-in (add S0 to the best no-topology stack) | **+0.0375** [+0.0173, +0.0510] · 10-look [+0.0111, +0.0639] **adds** | +0.0177 [−0.0023, +0.0367] **n.s.** |
| swap (S0 vs matched control, same slot) | **+0.0438** [+0.0253, +0.0554] · 10-look [+0.0203, +0.0673] **adds** | +0.0177 [−0.0023, +0.0367] **n.s.** |

Three things that stop this being over-read, all in
[`DUALKEY_RESULTS.md`](DUALKEY_RESULTS.md):

- **A weakening, not a reversal.** The strict estimate is positive at P = 0.93 and
  about half the size.
- **The bootstrap correction survives.** These *are* the multiplicity-respecting
  intervals — the ones the published draw made 12–29 % too narrow. What breaks
  the result is the block definition, not the resampling.
- **It cuts both ways.** Under the strict key the matched control T0w takes stack
  weight **0.00** (which is why the two contrasts collapse onto one number), and
  the *single-arm* encoder comparison S0 − T0w **strengthens**: +0.0376 → +0.0716.

Neither key is clean. The binned one mixes conditions; the strict one turns 552
blocks into 2,109 and discards the replicate averaging, so every arm drops. **Both
columns get reported from here on.**

## 3. The energetics were computed, and they make selectivity worse

957 rows of reference xTB calculations had sat queued and uncomputed for the whole
study; `FINDINGS.md` called them "the most promising untested feature available".
There was **not one energetic descriptor in the design matrix**, though a
separation factor *is* a difference of complexation free energies.

They are now computed for **953 of 956** geometries.

| | adj R² binned | adj R² strict | overall log D R² |
|---|---|---|---|
| baseline CatBoost | +0.1422 | +0.0819 | +0.4987 |
| + energy block | **−0.0350** | **−0.1994** | **+0.5068** |

Strict-key contrast **−0.2993 [−0.4566, −0.1792]**, surviving Bonferroni for all
16 looks. Overall accuracy improves to the best value in the study, reported
separately because the pre-registration fixed that split in advance.

**The mechanism is measured, not narrated.** A gate run *before* the campaign
substituted all 14 lanthanides into one **frozen cage** and found adjacent members
separated by 0.306 eV — 17.2× the 0.0178 eV a separation factor of 2 corresponds
to. The pre-registration said what that did and did not establish: *"it rules out
a specific failure mode; it is not evidence for the hypothesis."*

It held the geometry fixed. The dataset does not — every complex is one stochastic
conformer. Within a ligand family:

| feature | trend per series step | conformer scatter | SNR |
|---|---|---|---|
| `e_int_octanol` | 0.201 eV | 0.756 eV | **0.25** |
| `e_int_water` | 0.170 eV | 0.731 eV | **0.25** |
| `dg_transfer` | 0.014 eV | 0.099 eV | **0.17** |

98–100 % of families sit below SNR 1. The incumbent these must displace,
`Ionic Radius_metal`, is a **lookup table with zero scatter by construction**. A
tree only compares values, so it takes the noisy proxy and selectivity collapses.

**The useful consequence: the barrier is not the level of theory.** Nothing here
says GFN2's energies are wrong — only that one draw from a 0.73 eV distribution
cannot resolve a 0.20 eV step. DFT on single conformers would inherit the same
problem. *"Use DFT"* is the expensive conclusion and it is **not** the one the
data supports. Conformer averaging is, and it now has a target: **3.9× scatter
reduction, ≈16 effectively independent conformers**.
[`ENERGY_RESULTS.md`](ENERGY_RESULTS.md)

## 4. The compression is shrinkage, and cannot be rescaled away

Predictions span ~0.42× the true spread of adjacent-pair separations. A nested
per-extractant recalibration (scale / affine / isotonic) buys:

| key | gain | interval | verdict |
|---|---|---|---|
| binned | +0.0087 | [+0.0003, +0.0200] | clears zero by 3×10⁻⁴, best of three transforms, uncorrected |
| strict | −0.0004 | [−0.0137, +0.0119] | coin flip |

**Not established.** A best-of-three maximum at P = 0.95 that fails to replicate on
a second metric definition is the same object as the persistence-image "+0.0178
tuning gain" that replication reduced to +0.0003.

The real result is that recalibration **cannot** repair the compression even when
free to: span goes 0.42× → 0.53×, and isotonic — free to fit any monotone map —
makes R² *worse* on five of six models. With the ceiling, that is one sentence
instead of two findings:

> The models recover ~39 % of the attainable variance in adjacent-lanthanide
> separation and predict ~40–50 % of its true spread. Both follow from the signal
> being small relative to measurement noise, and neither is repaired post hoc.

[`CALIBRATION_RESULTS.md`](CALIBRATION_RESULTS.md)

## 5. The noise floor was not irreducible

`PI_SWEEP_PRECISION.md` measured an 8-seed ensemble moving by **0.0092** between
identical re-runs, showed more seeds does not fix it, and drew the design lesson
*"when per-measurement noise is irreducible, buy precision with design rather than
repetition"*.

The premise was false. Two runs of the identical S0 configuration now return
**byte-identical** out-of-fold vectors, `max |diff| = 0.000e+00`.

The published diagnosis blamed cuDNN autotuning — correct for the persistence-image
CNN, wrong for the simplicial network, **which has no convolutions at all**. Its
nondeterminism was `index_add_` atomics, so the fix had to be a reduction, not a
backend flag. Cost: ~7× slower, against a `GrpTRES` cap of one GPU node — so it is
for confirmatory runs, not exploratory sweeps.
[`DETERMINISM_RESULTS.md`](DETERMINISM_RESULTS.md)

---

## What is still running

| campaign | question | status |
|---|---|---|
| **encoder** (G0 `--no-triangles`, D0 `--arch dist`) | is *simplicial* or merely *3D message passing* the operative ingredient? — the study's oldest open question | 32 GPU runs, in flight |
| **objective** (level-weight × block-key) | the loss spends 68–91 % of its gradient on the block mean the metric never reads; does splitting it help? | 48 GPU runs, queued |
| **conformer pilot** | can metadynamics cut the 0.73 eV conformer scatter 3.9×? | CREST-lite smoke running |

## Two corrections to my own work, recorded

- **The compute plan was wrong.** It assumed 40 concurrent GPU jobs from
  `MaxSubmitJobs`; `GrpTRES` caps this account at **one node** on
  `xeon-g6-volta` (two concurrent) and two on `xeon-p8`. A 648-cell factorial was
  planned and abandoned for a 6-cell one sized to the real limit.
- **I wrote the corrected bootstrap the slow way.** 75 minutes without finishing,
  because it redid a groupby over 2,109 blocks 3,200 times. It did not need to: a
  twice-drawn extractant becomes two blocks holding the same rows, producing the
  same pairs twice — so **a single `np.unique` is the entire difference between an
  m-out-of-n subsample and a cluster bootstrap.** ~200× faster, and asserted
  against the literal statistic at 1e-9 rather than argued.

## Integrity

`control_guard --verify` passes on all 324 frozen artefacts after every change.
The standing precondition holds throughout: the published S0 ensemble re-scores to
exactly **+0.2382**. Nothing in `data/` was written. Existing reports received
appended errata rather than edits. Five pre-registrations
(`DUALKEY`, `ENCODER`, `ENERGY`, `OBJECTIVE`, plus the amendment) were committed
before their data existed.
