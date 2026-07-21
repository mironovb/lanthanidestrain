# Is this publishable? An honest assessment

> **Update, 21 July 2026 — blocker §3.1 is closed and §3.2 turned out larger than §3.1.** See [`CONTROL_RESULTS.md`](CONTROL_RESULTS.md). The verdict below stands; the recommended framing does not. Every number in this document is unchanged and reproduces exactly.

Written to be read by a supervisor and reused in a cover letter. Target as
stated: a **standalone methods paper**.

**Verdict up front: the internal result is solid, and it is not yet a standalone
methods paper.** One control is missing, and it is the one a reviewer will ask
for first. Closing it is ~1 day of compute. It is also the paper's best
opportunity, for reasons in §3.

---

## 1. What the result is

Adjacent-lanthanide separation R² rises from **+0.005** (FCNN on ECFP + RDKit)
to **+0.263**, on 4,746 measurements across 162 extractants and 14 lanthanides
under leave-extractants-out CV.

| test | seeds | Δ adjacent-pair R² | 90 % CI | P(better) |
|---|---|---|---|---|
| SNN ensemble vs **FCNN** | 16 | **+0.2426** | [+0.181, +0.333] | 1.00 |
| PI-CNN ensemble vs **FCNN** | 15 | +0.1984 | [+0.107, +0.266] | 1.00 |
| SNN ensemble vs **CatBoost** (standalone) | 16 | +0.0867 | [+0.025, +0.122] | 0.99 |
| SNN blend vs **CatBoost** (pre-registered w = 0.5) | 16 | +0.1004 | [+0.038, +0.140] | 1.00 |
| SNN blend vs **CatBoost** (nested, leakage-free) | 16 | **+0.1074** | [+0.039, +0.150] | 1.00 |

---

## 2. What is genuinely strong

- **The statistics are the right ones.** Every interval is a paired cluster
  bootstrap resampling whole *extractants* — the correct unit, since rows within
  an extractant are not independent. A row-level interval would be far too
  narrow.
- **Pre-registration.** The blend weight was fixed at 0.5 *before* the curve was
  computed. A separate nested variant chooses the weight per extractant using
  only the other 161, so no row influences the weight it is scored under. Both
  agree; the selected weight is stable at **zero-width IQR**.
- **Architecture-independent replication.** SNN +0.1972 and PI-CNN +0.1968 agree
  to 0.0004 while sharing no layers, readout, or inductive bias beyond the
  underlying geometry.
- **Negative controls fire.** `pi_topoonly` (−1.74) and `snn_allatom` (−0.41)
  return "worse" with intervals excluding zero. A test that can only win is not
  measuring anything.
- **Mechanism predicted before measurement.** The contrast loss was added
  *because* the metric scores differences while the objective optimised
  absolutes — not fitted to the result afterwards.
- **Complementarity without a p-value.** The blend curve has an interior maximum
  (+0.264 at w ≈ 0.7) above *both* endpoints. Models carrying identical
  information interpolate monotonically; only complementary information peaks in
  the middle.
- **Reproducible.** 115 tests pass, 0 tracked files modified, `data/` untouched.

---

## 3. What blocks publication

### 3.1 The missing control — decisive

Across all 51 topology runs (17 distinct configurations), **every arm uses a
topological encoder**: 27 `snn`, 24 `picnn`, zero without one. There is no arm
with *tabular features + contrast loss + adjacent selection + ensembling but no
topology*.

Since the identified mechanism is "train the contrast," a contrast-trained
tabular model may capture most of the +0.24. If so the finding is about the
**objective**, not topology, and the abstract's central claim is wrong even
though every number above is correct.

**This is also the opportunity.** If contrast-training alone explains the gain,
that is a *more* generalisable contribution — it transfers to any
separation-factor problem, in any domain, independent of persistent homology.
Either way the paper gets stronger. Only the title changes.

### 3.2 Asymmetric comparison — serious

| | FCNN baseline | topological arm |
|---|---|---|
| hyperparameters | sklearn defaults `(256,128)`, never tuned | swept over 17 configurations |
| ensembling | single model | 16-seed ensemble |
| objective | plain MSE | pairwise-contrast loss |
| early stopping | generic validation | selected on adjacent-pair R² |

Every one of those four differences favours the topological arm. The CatBoost
comparison is fairer (CatBoost was tuned in the prior study) and the topological
model still wins there — which is the stronger result to lead with.

### 3.3 The models under-predict separation magnitude

Visible in `topo_adjacent_parity.png`: true Δ log D spans roughly ±2 log units,
while predictions span ±0.5. Both models regress hard toward zero. The SNN wins
on R², but neither would tell a chemist *how much* separation to expect — only,
weakly, in which direction. **This limits the practical claim** and should be
stated rather than left for a reader to notice in the figure.

### 3.4 Smaller items, all disclosable

- **Selection on the reported metric.** `--select-on adjacent` tunes checkpoints
  for adjacent-pair R². Legal — inner validation only, folds grouped by
  extractant — but it must be disclosed prominently, not buried in methods.
- **Low-baseline reference.** The FCNN's +0.005 means it barely beats predicting
  the composition mean; large *relative* gains are easier against ~0.
- **Multiple comparisons.** 17 configurations were tried before the winner
  emerged.
- **Overall accuracy is not improved** — best topological arm 0.375 vs CatBoost
  0.499; the gain costs ~0.06 overall R² at the operating point.
- **Single dataset**, 162 groups, no external validation.
- **Ensembling is mandatory.** Single models are unstable (SNN seed SD 0.047;
  one seed scored +0.066 against another's +0.197). Honest, but it means the
  method is "train 16 models," not "train a model."

---

## 4. Recommendation

**Framing:** *the objective is the contribution, topology is the application.*
That framing survives either outcome of the control, and it is the more
generalisable claim.

**Before submission, in order:**
1. The tabular-only contrast control (~2 h). Determines the title.
2. Symmetric baselines — contrast-trained and seed-ensembled FCNN and CatBoost
   (~3 h). Removes §3.2 entirely.
3. State §3.3 (magnitude compression) in the results, with the parity figure.

**Realistic venue if the control holds:** a methods-focused ML-for-chemistry
venue, or a strong section of the larger lanthanide paper. The effect is real
and well-measured, but it is one derived metric on one dataset, and overall
accuracy does not improve — that is not a high-impact standalone story on its
own.

**If the control shows topology adds nothing beyond the objective:** publish it
as the contrast-training result with topology as a *negative* control. That is
still novel, more broadly useful, and considerably harder to dismiss.

---

## 5. Errors found and corrected during this work

Recorded because they bear on how much to trust the rest, and because two were
caught only by explicitly checking rather than by anything failing loudly.

| what | how it was caught | status |
|---|---|---|
| MPSN broke permutation invariance | invariance test; confirmed real in float64 (identical 2.6e-4) | fixed |
| 292 non-finite xTB charges NaN-poisoned 3 complexes | asset audit | fixed, imputed + flagged |
| Metal probe scored R² = 0.9995 by reading the element token | result was implausibly perfect | retracted; masked probe added |
| Parity figure re-enumerated pairs, **inverting the result** (baseline appeared to win) | rendering the figure and reading it | fixed; metric and figures now share `adjacent_pair_arrays` |
| Seed figure claimed "above every seed" — one seed beats the ensemble | checking the claim against the data | corrected |
| Palette: orange/green collapse under protanopia (ΔE 3.2) | ported CVD validator | replaced; no 4-colour subset passes, so shape carries the 4th distinction |
| Two architecture recommendations (PI-CNN over SNN) | matched-seed replication | reversed |
| Tight-geometry conclusion drawn from n = 2 seeds | third seed moved the mean | corrected |
| Octanol failures diagnosed as timeout | 3× timeout recovered 2 of 86; log said SCF non-convergence | re-diagnosed |
| Three verdict thresholds fired wrongly | comparing verdict strings against their own numbers | documented as calibrated for larger effects |

The measurements were correct in every case; the *interpretations* were what
needed fixing. That is the pattern to watch for in the remaining work.
