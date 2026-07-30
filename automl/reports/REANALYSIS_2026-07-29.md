# Re-analysis, 29 July 2026: nine measurements, and what is left of the claim

**Bogdan Mironov**
Every result below has a pre-registration committed **before its data existed**.
All nine campaigns are complete.

---

## The short version

The study's headline — *message passing over a Vietoris–Rips complex adds
adjacent-lanthanide selectivity that 2D fingerprints lack* — survives its own
bootstrap correction, and **two of its three load-bearing words do not survive
scrutiny**. It is not specific to a *simplicial* representation, and it is
specific to a *binned* definition of "identical conditions" that the codebase
itself flags as defective.

| # | measurement | result | what it changes |
|---|---|---|---|
| 1 | **Ceiling** on the metric | **+0.679** | best model is at **39 %**; +0.412 headroom exists |
| 2 | **Block-key robustness** | +0.0375 **adds** / +0.0177 **n.s.** | the effect is a *binned-key* effect |
| 3 | **Third encoder** | S0 vs D0 = **+0.0091, n.s.** | **it is 3D message passing, not simplicial** |
| 4 | **Reference energetics** (957 complexes) | **−0.2993**, significant | energetics *hurt*; barrier is conformer scatter, not theory |
| 5 | **Conformer ensembles** | SNR **0.29** even if perfect | the last physics lever is **closed** |
| 6 | **Magnitude compression** | gain +0.0087, not replicated | **shrinkage**, not miscalibration |
| 7 | **Run-to-run noise floor** | **0.0000** | the "irreducible" premise was false |
| 8 | **Recombination** of all 15 arms | **+0.2604** vs +0.2672 | stacking is exhausted |
| 9 | **Decomposed objective** | level weight helps *monotonically* | the loss was not the constraint; my diagnosis pointed the wrong way |

**The through-line.** This dataset's adjacent-pair signal is small relative to
its measurement noise, and most apparent gains are artefacts of how the metric,
the features, or the training loop were measured. **Five of the nine results above
are falsifications of things I or the study believed at the start of the day**,
four of them mine. The method for telling those apart is at least as much the
contribution as any single effect.

**Four routes to the remaining headroom are now closed with numbers**: a better
combination (§8), better features (§4), better geometries (§5), and a better
objective (§9). What is left is the data itself -- 953 distinct complexes, 905
adjacent pairs, and a ceiling of +0.679 that nothing here reaches half of.

---

## 1. How much of this metric is attainable

Nobody had asked. Against a ceiling of 1.0 the best model is poor; against 0.35
it is near-optimal, and those imply opposite next steps.

A **model-free** estimate — the same adjacent pair, same extractant, same binned
block, measured at two or more genuinely different *exact* condition sets, whose
disagreement is irreducible because they share one feature vector:

> **Ceiling: adjacent-pair R² ≤ +0.679.** The best model reaches **39 %** of it.

Two other estimators returned negative "ceilings" and are reported as failures
with their cause, which is a data-quality finding: a cell acquires a duplicate
row *precisely when two sources report the same system and disagree*, so
replicated cells are the worst case rather than a sample. Their pooled scatter
exceeds the entire observed spread of the predicted quantity.
[`ceiling_test.csv`]

## 2. The headline is a binned-key result

`adjacent_pair_metrics` blocks by `composition_key`, built from **binned**
condition columns. `strict_composition_key` has been in `dataset.py` since the
beginning with a comment saying the binned key *"turns a real log D difference
into label noise"*. The metric had never been computed with it.

| contrast | binned | strict |
|---|---|---|
| drop-in | **+0.0375** [+0.0173, +0.0510] · 10-look **adds** | +0.0177 [−0.0023, +0.0367] **n.s.** |
| swap | **+0.0438** [+0.0253, +0.0554] · 10-look **adds** | +0.0177 [−0.0023, +0.0367] **n.s.** |

A weakening, not a reversal (P = 0.93, half the size). The bootstrap correction
*survives* — these are the multiplicity-respecting intervals — so what breaks the
result is the block definition, not the resampling. And it cuts both ways: under
the strict key the matched control takes stack weight **0.00**, and the
single-arm encoder comparison S0 − T0w *strengthens*, +0.0376 → +0.0716.
[`DUALKEY_RESULTS.md`](DUALKEY_RESULTS.md)

## 3. It is not "simplicial" — the oldest open question, answered

Two new encoders over the **same** VR edges, same node inputs, same readout, same
16 seeds: **G0** with the triangles removed, **D0** a continuous-filter distance
network with no simplices, no boundary maps, no filtration.

| contrast (binned) | Δ | 13-look Bonferroni |
|---|---|---|
| add **G0** | **+0.0343** | [+0.0117, +0.0569] **adds** |
| add **D0** | **+0.0284** | [+0.0025, +0.0543] **adds** |
| **S0 vs D0, same slot** | +0.0091 | [−0.0292, +0.0474] **not distinguishable** |

As single arms both *outscore* the published simplicial one: D0 **+0.2474**,
G0 **+0.2459**, S0 +0.2382.

**The mechanism rule survives; the claim about the complex does not.** An arm
earns a slot only if it is both accurate and decorrelated — all three 3D encoders
satisfy it (err-corr 0.892–0.904), the persistence-image CNN (0.933) and the
tabular control (0.929) do not. Consequences: the model can drop a 9.3 M-triangle
level and a 46 MB cache; the persistence-image null reads better as "fixed loses
to learned"; and the 3.0/3.5/4.0 Å filtration replication worked because those
define the same neighbourhood graph.
[`ENCODER_RESULTS.md`](ENCODER_RESULTS.md)

## 4. The energetics were computed, and they make selectivity worse

957 rows had sat queued and uncomputed for the whole study; there was **not one
energetic descriptor in the design matrix**, though a separation factor *is* a
difference of complexation free energies. Now computed for **953 of 956**.

| | adj binned | adj strict | overall log D |
|---|---|---|---|
| baseline CatBoost | +0.1422 | +0.0819 | +0.4987 |
| + energy block | **−0.0350** | **−0.1994** | **+0.5068** |

Strict contrast **−0.2993**, surviving Bonferroni for all 16 looks. Overall
accuracy improves to the study's best, reported separately because the
pre-registration fixed that split in advance.

**The mechanism is measured.** A gate run *before* the campaign found GFN2
separating adjacent lanthanides by 0.306 eV in a **frozen cage** — 17× the
relevant scale. But the dataset's geometries are not frozen: within a ligand
family every energy feature carries the trend at **SNR ≈ 0.25**, against an
incumbent (`Ionic Radius_metal`) that is a lookup table with zero scatter. A tree
takes the noisy proxy and selectivity collapses.
[`ENERGY_RESULTS.md`](ENERGY_RESULTS.md)

## 5. The conformer lever is closed

Named in three reports as the untested physics lever, always qualitatively. A
CREST-lite metadynamics pilot over 120 complexes in 9 whole ligand families:

1. **The search works** — median 16 unique conformers, none degenerate.
2. **Boltzmann weighting cannot use it** — effective ensemble size **1.17**,
   because gaps are ~40× kT. A Boltzmann average of this ensemble *is* its
   minimum. That falsified the remedy the pilot was designed around.
3. **The surviving hypothesis is confirmed** — 79 % of shipped geometries are not
   the global minimum, and the within-family SD of the gap is **0.434 eV**, 59 %
   of the scatter.
4. **And it is still not enough** — removing that entirely leaves 0.588 eV
   against a 0.170 eV signal: **SNR 0.29**.

So a full campaign, succeeding completely, cannot rescue these features. **DFT on
single conformers inherits the same problem**, because the problem is the
conformer, not the Hamiltonian — evidence against the expensive next step.
[`CONFORMER_RESULTS.md`](CONFORMER_RESULTS.md)

## 6. The compression is shrinkage

Nested per-extractant recalibration of the predicted difference:

| key | gain | interval |
|---|---|---|
| binned | +0.0087 | [+0.0003, +0.0200] — clears zero by 3×10⁻⁴, best of three transforms, uncorrected |
| strict | −0.0004 | [−0.0137, +0.0119] — coin flip |

**Not established.** The real result is that recalibration *cannot* repair the
compression even when free to: span 0.42× → 0.53×, and isotonic makes R² worse on
five of six models. [`CALIBRATION_RESULTS.md`](CALIBRATION_RESULTS.md)

## 7. The noise floor was not irreducible

`PI_SWEEP_PRECISION.md` measured an 8-seed ensemble moving 0.0092 between
identical re-runs and drew the lesson *"buy precision with design, because the
noise is irreducible"*. The premise was false: `--deterministic` now returns
**byte-identical** out-of-fold vectors.

The published diagnosis blamed cuDNN autotuning — right for the persistence-image
CNN, wrong for the simplicial network, **which has no convolutions at all**. Its
nondeterminism was `index_add_` atomics. Cost ~7×, so it is for confirmatory runs.
[`DETERMINISM_RESULTS.md`](DETERMINISM_RESULTS.md)

## 8. Recombination is exhausted

Nested forward selection — arms, order and weights all chosen per held-out
extractant — over all **15** arms on disk gives **+0.2604**, against the published
three-arm **+0.2672**. Worse.

So the +0.412 of headroom is **not reachable by recombining what exists**. It
needs a model that is right about something none of these are. [`full_stack.csv`]

---

## What a paper should now claim

1. **A learned 3D representation supplies adjacent-lanthanide selectivity that 2D
   fingerprints do not**, worth ~+0.03 to +0.04 R² in the best stack **under the
   binned metric**, and not distinguishable from zero under the strict one.
2. **The representation family is broad, not narrow.** Simplicial complexes,
   their underlying graphs, and continuous-filter distance networks all work
   equally; fixed persistence images do not. The transferable object is the
   **rule** — accurate *and* decorrelated — not the complex.
3. **The problem is measurement-limited, and by how much is now known.** Ceiling
   +0.679, best model 39 % of it, predictions at 40–50 % of the true spread,
   neither repairable post hoc.
4. **Three routes to the headroom are closed with numbers**: recombination,
   energetics, and conformer search.

## Two corrections to my own work

- **The compute plan misread the cluster.** It assumed 40 concurrent GPU jobs from
  `MaxSubmitJobs`; `GrpTRES` caps this account at **one node** (two concurrent). A
  648-cell factorial was planned and abandoned for a 6-cell one.
- **I wrote the corrected bootstrap the slow way** — 75 minutes without finishing.
  A twice-drawn extractant becomes two blocks holding the same rows, so **a single
  `np.unique` is the entire difference between an m-out-of-n subsample and a
  cluster bootstrap**. ~200× faster, asserted at 1e-9 rather than argued.

And one confound in my own design, caught before it ran: the contrast objective
only batches blocks with ≥2 rows, which would have given the strict-key cells 67 %
of the rows against the binned cells' 96 %. Fixed for the decomposed objective and
recorded as Amendment 1 — including the part *not* fixed, that every published
topological arm still never sees its 202 singleton-block rows.

## Integrity

`control_guard --verify` passes on all **324** frozen artefacts after every
change. 145 tests pass. The frozen tune/confirm split still hashes to
`6070dc55…`. The standing precondition holds throughout: published S0 re-scores
to exactly **+0.2382**. Nothing in `data/` was written. Existing reports received
appended errata rather than edits. Six pre-registrations plus two amendments were
committed before their data existed.

---

## 9. The objective was not the constraint either — and my diagnosis pointed the wrong way

**Added 29 July 2026, on completion of the 48-run sweep.**
[`OBJECTIVE_RESULTS.md`](OBJECTIVE_RESULTS.md)

§1 of `OBJECTIVE_PREREGISTRATION.md` measured that the published loss spends
**68–91 % of its gradient** on the block mean the metric never reads. The
arithmetic was right; the inference was wrong. The `level_weight` main effect is
monotone in the **opposite** direction to the one predicted:

| `level_weight` | mean tune adj R² (binned), 16 runs each | overall log D |
|---|---|---|
| 0.1 | +0.2286 | +0.289 |
| 0.3 | +0.2351 | +0.335 |
| **1.0** | **+0.2412** | **+0.385** |

**More weight on the block mean is better.** The level term is not wasted
gradient — it anchors the representation, and a network told only to get contrasts
right places the blocks worse and learns a worse encoder for it.

Training against the strict key is also worse (+0.2298 vs +0.2400), even after
Amendment 1 restored the singleton blocks. So the strict key is a better *metric*
and a worse *training signal*; those are separate questions and this separates
them.

Confirmatory, on the 78 confirm extractants: the decomposed arm **does not beat
S0** (−0.0233 binned, −0.0170 strict). It does add to the no-topology stack, and
under the strict key that survives 19-look Bonferroni — **+0.0150 [+0.0011,
+0.0289]**, the only strict-key contrast in this re-analysis that does. That is one
look on 78 extractants and the direct OBJ-vs-S0 comparison is a null, so it is
reported as narrow and in need of replication, not as a win.

**This closes the fourth route to the headroom.** Recombination, features,
geometries and the objective are all now measured and closed. The
pre-registration said a negative here points at *the representation or the data*;
`ENCODER_RESULTS.md` has since shown the representation family is broad and
interchangeable, which leaves **the data** — 953 complexes, 905 adjacent pairs,
and a ceiling of +0.679 that nothing here reaches half of.

**Five of the nine results in this document are falsifications of things believed
at the start of the day**, four of them mine.
