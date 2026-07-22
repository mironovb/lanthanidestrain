# Pre-registration: does a *tuned* persistence image contribute?

**Bogdan Mironov · 22 July 2026**
Written and committed **before any sweep run existed**. The frozen split, the
configuration grid, the selection rule, the endpoint, the multiplicity count and
the meaning of every outcome are fixed here.

---

## 1. Why this test exists

The study's positive result is that message passing over a Vietoris–Rips complex
adds **+0.0381** adjacent-pair R² to the best no-topology stack. A
persistence-image CNN (P0) did **not** replicate it (−0.0041, n.s.).

Every scope statement in this repository declines to draw a conclusion from that,
for a stated reason: the images used **the shipped asset's fixed settings —
resolution 20, spread 0.08, range (0, 2.5), H0+H1 summed — never tuned at any
point in this study**. That bounds what has been *demonstrated*, not what is
*possible*.

The PI (Kostas) has since named the two axes to benchmark: **image resolution**
("we usually go from 20×20 up to 150×150 (not recommended) … benchmark this
hyperparameter") and **Gaussian spread** ("becomes important when we have many
points in the persistence diagrams"). He attached publications describing the
procedure; **those attachments are not in this repository and were not read**, so
this grid follows the two axes he named explicitly plus three the diagnostics
below implicate.

### Diagnostics, measured before the grid was chosen

Over 120 complexes / 59,171 persistence points:

| observation | value |
|---|---|
| Gaussian spread ÷ pixel spacing | 0.08 ÷ 0.132 = **0.61** — each point lands in ~one pixel, so the "image" is a sparse histogram |
| persistence points outside the (0, 2.5) window | **13.5 %** discarded; deaths reach 20.6, p95 = 3.24 |
| kept points with persistence < 1 pixel, under `weight = death − birth` | **24.7 %**, effectively invisible |
| channel layout | H0 (death median 0.30) and H1 (death median 1.98) **summed into one plane** |
| points per complex | ~493 — Kostas's "many points" regime |

Under the study's mechanism — an arm helps only if **both** strong on the metric
**and** decorrelated from its partners — P0 currently fails both: **+0.2101** and
error correlation **+0.933**. To contribute it must reach roughly ≳0.23 and
≲0.91. That is the quantitative target, stated in advance.

---

## 2. The frozen split, and why it is 50/50

A sweep picks a winner, so the winner's score is optimistically biased. The
protection is a split of **extractants**: the sweep sees only the tune half, and
the single winning configuration is scored once on the confirm half.

The ratio was calibrated from data, using the **already-published S0 arm** — a
different model from the one under test — before any sweep run existed:

| candidate split | confirm side | S0's known effect measured there |
|---|---|---|
| 2/3 – 1/3 | 18 ext / 302 pairs | +0.0398 **[−0.0005, +0.0549]** — fails |
| **50 / 50** | **78 ext / 453 pairs** | **+0.0415 [+0.0185, +0.0519]** — clears |

Only **76 of 162** extractants have any adjacent pair and the top five carry
~36 % of them, so the effective cluster count is small. A one-third confirm side
cannot detect an effect we already know is real, and would therefore report a
null whatever the sweep found.

**Frozen now** (`automl/artifacts/pi_sweep/split.json`):

```
rule    pairs-balanced snake draft on adjacent-pair count, ordered by (-pairs, name);
        zero-pair extractants alternated by name
TUNE    84 extractants / 452 pairs
CONFIRM 78 extractants / 453 pairs  (50.0 %)
sha256  6070dc55ee5ff2d0e285f3169e18e8fe9184b829408e788f1d131868f75f74ec
```

`pi_split.py --verify` recomputes the rule and refuses to proceed if the hash
moves.

---

## 2a. AMENDMENT, 22 July — how the tune half is used

**Superseded:** the sweep was to be confined with `train.py --restrict-groups`,
so that sweep models never *trained* on confirm extractants.

**Amended to:** every sweep configuration trains on **all 162 extractants**;
selection is scored on **tune-half rows only**. The winner is then evaluated on
confirm-half rows.

### Why — the original design does not work, measured

The first complete 8-seed read showed the flaw. `--restrict-groups` removes 57 %
of the training rows (4,742 → **2,030**; 953 → **494** complexes), and the
persistence-image CNN cannot survive that. Scored on identical tune-half rows:

| the same representation | trained on | tune-half adj R² |
|---|---|---|
| published P0 | all 4,742 rows | **+0.1562** |
| sweep anchor | 2,030 tune rows | **+0.0362** |

At +0.036 against a tune-half baseline stack of **+0.2473**, the stack assigns
**every** configuration weight 0.00 and gain exactly **+0.0000**. All 25 Stage A
configurations tie, and the selection rule cannot rank anything. The design was
unable to answer its own question.

Two further reasons the amended form is the correct one:

* **Comparability.** The published S0, P0 and T0w arms were all trained on the
  full dataset. An arm trained on 2,030 rows is not comparable to them, so even a
  clean tune-half ranking could not be carried across to the confirmatory test.
* **Transfer.** The optimal smoothing at 2,030 rows need not be the optimal
  smoothing at 4,742. Selecting in a crippled-data regime and hoping the choice
  transfers is a poor inference even when the ranking is well estimated.

### What this costs, stated honestly

The original claim was that selection *never saw* the confirm extractants in any
capacity, so no multiplicity penalty could be warranted. That is now weaker:
sweep models do train on confirm extractants in other folds.

The endpoint remains unbiased, and this is the standard selection-on-a-holdout
argument: the winner is chosen as a function of **tune-row outcomes only**, and
the reported statistic is computed from out-of-fold predictions for **confirm
rows**, made by models that never saw those rows. No confirm-row label
influences which configuration is selected, so the confirm estimate is unbiased
for the selected configuration.

What is lost is the stronger, simpler guarantee. `N_LOOKS` stays at **8** on that
basis, and because the tune and confirm halves are disjoint in *rows scored*. To
keep this falsifiable rather than merely argued, `PI_SWEEP_RESULTS.md` will
additionally report the interval under a deliberately punitive
`N_LOOKS = 8 + 57` — one look per configuration swept — so a reader who rejects
the argument above can still see whether the conclusion would survive without it.

### A result in its own right

The collapse from **+0.1562 to +0.0362** on 57 % fewer training rows is worth
recording independently: the persistence-image CNN is severely data-limited on
953 images. That is a plausible partial explanation for why P0 underperforms S0
generally, and it is a concrete argument that the representation is not yet at
its ceiling — which is the question this sweep exists to probe.

The 66 tune-trained runs are **kept**, not discarded; they are the evidence for
this amendment.

---

## 3. Positive control — the endpoint is gated on it

**The confirm half must reproduce S0's known effect before any
persistence-image number is reported: +0.0415 [+0.0185, +0.0519].** Frozen in
`automl/artifacts/pi_sweep/positive_control.csv`.

If it does, the harness demonstrably has the power to see an effect of that size,
and a null from the persistence-image arm is interpretable. If it does not, the
test is void and no conclusion may be drawn in **either** direction.

---

## 4. A provenance finding that changes the comparison point

The gate on the diagram cache found that **18 of the 953 complexes cannot be
reproduced bit-for-bit from the shipped asset in this environment** — and that
this is not a defect in the cache. Calling the shipped `persistence_diagram` /
`persistence_image` *directly* on the same coordinates reproduces each mismatch
exactly and deterministically under gudhi 3.13.0. All 18 are large (268–361
atoms), where alpha-complex construction is most sensitive to the CGAL
predicates. The shipped asset was built under a different gudhi/CGAL. Recorded
in `automl/artifacts/pi_sweep/shipped_reproduction.json`; 935/953 match exactly.

**Consequence, fixed in advance:** the untuned comparison point is the **shipped
configuration rendered from the same cache** (the *reproduction anchor*), not the
shipped `.npz`. Every tuned-versus-untuned comparison is then between two image
sets built the same way in the same environment. The anchor is swept like any
other configuration and doubles as a reproduction check: it should land near
P0's published **+0.2101**.

---

## 5. The grid, fixed now

**Stage A — the benchmark Kostas asked for.** Resolution × spread, at the shipped
range and channel layout, so the curve is clean. 150×150 is excluded: he advises
against it, and 953 images do not support that many free pixels.

```
resolution  20, 32, 48, 64, 96, 128
spread      {0.5, 1, 2, 4} x pixel spacing   (pixel-relative, so it means the
                                              same thing at every resolution)
plus        the reproduction anchor (res 20, spread 0.08 = 0.61 pixels)
                                              -> 25 configurations
```

**Stage B — the axes the diagnostics implicate**, at Stage A's winning resolution
and spread. Spread is held at the same *multiple of pixel spacing* so that
"wider range" and "more smoothing" are not confounded.

```
range     (0, 2.5) | (0, 4.0) | (0, 6.0) | auto (p99 of observed deaths = 4.97)
channels  H0+H1 summed | H0 and H1 as separate channels
weight    linear (shipped) | constant | squared | arctan
                                              -> 32 configurations
```

> **Erratum, 22 July, before Stage B ran.** This line originally read "24
> configurations". That was an arithmetic slip: 4 ranges x 2 channel layouts x
> 4 weightings is **32**. The grid itself is exactly as enumerated above and is
> unchanged — only my multiplication was wrong. Sweep total is therefore
> **25 + 32 = 57** configurations, not 49.
>
> This changes nothing about the endpoint or its multiplicity. `N_LOOKS` stays
> at 8 because the correction for the sweep's size is the frozen split, not a
> Bonferroni term: selection happens on extractants the endpoint never sees, so
> the confirmatory interval is insensitive to how many configurations were tried.

**Seeds.** 8 per configuration exploratory (drawn from the published matched
set), 16 for the winner. **Readout unchanged** — same CNN, same
`--pair-loss-weight 2.0 --select-on adjacent`, same 5×3 folds — so any change is
attributable to the representation and not to the model.

> **Seed count, raised 3 → 8 before Stage A launched.** The smoke run showed a
> configuration costs 40 s, not the ~2 min assumed, and that a *single* seed
> scores +0.0401 against the published 16-seed +0.2101. Three seeds would have
> made selection mostly ensembling noise; eight costs ~6 min per configuration.
> See [`PI_SWEEP_HARNESS_CHECK.md`](PI_SWEEP_HARNESS_CHECK.md).

### A stated confound on the resolution axis specifically

Holding the readout fixed is what makes a gain attributable to the
representation — but it does mean the **resolution** axis is not a clean
manipulation, and this is written down now rather than discovered afterwards.

`PersistenceCNN` is three 3×3 convolutions with no striding or pooling between
them, so its receptive field is **7×7 regardless of resolution**, followed by a
*global* mean+max pool over the whole plane. The fraction of the image any unit
can integrate over therefore shrinks as resolution rises:

| resolution | receptive field as a fraction of the image |
|---|---|
| 20 | 35 % |
| 48 | 15 % |
| 128 | **5.5 %** |

So higher resolution does not simply supply more detail to the same model: it
also gives that model a proportionally narrower view before everything is pooled
away. **If the benchmark curve is flat or declining in resolution, the honest
reading is "at fixed readout capacity and a fixed 7×7 receptive field", not
"resolution does not help persistence images."** Distinguishing those would need
the readout swept too, which is deliberately out of scope here and is recorded
as the obvious follow-up.

The **spread** axis carries no such confound: smoothing changes the image
in-place without changing what fraction of it a unit sees, which is a further
reason to read the spread result as the more interpretable of the two axes
Kostas named.

---

## 6. Selection rule — tune half only

The winner **P\*** is the single configuration maximising the **stack gain on the
tune half**: `nested_stack(CatBoost, repaired, P) − nested_stack(CatBoost,
repaired)`, restricted to tune extractants. Ties broken by the earlier grid
position (deterministic). **One winner. No re-selection after any confirm-half
number is seen.**

Reported alongside, for every configuration, the mechanism's two axes —
standalone adjacent-pair R² and error correlation with the repaired baseline — so
a failure can be attributed rather than left unexplained. That is how the
original P0 failure was diagnosed and what made the mechanism predictive.

---

## 7. Endpoints

**Primary.** Δ adjacent-pair R² = `nested_stack(CatBoost, repaired, P*)` −
`nested_stack(CatBoost, repaired)`, computed **on confirm-half rows only**,
paired cluster bootstrap resampling whole extractants, **400 draws, seed 0, 90 %
interval**, multiplicity-respecting (no collapsing of duplicate clusters).

**Secondary, decisive.** P\* versus the matched 2D control (T0w) in the same
stack slot — the identical contrast S0 faced.

**Multiplicity.** `N_LOOKS = 8` (S0, S2, stack primary, stack decisive, S0X, F30,
F40, and this test). Bonferroni applied via `stack_test._corrected`, as
throughout.

**Descriptive.** The resolution benchmark curve; P\* on the mechanism's two-axis
plane against S0 / P0 / T0w / CatBoost; the anchor's reproduction of +0.2101.

---

## 8. What each outcome means — fixed before the data exists

| outcome | consequence |
|---|---|
| CI excludes 0, positive, **and** P\* beats the matched 2D control | **Tuned persistence homology also contributes.** The claim broadens from one representation to a class of them. Rewrite the scope paragraphs in `README.md`, `SYNTHESIS.md` and `PI_EMAIL.md`; the mechanism gains a second confirmed prediction. |
| point estimate positive, CI spans 0 | Tuning helps but not demonstrably at this sample size. Current scope stands unchanged; report the benchmark curve and the point estimate as suggestive only. |
| ≤ 0, **with the positive control passing** | Persistence images do not contribute here even when tuned across 49 configurations. This is **strictly stronger than today's caveat**: the honest statement becomes "given a fair test, it did not replicate", not "untested". |
| positive control fails | The test is **void**. No conclusion in either direction. Report the power failure. |

All four are reportable, which is what makes this pre-registration honest rather
than decorative.

**The honest prior:** P0 must gain ~0.02 adjacent-pair R² *and* shed ~0.02 of
error correlation to enter the stack. Tuning has moved representations that far
before, but the second condition is the harder one, and the diagnostics say
nothing about whether smoothing decorrelates P0 from a fingerprint baseline. I
expect the middle outcome and would not be surprised by the third.

---

## 9. Guards

* `control_guard --verify` — **324 pinned artefacts** byte-identical, before and
  after. Nothing published moves.
* Standing harness check: **published S0 must re-ensemble to +0.2382** or the
  analysis module refuses to report.
* The anchor must reproduce P0's **+0.2101**; a large gap is itself reportable and
  would qualify every comparison against the published number.
* `data/` never written. All output under `automl/artifacts/pi_sweep/` and
  `automl/reports/PI_SWEEP_*`. Existing reports append-only.
* Correctness tests (`automl/tests/test_pi_sweep.py`, 21 passing) run **before**
  any fit: shipped-settings round-trip, mismatch attribution, purity of the
  render, injective configuration hashes, split partition and reproducibility,
  per-channel normalisation backward compatibility.
* No source edits while a fit array is in flight.
