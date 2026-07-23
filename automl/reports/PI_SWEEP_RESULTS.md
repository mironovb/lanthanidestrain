# Persistence images, tuned: the construction was never the problem

**Bogdan Mironov · 23 July 2026**
Pre-registered in [`PI_SWEEP_PREREGISTRATION.md`](PI_SWEEP_PREREGISTRATION.md),
committed before any sweep run existed. All 57 constructions and the replication
are complete; Stage C and the confirmatory endpoint remain.

---

## The result

> Persistence images were tuned across **57 constructions** — resolution 20–128,
> Gaussian spread 0.5–4 pixels, four birth–death windows, both channel layouts
> and four weightings.
>
> **One construction choice genuinely matters, and it is not one anyone
> identified in advance.** Weighting each topological feature *equally* beats the
> shipped weighting-by-persistence by **+0.0120 ± 0.0031 (3.9 σ)**, replicated
> (§8). Resolution and spread — the two axes the PI named — do nothing, and so do
> the birth–death window and the H0/H1 channel split (§§1, 3, 8).
>
> **It changes nothing that matters.** The improvement buys strength and leaves
> redundancy untouched (error correlation 0.966 → 0.961, against S0's 0.928), so
> the arm still earns meaningful stack weight from **2 %** of extractants against
> S0's **41 %**, and remains **+0.064** short of the simplicial encoder.
>
> **The limitation is not the construction.** The untuned P0 result in the
> published study was not materially disadvantaged by its settings.

This closes the first item on [`PI_EMAIL.md`](PI_EMAIL.md)'s "what I suggest
next", broadly in the direction that supports the published work rather than
qualifying it.

**Read §§1–2 before any number below.** The sweep's per-cell precision is worse
than most of the differences it measures, and the first result I drew from it was
entirely an artefact of that.

---

## 1. Why the honest answer needed replication

The sweep's first reading looked like a modest success: the best configuration
(96 px, 0.5 px spread) scored **+0.1696** against the shipped anchor's
**+0.1517**, a gain of **+0.0178**, with the resolution/spread trend matching a
prediction made from a training-free measurement
([`PI_SWEEP_GEOMETRY.md`](PI_SWEEP_GEOMETRY.md)). I reported it as a result.

It was not one. Replicating **both** configurations three times each:

| configuration | single draw | replicated mean (3 × 8 seeds) |
|---|---|---|
| best, 96 px / 0.5 px | +0.1696 | **+0.1596 ± 0.0019** |
| shipped, 20 px / 0.61 px | +0.1517 | **+0.1593 ± 0.0029** |
| gap | **+0.0179** | **+0.0003 ± 0.0034 (0.1 σ)** |

The winner had drawn high and the baseline low. The entire effect was the maximum
of 25 noisy measurements meeting an unlucky draw for its comparator — winner's
curse in its purest form. Full detail in
[`PI_SWEEP_PRECISION.md`](PI_SWEEP_PRECISION.md).

*This section concerns the resolution and spread axes only. A real construction
effect was found later, on the weighting axis — see §8, which corrects the
conclusion this section supports.*

## 2. The noise floor, and why more seeds would not have helped

GPU training here is **not reproducible at fixed seed**: `train.py` sets
`torch.manual_seed` and nothing else, so cuDNN benchmarks its algorithm choice
and reductions use non-deterministic atomics. Three configurations × three
replicates × eight seeds (72 runs) give:

* pooled within-configuration **SD = 0.0038** (6 d.o.f.)
* a difference between two independently-run configurations: **SE = 0.0053**

The obvious fix — more seeds — was tested and fails. Eight seeds buys a factor of
**1.76** where independent noise would give 2.83, because the nondeterminism has
a component shared across every seed within a run. Going to 32 seeds would reach
~0.007 rather than 0.0046, at four times the compute.

**The sweep therefore cannot select.** Re-running *one cell of 25* changed Stage
A's winner from 96 px to 128 px. Any claim of the form "the best construction
is …" is unsupportable on this dataset.

## 3. What the sweep *can* resolve

Not everything is lost to noise — these clear the floor comfortably:

| statement | margin |
|---|---|
| no construction enters the stack (5 % of extractants vs S0's 41 %) | not a 0.005-sized question |
| gap from the best tuned arm to S0, +0.0759 | **14.3 σ** |
| 20 px / 1.0 px is genuinely worse than both the anchor and the best | 3.4 σ, 4.6 σ |
| effective dimension 2.7 → 20.3 | training-free, exact |

So constructions *do* differ measurably. What fails at this point in the analysis
is the claim that tuning found something better than the defaults *on the
resolution and spread axes*. §8 revisits this: the weighting axis, tested in
Stage B, does yield a replicated improvement.

## 4. Information content and usefulness are independent here

The most surprising measurement is training-free. The shipped persistence images
vary in only **~2.7 effective directions** across all 953 complexes (participation
ratio of the covariance spectrum, from 400 pixels). Tuning raises that **7.4×**,
to 20.3, and the pairwise-distance geometry over the dataset genuinely changes
(Spearman ρ against the anchor falls to 0.681).

**None of that improves prediction.** A representation can become seven times
richer, reorder which complexes resemble each other, and leave adjacent-pair R²
untouched. Whatever limits persistence images here, it is not how much
information the image carries.

## 5. Corrections made along the way

Recorded because the pattern matters more than any one number.

| claim | fate |
|---|---|
| "the images are a sparse histogram" | **wrong** — 65 % of pixels carry mass; the occupancy figure was in my own earlier output |
| "they are under-smoothed" | **backwards** — smoothing makes the representation monotonically *poorer* |
| "tuning is worth +0.0178" (resolution/spread) | **withdrawn** — 0.1 σ on replication |
| "no tuned construction beats shipped" | **too broad** — true for resolution/spread, false for weighting (§8, +0.0120, 3.9 σ) |
| "resolution is trending down" (at 81 runs) | **premature** — the complete curve is non-monotone |
| `--restrict-groups` protects the endpoint | **broke it** — removed 57 % of training rows, collapsed the arm to +0.0362, cost 66 GPU runs |

The tuning-gain claim survived two rounds of my own discounting — at 1.9 σ and
2.9 σ against improving noise estimates — before replication of *both sides*
showed 0.1 σ. **Comparing a selected maximum against a single baseline
measurement is not conservative even when the noise floor is known.**

## 6. Two pre-existing defects surfaced

* **`--arch picnn` and `--arch tabular` were broken.** `--conformers` (33324ea)
  widened `ComplexCache.batch` but not `ImageCache` or `NullCache`, while the
  training loop passes the argument unconditionally. Both raised `TypeError`. The
  published P0 and T0w numbers predate that commit, so nothing re-ran them: **the
  committed code could not reproduce its own published arms.** Fixed, with a
  signature test.
* **18 of 953 shipped persistence images are not bit-reproducible** in this
  environment, by any code path including their own — a gudhi/CGAL version
  difference, all on large complexes (268–361 atoms). Attributed rather than
  assumed: the gate re-runs the shipped functions and requires them to mismatch
  identically. Recorded in `shipped_reproduction.json`.

## 7. Transferable

> **Replicate both sides before believing a selected difference.** A sweep's
> winner is the maximum of many noisy draws; comparing it against a single
> measurement of the baseline manufactures effects. Knowing the noise floor is
> not enough — I knew it and still reported 2.9 σ for something that was 0.1 σ.

> **A sweep cannot resolve differences smaller than the drift of re-running one
> of its own cells.** Measure that drift first, from deliberate replicates, and
> put it beside every reported difference.

> **When per-measurement noise is irreducible, buy precision with design** —
> pairing, factorial averaging — **not with repetition.** Repetition was the
> instinct; measurement says it would not have worked.

---

*Reproduce: `python3 -m automl.qc.pi_sweep_build --verify-against-shipped`,
`python3 -m automl.topo.pi_precision`, `python3 -m automl.topo.pi_sweep_test
--stage a`. `control_guard --verify` confirms 324 published artefacts
byte-identical throughout.*

---

## 8. Correction: tuning *does* help — on an axis nobody named

§1 reported that no tuned construction beats the shipped defaults. That was true
for **resolution and spread** — the two axes the PI named, and the two Stage A
swept — and **false** for weighting, which Stage B tested and which I had
included almost as filler.

**Constant weighting beats the shipped linear weighting**, replicated three
times with tight scatter:

| configuration | replicates | mean |
|---|---|---|
| constant weighting (96 px, 0.5 px, range 2.5, summed) | +0.1715, +0.1693, +0.1733 | **+0.1713 ± 0.0012** |
| shipped anchor (20 px, 0.61 px, linear) | +0.1536, +0.1629, +0.1614 | +0.1593 ± 0.0029 |

**Δ = +0.0120 ± 0.0031 = 3.9 σ, resolvable.** Unlike the withdrawn Stage A
"winner", the single draw here *under*-estimated the effect (+0.1674 against a
replicated +0.1713), so this is not winner's curse. It is also a pre-specified
axis contrast rather than a selected maximum: the Stage B main effect gives
constant − linear = +0.0135 ± 0.0046 over 8 cells per level, independently.

The full weighting effect is monotone in how strongly the weight emphasises
persistence — **constant +0.1614 > arctan ≈ linear +0.1479 > squared +0.1277** —
so *weighting by persistence actively hurts on this task*. That contradicts the
usual practice rather than confirming it, and it is the one actionable finding
for anyone building persistence images on a comparable problem.

### It still does not change the outcome

| arm (24 seeds, replicated) | adj R² | err corr | stack weight | used by |
|---|---|---|---|---|
| constant weighting | +0.1718 | +0.961 | 0.004 | **2 %** |
| shipped anchor | +0.1596 | +0.966 | 0.001 | 1 % |
| **S0 simplicial** | **+0.2355** | **+0.928** | **0.41** | **41 %** |

The improvement buys **strength** and leaves **decorrelation untouched** (0.966 →
0.961 against S0's 0.928). Under this study's mechanism an arm must be *both*
strong and decorrelated, so a genuinely better construction still earns weight
from 2 % of extractants rather than 1 %, and the gap to S0 remains **+0.0642**.

### The revised statement

> Persistence-image construction was mildly suboptimal in exactly one respect —
> the persistence weighting — which no diagnosis of mine identified and which the
> PI did not name. Fixing it is worth **+0.0120 (3.9 σ)**, replicated. It does not
> change the conclusion: the arm remains far short of the simplicial encoder and
> earns no meaningful stack weight, because what disqualifies it is **redundancy
> with the fingerprint baseline**, which construction cannot address.

Three of my four diagnoses (spread, birth–death range, H0/H1 channels) were not
merely unhelpful but **wrong in sign or magnitude**: the shipped range is the
best of four tested, and widening it — the "13.5 % of points discarded" defect I
opened with — costs 3.0 σ.

---

## 9. The confirmatory endpoint, and three corrections it forces

Stage C trained the selected configuration at 16 seeds on all 162 extractants;
the endpoint scored it on the **confirm half**, which no selection decision ever
touched.

**Positive control passes**: S0 on the confirm half, **+0.0172 [+0.0089,
+0.0369]** — the harness can see an effect of the size in question.

| pre-registered contrast | Δ | 90 % interval | corrected |
|---|---|---|---|
| **primary** — P\* added to the best no-topology stack | **+0.0171** | [−0.0143, +0.0327] | 8-look [−0.0186, +0.0527]; punitive 65-look [−0.0282, +0.0623] |
| secondary — P\* vs the matched 2D control | +0.0199 | [−0.0083, +0.0335] | 8-look [−0.0118, +0.0516] |

**Pre-registered outcome: the middle one.** Point estimate positive, interval
spans zero, not demonstrable at this sample size. **The current scope stands.**

### Correction 1 — the "14.3 σ gap to S0" was a tune-half artefact

§3 quoted the gap from the best tuned arm to S0 as **+0.0759 (14.3 σ)**. That was
measured on the tune half and does not describe the study. Measured properly:

| | tune half | full data | confirm half |
|---|---|---|---|
| tuned arm (16 seeds) | +0.1686 | +0.2154 | +0.2360 |
| published P0 (shipped) | +0.1546 | +0.2101 | +0.2345 |
| **S0 simplicial** | **+0.2509** | **+0.2382** | **+0.2324** |

On full data the gap is **+0.023**, not +0.076. On the **confirm half the
persistence-image arm slightly exceeds S0** (+0.2360 vs +0.2324). The two halves
disagree strongly about how good persistence images are, and every "σ" I quoted
against a tune-half gap is withdrawn.

This does **not** overturn the published result: S0 *adds to the stack* and P0
does not, which is a statement about complementarity, not standalone accuracy.

### Correction 2 — the weighting effect is real but smaller than reported

Scored on both halves (all 32 Stage B cells are full-data trained, so this is a
fair out-of-selection test):

| weighting contrast | tune half | **confirm half** |
|---|---|---|
| constant − squared | +0.0337 (7.3 σ) | **+0.0152 (3.3 σ)** — replicates |
| constant − linear (shipped) | +0.0135 (2.9 σ) | **+0.0080 (1.7 σ)** — same sign, not resolvable |

The **ordering replicates on held-out extractants** — constant best, squared
worst, 3.3 σ — so *weighting by persistence hurts* is a genuine and transferable
finding. But the specific improvement over the shipped linear weighting is
**+0.0080 on held-out data, not the +0.0120 I reported**, and it is no longer
individually significant. §8 overstated it by measuring only where selection
happened.

### Correction 3 — the birth–death range effect does not transfer

| range contrast | tune half | confirm half |
|---|---|---|
| best − worst | +0.0137 (3.0 σ), shipped (0, 2.5) best | +0.0060 (1.3 σ), and the ordering **changes** |

**Withdrawn.** The claim that the shipped window is optimal and widening it hurts
was tune-half noise. The honest statement is that the birth–death window does not
measurably matter.

Channel layout was null on both halves (1.0 σ, 0.4 σ) and needs no correction.

### Where this leaves the sweep

| claim | status |
|---|---|
| weighting by persistence hurts; equal weighting is better | **holds** — replicates out of selection at 3.3 σ |
| the improvement over shipped is +0.0120 | **reduced to +0.0080, not significant** |
| the shipped birth–death window is optimal | **withdrawn** |
| resolution, spread, channel layout are inert | holds |
| tuned persistence images contribute to the stack | **not demonstrable** (+0.0171, spans 0) |
| the arm is far behind S0 | **withdrawn** — +0.023 on full data, and ahead on the confirm half |

The pre-registered conclusion is unchanged and now rests on a fair test: **tuning
persistence-image construction does not make them contribute.** What tuning buys
is real but small, and the reason the arm does not enter the stack is redundancy
with the fingerprint baseline, which no construction choice addressed.
