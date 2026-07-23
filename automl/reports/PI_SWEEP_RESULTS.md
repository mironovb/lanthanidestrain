# Persistence images, tuned: the construction was never the problem

**Bogdan Mironov · 23 July 2026**
Pre-registered in [`PI_SWEEP_PREREGISTRATION.md`](PI_SWEEP_PREREGISTRATION.md),
committed before any sweep run existed. Stage B main effects and the
confirmatory endpoint are still running; everything below is complete and will
not be changed by them.

---

## The result

> Persistence images were tuned across **57 constructions** — resolution 20–128,
> Gaussian spread 0.5–4 pixels, four birth–death windows, both channel layouts
> and four weightings. **The best construction found is indistinguishable from
> the shipped defaults**: +0.0003 ± 0.0034 (**0.1 σ**). The arm remains
> **+0.0759 (14.3 σ)** short of the simplicial encoder, and no construction
> earned meaningful weight in the stack.
>
> **The limitation is not the construction.** The untuned P0 result in the
> published study was not disadvantaged by its settings.

This closes the first item on [`PI_EMAIL.md`](PI_EMAIL.md)'s "what I suggest
next" — in the direction that supports the published work rather than qualifying
it.

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

So constructions *do* differ measurably. What fails is specifically the claim
that tuning found something **better than the defaults**.

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
| "tuning is worth +0.0178" | **withdrawn** — 0.1 σ on replication |
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
