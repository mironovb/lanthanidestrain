# The topology result does not survive a stricter definition of "identical conditions"

**Bogdan Mironov · 29 July 2026**
Pre-registered in [`DUALKEY_PREREGISTRATION.md`](DUALKEY_PREREGISTRATION.md),
committed at `836ad30` before any contrast below existed.
Data: `dualkey_test.csv`, `dualkey_arms.csv`. Job 5278275.

---

## The verdict the pre-registration commits me to

> **CLAIM DOWNGRADED — survives only under the binned key.**

Both published contrasts add under `composition_key` and neither is
distinguishable from zero under `strict_composition_key`. §4 of the
pre-registration says what that means and what to do, and it was written before
the numbers existed:

> the published effect is partly an artefact of averaging measurements taken
> under different conditions into one cell. The headline is rewritten as a
> binned-key result with the strict-key null stated in the abstract, not the
> appendix.

## The numbers

Multiplicity-respecting cluster bootstrap over whole extractants, 400 draws,
seed 0, 90 % interval, 10-look Bonferroni.

| contrast | binned `composition_key` | strict `strict_composition_key` |
|---|---|---|
| **drop-in** — add S0 to the best no-topology stack | **+0.0375** [+0.0173, +0.0510] · 10-look [+0.0111, +0.0639] · **adds** | **+0.0177** [−0.0023, +0.0367] · 10-look [−0.0128, +0.0482] · **not distinguishable** |
| **swap** — S0 vs the matched control in the same slot | **+0.0438** [+0.0253, +0.0554] · 10-look [+0.0203, +0.0673] · **adds** | **+0.0177** [−0.0023, +0.0367] · 10-look [−0.0128, +0.0482] · **not distinguishable** |

Single arms:

| arm | binned | strict | overall log D R² |
|---|---|---|---|
| CatBoost | +0.1422 | +0.0819 | **+0.4987** |
| repaired FCNN | +0.2206 | **+0.1741** | +0.3218 |
| **S0 simplicial** | **+0.2382** | +0.1683 | +0.3678 |
| T0w matched control | +0.2006 | +0.0967 | +0.2963 |

Stacks:

| stack | binned | strict |
|---|---|---|
| CatBoost + repaired + S0 | +0.2672 | +0.1918 |
| CatBoost + repaired | +0.2263 | +0.1737 |
| CatBoost + repaired + T0w | +0.2208 | +0.1737 |

## Three things worth saying before anyone over-reads this

**1. It is a weakening, not a reversal.** The strict-key point estimate is
**+0.0177 and positive**, with `P(Δ>0) = 0.93` and a lower bound of −0.0023. It
is about half the binned effect and it misses significance narrowly. Nothing here
says topology hurts; it says the evidence that it helps is roughly half as large
as published and no longer clears zero.

**2. The published bootstrap correction survives.** These are the *corrected*,
multiplicity-respecting intervals — the ones `bootstrap_check.py` showed the
published draw made 12–29 % too narrow. Under the binned key the contrasts still
add after that correction *and* 10-look Bonferroni (+0.0375 → [+0.0111, +0.0639]).
So the thing that breaks the result is the **block definition**, not the
resampling.

**3. The swap contrast degenerates, and that is itself the finding.** Under the
strict key the matched control T0w receives a fitted stack weight of **0.00** —
it is worthless in the stack, so "topology swapped for control" becomes
numerically identical to "no topology", and the two contrasts collapse onto one
number. T0w falls from +0.2006 to +0.0967, much further than S0 falls
(+0.2382 → +0.1683).

That last point cuts the other way from the headline, and it belongs in the
paper. **The encoder comparison gets *stronger* under the strict key:**

| | binned | strict |
|---|---|---|
| S0 − T0w, single arms | +0.0376 | **+0.0716** |

So a stricter definition of "identical conditions" makes the simplicial encoder
look *better* against its matched control and *worse* inside the stack. Both are
true and they are not in conflict: under the strict key the repaired FCNN becomes
the strongest single arm (+0.1741, above S0's +0.1683) and absorbs most of what
the stack needs, leaving less for S0 to add on top.

## Which key is right?

**Neither is clean, and the paper should say so rather than pick a winner.**

- `composition_key` bins the conditions. `dataset.py:387` says the consequence in
  its own words: two rows can share a block while differing in extractant
  concentration, "which turns a real log D difference into label noise". Under
  this key the model is partly rewarded for predicting condition effects rather
  than selectivity.
- `strict_composition_key` fixes that and pays for it: 552 blocks become 2,109,
  so far less replicate averaging survives and every cell mean is noisier. The
  true adjacent-pair spread falls from SD 0.271 to 0.224 while the measurement
  noise does not, so the metric is simply harder — every arm drops, including the
  baselines.

The defensible position is the one the pre-registration already fixed: **report
both columns in every table from here on**, and let the reader see that the
effect is a binned-key effect.

## What this changes

- **`SYNTHESIS.md`, `README.md` and `PI_EMAIL.md` overstate the result** as it
  now stands. Each needs an erratum pointing here. They are append-only, so the
  erratum is appended rather than the text edited.
- The headline "+0.2263 → +0.2672" is correct **under the binned key** and must
  be labelled that way, not left unqualified.
- **All new work is scored under both keys.** `ENCODER_PREREGISTRATION.md` and
  `OBJECTIVE_PREREGISTRATION.md` already require this; they were written after
  this test was pre-registered and before it was run.
- The look count for the topology claim is now **10**.

## What was NOT affected

`control_guard --verify` passes on all 324 frozen artefacts. The standing
precondition holds: the published S0 ensemble re-scores to exactly **+0.2382**.
No published file was edited and nothing was retrained — this is a re-scoring of
out-of-fold vectors that already existed.

## A methods note that came out of running it

The literal implementation of the corrected bootstrap — suffix a copy index onto
the block key, recompute the metric per draw — ran for 75 minutes without
finishing, because it redoes a groupby over 2,109 blocks 3,200 times.

It did not need to. Tagging each drawn copy turns a twice-drawn extractant into
two blocks holding the same rows, which produce the same pairs twice. So the
corrected resample is exactly the precomputed per-cluster pair vectors
concatenated **with repetition**, where the published collapsing version
concatenated the *distinct* clusters. **A single `np.unique` is the entire
difference between an m-out-of-n subsample and a cluster bootstrap** — and
removing it is ~200× faster. Asserted against the literal statistic at 1e-9 on
ten draws per key (`_assert_corrected_matches`), not argued.

---

**Reproduce**

```bash
module load anaconda/Python-ML-2025a
export PYTHONPATH=$PWD
python3 -m automl.topo.dualkey_test --n-boot 400
python3 -m pytest automl/tests/test_dualkey.py -q
```
