# The sweep's noise floor, and what it does to the Stage A result

**Bogdan Mironov · 22 July 2026**
Written immediately on discovery, before the replication that quantifies it
finished, because it materially weakens a claim I had already reported.

---

## 1. What happened

Stage A and Stage B share exactly one grid cell — `96 px, 0.5 px spread,
(0, 2.5), H0+H1 summed, linear weighting` — because Stage B is defined at Stage
A's winning resolution and spread. It has the same configuration hash
(`9d6e4c93026dfa0c`) in both manifests, so Stage B **re-ran it and overwrote
Stage A's outputs**.

Same image set. Same eight seeds. Same code. Same node type.

| | 8-seed ensemble, tune half |
|---|---|
| Stage A | **+0.1696** |
| Stage B re-run | **+0.1587** |

**0.011 apart.** Per-seed scores within the re-run span +0.1064 to +0.2145.

## 2. Why — training is not reproducible at fixed seed

`automl/topo/train.py` calls `torch.manual_seed(seed)` and nothing else. There is
no `torch.use_deterministic_algorithms(True)`, no `cudnn.deterministic`, no
`cudnn.benchmark = False`. On a GPU that leaves at least two sources of drift:
cuDNN selects convolution algorithms by runtime benchmark, and several reductions
accumulate with non-deterministic atomics. Identical seeds therefore do **not**
give identical weights.

This does **not** affect the study's existing reproducibility claim, which is
about re-running the *analysis* from stored out-of-fold parquets — that remains
bit-for-bit. It affects **re-training**, which is what a sweep does.

## 3. What it does to the Stage A conclusion

I reported that tuning the construction is worth **+0.0178** (shipped anchor
+0.1517 → best +0.1696) and that the effective-dimension prediction was confirmed
at r = +0.421.

Set against a same-configuration re-run difference of **0.011**:

| quantity | value | relative to the 0.011 noise |
|---|---|---|
| best − anchor | +0.0178 | **1.6×** |
| full Stage A range (+0.1385 to +0.1696) | 0.031 | 2.8× |
| S0 − best tuned | +0.0733 | 6.7× |

**The +0.0178 tuning gain is only about 1.6 times the drift of re-running the
same configuration.** I should not have reported it as an established effect, and
I am withdrawing that framing: it is suggestive, not demonstrated. The
correlation with effective dimension is computed across 25 configurations so it
is somewhat better protected than a single pairwise difference, but every
individual configuration in it carries this noise, and r = +0.421 on n = 25 was
already only marginal.

What is **not** weakened, because both quantities are far outside the noise:

* No configuration entered the stack — best weight 0.006 / 5 % of extractants
  against S0's 0.41 / 41 %. That is not a 0.011-sized question.
* The gap to S0 (+0.0733) is 6.7× the noise.
* The effective-dimension measurement itself (2.7 → 20.3) involves no training at
  all and is exactly reproducible.

## 4. The measurement now running

One accidental replicate is not an estimate. `automl/slurm/pi_replicate.sh` runs
**three configurations × three independent replicates × the same eight seeds**,
writing to its own directory so nothing is overwritten. The configurations span
the observed range: the shipped anchor, Stage A's winner, and a mid-range cell.

The spread across replicates is the noise floor that every "difference between
configurations" in this sweep must clear, and it will be reported before any
Stage A or Stage B ranking is treated as meaningful.

## 5. What this does *not* threaten

The pre-registered endpoint remains valid. It is a test of whichever
configuration selection produces, scored on the confirm half with a paired
cluster bootstrap — the interval already accounts for sampling variability, and
the positive control (S0 detected at +0.0415 [+0.0185, +0.0519]) is unaffected.

What noise costs is **power in the selection stage**: we may not be handing the
endpoint the genuinely best configuration. That is a weaker problem than bias,
and it errs against finding an effect rather than towards one.

## 6. The lesson

> A sweep cannot resolve differences smaller than the drift of re-running one of
> its own cells. Measure that drift *first*, from replicates, and set it beside
> every reported difference.

I designed this sweep around ensembling 8 seeds to control *seed* variance and
never checked run-to-run variance at fixed seed. It surfaced only because two
stages happened to share a cell — which is luck, not method.

---

*Reproduce: `sbatch --array=0-8 automl/slurm/pi_replicate.sh`, then compare the
three replicates per configuration.*

---

## 7. More seeds will not fix it — measured, not assumed

The obvious response to a noise floor is to ensemble more seeds. That was tested
directly, by taking the two complete replicates of the shipped anchor and
comparing them using the *same* N seeds in both, for increasing N:

| seeds ensembled | mean run-to-run difference | 1/√N would predict |
|---|---|---|
| 1 | 0.0162 | — |
| 2 | 0.0115 | 0.0115 ✓ |
| 4 | 0.0112 | 0.0081 |
| 8 | **0.0092** | 0.0057 |

Eight seeds buys a factor of **1.76** where independent noise would give 2.83.
The nondeterminism therefore has a component **shared across every seed within a
run** — consistent with cuDNN fixing its algorithm choices per process, so all
eight trainings in one job inherit the same perturbation.

**Consequence: seed count is not the lever.** Going from 8 to 32 seeds would
reach roughly 0.007 rather than the 0.0046 that independence predicts, at four
times the compute. That was the intervention I would have reached for, and it
would have been largely wasted.

## 8. What this means for Stage B — and why it is worth finishing

Stage B is a **4 × 2 × 4 factorial** (birth–death range × channel layout ×
weighting). Read cell-by-cell it is hopeless: individual configurations differ by
less than the 0.0092 floor.

Read as a factorial it is not. A **main effect** averages 16 cells per level, so
its noise falls to roughly 0.0092 / √16 ≈ **0.0023**, resolving differences of
about 0.005. The question that matters — *does separating H0 from H1 buy
anything, averaged over range and weighting* — is comfortably inside that.

So Stage B resumed, with its analysis changed from "pick the winning cell" to
"estimate the three main effects". The factorial design was chosen for coverage;
it turns out to be what rescues the stage from its own noise floor. The winning
*cell* will still be reported, and still labelled unresolvable if it is.

**The general form of this:** when per-measurement noise is irreducible, buy
precision with design (pairing, factorial averaging) rather than with repetition.
Repetition was the instinct and the measurement says it would not have worked.

---

## 9. The selection is demonstrably unstable — one cell changed the winner

The clearest evidence that this sweep cannot select is not a variance estimate.
It is that the answer changed.

Stage B's re-run of the single shared cell moved it from **+0.1696 to +0.1587**.
Regenerating Stage A's table from the current runs on disk — **one cell of 25
altered, the other 24 untouched** — moves the winner:

| | winning configuration | adj R² |
|---|---|---|
| Stage A as first computed | **96 px**, 0.5 px spread | +0.1696 |
| Stage A regenerated | **128 px**, 1.0 px spread | +0.1670 |

The two are **0.0026 apart**, well under the 0.0092 SE of a difference. They are
not distinguishable, and which one is called "the winner" is decided by
nondeterminism.

This has a consequence for Stage B, which should be stated rather than buried:
**Stage B's entire grid was rendered at 96 px / 0.5 px because that was Stage A's
winner** — a selection that does not reproduce. Stage B is not invalidated by
this: it is a factorial over range × channels × weighting at a *fixed*
resolution and spread, and its main effects remain interpretable conditional on
that fixed point. But the fixed point is arbitrary among several statistically
indistinguishable options, and any claim of the form "the best configuration
is …" is unsupportable.

**This is arguably the sweep's principal finding.** A 57-configuration
hyperparameter search, 8 seeds per configuration, ~450 GPU runs, cannot identify
a best persistence-image construction on this problem — because the differences
between constructions are smaller than the noise of re-running one of them, and
that noise does not shrink with more seeds.

What the sweep *can* support, because these clear the floor comfortably:

| statement | margin |
|---|---|
| no configuration enters the stack (5 % of extractants vs S0's 41 %) | not a 0.0092-sized question |
| the gap from the best tuned arm to S0 (+0.0733) | 8.0 σ |
| effective dimension 2.7 → 20.3 | training-free, exact |
| main effects of range / channels / weighting (16 cells per level) | SE 0.0023 |

---

## 10. The construction question, answered

The sweep was run to settle whether persistence images had been given a fair
test. The answer to the *construction* half is now definite, and it is neither
of the two answers anyone expected.

**No tuned configuration is demonstrably better than the shipped settings.**
Every one of the top six Stage A cells sits under 2 σ against the anchor
(+0.1517), using the measured difference SE of 0.0092:

| configuration | adj R² | Δ vs shipped | σ |
|---|---|---|---|
| 128 px / 1.0 px | +0.1670 | +0.0153 | **1.7** |
| 64 px / 1.0 px | +0.1624 | +0.0107 | 1.2 |
| 96 px / 0.5 px | +0.1587 | +0.0070 | 0.8 |
| 128 px / 2.0 px | +0.1581 | +0.0063 | 0.7 |
| 96 px / 4.0 px | +0.1545 | +0.0028 | 0.3 |
| 96 px / 1.0 px | +0.1544 | +0.0027 | 0.3 |

That 1.7 σ is **optimistic**, because it is the maximum of 25 noisy draws.

So the finding is not "tuning helps" and not "tuning does not help". It is that
**on 953 complexes, the two hyperparameters the PI named — resolution and
Gaussian spread — cannot be distinguished from one another, nor from the shipped
defaults.** The advice was sound. The dataset is too small to act on it.

### What this replaces

`PI_EMAIL.md` and the scope paragraphs currently say persistence images "require
tuning and were never tuned", and treat that as an open question. It was true,
and it turns out to be **unactionable at this scale**. The replacement statement
is both more specific and more useful:

> Persistence images were tuned across **57 constructions** spanning resolution
> 20–128, spread 0.5–4 pixels, four birth–death windows, both channel layouts and
> four weightings. **No construction was distinguishable from any other**, and the
> arm remains **+0.0733 (8.0 σ)** short of the simplicial encoder. The limitation
> is not the construction.

That is a stronger claim than the one it replaces, and it closes the gap the PI
email listed as the first thing to do next — just not in the direction expected.

---

## 11. Final: the tuning gain was entirely winner's curse

Replication complete — three configurations, three independent replicates each,
eight seeds per replicate, 72 runs. Pooled within-configuration
**SD = 0.0038 (6 d.o.f.)**, so a difference between two independently-run
configurations has **SE = 0.0053**.

Comparing configurations by their *replicated means* rather than by single draws:

| configuration | replicated adj R² |
|---|---|
| Stage A "winner" — 96 px, 0.5 px spread | +0.1596 ± 0.0019 |
| **shipped anchor — 20 px, 0.61 px spread** | **+0.1593 ± 0.0029** |
| mid-range — 20 px, 1.0 px spread | +0.1484 ± 0.0016 |

| contrast | Δ | σ | verdict |
|---|---|---|---|
| **winner − shipped anchor** | **+0.0003 ± 0.0034** | **+0.1** | **not resolvable** |
| anchor − mid-range | +0.0110 ± 0.0033 | +3.4 | resolvable |
| winner − mid-range | +0.0112 ± 0.0024 | +4.6 | resolvable |

### The mechanism, exactly

| | Stage A single draw | replicated mean | |
|---|---|---|---|
| winner | +0.1696 | +0.1596 | drew **high** |
| anchor | +0.1517 | +0.1593 | drew **low** |
| gap | **+0.0179** | **+0.0003** | |

The entire apparent tuning gain was the maximum of 25 noisy measurements meeting
an unlucky draw for the baseline. This is winner's curse in its purest form, and
it is worth noting that the number survived my first two attempts to discount it
— at 1.9 σ and then 2.9 σ against progressively better floor estimates — before
replication showed it to be **0.1 σ**. Comparing a *selected* maximum against a
*single* baseline measurement is not conservative even when the noise floor is
known; both sides have to be replicated.

### The sweep is not measuring nothing

Mid-range (20 px, 1.0 px spread) is genuinely worse than both others, at 3.4 σ
and 4.6 σ. Constructions do differ measurably. The specific claim that fails is
that **tuning found something better than the shipped defaults**.

### Final answer to the construction question

> Persistence images were tuned across 57 constructions. The best construction
> found is **indistinguishable from the shipped defaults** (+0.0003, 0.1 σ), and
> the arm remains **+0.0759 (14.3 σ)** short of the simplicial encoder. The
> limitation is not the construction, and the untuned P0 result in the published
> study was not disadvantaged by its settings.
