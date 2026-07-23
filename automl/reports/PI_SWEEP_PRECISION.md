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
