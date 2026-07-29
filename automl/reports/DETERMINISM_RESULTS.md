# The run-to-run noise floor is removable, and it costs about 7x

**Bogdan Mironov · 29 July 2026**

---

## Summary

`--deterministic` makes GPU training **bit-for-bit reproducible**. Two runs of
the identical configuration now return byte-identical out-of-fold vectors:
`max |diff| = 0.000e+00`.

The floor it removes is the one `PI_SWEEP_PRECISION.md` called "arguably the
sweep's principal finding": an 8-seed ensemble moving by **0.0092** between
identical re-runs, larger than most of the differences this study argues about,
and the reason re-running one cell of 25 changed Stage A's winner.

It is not free: the deterministic path is roughly **7x slower**
(~200 s per fold against ~26 s). That ratio, not the correctness, is what
decides where it gets used.

## What was wrong, and why the obvious fix was not the fix

The published diagnosis attributed the shared-across-seeds component to **cuDNN
autotuning** — a different convolution algorithm chosen per process. That is
correct for the persistence-image CNN.

It is not the cause for the simplicial network, **which has no convolutions at
all**. `SimplicialNet` is `Linear`, `LayerNorm` and scatter reductions. Its
nondeterminism comes from `index_add_`, which accumulates with atomics: the
summation order depends on which thread arrives first, so the low bits differ
between runs and the difference then amplifies through 60 epochs of training and
a checkpoint-selection rule.

So the fix had to be a reduction, not a backend flag:

| source | switched off by | applies to |
|---|---|---|
| scatter atomics | `snn.set_deterministic(True)` — sorted-segment sums in float64 | the SNN (**dominant**) |
| cuDNN autotuning | `cudnn.benchmark = False`, `cudnn.deterministic = True` | the PI-CNN |
| cuBLAS workspace reuse | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, exported before the process starts | both |

Two deliberate non-changes, both recorded because each looks like an omission:

- **`scatter_max` is untouched.** A maximum is exact and associative, so
  `index_reduce_` returns the same value whichever order the atomics land in.
  It has no deterministic CUDA kernel registered, so it *warns* under
  `use_deterministic_algorithms(True, warn_only=True)`. That warning is expected
  and is not evidence of nondeterminism.
- **The mean's denominator stays on the original path.** It accumulates 1.0 into
  segments of at most ~10⁶ elements, which float32 represents exactly, so its
  value cannot depend on ordering.

The sorted sum accumulates in **float64**. A float32 prefix sum over ~10⁶
triangles loses more precision than the atomics it replaces, which would have
traded one error for another.

## The measurement

Nothing here is argued from the source. Arguing determinism from the code is the
mistake this project has been caught by before, so it was run.

The published S0 configuration — simplicial encoder, contrast objective,
adjacent-pair checkpoint selection, 5 folds, seed 7 — executed twice in the same
job, and the out-of-fold vectors compared element by element.

| | result |
|---|---|
| `max |oof_a − oof_b|` | **0.000e+00** |
| bit-identical | **True** |
| per-fold R² | identical to every printed digit across both runs |
| wall clock per fold | 191 s / 212 s / 263 s (against ~26 s on the published path) |

Job 5278050, `automl/logs/detsmoke_5278050.out`.

## What this changes, and what it does not

**Changes.** A sweep can now select. Every configuration comparison in this study
so far has had to clear ~0.009 of pure re-run noise before it meant anything;
under this flag two configurations differing by 0.001 differ by 0.001. The design
lesson from `PI_SWEEP_PRECISION.md` — *when per-measurement noise is irreducible,
buy precision with design rather than repetition* — turns out to have a false
premise: the noise was not irreducible.

**Does not change.** Any published number. Determinism is off by default and
every existing arm is byte-reproducible from the current source; `control_guard
--verify` passes on all 324 frozen artefacts after the change.

**Does not change the encoder test either**, deliberately. The 16-seed S0
ensemble that G0 and D0 are compared against was trained *without* the flag, and
an arm under a different reduction order is not matched to it. That is
Amendment 1 to `ENCODER_PREREGISTRATION.md`, written before those runs started.

## Where it should be used

The 7x cost is the constraint, and this account is capped by `GrpTRES` at **one
node on `xeon-g6-volta`** — two concurrent jobs, about 17 published-path runs per
hour, so about 2.5 deterministic ones.

| use | flag | why |
|---|---|---|
| exploratory sweeps | **off** | 7x more configurations per hour beats a noise floor that seed-ensembling and a factorial design already partly absorb |
| confirmatory runs, published arms, anything a later result is compared against | **on** | reproducibility is the point, and there are few of them |
| replicating a comparison before believing it | **on** | this is the specific failure `PI_SWEEP_PRECISION.md` documents — a +0.0178 tuning gain that replication reduced to +0.0003 |

## Caveat, stated rather than buried

This was measured on **one configuration, one seed, five folds**, twice. It shows
bit-identity where bit-identity previously failed, which is the claim. It does
not show that every architecture and every flag combination reaches bit-identity
— `--conformers`, `--block-centre` and `--arch picnn` each take different code
paths and none has been checked. `automl/slurm/determinism.sh` runs the wider
3-replicate × 4-seed design and `automl/topo/determinism_test.py` reads it; that
measurement has not been made, and the GPU budget went to the encoder test
instead. Any arm that needs the guarantee should be verified, not assumed to
inherit it.

---

**Reproduce**

```bash
sbatch automl/slurm/determinism.sh          # MODE=det / MODE=nondet
python3 -m automl.topo.determinism_test
python3 -m pytest automl/tests/test_determinism.py -q
```
