# The "~0.04 Å optimisation-noise floor" is wrong by a factor of 200

**Measured, against a criterion fixed in advance**
([`C7_PREREGISTRATION.md`](C7_PREREGISTRATION.md) §4). 390 GFN2-xTB
optimisations, 389 successful, 30 structures stratified by atom count
(34–430 atoms).

---

## 1. The claim being tested

Three reports dismiss the 0.013 Å adjacent-lanthanide radius step on the same
grounds:

> `SYNTHESIS.md` — "The adjacent-lanthanide contrast is redundant with the
> tabular ionic radius and sits **below the ~0.04 Å optimisation-noise floor**."
>
> `WO_PREREGISTRATION.md` — "…below the ~0.04 Å optimisation-noise floor. So
> topology is capped for selectivity … This is accepted, not retried."
>
> `WO_RESULTS.md` — "0.013 Å signal is … below the 0.04 Å noise floor."

**The number was never measured.** It traces to an asserted "~0.05 Å conformer
scatter" carrying no derivation. And

    xtb_backend.OPT_LEVELS["tight"] = 8.0e-4 Eh/bohr × 51.42 = 0.041 eV/Å

is a **force** convergence target, not a **distance**. The numeral agrees to two
significant figures with a claim carrying Ångström units.

## 2. The measurement

Displace a converged structure by a seeded isotropic Gaussian, re-optimise under
settings **byte-identical** to the shipped control (GFN2 / ALPB water /
`--opt tight` / maxcycle 750 / `--norestart` / uhf 0 / independent `--grad`
force check), and compare the relaxed results.

| σ (Å) | runs | basin escape | median \|Δ⟨M–D⟩\| | P90 | max | median RMSD |
|---|---|---|---|---|---|---|
| 0.00 (idempotency) | 30 | 0.0 % | **0.00000** | 0.00000 | 0.00000 | 0.00000 |
| 0.02 | 120 | 0.0 % | 0.00013 | 0.00040 | 0.00186 | 0.00734 |
| **0.05** | 120 | **0.0 %** | **0.00019** | **0.00064** | 0.00125 | 0.01316 |
| 0.10 | 119 | 0.8 % | 0.00025 | 0.00064 | 0.00136 | 0.01992 |

## 3. Verdict against the pre-registered bar

At σ = 0.05 Å the criterion required median |Δ⟨M–D⟩| ≤ 0.005 Å and P90 ≤ 0.013 Å:

| quantity | bar | measured | margin |
|---|---|---|---|
| median | ≤ 0.005 Å | **0.00019 Å** | passes by 26× |
| P90 | ≤ 0.013 Å | **0.00064 Å** | passes by 20× |

**The claim is overturned.**

- The true optimiser reproducibility floor is **≈ 0.0002 Å**, not 0.04 Å —
  **200× smaller** than claimed.
- The 0.013 Å adjacent-lanthanide radius step is **≈ 68× ABOVE** the measured
  floor, not below it.

**One number suffices here, not two.** The pre-registration warned that if basin
escape were common the honest replacement would be a within-basin floor plus an
escape rate. Escape is **0.0 % at σ = 0.02 and 0.05** and 0.8 % at σ = 0.10, so
there is no second regime to report. Idempotency at σ = 0 is exactly zero, as
expected — ANCopt from an identical start is deterministic.

## 4. What this does and does not change

**Does change.** The stated reason for closing the "geometry cannot carry
adjacent-lanthanide selectivity" question was wrong. Whatever limits geometry
here, it is *not* that the signal is below the optimiser's noise.

**Does not change.** The empirical nulls themselves stand — four independent
tests found no adjacent-lanthanide selectivity in the tabular geometry
summaries, and campaign 6 measured eight 3D encoders to be interchangeable.
Those results were right; the *explanation* attached to them was not.

**What the real limit appears to be**, from campaign 7's other lines: adjacent
lanthanide structures were generated *independently* and land in different
conformer basins — median heavy-atom RMSD **5.46 Å**, and flat in
|Δ index|, so it is sampling and not contraction. That is a
**data-generation** limit, not a numerical-precision one, and unlike a noise
floor it is fixable — which is what `automl/qc/serial_metals.py` tests.

## 5. Corrections issued

Errata added to `SYNTHESIS.md`, `WO_PREREGISTRATION.md` and `WO_RESULTS.md`
pointing here. The original text is left in place; the repo's convention is to
append an erratum rather than rewrite history.

Recorded in [`SCIENTIFIC_FINDINGS.md`](SCIENTIFIC_FINDINGS.md) §H1.

## 6. Reproducing

```bash
export XTB_BIN=$HOME/opt/xtb-dist/bin/xtb
python3 -m automl.qc.opt_reproducibility --shard 0 --num-shards 2 --workers 48
```

Every replicate's starting geometry is regenerable from its record: the seed is
`hash((stem, sigma, rep))` and is stored in each JSON.
