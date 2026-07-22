# Extended S0: the ensemble had already converged at 16 seeds

**Bogdan Mironov · 22 July 2026**
Pre-registered in [`S0X_PREREGISTRATION.md`](S0X_PREREGISTRATION.md) (commit
`86a35eb`) before any extra seed finished; analysis
([`s0x_test.py`](../topo/s0x_test.py)) written before the data landed.

---

## Verdict

**Negative, and the pre-registration predicted it.**

> **S0X (48 seeds) − repaired baseline = +0.0244, 90 % CI [−0.0056, +0.0727],
> P = 0.85 — not distinguishable.** Four-test corrected: [−0.0289, +0.0778].

S0X (+0.2369) is in fact marginally *below* the published 16-seed S0 (+0.2382);
convergence bought **−0.0017 [−0.0055, +0.0008]**.

The pre-registration stated: *"convergence plausibly buys a few thousandths, not
two centi-units. The honest prior is that this does not clear."* It did not.

---

## Why: the seed-count curve

| n seeds | 4 | 8 | 16 | 24 | 32 | 40 | 48 |
|---|---|---|---|---|---|---|---|
| ensemble adj R² | +0.2084 | +0.2285 | **+0.2382** | +0.2392 | +0.2408 | +0.2365 | +0.2369 |

The curve rises steeply to 16 seeds and then **oscillates in a ±0.002 band**.
The published 16-seed ensemble was already at the asymptote, so there was no
headroom for more seeds to find — the premise that the curve "had not visibly
flattened at 16" was wrong, and the curve is the evidence.

This **independently corroborates the split-half result**
([`STACK_RESULTS.md`](STACK_RESULTS.md) §10): 8 seeds already reach +0.2285, and
both 8-seed halves add to the stack, so the positive stack result never depended
on a large ensemble.

---

## What this does and does not change

- **The positive stack result is untouched.** It uses the published 16-seed S0,
  which still re-ensembles to **+0.2382** (asserted as a precondition before any
  number here was reported), and it was already replicated on two independent
  8-seed halves.
- **The fourth attempt at "topology beats the repaired baseline alone" fails**,
  like the three before it. Topology's value remains complementarity, not
  superiority — which is what [`SYNTHESIS.md`](SYNTHESIS.md) already says.
- **Nothing needs re-running.** 32 extra GPU-hours bought a decisive answer to a
  question that had to be asked: the ensemble was converged, so no amount of
  additional seeds was going to close a +0.026 gap.

---

*Reproduce: `python3 -m automl.topo.s0x_test --n-boot 400`. Seed-count curve in
`s0x_curve.csv`.*
