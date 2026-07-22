# Both alternative radii add: the finding is about the complex, not a tuned radius

**Bogdan Mironov · 22 July 2026**
Pre-registered in [`FILT_PREREGISTRATION.md`](FILT_PREREGISTRATION.md) before any
filtration run finished; analysis ([`filt_test.py`](../topo/filt_test.py))
committed before the data landed.

---

## Verdict

**Reading (a) of the pre-registration.** A different filtration radius also adds,
at both radii tested, so the positive result is **not** an artefact of tuning the
complex to 3.5 Å.

| radius | own adj R² | corr with baseline error | stack adj R² | Δ vs no-topology stack | 7-look corrected |
|---|---|---|---|---|---|
| **3.0 Å** | +0.2366 | +0.898 | +0.2675 | **+0.0382** [+0.0178, +0.0503] | **[+0.0141, +0.0624] adds** |
| **3.5 Å** (published) | +0.2382 | +0.897 | +0.2672 | **+0.0381** [+0.0191, +0.0495] | **[+0.0154, +0.0607] adds** |
| **4.0 Å** | +0.2319 | +0.907 | +0.2619 | **+0.0327** [+0.0140, +0.0435] | **[+0.0108, +0.0547] adds** |

All three survive Bonferroni correction for **all seven confirmatory looks** this
project has taken at "does topology add" (S0, S2, stack primary, stack decisive,
S0X, F30, F40).

These are genuinely different complexes, checked before the runs: relative to
3.5 Å, the 3.0 Å complex has 0.78× the edges and **0.59×** the triangles, and the
4.0 Å complex 1.47× the edges and **2.29×** the triangles. A near-identical
complex would have made the test uninformative whatever it returned.

---

## The mechanism predicted this, and that is the point

The mechanism from [`STACK_RESULTS.md`](STACK_RESULTS.md) §7–8 says an arm adds
only if it is **both** strong on the metric **and** decorrelated from the
partner. Every radius satisfies both:

| arm | strong? | decorrelated? | adds? |
|---|---|---|---|
| 3.0 / 3.5 / 4.0 Å | +0.232 – +0.238 | 0.897 – 0.907 | **yes, all three** |
| PI-CNN | +0.210 | 0.933 | no |
| tabular control | +0.203 | 0.928 | no |

The three radii cluster tightly on both axes and all add; the PI-CNN sits worse
on both and does not. **The mechanism was stated before this test and correctly
predicted its outcome** — which is what separates it from a story fitted to the
data after the fact.

Note also that the effect **declines monotonically as the radius grows**
(+0.0382 → +0.0381 → +0.0327) while the error correlation rises
(0.898 → 0.897 → 0.907). That is the mechanism operating within the family: a
looser complex adds more distant, less discriminating simplices, which makes it
marginally more redundant with the fingerprint model.

---

## What the claim now is

> **Message passing over a Vietoris–Rips complex of the 3D structure** supplies
> adjacent-pair selectivity information that fingerprint models lack, across a
> range of filtration radii — and blending it with the best no-topology stack
> raises adjacent-pair R² from +0.2263 to +0.2619–0.2675.

This is broader than the pre-filtration claim ("the simplicial encoder at
3.5 Å"). It is stated for the Vietoris–Rips complex because that is what was
tested across radii; the one alternative representation tried — a
persistence-image CNN at the shipped, **untuned** settings — did not reproduce
it, which bounds what has been demonstrated rather than establishing that
persistent homology cannot work here. Tuning the image construction is the
obvious untried experiment.

---

## Guards

The published S0 re-ensembles to **+0.2382**, asserted as a precondition before
any number above was reported. `control_guard --verify` passes: 324 artefacts
byte-identical.

---

*Reproduce: `python3 -m automl.topo.filt_test --n-boot 400`. Mechanism numbers in
`filt_mechanism.csv`.*
