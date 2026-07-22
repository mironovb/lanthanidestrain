# S2: the variance lever worked, and it made the ensemble worse

**Bogdan Mironov · 21 July 2026**
Pre-registered in [`S2_PREREGISTRATION.md`](S2_PREREGISTRATION.md) (commit
`33324ea`, with Amendment 1), executed by
[`automl/topo/s2_test.py`](../topo/s2_test.py) (written before the runs, commit
`0b18efd`). Baseline re-pinned after S2 legitimately re-rendered one figure;
new manifest `sha256 = 3ad7c803b75d8562…` (324 artefacts, 141 OOF parquets, all
verified byte-identical except this analysis's own figure).

---

## Verdict

**The pre-registered primary endpoint is negative, and the way it is negative is
the finding.**

> **S2 − repaired FCNN baseline = +0.0066, 90 % CI [−0.025, +0.059], P = 0.39.**
> Corrected for two tests: [−0.043, +0.056]. Not distinguishable from zero
> either way.

S2 does **not** clear the strongest baseline — and it is significantly *worse*
than S0, the arm it was built to improve on (S2 − S0 = **−0.0195**, 90 % CI
[−0.025, −0.012], P = 0.00). Yet every lever did mechanically what it was
designed to. This is not a null from a broken intervention; it is a null that
teaches something about the method.

---

## 1. What was built, and that it worked

| lever | intended effect | measured effect |
|---|---|---|
| conformer augmentation | 956 → 2,797 structures (2.93×) | delivered; mean \|Δ M–L\| 0.306 Å between conformers |
| conformer pretraining | encoder learns chemistry before log D | pretrain loss 0.064 → 0.0019 |
| 32 seeds | more averaging | delivered |
| block-centred embedding | cancel common-mode block noise | delivered |

**The variance diagnosis was correct.** S2's per-seed SD is **0.0293** against
S0's 0.0468 — a **37 % reduction**. The levers reduced model variance exactly as
intended.

The per-seed *mean*, though, did not rise: S2 **+0.163 ± 0.029** vs S0
**+0.178 ± 0.047**. Lower spread around a slightly lower centre.

---

## 2. Why lower variance produced a lower ensemble

The seed-paired view (16 shared seeds) is unambiguous:

| | |
|---|---|
| mean paired difference (S2 − S0) | −0.0108 |
| S2 better on | 6 / 16 seeds |
| per-seed range | S0 spans 0.174; S2 spans 0.093 |

S2 lifts S0's **worst** seeds (7: 0.066 → 0.132; 263: 0.116 → 0.136) and pulls
down its **best** (37: 0.240 → 0.170; 51: 0.236 → 0.195). That is regression
toward the mean — textbook variance reduction, with the mean unmoved.

**The pre-registration's premise was that variance was the enemy. It was
half-right.** Variance *of a single model* is bad, but ensembling already
converts most of it into signal: S0 turns its high per-seed variance into a
+0.060 ensemble gain (+0.178 → +0.238). Reducing that variance *upstream* — with
conformers, pretraining, block-centring — leaves **less for the ensemble to
harvest**, and no higher centre to harvest it around. The two mechanisms are
**substitutes, not complements.**

Numerically: S0's ensemble sits ~+0.060 above its seed mean; S2's ~+0.055 above
its own, but from a lower mean, landing at +0.218 < +0.238. The gain per unit of
per-seed variance was actually *higher* for S2 (0.055 / 0.029 vs 0.060 / 0.047),
but there was less variance to convert.

---

## 3. Where this leaves the physics conjecture

`PI_REPORT.md` §4 named multi-conformer Boltzmann sampling "the physics lever
worth spending compute on," on the argument that single-conformer scatter in an
M–L distance (~0.05 Å) swamps the adjacent-lanthanide signal (~0.013 Å), so
averaging conformers should recover the signal.

**Averaging three conformers did not recover it** — the per-seed mean did not
move. Two readings, and this experiment cannot separate them:

1. **Three arbitrary minima in two solvents are not a Boltzmann ensemble.** The
   re-optimisations were seeded from one starting structure each and relaxed to
   whatever local minimum was nearest (median RMSD 1.87 Å); a proper thermal
   average would sample many more, weighted by energy. This is the optimistic
   reading and it is testable — the physics lever was approximated, not run.
2. **The adjacent-Ln signal is not in the geometry at this resolution.** GFN2-xTB
   may not place the coordination sphere accurately enough at 0.013 Å for any
   amount of conformer averaging to expose it.

Distinguishing these needs a real conformer search (a crest/metadynamics
ensemble, not two ANCopt relaxations) — which is the honest next step, and a
much larger compute commitment than this was.

---

## 4. Correctness, so the null is trustworthy

A negative result is only worth reporting if the machinery is sound. Five checks,
each of which caught something:

- **Featuriser identity**: rebuilding the shipped geometries through the new path
  reproduces the shipped asset on 20/20 complexes. Caught two donor-rule bugs
  (a distance cutoff where the rule is a count; a `coreCN` column that disagrees
  with what was built) that would each have marked every augmented structure.
- **No augmentation marker**: `charge_missing` is 0.0000 on originals *and*
  conformers — the 2,378 Mulliken single points exist precisely so the model
  cannot tell augmented from original.
- **Packed-batch exactness**: block-centred inference is bit-identical whether
  blocks are packed or sent singly; a split block is asserted to differ.
- **Seed pairing**: the first 16 S2 seeds are S0's, so S2 − S0 is a matched
  comparison, not two independent means.
- **Harness integrity**: S0 still re-ensembles to **+0.2382** and the published
  headline to **+0.2426 [+0.181, +0.333]**. `control_guard --verify` confirms
  every OOF parquet and result CSV byte-identical; the only changed artefact is
  this analysis's own figure.

---

## 5. What holds, and what to do next

**The control's conclusions are untouched.** Topology adds +0.049 over a matched
control and +0.026 (n.s.) over the repaired baseline; S2 does not change either.
The `QuantileTransformer` finding — a rank transform costs ~0.11 adjacent-pair R²
across every neural arm — stands regardless.

**S2 adds one transferable lesson:** on a metric where ensembling is already the
headline method, reducing single-model variance is not free improvement — it can
cannibalise the ensemble's own gain. "Train more stable models" and "ensemble
more models" compete for the same variance.

**Recommended, in order:**
1. **A real conformer ensemble** (crest / metadynamics, energy-weighted), not two
   ANCopt relaxations, to separate reading 1 from reading 2 in §3. This is the
   actual physics lever; S2 only approximated it.
2. **Drop block-centring and pretraining, keep conformers, re-test** — the
   ablation (`topo_s2_ablate`, descriptive) isolates whether any single lever
   helps while the bundle does not. Cheap; already scripted.
3. Accept that on *this* dataset and *these* geometries, topology's ceiling is
   the matched-control result, and let the objective and the scaler findings
   carry the paper.

---

*Reproduce: `python3 -m automl.topo.s2_test --n-boot 400`. Per-seed and
seed-paired views regenerate from `automl/artifacts/topo_s2/run_*.json`.*

---

## 6. Ablation (added after the fact): every lever hurt

Leave-one-out ablation, 8 seeds per cell, descriptive. Unambiguous — dropping
*any* lever **raises** the per-seed mean:

| configuration | per-seed adjacent-pair R2 |
|---|---|
| full S2 (all three levers) | +0.163 +/- 0.029 |
| drop pretraining | +0.174 +/- 0.026 |
| drop conformers | +0.184 +/- 0.021 |
| drop block-centring | +0.189 +/- 0.031 |

Each lever actively hurt; the best row is the one closest to plain S0. The same
finding as sections 1-2 from the other side: conformers, block-centring and
pretraining all shrink the per-seed spread the ensemble was harvesting. Nothing
here was worth adding. (Ablation runs: automl/artifacts/topo_s2_ablate/.)
