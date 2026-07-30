# It is not "simplicial". It is 3D message passing.

**Bogdan Mironov · 29 July 2026**
Pre-registered in [`ENCODER_PREREGISTRATION.md`](ENCODER_PREREGISTRATION.md)
(commit `6abaf35`, plus Amendment 1), committed before either arm had been run.
Data: `encoder_test.csv`, `encoder_arms.csv`. Job 5278721.

---

## The verdict the pre-registration commits me to

> **THE CLAIM BROADENS.** A non-simplicial 3D encoder adds as much as the
> simplicial one, so the effect is *a learned 3D representation*, not *a
> simplicial complex*.

This is the question named as the outstanding one in three places — `PI_EMAIL.md`
§9 next step 2, `SYNTHESIS.md` ("the one boundary the filtration test could not
probe"), and `STACK_RESULTS.md` §8 ("what would settle it is a third encoder").
It is now settled.

## The arms

Both reuse the shipped Vietoris–Rips asset, so the neighbourhood definition is
identical and only the algebra over it differs. Same node inputs, same
metal-centred readout, same radial shell histogram, same embedding width (see
the capacity correction below), same
16 seeds, same folds. Asserted in `automl/tests/test_encoders.py`, including
permutation and edge-direction invariance.

| arm | what it is |
|---|---|
| **S0** | the published simplicial network (nodes, edges, **triangles**) |
| **G0** | the same network with the 2-simplex level **removed** — a graph over the same complex |
| **D0** | SchNet-style continuous filters over interatomic distance — **no boundary maps, no filtration, no simplices at all** |

## Single arms

| arm | adj R² binned | adj R² strict | overall log D R² | err-corr with repaired |
|---|---|---|---|---|
| CatBoost | +0.1422 | +0.0819 | **+0.4987** | 0.881 |
| repaired FCNN | +0.2206 | +0.1741 | +0.3218 | — |
| **S0** simplicial | +0.2382 | +0.1683 | +0.3678 | 0.897 |
| **G0** no triangles | **+0.2459** | **+0.1795** | +0.3726 | 0.904 |
| **D0** no simplices | **+0.2474** | +0.1603 | +0.3436 | 0.892 |
| T0w control | +0.2006 | +0.0967 | +0.2963 | 0.929 |

**Both new encoders outscore the published simplicial arm.** Removing the
triangles does not cost anything; it gains a little.

## The pre-registered contrasts

Multiplicity-respecting cluster bootstrap, 400 draws, 90 % interval, 13-look
Bonferroni.

### Binned key (the published metric)

| contrast | Δ | 90 % CI | 13-look | verdict |
|---|---|---|---|---|
| add **G0** to the no-topology stack | **+0.0343** | [+0.0176, +0.0455] | [+0.0117, +0.0569] | **adds** |
| add **D0** to the no-topology stack | **+0.0284** | [+0.0154, +0.0474] | [+0.0025, +0.0543] | **adds** |
| **S0 vs D0 in the same slot** | +0.0091 | [−0.0165, +0.0308] | [−0.0292, +0.0474] | **not distinguishable** |

### Strict key

| contrast | Δ | 90 % CI | 13-look | verdict |
|---|---|---|---|---|
| add **G0** | +0.0178 | [+0.0046, +0.0306] | [−0.0033, +0.0388] | n.s. after correction |
| add **D0** | +0.0186 | [+0.0062, +0.0319] | [−0.0022, +0.0395] | n.s. after correction |
| S0 vs D0 | −0.0009 | [−0.0263, +0.0271] | [−0.0442, +0.0423] | not distinguishable |

Consistent with [`DUALKEY_RESULTS.md`](DUALKEY_RESULTS.md): everything weakens
under the strict key, and the *ordering* of the encoders is unchanged.

## What this means

**The simplicial structure was never the operative ingredient.** A continuous-
filter network over interatomic distances — no boundary maps, no filtration, no
2-simplices — earns the same stack slot, with a decisive contrast that is
indistinguishable from zero in both directions.

That is a **larger** result than the one it replaces, and the pre-registration
said so before the numbers existed rather than defending against it:

> The paper's contribution becomes the *rule* for what any candidate
> representation must satisfy, and the VR complex is one instance.

The rule already exists in `SYNTHESIS.md`: an arm earns a slot only if it is
**both** strong on the scored metric **and** decorrelated from its partner. All
three 3D encoders satisfy it (err-corr 0.892–0.904, adj R² +0.238 to +0.247);
the persistence-image CNN did not (0.933); the tabular control did not (0.929).
**The rule survives; the claim about the complex does not.**

## Three consequences that should be acted on

1. **The published model can be simplified.** G0 matches or beats S0 on both
   metrics (+0.2629 vs +0.2672 in-stack binned, but +0.4400 vs +0.4369 overall,
   and +0.1919 vs +0.1918 strict) while carrying **no triangle level at all**.
   The shipped Vietoris–Rips asset devotes 9.3 M triangles and a 46 MB
   triangle→edge cache to a structure the model does not need.
2. **The persistence-image null is reframed.** It was read as evidence that
   *topology specifically* matters. It is better read as evidence that a *fixed*
   representation loses to a *learned* one — which is a different and more
   ordinary claim.
3. **The filtration replication is reinterpreted.** 3.0 / 3.5 / 4.0 Å all worked
   because they all define the same neighbourhood graph to within a scale factor,
   not because the filtration is doing topological work.

## Limits

- **`--deterministic` is off** for these arms (Amendment 1), matching the
  published S0 they are compared against. All three carry the same ~0.009
  run-to-run floor, which is why no contrast below ~0.01 is called a difference.
- **The decisive contrast is a null, not a proof of equality.** +0.0091
  [−0.0165, +0.0308] is consistent with S0 being slightly better *or* slightly
  worse. The claim is that the simplicial structure is not *necessary*, not that
  it is worthless.
- **One architecture family each.** "3D message passing" here means these two
  concrete networks over these edges, not the class in general.
- **Look count is now 13.** Every interval above is reported uncorrected and
  Bonferroni-corrected side by side.

---

**Reproduce**

```bash
ARM=G0 sbatch --array=0-15 automl/slurm/topo_encoder.sh
ARM=D0 sbatch --array=0-15 automl/slurm/topo_encoder.sh
python3 -m automl.topo.encoder_test --n-boot 400
python3 -m pytest automl/tests/test_encoders.py -q
```


---

## Correction, 30 July 2026 — equal width is not equal information

`AUDIT_2026-07-30.md` E5. This document said the arms share an embedding width so
that "no capacity difference is smuggled in". Equal *width* is not equal
*information*: `DistanceNet` holds **2 of its 9 readout blocks at exactly zero** —
the slots `SimplicialNet` fills from the triangle level — so **D0 carries strictly
less than S0 at the same width**.

That makes D0 matching S0 a **conservative** comparison and *strengthens* the
conclusion of this document. The original phrasing was nonetheless inaccurate and
is corrected here rather than left to be found.
