
---

## Erratum, 29 July 2026 — the headline is a *binned-key* result

The adjacent-pair metric blocks by `composition_key`, which **bins** the
condition columns. `strict_composition_key` — every numeric condition matched,
so the only thing varying inside a block is the lanthanide — has been in
`dataset.py` since the beginning, with a comment saying the binned key "turns a
real log D difference into label noise". The metric had never been computed with
it.

It has now been, under a pre-registered decision rule
([`DUALKEY_PREREGISTRATION.md`](DUALKEY_PREREGISTRATION.md), committed before the
contrasts existed). Both published contrasts **add under the binned key and are
not distinguishable from zero under the strict key**:

| contrast | binned | strict |
|---|---|---|
| drop-in | +0.0375 [+0.0173, +0.0510] **adds** | +0.0177 [−0.0023, +0.0367] **n.s.** |
| swap | +0.0438 [+0.0253, +0.0554] **adds** | +0.0177 [−0.0023, +0.0367] **n.s.** |

Every number in this document is therefore a **binned-key** number and should be
read as one. The effect is a weakening rather than a reversal — the strict-key
estimate is positive at P=0.93 and about half the size — and the encoder
comparison against the matched control actually *strengthens* under the strict
key (+0.0376 → +0.0716). Full account, including which key is defensible and why
neither is clean: [`DUALKEY_RESULTS.md`](DUALKEY_RESULTS.md).

---

## Erratum 2, 29 July 2026 — it is not "simplicial", it is 3D message passing

The study's oldest open question — named here, in `PI_EMAIL.md` §9 and in
`STACK_RESULTS.md` §8 — has been answered, and it broadens the claim rather than
supporting it.

Two new encoders, both over the **same** Vietoris–Rips edges, same node inputs,
same readout, same 16 seeds
([`ENCODER_PREREGISTRATION.md`](ENCODER_PREREGISTRATION.md), committed before
either was run):

- **G0** — the same network with the **triangles removed**;
- **D0** — a continuous-filter distance network with **no simplices at all**.

Both earn the stack slot, and the decisive contrast is a null:

| contrast (binned key) | Δ | 13-look Bonferroni |
|---|---|---|
| add G0 (no triangles) | **+0.0343** | [+0.0117, +0.0569] **adds** |
| add D0 (no simplices) | **+0.0284** | [+0.0025, +0.0543] **adds** |
| **S0 vs D0, same slot** | +0.0091 | [−0.0292, +0.0474] **not distinguishable** |

As single arms both *outscore* the published simplicial one: G0 +0.2459, D0
+0.2474, S0 +0.2382.

**So the simplicial structure is not the operative ingredient.** The mechanism
rule stated in this document — an arm earns a slot only if it is both accurate on
the scored metric and decorrelated from its partner — survives intact and is the
transferable contribution. The claim about the *complex* does not. Full account:
[`ENCODER_RESULTS.md`](ENCODER_RESULTS.md).
