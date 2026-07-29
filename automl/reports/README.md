
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
