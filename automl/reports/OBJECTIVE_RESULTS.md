# The level term was not the binding constraint — and my diagnosis pointed the wrong way

**Bogdan Mironov · 29 July 2026**
Pre-registered in [`OBJECTIVE_PREREGISTRATION.md`](OBJECTIVE_PREREGISTRATION.md)
plus Amendment 1, both committed before the first run existed.
Data: `objective_cells.csv`, `objective_test.csv`. Sweep 48/48, job 5280890.

---

## The verdict the pre-registration commits me to

Two of its four outcome rows fire together:

> **The level term was NOT the binding constraint.** An objective spending ~91 %
> of its gradient on nuisance is not what limits this problem, which points at
> the representation or the data rather than the loss.

and

> **The best cell is `level_weight = 1.0` + binned key.** The published objective
> was already near-optimal on this axis and the diagnosis in §1, though
> arithmetically correct, does not bind. Say so.

## The diagnosis, and why it failed

§1 measured that the composition-block mean carries Var 1.82 (binned) / 2.41
(strict) against the within-block contrast's 0.84 / 0.25 — so the published Huber
objective spends **68–91 % of its gradient on a quantity the metric never
reads**. That arithmetic is correct. The inference from it was wrong.

The `level_weight` main effect is **monotone in the wrong direction**:

| `level_weight` | mean tune adj R² (binned), 16 runs each |
|---|---|
| 0.1 | +0.2286 |
| 0.3 | +0.2351 |
| **1.0** | **+0.2412** |

**More weight on the block mean is better, not worse.** The level term is not
wasted gradient — it anchors the representation. A network told only to get
within-block contrasts right has no reason to place the blocks anywhere sensible,
and evidently learns a worse encoder for it. The same ordering appears in overall
log D, which is what one would expect if the level term is doing real work:
+0.289 → +0.335 → **+0.385** as its weight rises.

That is a clean falsification of a diagnosis I was confident enough about to
build a 48-run campaign on. It is the fourth such falsification today, and the
reason the pre-registration wrote all four outcomes down in advance.

## The block-key main effect

| `block_key` used for training | mean tune adj R² (binned), 24 runs each |
|---|---|
| **`composition_key`** | **+0.2400** |
| `strict_composition_key` | +0.2298 |

Training against the strict definition of "identical conditions" is **worse**,
even after Amendment 1 restored the 1,573 singleton blocks the contrast objective
would otherwise have dropped. So the strict key is a better *metric* (fewer
condition effects mistaken for selectivity) and a worse *training signal* (fewer
usable pairs per block, noisier per-block means). Those are separate questions and
this separates them.

The effect is much larger on overall log D, where the strict-key cells collapse
(+0.204, +0.290, +0.373 against the binned cells' +0.289, +0.335, +0.385) — again
consistent with the level term being load-bearing.

## The confirmatory contrasts

Selected cell: `level_weight = 1.0`, `composition_key`. Scored **once** on the 78
confirm extractants. Multiplicity-respecting cluster bootstrap, 400 draws, 90 %
interval, 19-look Bonferroni.

| key | contrast | Δ | 90 % CI | 19-look | verdict |
|---|---|---|---|---|---|
| binned | OBJ − **S0** | −0.0233 | [−0.0325, −0.0044] | [−0.0472, +0.0005] | worse uncorrected, n.s. corrected |
| binned | OBJ − no topology | +0.0172 | [+0.0029, +0.0245] | [−0.0011, +0.0354] | adds uncorrected, n.s. corrected |
| strict | OBJ − **S0** | −0.0170 | [−0.0324, +0.0052] | [−0.0489, +0.0150] | not distinguishable |
| strict | OBJ − no topology | **+0.0150** | [+0.0068, +0.0232] | **[+0.0011, +0.0289]** | **adds** |

**The decomposed arm does not beat S0** in the same slot, on either key. It is
worse by about 0.02 under the binned key.

## One positive worth stating carefully

The last row is the **only strict-key contrast in the whole re-analysis that
survives Bonferroni**: the decomposed arm adds **+0.0150 [+0.0011, +0.0289]** to
the no-topology stack after correction for all 19 looks. S0's own strict-key
contrast did not
([`DUALKEY_RESULTS.md`](DUALKEY_RESULTS.md): +0.0177, 10-look [−0.0128, +0.0482]).

**Do not read that as the decomposed arm being better than S0 on the strict key.**
Two reasons, and both matter:

1. **Different row sets.** This contrast is on the 78 **confirm** extractants
   only, by design — one confirmatory look. The dual-key S0 contrast used all 162.
   The intervals are not comparable, and the tighter one here is partly a
   different sample, not a better model.
2. **The direct comparison exists and is a null.** OBJ − S0 on the strict key is
   −0.0170 [−0.0324, +0.0052]. If the decomposed arm were genuinely better on
   this metric, that contrast would say so, and it does not.

What it *does* support is narrower and still worth having: **a 3D arm can add to
the no-topology stack under the strict key**, which nothing else in this
re-analysis demonstrated. It is one confirmatory look on 78 extractants and should
be replicated before it is leaned on.

## What this closes, and what it leaves

**Closed.** The loss is not the constraint. Together with
[`full_stack.csv`](full_stack.csv) (recombination exhausted),
[`ENERGY_RESULTS.md`](ENERGY_RESULTS.md) (energetics hurt) and
[`CONFORMER_RESULTS.md`](CONFORMER_RESULTS.md) (conformer search cannot help),
**four routes to the +0.412 of headroom are now measured and closed**: a better
combination, better features, better geometries, and a better objective.

**Left open.** The pre-registration named where a negative here points: *the
representation or the data*. [`ENCODER_RESULTS.md`](ENCODER_RESULTS.md) already
showed the representation family is broad and interchangeable — simplicial,
graph and distance encoders are indistinguishable — which makes the
representation an unpromising place to look next. That leaves **the data**: 953
distinct complexes, 905 adjacent pairs, and a measured ceiling of +0.679 that no
model here reaches half of.

## Limits

- `--deterministic` off, matching the arms compared against (Amendment 1 to
  `ENCODER_PREREGISTRATION.md`, same reasoning). All cells carry the ~0.009
  run-to-run floor; the level-weight main effect spans 0.0126, which is above it
  but not far above it. Eight seeds per cell gives a per-cell SE around 0.017, so
  **the cells are not individually rankable** and only the main effects are read —
  as fixed in advance.
- The `level_weight = 1.0` cell is **not** the published objective. It is a
  per-block Huber on the level plus the same pair term; the published one is a
  per-row Huber on the raw target. So "the published objective was near-optimal on
  this axis" means exactly that — on this axis — not that the two are equivalent.
- The look count is now **19**. Every interval is reported uncorrected and
  corrected; the corrected column is the one that counts for a claim looked at
  nineteen times.

---

**Reproduce**

```bash
automl/slurm/campaign_driver.sh automl/slurm/topo_objective.sh 48 8 34
python3 -m automl.topo.objective_test --n-boot 400
```
