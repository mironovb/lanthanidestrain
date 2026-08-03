# Putting the scored quantity into the loss does not help, and the ceiling is unknowable

**Bogdan Mironov · 31 July 2026**
Pre-registered in `CAMPAIGN3_PREREGISTRATION.md`, committed before any code.
Data: `c3_cells.csv`, `reliability.csv`, `speciation_audit.csv`.
32 GPU runs (24 screen + 8 post-hoc); T1 and T4 were CPU-only.

---

## Summary

| track | what it tested | outcome |
|---|---|---|
| **T1** ceiling | split-half reliability of the metric | **not identifiable** — see `CEILING_CLOSED.md` |
| **T2** pairwise head | predict `dy` directly, with its own parameters | **null, and monotone the wrong way** |
| **T3** FiLM | let 45 diluents / 9 acids reach the structural embedding | **null** |
| **T4** speciation | cation-exchange vs solvating mechanism | **not testable on this dataset** |

No cell cleared the +0.005 tune-half gate, so **no confirmatory run was spent** —
the rule was fixed in advance precisely so a negative "best" cannot be laundered
into a candidate.

## 1. The screen — 84 tune extractants, anchor D0 = +0.2362

| cell | change | tune adj-R² | vs D0 | tune strict | overall R² |
|---|---|---|---|---|---|
| D0 | anchor (sweep2's A0, unchanged) | +0.2362 | — | +0.1369 | +0.4743 |
| T2 | pair head w=1, + surrogate | +0.2109 | −0.0253 | +0.1067 | +0.4668 |
| T2W | pair head w=3, + surrogate | +0.2042 | −0.0321 | +0.1087 | +0.5101 |
| **T2X** | **pair head alone, surrogate off** | +0.1531 | **−0.0832** | +0.0761 | +0.5349 |
| T3 | FiLM on conditions | +0.1786 | −0.0577 | +0.1105 | +0.4769 |
| T23 | both | +0.1704 | −0.0658 | +0.1413 | +0.4668 |

D0 reproduces sweep2's A0 to six decimals, so the two campaigns are directly
comparable — pinned by a test, not assumed.

## 2. T2: a parametrised difference is *worse* than differencing two levels

The ordering is monotone in reliance on the pair head: −0.0253 at weight 1,
−0.0321 at weight 3, **−0.0832** with the scalar surrogate removed entirely.
The existing `--pair-loss-weight` surrogate — which constrains the difference of
two level predictions and has no parameters of its own — beats a 254 k-parameter
head trained on the same pairs.

**My first explanation was wrong, and the test that refuted it is the result.**
I proposed that the pair head's skill simply never reaches the metric: the
metric differences *level-head* predictions, while the pair head sits on a
pathway evaluation never touches. That is true as a description of the
plumbing, and it makes a prediction — route the pair head's predictions into the
per-row values the metric consumes, and the loss should be recovered.

`--pair-reconcile` does exactly that: keep each block's overall level, replace
the adjacent increments with the pair head's, walk the chain.

| cell | tune adj-R² | vs its own non-reconciled twin |
|---|---|---|
| T2REC | −1.0882 | **−1.2991** |
| T2XREC | −0.1182 | **−0.2712** |

Catastrophically worse. **There is no skill to route.** The pair head cannot
predict adjacent-lanthanide differences as well as the level head's own
differences do, so T2's failure is not a plumbing problem and the mechanism I
proposed is refuted by its own test.

A plausible remaining explanation, stated as hypothesis and *not* tested here:
the level head trains on all 4,746 rows, while the pair head sees only
within-block adjacent pairs — a far smaller and harder signal. Anyone pursuing
it should test it rather than accept it; this study's record on mechanisms
asserted without measurement is poor.

## 3. T3: the conditions do not need to reach the structure

FiLM makes the structural embedding depend on the medium, so kerosene and
nitrobenzene no longer produce byte-identical structure representations. It
costs −0.0577.

This is the genuinely surprising null. A partition coefficient is a property of
two phases, and the encoder is blind to one of them. The result says the
existing treatment — conditions as tabular columns concatenated after pooling —
is adequate, and that the interaction FiLM adds is not worth its variance on
162 extractants.

## 4. T4: the dataset cannot ask the speciation question

89 % of modelled complexes are bare Ln³⁺ cations while the species that
partitions is neutral, so charge neutralisation was the obvious missing physics.
It cannot be tested here: **97.5 % of the 162 extractants are neutral solvating
agents** — diglycolamides (CITAM, TBDGA, TEHDGA, DOODA) and BTBPs — and only
6.8 % of extractant SMILES contain P(=O) at all.

| derived column | SD | status |
|---|---|---|
| `is_cation_exchanger` | 0.041 | degenerate |
| `dentate` | 1.295 | redundant, r = 1.00 with `DENTATE` |
| `nitrate_available` | 0.412 | redundant, r = 0.99 with `cond__acid__hno3` |
| `acidconc_x_solvating` | 1.605 | redundant, r = 1.00 with `cond__acid_concentration` |
| `n_neutral_donor_groups` | 0.453 | genuinely new |

One new column is not a track. Reported as **not tested**, per the
pre-registration's own clause: a null here would read as "speciation does not
matter", when what happened is that the dataset lacks the acidic extractants —
HDEHP, HEH[EHP], Cyanex 272 — the comparison requires. 16 GPU runs saved.

## 5. T1: no ceiling can be quoted for this metric

Full account in `CEILING_CLOSED.md`. Three methods, three failures, cause
localised: 70 % of the metric's pairs come from cells carrying no replicate, and
the replicated cells scatter 26× more than those pairs imply — because 46 % of
binned blocks pool multiple exact-condition sets, and a cell acquires duplicates
preferentially when sources disagree.

**Consequence for the study:** +0.2382 cannot be expressed as a fraction of
attainable. It is answerable by a *designed* replicate study, which is a
data-collection recommendation.

## 6. What three campaigns now establish

Sweep2 and campaign 3 together tested angular information (3 routes), auxiliary
targets (3), readout and optimisation (4), a parametrised difference (3), and
condition conditioning (2) — **15 cells, all null or negative**, against an
anchor of +0.2382.

The consistent finding across all of them: **information that varies within a
composition block does not help this metric, and often hurts.** Established
directly by sweep2's A1BM post-hoc, where removing only the within-block
variation of 119 geometry columns recovered 93 % of a −0.3167 collapse.

Combined with T1, the honest position is that the +0.2382 arm is close to what
this dataset supports, and that the binding constraint is the data — arbitrary
geometries (`CONFORMER_RESULTS.md`), unrecordable condition variation, and a
noise floor that cannot even be measured — rather than the representation.

## 7. Limits, stated

- The screen is 4 seeds per cell, legitimate only because `--deterministic`
  makes runs bit-identical.
- Sections 2 (reconciliation) and 4 are **post-hoc**; neither claims
  confirmatory status and neither entered the pre-registered gate.
- T3's null is for FiLM at the embedding level. Node-level FiLM was measured to
  cost 5× (4,746 rows against 956 complexes) and was not run; it is a different
  experiment and remains untested.
- No confirmatory look was spent, so the ≥ 26-look budget is intact for a
  future campaign.

---

**Reproduce**

```bash
python3 -m automl.topo.reliability --n-splits 200
python3 -m automl.topo.speciation
automl/slurm/campaign_driver.sh automl/slurm/campaign3.sh 24 8 30
python3 -m automl.topo.c3_test --n-boot 400
sbatch --array=0-7 automl/slurm/c3_reconcile.sh
python3 -m automl.topo.c3_test --posthoc
```
