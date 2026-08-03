# Pre-registration: attack the target the metric actually scores, and bound what is attainable

**Bogdan Mironov · 31 July 2026** — committed before the first run of this campaign.
Follows `SWEEP2_RESULTS.md` (null on all three axes) and `AUDIT_2026-07-30.md`.

---

## 1. Why the previous campaign failed to move the metric

Sweep2 varied *what the encoder sees* (angular information), *what it is asked
to predict on the side* (auxiliary targets), and *how it reads out*. All three
were nulls, and the one large effect went the wrong way. The diagnosis from
`within_block_signal.csv` and the A1BM post-hoc is specific:

> The metric scores the **difference** between adjacent lanthanides inside a
> composition block. Almost everything the encoder was given varies *within* a
> block in ways uncorrelated with that difference (87 of 119 geometry columns,
> median |r| with `dy` = 0.049). Adding such information cannot help and can
> hurt: A1 lost 0.3167, and removing only its within-block variation recovered
> 93 % of that.

So the problem is not the *amount* of information. It is that the model
optimises levels while the metric scores differences, and that the information
which distinguishes adjacent lanthanides is largely absent from what is
modelled. This campaign attacks both, and bounds what is attainable at all.

## 2. Four tracks

### T1 — An identifiable ceiling, by split-half reliability *(no GPU)*

`ceiling_test` was withdrawn (audit E1) for measuring the wrong quantity;
`ceiling_v2` returned **NOT IDENTIFIABLE** because it estimated a noise SD on
273 duplicate cells and assumed it transferred to all 905 pairs — cells acquire
duplicates preferentially when sources disagree, so it does not.

Split-half reliability avoids the transfer assumption entirely, because it
measures the metric on **itself**:

1. Keep every `(block, metal)` cell with ≥ 2 rows (653 cells, 3,550 rows).
2. Randomly split each cell's rows into halves A and B; average each.
3. Form adjacent pairs from A alone and from B alone → `dy_A`, `dy_B`.
4. `r = corr(dy_A, dy_B)` is the reliability of a **half-sized** measurement.
   Spearman–Brown up to full size: `r_full = 2r / (1 + r)`.
5. Repeat over many random splits; report the median and a percentile interval.

Because measurement error is independent between halves, `r_full` is an upper
bound on the R² any model can achieve against this target. **This is
identifiable by construction and needs no representativeness assumption.**

*Pre-registered reading:* report `r_full` with its interval. If the observed
model R² (+0.2382) sits close to `r_full`, the task is measurement-limited and
that is the campaign's headline. If far below, there is real headroom and T2–T4
are worth their GPU.

### T2 — Predict the difference, not the level *(the main upgrade)*

Every model in this study predicts `log_D` per row; the metric then differences
block-averaged predictions. Nothing in training ever sees a pair.

**Pairwise head:** sample adjacent-lanthanide pairs *within a block*, encode
both complexes with the shared encoder, and predict `dy` directly from the pair
representation `[h_i, h_j, h_i − h_j, cond]`. Trained jointly with the existing
per-row loss so the level task still regularises the encoder.

This is the one change that puts the metric's own quantity into the objective.
`--pair-loss-weight` already re-weights a *surrogate*; this predicts `dy` itself.

### T3 — Let the conditions reach the structure *(never done)*

The dataset records **45 diluents, 9 acids**, acid and extractant concentration,
temperature. The encoder never sees any of it: conditions enter only as tabular
columns concatenated *after* pooling. So kerosene and nitrobenzene produce
identical structural embeddings.

**FiLM conditioning:** a small MLP maps the condition vector to per-channel
scale and shift applied to node features at each message-passing layer. The
structure representation becomes condition-dependent, which is what a partition
coefficient requires.

### T4 — Speciation features *(the chemistry gap)*

89 % of the modelled complexes carry net charge **+3**: they are bare Ln³⁺–ligand
cations. The species that actually partitions into kerosene is neutral —
Ln(NO₃)₃·nL or LnA₃·nL — and charge neutralisation is the physics of extraction.
Rebuilding 3D structures is out of scope here; instead encode the demand
explicitly:

- net complex charge, and counter-ions required to neutralise it
- the recorded acid's anion identity (nitrate / chloride / sulfate / perchlorate)
- ligand denticity × stoichiometry vs the metal's coordination number
- whether the ligand is acidic (cation-exchange) or neutral (solvating) — these
  extract by different mechanisms and the distinction is currently invisible

## 3. Fixed across every cell

`--arch snn --no-triangles --pair-loss-weight 2.0 --select-on adjacent
--deterministic --folds 5 --repeats 3`, dim 96, layers 3, dropout 0.15,
filtration 3.5, heavy-only, seeds {7, 11, 23, 37}. Anchor **A0** = the sweep2
anchor, unchanged, so results are directly comparable.

## 4. Decision rules, fixed now

| situation | what is reported |
|---|---|
| T1 `r_full` computed | Reported regardless of outcome. It is the campaign's one guaranteed deliverable and it is a **positive** methodological result either way: the field currently has no identifiable ceiling for this metric. |
| a cell beats A0 by **> 0.005** on the 84 tune extractants | it becomes the single confirmatory candidate |
| the confirmatory contrast excludes zero after correction for **≥ 26 looks** | **a real improvement.** Reported as the study's first. |
| it spans zero | screening noise; report the null, as in sweep2 |
| no cell clears +0.005 on tune | report the null; **do not** spend the confirmatory run |

Confirmatory stage: **16 seeds a side, both replicated**, scored **once** on the
78 confirm extractants, under **both** block keys, with the
multiplicity-respecting cluster bootstrap. Look count rises to **≥ 26** (21 from
sweep2, plus one per new cell here).

**The confirm extractants have been used once already** (sweep2's C1 contrast).
That is recorded here explicitly rather than hidden: the Bonferroni count
carries it, and any T2–T4 confirmatory claim is therefore made under a stricter
correction than sweep2's was.

## 5. What would make this campaign positive

Two independent routes, and they do not depend on each other:

1. **T1 succeeds by existing.** An identifiable ceiling is publishable whether it
   is high or low, and it is what `ceiling_test` and `ceiling_v2` both failed to
   deliver. If it is low, the study's existing +0.2382 becomes a much stronger
   result than it currently reads as.
2. **T2 is the best-motivated architecture change in the project.** It is the
   only one that puts the scored quantity into the loss.

## 6. Pre-committed honesty clauses

- Every cell is smoked for one fold before the campaign runs. Sweep2's smoke
  caught two real bugs; it is not optional.
- Any cell whose new inputs are absent or all-NaN is reported as **not tested**,
  never as a null (`sweep2_coverage.py` extended to cover T3/T4 inputs).
- T1 is computed with a fixed seed list and reports its own split-to-split
  spread, so a single lucky split cannot set the ceiling.
