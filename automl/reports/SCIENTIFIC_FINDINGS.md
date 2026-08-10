# Scientific findings: what is established, what is hypothesis, what is dead

A standing register, kept separate from any one campaign's results file. Each
entry states the claim, its **status**, the measurement behind it, and — where a
mechanism is asserted — **the test that would falsify it and whether that test
has run**.

The rule this file exists to enforce, learned the hard way in this project: *a
mechanism read off the code is a hypothesis, not a finding.* Four causal
explanations in earlier campaigns were derived by reading the implementation and
all four were wrong. Nothing here is promoted to ESTABLISHED without a test that
could have come out the other way.

**Status vocabulary**

| status | meaning |
|---|---|
| **ESTABLISHED** | measured, with a held-out or replicated comparison and an interval |
| **SUPPORTED** | measured, but selection-exposed or interval touching zero |
| **HYPOTHESIS — TEST PENDING** | stated, falsifying test designed and running |
| **HYPOTHESIS — UNTESTED** | stated, no test designed yet. Do not cite as a reason |
| **FALSIFIED** | its own test refuted it. Kept, because the refutation is the finding |

---

## A. Findings about the objective

### A1. Training the contrast, not the absolute value, is the single largest lever
**ESTABLISHED** (earlier campaign, reproduced here). `--pair-loss-weight 2.0`:
+0.186 for the SNN, +0.042 PI-CNN, +0.030 tabular; Shapley 52.5 % of total gain.

### A2. Weighting the contrast term toward adjacent pairs helps; **collapsing replicates does not**
**ESTABLISHED** (39-cell screen, 4-seed ensembles, matched control).

| change | Δ vs control |
|---|---|
| adjacent emphasis 3 → 10, replicates kept | **+0.0519** |
| adjacent pairs only, replicates kept | **+0.0365** |
| replicates collapsed to (block, metal) cells — 16 cells | **−0.0001 to −0.1497, all negative** |
| emphasis 10 **and** collapsed | −0.0167 |

Every one of 16 cells using `--pair-metric-align` scored at or below control.

### A3. Why collapsing fails: data poverty
**FALSIFIED, decisively.** The `--pair-subsample` control thins the pair set to
the *same count* alignment leaves (7.5 %, ~1,349 of 18,065) **without**
collapsing replicates. Starvation predicts it should hurt about as much.

| cell | pair terms kept | Δ vs control |
|---|---|---|
| thin to 7.5 %, replicates kept | ~1,349 | **+0.0791** |
| thin to 25 % | ~4,500 | +0.0010 |
| thin to 50 % | ~9,000 | −0.0226 |
| **collapse** to (block, metal) cells | ~1,349 | **−0.0636** |

Identical pair counts, opposite signs. **It is not data poverty.** The damage is
the collapsing itself — i.e. *what* is differenced, not *how many* differences
there are. Collapsing replaces row targets with smoothed cell means; random
thinning leaves every target intact and merely varies which pairs are seen each
epoch, so over training the model still sees them all.

This is the fourth mechanism I proposed in this campaign and the fourth
falsified by its own test (with B2, B3, B6-left-tail). The pattern is now the
finding: on this problem, mechanisms inferred from the code have a very poor
hit rate, and only the pre-designed falsifying test settles anything.

### A5. Aggressive random pair subsampling: real but **much smaller than first measured**
**SUPPORTED, at a third of the headline.** At 4 seeds the 7.5 % cell scored
**+0.0791**. At 12 seeds it is **+0.0437**. The effect survives — it is still a
positive, cheap, encoder-side lever — but the number I first reported was
inflated by seed noise, exactly as I flagged when reporting it.

The falsification it delivered (A3) is unaffected: thinning to the same pair
count as collapsing gives +0.0437 against collapsing's −0.0636, so the sign
difference that kills the starvation hypothesis is intact and large.

---

## B. Findings about the loss function

### B1. MAE instead of RMSE nearly doubles the tabular arm's selectivity
**ESTABLISHED.** `CatBoostRegressor(loss_function="MAE")`, nothing else changed:

| | adjacent-pair R² | 90 % cluster bootstrap | log D R² |
|---|---|---|---|
| full set (905 pairs) | +0.1422 → **+0.2487** (+0.1066) | [+0.0375, +0.1420] **excludes 0** | +0.4987 → **+0.5102** |
| held-out third (301) | +0.2261 → **+0.2812** (+0.0552) | [+0.0177, +0.1009] **excludes 0** | — |

Improves **both** scored quantities simultaneously — no metric tension.

### B2. The mechanism is **not** robustness to outliers
**FALSIFIED** (my own hypothesis, refuted by its own test). Robustness predicts
Huber should also help. It does not:

| loss | adjacent-pair R² |
|---|---|
| RMSE | +0.1594 |
| Huber δ = 1.0 | +0.1649 |
| Huber δ = 0.3 | +0.1725 |
| **MAE** | **+0.2188** |

### B3. The mechanism is median-seeking rather than mean-seeking
**FALSIFIED.** `Quantile:alpha=0.5` reproduces MAE exactly (+0.2188 vs +0.2188),
confirming the test is well-posed — and then the **0.7 quantile beats it**:

| CatBoost loss | adjacent-pair R² |
|---|---|
| RMSE | +0.1594 |
| Quantile α = 0.3 | +0.1523 |
| Quantile α = 0.5 ≡ MAE | +0.2188 |
| **Quantile α = 0.7** | **+0.2384** |

The median is not optimal, so "MAE works because it targets the median" is
refuted. Two of my mechanisms for B1 have now been falsified by their own tests
(B2 robustness, B3 median). The measurement was right both times; the story
was wrong both times.

### B6. An UPPER quantile is optimal. **Three mechanisms proposed, three falsified.**
**EFFECT: SUPPORTED. MECHANISM: UNKNOWN — and that is the honest state.**

The effect is large and reproducible on the selection half:

| CatBoost loss | adjacent-pair R² |
|---|---|
| RMSE (published) | +0.1594 |
| Quantile α = 0.3 | +0.1523 |
| Quantile α = 0.5 ≡ MAE | +0.2188 |
| **Quantile α = 0.7** | **+0.2384** |

Three explanations have been offered and each was killed by its own test:

1. **Robustness to outliers** — predicts Huber helps. It does not (δ=1.0 +0.1649,
   δ=0.3 +0.1725, against MAE's +0.2188). **FALSIFIED.**
2. **Median-seeking** — predicts α=0.5 is optimal. α=0.7 beats it. **FALSIFIED.**
3. **Down-weighting an untrustworthy left tail** — log D *is* strongly left-skewed
   (skew −0.712; mean +0.267 below median +0.352; 58 % of replicate cells
   left-skewed), so the story was plausible. It predicts the gain concentrates in
   blocks containing low log D. Measured, and it goes the **other way**:

   | blocks | q70 | RMSE | gain |
   |---|---|---|---|
   | **with** a low-log D row (≤10th pct) | +0.0874 | +0.0269 | +0.0604 |
   | **without** one | +0.3808 | +0.2844 | **+0.0964** |

   The gain is *larger* where the left tail is absent. **FALSIFIED.**

What is left standing: an asymmetric pinball loss with α ≈ 0.7 buys ~+0.08
adjacent-pair R² on the tabular arm, the effect is broad rather than localised to
any identifiable subpopulation, and **no proposed mechanism survives**. A
constant offset is excluded a priori, because the metric scores within-block
differences in which any block-common shift cancels exactly.

This is recorded as an unexplained empirical regularity rather than dressed in
the first story that fits. Following this project's own history — four mechanisms
read off the code in earlier campaigns, four wrong — the prior on any untested
explanation here should be low.

**The shape prediction was CONFIRMED.** B6 pre-registered that a real effect
should show *a smooth interior maximum above 0.5, not a monotone climb to 1.0*.
The sweep:

| α | 0.30 | 0.50 | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| adjacent R² | +0.1523 | +0.2188 | +0.2304 | +0.2277 | **+0.2384** | +0.2267 | +0.1881 | +0.1730 | +0.1114 |
| log D R² | +0.4870 | +0.4963 | +0.4952 | +0.4841 | +0.4783 | +0.4634 | +0.4399 | +0.3859 | +0.2852 |

A clean interior maximum at α ≈ 0.70, rising from 0.30 and collapsing by 0.90.
So the effect is well-behaved and tunable even though no mechanism for it
survives — which is a stronger position than a monotone trend, because a
monotone trend to the boundary usually indicates an artefact.

**Important nuance: α = 0.7 is NOT the free lunch that α = 0.5 is.** log D R²
degrades monotonically above ≈ 0.6 (+0.4963 → +0.4783 at 0.70 → +0.2852 at
0.90), so α = 0.7 *trades* log D accuracy for selectivity, whereas MAE (α = 0.5)
improved both. Which to prefer depends on which quantity is being bought, and
that should be stated rather than hidden behind a single headline number.

*Still to run:* full-data confirmation of α = 0.70 and 0.60 at 16 seeds, and
whether the effect is architecture-general (pinball loss on the 3D neural arm)
or tree-specific.

### B4. Hyperparameters around MAE are already right
**SUPPORTED.** Re-gridding depth / learning rate / l2 / rsm *under MAE* does not
beat plain MAE (best `rsm=0.3` at +0.2209 vs +0.2188, within noise). The loss
function is the whole effect.

### B5. The robust-loss lever is largely tree-specific
**SUPPORTED.** The neural analogue (`--level-loss mae`) is worth only **+0.0216**
against CatBoost's ~+0.107 equivalent, and `--level-loss mse` is worth +0.0043.
Huber δ = 0.2 on the neural level term gives +0.0660 — so *some* of it transfers,
but not the bulk.

---

## C. Findings about the 3D representation

### C1. A threshold was discarding usable 3D signal
**SUPPORTED** (selection-exposed; effect size replicates across partitions).
Every published run thresholded the Vietoris–Rips asset at 3.5 Å although it
contains edges to 4.0 Å. Using all of them, with the radial basis widened to
match (`--filtration-max 4.0 --rbf-bins 64`):

| partition | Δ vs incumbent D0 |
|---|---|
| full set | +0.0139 |
| selection half | +0.0149 |
| held-out third | **+0.0106**, 90 % CI [−0.0015, +0.0227] |

The **stability of the effect size across all three partitions** is the evidence;
the interval alone touches zero.

*Securing tests, now running:* the same widening on a **different architecture**
(`snn --no-triangles`, 3.5 vs 4.0 Å) and the widening **without** the wider basis
(`dist`, 4.0 Å at 32 bins). Architecture replication is the strongest securing
move available on a dataset with no unspent extractants.

### C2. The receptive field is exhausted at 4 Å — **reinstated, now properly evidenced**
**ESTABLISHED at 12 seeds.** This claim was stated early, **withdrawn** when the
8 Å cell arrived at 4 seeds looking like the largest, and is now reinstated on
adequate data. All cells at 12 seeds, matched rows:

| graph | seeds | Δ vs control | (was, at 4 seeds) |
|---|---|---|---|
| **4.0 Å (shipped ceiling)** | 12 | **+0.0649** | +0.0711 |
| 8.0 Å rebuilt, 11.5 M edges | 12 | +0.0579 | +0.0880 |
| 6.0 Å rebuilt, 6.1 M edges | 12 | +0.0530 | +0.0718 |
| 5.0 Å rebuilt, 4.0 M edges | 12 | +0.0323 | +0.0551 |
| k-NN 24, degree-based | 4 | +0.0306 | — |

4.0 Å is the optimum; everything wider is worse, monotonically in the seed-noise
sense. The 8 Å cell that triggered the withdrawal fell from **+0.0880 to +0.0579**
when taken from 4 to 12 seeds — a −0.030 move, squarely in line with E5.

The episode is worth keeping as a worked example: the original claim was right,
the *evidence* for it was inadequate, withdrawing it on 4-seed data was correct
procedure, and only 12 seeds settled it. Being right for inadequate reasons is
not a reason to keep a claim.

**Practical consequence:** the shipped asset's 4.0 Å ceiling sits at the optimum,
so the `build_neighbor_graph.py` machinery — although verified exact (0
disagreements of 2,301,232 edges) — buys nothing here. That is a useful negative:
nobody needs to rebuild these graphs again.

### C5. Solvent-relaxed geometry does not help the encoder
**ESTABLISHED (negative).** Three geometry arms on an **identical 96-extractant
support** (the intersection all three assets cover — the campaign-4 discipline,
without which a cross-arm contrast compares datasets rather than arms), 8 seeds,
584 pairs:

| geometry | adjacent-pair R² |
|---|---|
| **shipped (gas-phase GFN2-xTB)** | **+0.1866** |
| water-reoptimised | +0.1788 |
| octanol-reoptimised | +0.1060 |

Neither solvent-relaxed set improves on the shipped structures; octanol is much
worse (it also covers only 149 of 162 extractants). This closes for the *encoder*
the question `WO_RESULTS.md` closed for the tabular block: the solvent
reorganisation assets, ~2,400 additional GFN2-xTB optimisations, do not carry
usable adjacent-pair signal in either form.

### C3. Widening helps the distance encoder but **hurts** the simplicial one
**SUPPORTED.** Published SNN runs by radius (per-seed mean): 3.0 Å +0.1914,
3.5 Å +0.1850, 4.0 Å +0.1800 — monotone *decreasing*. The distance net gains
from the same change. Candidate explanation: for the SNN, widening also
multiplies 2-simplices (they scale ~r⁶), so "more neighbourhood" and "more
triangles" are confounded in that arm. **HYPOTHESIS — UNTESTED**; the `g0`
(no-triangles) replication now running is exactly the discriminating case.

### C4. Adjacent-pair emphasis saturates by 10
**ESTABLISHED.** Emphasis 10 +0.0519; 20 +0.0704 and 40 +0.0760 both score
*below* the metal-physics block alone, and neither survives confirmation.

---

## D. Findings about features

### D1. A free pre-screen predicts A1-style collapses but not generalisation
**ESTABLISHED, with an important limit.** `within_block_signal`-style screening —
correlate each candidate column's within-block difference against `dy` before
training on it — costs no GPU and no confirmatory look. The aqueous/f-shell
metal block cleared it decisively (`mphys__dG_hyd` at 0.215, above the incumbent
ionic radius's 0.171 and above the best of sweep2-A1's geometry columns at
0.183; A1's median was 0.0495 and it cost −0.3167).

The block then **did not repeat A1's collapse** (+0.0853 on screening) and **did
not generalise** (−0.0044 vs D0 on held-out). So the pre-screen is **necessary
and not sufficient**: it predicts harm, not benefit.

### D2. Aqueous-phase and f-shell descriptors do not generalise
**ESTABLISHED (negative).** +0.0853 on selection data, −0.0044 on held-out.

---

## E. Findings about methodology

### E1. Selection on ~40 cells inverts the ranking
**ESTABLISHED, and the most transferable result here.** Screening ranked
z1 > e0 > a7 > z0 > b7 > control. On extractants that took no part in choosing,
the order was b7 > **control** > a7 > e0 > z0 > **z1**. The pre-declared endpoint
came **last of six**; the untouched control came second.

### E2. Held-out partitions differ in difficulty, and the bar must be set on the partition
**ESTABLISHED.** The incumbent D0 scores +0.2474 on the full set and **+0.3030**
on the report third. An endpoint of +0.2903 there reads as a large win against
the published headline and is in fact a regression. Fix the bar *per partition*,
before the look.

### E3. Shrinkage from screening to held-out is worse than "about half"
**ESTABLISHED.** Predicted ~50 % in advance. Actual: 23–46 % from screening to
confirmation, then **to zero or negative** on held-out. A screening gate of
+0.02 was far too permissive — three axes cleared it four times over and none
survived.

### E4. Cell identity must key on recorded config, never on tag
**ESTABLISHED (by catching a real error).** When `--level-loss` was added without
registering it as a cell key, the analysis silently treated the MAE arm and the
control as the same cell — caught only by a duplicate-seed guard.

### E5. Within-screening winner's curse is large: ~0.04 on the top cells
**ESTABLISHED, and it is the most operationally useful number here.** Five cells
were taken from 4 seeds to 12:

| cell | Δ at 4 seeds | Δ at 12 seeds | move |
|---|---|---|---|
| `x6_mae_f40` | +0.0910 | +0.0493 | **−0.042** |
| `h0_sub075` | +0.0791 | +0.0437 | **−0.035** |
| `e0_mphys` | +0.0853 | +0.0773 | −0.008 |
| `b7_f40_fb64` | +0.0711 | +0.0649 | −0.006 |
| `a7_w10_only` | +0.0519 | +0.0630 | +0.011 |

**The two largest 4-seed deltas shrank the most** — the signature of selection
noise, not of a real ordering. A 4-seed screen on this metric cannot rank cells
separated by less than ~0.04, and the top of the table is systematically
inflated because the maximum of many noisy cells is biased upward.

Operational consequence: **no 4-seed cell may be promoted, quoted, or used to
close an axis.** Both leads I reported this campaign at 4 seeds
(`mae_f40` +0.0910, `sub075` +0.0791) lost roughly half their apparent size at
12. This is the same failure that produced the campaign's null endpoint, one
level down.

---

## F. The robust-loss optimum (added after §10 of C6_RESULTS)

### F1. The 3D arm's gain decomposes cleanly, and the two families want OPPOSITE losses
**ESTABLISHED** (16-seed arms, full protocol, matched rows).

| arm | graph | level loss | full (905) | held-out (301) |
|---|---|---|---|---|
| `d0_dist` incumbent | 3.5 Å | Huber δ=1.0 | +0.2474 | +0.3030 |
| `s2_dist_f40_fb32` | 4.0 Å / 32 | Huber δ=1.0 | +0.2586 | +0.3124 |
| `b7_f40_fb64` | 4.0 Å / 64 | Huber δ=1.0 | +0.2613 | +0.3136 |
| `t2_mae_only` | 3.5 Å | **MAE** | +0.2318 | +0.3053 |
| `t0_mae_f40` | 4.0 Å / 64 | **MAE** | +0.2300 | +0.3048 |
| **`t1_d02_f40`** | 4.0 Å / 64 | **Huber δ=0.2** | **+0.2704** | **+0.3318** |

Reading the rows:

- **the graph change alone** is worth +0.0139 full / +0.0106 held-out;
- **MAE alone HURTS the neural arm** (−0.0156 full) and *cancels* the graph gain
  when combined with it (t0: −0.0174 full);
- **Huber δ = 0.2 plus the graph** is worth +0.0230 / +0.0288 — more than either
  component, and the only combination that beats the incumbent decisively.

**The two model families want opposite amounts of robustness.** For CatBoost,
full L1 (MAE) is the single largest lever in the campaign (+0.1066). For the
neural encoder, full L1 is *harmful* and the optimum sits at partial robustness
(δ ≈ 0.2). Any recommendation to "use a robust loss" that does not distinguish
the two is wrong for one of them.

### F2. Prediction recorded in advance — **FALSIFIED**
I predicted a *smooth interior maximum between δ ≈ 0.1 and 0.3*, and stated the
falsification condition: flat, monotone, or peaking at an endpoint. The 12-seed
sweep, all cells carrying the 4.0 Å graph:

| δ | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.50 | 1.00 (published) |
|---|---|---|---|---|---|---|---|
| Δ vs control | +0.0147 | **−0.0345** | +0.0271 | +0.0397 | +0.0431 | **+0.0665** | +0.0649 |

There is **no interior maximum**. The curve climbs to δ ≈ 0.5 and is flat to
δ = 1.0 — the published value is already at the plateau. δ = 0.2, the value in
the arm I had promoted, is *worse* than the published δ = 1.0 (+0.0397 vs
+0.0649). The δ = 0.10 point at −0.0345 is a non-monotone outlier even at 12
seeds, which is itself a warning about this axis's noise.

**Consequence — F3.**

### F3. The robust-loss component of the "best arm" is NOT established
**CORRECTION to C6_RESULTS §10 as first written.** `t1_d02_f40` (δ = 0.2) was
reported as the best single arm in the study. Tested directly against
`b7_f40_fb64` (δ = 1.0), *same graph, same protocol, same 16 seeds*:

| partition | t1 (δ=0.2) | b7 (δ=1.0) | Δ | 90 % CI |
|---|---|---|---|---|
| held-out (301) | +0.3318 | +0.3136 | +0.0182 | [−0.0072, +0.0397] |
| selection (604) | +0.2527 | +0.2460 | +0.0066 | [−0.0053, +0.0123] |
| full (905) | +0.2704 | +0.2613 | +0.0090 | [−0.0006, +0.0166] |

**Every interval spans zero.** The two arms are not distinguishable, and the
independent 12-seed δ sweep puts δ = 0.5–1.0 ahead of δ = 0.2. So the robust-loss
half of that result does not survive; `t1`'s higher held-out number is best read
as seed noise on top of the graph change.

**What survives is the graph change alone** (C1): 4.0 Å + 64-bin basis, worth
+0.0106 to +0.0139 consistently across all three partitions. That is the real —
and modest — 3D-specific result of this campaign.

This is the fifth mechanism proposed and falsified here, and the second time a
promoted arm dissolved under a direct matched contrast. Both times the failure
mode was the same: a difference of ~0.02 taken at face value on an axis whose
seed noise is ~0.03.

---

## G. Why 3D engineering keeps returning nothing — the family is one-dimensional

### G1. Eight 3D encoder variants have effective rank 1.05
**ESTABLISHED, and it is the most consequential 3D result in the study.**

Take every 3D arm built at full protocol (16 seeds, 905 adjacent pairs) — two
architectures (simplicial MPSN and continuous-filter distance convolution),
three graph cutoffs (3.5 / 4.0 Å + rebuilt 5–8 Å), four level losses (Huber
δ=1.0/0.2, MAE, quantile), and two feature blocks (baseline_2d, +mphys) — and
decompose their adjacent-pair predictions:

| | PC1 | PC2 | PC3 | effective rank |
|---|---|---|---|---|
| **8 3D arms** | **97.36 %** | 1.20 % | 0.55 % | **1.05 of 8** |
| + CatBoost-MAE and the fingerprint net | 87.49 % | 7.59 % | 2.96 % | 1.30 of 10 |

Pairwise error correlation, mean [min, max]:

| pair type | correlation |
|---|---|
| **3D vs 3D** | **0.990 [0.982, 0.997]** |
| tabular vs tabular | 0.918 [0.881, 0.964] |
| 3D vs tabular | 0.877 [0.834, 0.905] |

Some correlation is expected — all arms fit the same target. The finding is
**relative**: the 3D family is far more internally redundant (0.990) than the
tabular family is (0.918), and *no* pair of 3D arms is as decorrelated as the
*average* 3D-tabular pair.

**Interpretation: on this metric the 3D representation family has essentially
one degree of freedom.** Changing the architecture, the neighbourhood
definition, the loss, or the feature block moves the prediction along the same
axis. Adding a second 3D arm to a stack therefore contributes nothing, and the
nested NNLS confirms it directly — given `b7`, `g0` and `t1` together, it puts
0.99 of the weight on one and ~0.00 on the others.

**This retrodicts the whole history of the project.** Seven campaigns of encoder
variation — simplicial vs graph vs distance, filtration radii, persistence
images, conformers, angular readouts, attention pooling, capacity — all returned
≈ 0 in combination. G1 says why: they were all sampling one direction. It also
predicts that further 3D *encoder* engineering on this dataset is futile, and
that the only way to extract more from geometry is a representation whose errors
point somewhere else — which none of simplicial, graph, distance or
persistence-image achieved.

### G1b. Strength confound — **partial correction to G1**
**TESTED, and it removes one of G1's two claims.**

Better models necessarily agree more: as R² rises, the residual shrinks toward
the shared irreducible noise. Across all 153 arm pairs, error correlation and
model strength are related at **Pearson r = +0.696**. So G1's raw comparison
(3D-vs-3D 0.990 against tabular-vs-tabular 0.918) was **partly a strength
artefact** — the tabular arms are simply weaker.

Restricting to pairs where **both** arms exceed adj R² = 0.23:

| pair type | n | mean correlation |
|---|---|---|
| both 3D | 91 | **0.991** [0.982, 0.999] |
| 3D + tabular | 14 | **0.893** [0.884, 0.897] |
| both tabular | **0** | — cannot be computed |

**What survives:** at *matched strength*, 3D arms are still far more like each
other (0.991) than like a tabular arm (0.893). The core of G1 holds.

**What does not survive:** the claim that "the tabular family is more internally
diverse than the 3D family". Only one tabular arm clears R² = 0.23, so that
comparison cannot be strength-matched and is withdrawn.

**Why G2 is unaffected:** G2 compares an arm against *itself* under reseeding,
so both sides have identical strength by construction. It is the one form of
this measurement that no strength confound can touch — which is why it, not G1,
is the finding to rely on.

### G2. The matched control: **architecture moves the prediction no more than the random seed does**
**ESTABLISHED.** G1's PCA could be an artefact of ensembling — averaging 16 seeds
removes seed noise and mechanically inflates correlation. The matched test uses
**8-seed ensembles on both sides**:

- *within* a config: two **disjoint** 8-seed halves of the identical configuration;
- *across* configs: an 8-seed ensemble of arm A vs an 8-seed ensemble of arm B.

| comparison | mean error correlation |
|---|---|
| **within** one config (disjoint seed halves) | **0.9900** [0.9862, 0.9916] |
| **across** configs (architecture, graph, loss, features all differ) | **0.9864** [0.9806, 0.9925] |
| gap | **−0.0036** |

**They are the same number.** Re-drawing the random seeds of one configuration
perturbs its adjacent-pair predictions as much as replacing a simplicial complex
with a continuous-filter distance convolution, changing the neighbourhood
definition, and changing the loss.

This is the rigorous form of G1 and it is far stronger than the PCA statement:
the architecture axis carries **no signal beyond seed noise** on this metric.

**Scope, stated precisely.** This is a claim about `sel_adj_logSF_r2`, not about
the encoders in general. The same arms *do* differ on overall log D R² (G0
+0.3726, D0 +0.3436, T0w +0.2963), so they are not interchangeable models —
they are interchangeable *for the adjacent-pair contrast*. The saturation is a
property of the metric-representation pair, not of the representations alone.

**Falsification attempted, 11 representations, none escapes.** The condition:
error correlation with `b7` below 0.95 — i.e. further from `b7` than `b7` is
from its own reseeding — while keeping adjacent-pair R² above +0.20.

| arm | corr with `b7` | adj R² | escapes? |
|---|---|---|---|
| `T0w` — **tabular control, no 3D** | **0.949** | +0.2031 | yes, but it is not a 3D arm |
| `P0` — persistence-image CNN | **0.958** | +0.2101 | no |
| every learned 3D encoder (9 arms) | 0.986 – 0.997 | +0.2328 – +0.2816 | no |

Two things worth noting. The **only** arm that escapes is the one with no
geometry in it — which is the positive half of the finding restated: the tabular
family really does occupy a different direction. And the closest 3D arm to
escaping is `P0`, a *fixed* topological descriptor fed to a CNN rather than
learned message passing — so representation **class** buys slightly more
independence than architecture within a class, but still not enough.

---

## H. Corrections to earlier campaigns

### H1. The "~0.04 Å optimisation-noise floor" is wrong by 200×
**ESTABLISHED, against a bar fixed in advance** (C7_PREREGISTRATION §4).
390 GFN2-xTB optimisations, 30 structures, 34–430 atoms.

Three reports (`SYNTHESIS.md`, `WO_PREREGISTRATION.md`, `WO_RESULTS.md`) dismiss
the 0.013 Å adjacent-lanthanide step as "below the ~0.04 Å optimisation-noise
floor". The number was **never measured**; it traces to an asserted conformer
figure, and `0.041` is exactly the `tight` convergence target **in eV/Å (force),
not Å (distance)**.

Measured by perturbing a converged structure and re-optimising under identical
settings:

| σ (Å) | escape | median \|Δ⟨M–D⟩\| | P90 |
|---|---|---|---|
| 0.00 | 0 % | 0.00000 | 0.00000 |
| 0.02 | 0 % | 0.00013 | 0.00040 |
| **0.05** | **0 %** | **0.00019** | **0.00064** |
| 0.10 | 0.8 % | 0.00025 | 0.00064 |

Bar: median ≤ 0.005 and P90 ≤ 0.013. Passed by 26× and 20×.

- true floor ≈ **0.0002 Å**, not 0.04 — **200× smaller**;
- the 0.013 Å step is **≈ 68× ABOVE** the floor, not below it;
- escape ≈ 0, so one number suffices — the two-regime caveat the
  pre-registration reserved does not apply.

**The empirical nulls stand; the explanation attached to them does not.**
Whatever caps geometry here, it is not numerical noise. See
[`NOISE_FLOOR.md`](NOISE_FLOOR.md).

### H2. GFN2-xTB's lanthanide parameters are linear in Z — **DOCUMENTED BY THE METHOD'S AUTHORS, not discovered here**
**ESTABLISHED, and NOT novel.** The primary source states it outright:

> **Bannwarth, Ehlert & Grimme, *J. Chem. Theory Comput.* 2019, 15, 1652–1671**
> (DOI 10.1021/acs.jctc.8b01176), §2.4 Technical Details, p. 1660:
>
> *"For the lanthanides, only the parameters for Ce and Lu were freely fitted,
> while a linear interpolation with the nuclear charge Z has been used for the
> other elements."*
>
> and §2.1, p. 1655:
>
> *"As in GFN-xTB, the 'f-in-core' approximation is employed for lanthanides."*

Our contribution is therefore **not** the fact. It is (a) an independent numerical verification that the shipped implementation matches the stated intent, and (b) the consequence for machine learning, which the paper does not draw.

Verification from the shipped parameter file:
`~/opt/xtb-dist/share/xtb/param_gfn2-xtb.txt`, Ce(58)→Lu(71), n=14: every
parameter — `lev`, `exp`, `GAM`, `GAM3`, `REPA`, `REPB`, `DPOL`, `QPOL`,
`POLYS`, `POLYD`, `LPARD`, `KCNS/P/D` — is linear in Z to a worst residual of
**5.67e-07**, the file's printed precision. Ce and Lu are fitted anchors;
everything between is interpolation. La(57) is a separate anchor, off-trend by
15× (`lev` step −1.577 vs Ce→Pr −0.101).

**Inside GFN2 the lanthanide identity is one scalar linear in atomic number.**
No f-shell occupation, no crystal field, no nephelauxetic effect, no gadolinium
break, no tetrad effect. Any geometry it produces can carry at most a rank-1,
linear-in-Z deformation of metal identity — which *derives* the empirical
effective rank 1.05 of G1/G2 from the method rather than from our models.

### H3. Prediction from H2, confirmed on two independent arms
**ESTABLISHED.** H2 says La is a parameter outlier; C7_PREREGISTRATION §3
declared before looking that La pairs should be predicted worse by ≥ 0.05.

| stratum | `b7_f40_fb64` | `d0_dist` |
|---|---|---|
| **La→Ce** (parameter discontinuity) | +0.1477 | +0.1333 |
| **Gd→Tb** (CN 9→8 switch) | +0.0308 | +0.0299 |
| all other adjacent pairs | +0.2358 | +0.2193 |
| **deficit** (bar ≥ +0.05) | **+0.0880** | **+0.0860** |

The models' two worst strata are exactly the two places the *method* and the
*dataset construction* are discontinuous — the GFN2 parameter break at La and
the coordination-number switch at Gd/Tb. Replicates across two architectures.

### H4. Correspondence is recoverable — the conformer problem is fixable
**ESTABLISHED.** Adjacent-lanthanide structures were generated *independently*
per (ligand, metal) and land in different conformer basins: median heavy-atom
RMSD **5.46 Å**, and — decisively — **flat in |Δindex|** (5.46 at Δ=1, 5.77 at
Δ=7), so La-vs-Ce looks like La-vs-Lu. The difference was ~99 % sampling.

Rebuilding each family from ONE relaxed anchor by metal substitution
(`automl/qc/serial_metals.py`, 786 structures, 146/158 families clean):

| quantity | independent build | serial build |
|---|---|---|
| median adjacent-pair RMSD | 5.46 Å | **0.0120 Å** (455× down) |
| contraction SNR | 0.14 | **0.799** (5.7× up) |
| residual sd of Δ⟨M–D⟩ | 0.076 Å | **0.0061 Å** |
| response correlation r | 0.197 | **0.574** |
| RMSD vs \|Δindex\| | **flat** (5.46→5.77) | **rises 5.12×** (0.0120→0.0613) |

Seven gates pass as written (G1, G2, G3, G5, G6, G8, G9); G4 and G7 pass under
amended specifications (see C7_PREREGISTRATION Amendments 1–2, which record the
original failures alongside). **G8 is exact**: the anchor re-run reproduces its
own input to 0.00000 Å median, so the entire displacement is the metal
substitution and none of it is pipeline drift.

**Consequence.** The apparent response slope in the independent set (0.505) was
roughly **half conformer covariance** — the clean value is 0.255, at 3× better
correlation. And the practical instruction is: if a method with real f-electron
structure is ever used for this problem, generate the series *in correspondence*.
Independent per-metal optimisation throws the signal away.

### H5. The construction's failure mode confirms H2 independently
**SUPPORTED** (n = 9, suggestive not decisive). Of 10 rejects in 796, nine are
basin hops. They are **not** a size effect — failed families are *smaller* on
median (190 vs 233 atoms). They are **La**:

| | hop rate | statistic |
|---|---|---|
| substitution to **La** | 4/70 = **5.71 %** | OR 8.7, Fisher one-sided p = 0.0049 |
| every other metal | 5/726 = 0.69 % | |
| excluding the one 430-atom family (5 hops alone) | 3/69 vs 1/721 | OR 33, p = 0.0024 |

H2 says La is the parameter outlier, off-trend by 15× where Ce…Lu are linear in
Z to 5.67e-07. Substituting *to* La is the largest available perturbation of the
Hamiltonian, and that is exactly where relaxation escapes its basin.
**This was not designed as a test of H2** — it fell out of the construction's
failure mode, which is what makes it independent of §1's parameter read and of
H3's model stratum.

---

## I. The ceiling is the Hamiltonian's, and a better one removes it

### I0. Correspondence makes the geometry 455× cleaner and prediction slightly WORSE
**ESTABLISHED**, 8 paired seeds, `--deterministic`, identical rows and build
ids; the two assets differ **only** in where the atoms are (verified: identical
`build_ids`, `node_ptr`, `atomic_numbers`, `is_metal`).

| metric | serial (in correspondence) | original | Δ |
|---|---|---|---|
| `sel_adj_logSF_r2` | +0.1702 | +0.1831 | **−0.0129** (t = −2.67, 2/8 seeds up) |
| `sel_adj_pearson` | +0.4228 | +0.4390 | −0.0162 (t = −2.48, 2/8 up) |
| `sel_adj_sign_accuracy` | +0.6455 | +0.6379 | +0.0076 (n.s.) |

The serial construction was a large success *as geometry*: adjacent-pair
heavy-atom RMSD fell 5.46 → 0.0120 Å (455×), the contraction signal-to-noise
rose 0.14 → 0.799 (5.7×), and the response correlation with the Shannon radius
step went 0.197 → 0.574. None of it transfers. The cleaner set predicts
**worse**, consistently.

This is the sharpest available confirmation of H2/C-I, and it is a *positive*
scientific statement rather than another null: geometric noise was never the
binding constraint. Under GFN2 the metal enters as one linear-in-Z scalar, that
scalar is already supplied to the model as the tabular ionic radius, and so
removing 99 % of the conformer noise around it adds nothing — it only costs the
regulariser the incidental diversity the noisy set happened to provide.

I had explicitly reserved judgement here: "one degree of freedom" and "no usable
signal" are different claims, and a single *clean* degree of freedom might have
been worth more than a noisy one. It is not. The claim is now tested rather
than assumed.

*Falsifying test:* rerun the same contrast on geometries from a Hamiltonian
whose metal response is **not** rank-1 (§I3). If correspondence still fails
there, the problem is the representation; if it succeeds, the problem was the
Hamiltonian all along.

### I1. g-xTB is usable on these complexes at ~2× GFN2 cost
**ESTABLISHED.** `grimme-lab/g-xtb` (GPL-3.0, `xtb-6.7.1-gxtb-140526`,
sha256 verified against the published checksum). Parameters are compiled into
the binary; no external files. On a 58-atom Ce complex `--opt tight` converged
in 44 cycles, **20.2 s vs GFN2's 10.8 s**, with **analytical** gradients. The
same binary runs GFN2, so every comparison below carries no build confound.
A full re-optimisation of all 956 complexes is therefore ~900 CPU-h ≈ 10 h wall.

Two operational facts that would silently corrupt a campaign:

| | |
|---|---|
| `--alpb water` | **hard error** — no ALPB/GBSA parameters for g-xTB |
| `--cosmo water` | works (NH₄⁺ shifts −67 kcal/mol vs GFN2/ALPB's −91) |
| `--cpcmx water` | **accepted and silently ignored** — energy bit-identical to gas phase |

The production set was built with ALPB, so a g-xTB arm is *not* a drop-in
replacement: it changes the solvation model as well as the Hamiltonian, and any
comparison has to be matched on both.

*Falsifying test:* re-run the checksum; run `--cpcmx` on a charged species and
show a non-zero solvation shift.

### I2. GFN2's lanthanide response is flat; g-xTB's has reproducible f-shell structure
**ESTABLISHED**, one anchor (104-atom Nd nitrate complex), all 15 lanthanides
substituted at **fixed geometry**, so every difference is electronic structure
and nothing else. Ln(III) run at the Hund high-spin `uhf` under g-xTB, `uhf 0`
under GFN2 (correct there — f is in the core). 15/15 SCF converged in both arms.

Residual of the HOMO–LUMO gap after removing the linear-in-Z trend, Ce→Lu
(La excluded — it is a separate GFN2 anchor, H2):

| arm | residual sd | reproducibility gas↔water |
|---|---|---|
| GFN2 | **0.00075 eV** | r = +0.41 |
| g-xTB | **0.278 eV** — 370× | **r = +0.97** |

The gas-phase and water runs are independent SCF solutions, so that r = +0.97
is the control that matters: g-xTB's departure from linearity is *physics*, not
SCF noise, while GFN2 has essentially nothing to reproduce.

**The gadolinium break.** Mean gap for f⁷–f¹⁴ minus f¹–f⁶:

| arm | gas | water |
|---|---|---|
| GFN2 | +0.0115 eV | +0.0048 eV |
| g-xTB | **+1.151 eV** | **+1.220 eV** |

GFN2's "break" is not a break at all — its gap is a smooth monotone ramp
(2.146 → 2.166 eV over the whole series) and splitting a straight line in the
middle trivially yields a difference. g-xTB shows a genuine discontinuity at the
half-filled shell, ~100× larger, reproduced in two independent runs.

The metal Mulliken charge tells the same story: GFN2 spans **0.0065 e** across
Ce→Lu (i.e. constant), g-xTB spans **0.66 e** (gas) / **0.97 e** (water), and
that variation reproduces at r = +0.99 between the two.

*Falsifying test:* repeat on other ligand families; if the break is anchor-
specific it is a property of that complex, not of the Hamiltonian.

### I3. Open question — does the structure survive into the *geometry*?
**RUNNING**, and this is the one that matters. Models are given coordinates, not
wavefunctions. I2 is an electronic result and does not by itself buy anything.

270 optimisations: 6 diverse anchors (93–130 atoms, CN 8 and 9, distinct ligand
classes) × 15 lanthanides × 3 arms — GFN2, g-xTB high-spin, and **g-xTB forced
closed-shell as a deliberate wrong-physics control**: if the geometric break is
f-shell in origin, `uhf 0` should damage it. Run in gas and in solvent.

The claim under test: *is the optimised M–donor response to lanthanide identity
more than one linear-in-Z scalar under g-xTB?* Under GFN2 it provably is not
(H2), which is the rank-1 ceiling that made eight 3D encoders interchangeable
at effective rank 1.05 (G1, G2). If g-xTB's relaxed geometry carries structure
GFN2 cannot represent, then re-generating the set is justified and the ceiling
is a property of the method, not of the problem.

If it does *not* survive relaxation, then I2 is a true but useless result — the
extra physics lives in the wavefunction and never reaches the coordinates — and
the honest conclusion is that geometry-only 3D modelling of adjacent-lanthanide
selectivity is capped regardless of the electronic-structure method.


### I4. GFN2-xTB underestimates the lanthanide contraction by 2.5×; g-xTB reproduces it
**ESTABLISHED**, 71 distinct ligands × 15 lanthanides × 2 Hamiltonians, one
binary, one protocol, gas phase, 2130 optimisations. Per-ligand compliance
`c_L = d⟨M–donor⟩ / d r_Shannon`, where **1.00 is exact agreement with the
Shannon (1976) effective ionic radii**:

| arm | c_L | vs experiment | t vs 1.0 |
|---|---|---|---|
| **GFN2** | **0.405 ± 0.145** | **under by 2.47×** | −34.5, p = 1.1e−45 |
| **g-xTB** | **1.078 ± 0.094** | over by 1.08× | +7.0, p = 1.4e−09 |

Paired difference **+0.673**, improving on **71 of 71 ligands**
(t = +43.0, p = 4.9e−52; Wilcoxon p = 2.4e−13). Reproduced independently in
solvent (GFN2 0.398, g-xTB 1.313) and on the 6-ligand pilot (0.386 / 1.142).

This is externally validated against experiment rather than an internal
contrast, and it is the explanation for a long run of null 3D results: **the
geometries the models are given barely encode the lanthanide contraction at
all.** GFN2's per-ligand slope is also mostly noise — cv 0.358 with only 23 %
of its non-linear response shared across ligands, against g-xTB's 96 %.

*Falsifying test:* run a third Hamiltonian with explicit f electrons; if it
lands near 0.4 rather than 1.0, the effect is not about f-in-valence.

### I5. Per-ligand compliance does NOT predict measured selectivity
**ESTABLISHED (null)**, 71 ligands, 44 matched to measured separations.

| arm | Pearson r | p | partial (size, CN) |
|---|---|---|---|
| GFN2 | +0.110 | 0.48 | +0.194 |
| g-xTB | **−0.020** | 0.90 | +0.092 |

n = 44 needs |r| ≈ 0.30 for p < 0.05. Both null; g-xTB's is essentially zero.

This closes the last mechanism by which better geometry could have helped.
Adjacent-lanthanide selectivity *is* ligand-dependent discrimination, so if
geometry carried it, the per-ligand compliance was where it had to live. It
does not. Note also that g-xTB makes c_L **more** uniform, not less
(cv 0.358 → 0.087): the better Hamiltonian moves the response closer to one
universal constant times the tabular ionic radius, which is the quantity the
model already has.

*Falsifying test:* the C8 training contrast. If g-xTB geometries raise
`sel_adj_logSF_r2`, then geometry carries something this scalar summary misses
and I5 is too strong a reading.
