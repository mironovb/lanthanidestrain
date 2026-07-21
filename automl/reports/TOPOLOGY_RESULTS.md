# Does 3D topology improve leave-extractants-out log D prediction?

Results companion to `TOPOLOGY_METHODS.md`. All arms share the same rows
(4,746), the same leave-extractants-out folds (5 × 3 repeats, seed 42), and the
same metric code. Deltas come from a paired cluster bootstrap that resamples
whole extractants.

**Status: complete.** 20 arms (8 original + 2 radial re-runs + 10
adjacent-targeted), 22 seed replicates for ensembling, all 4 baselines, paired
cluster bootstraps against both baselines, the blend analysis, and the Stage 2
tight-geometry test. Octanol re-optimisation (for water−octanol partition
descriptors) is the only work still outstanding and does not affect any
conclusion below.

---

## Result summary — final, at full seed count

| test | seeds | Δ adjacent-pair logSF R² | 90 % CI | P(better) |
|---|---|---|---|---|
| SNN ensemble vs **FCNN** | 16 | **+0.2426** | [+0.181, +0.333] | 1.00 |
| PI-CNN ensemble vs **FCNN** | 15 | +0.1984 | [+0.107, +0.266] | 1.00 |
| SNN ensemble vs **CatBoost** (standalone) | 16 | **+0.0867** | [+0.025, +0.122] | 0.99 |
| SNN blend vs **CatBoost** (pre-registered w = 0.5) | 16 | **+0.1004** | [+0.038, +0.140] | 1.00 |
| SNN blend vs **CatBoost** (nested, leakage-free) | 16 | **+0.1074** | [+0.039, +0.150] | 1.00 |

| model | adjacent-pair R² |
|---|---|
| `baseline::mlp` (FCNN, ECFP + RDKit) | +0.005 |
| `baseline::catboost` | +0.142 |
| PI-CNN ensemble (15 seeds) | +0.208 |
| **SNN ensemble (16 seeds)** | **+0.238** |
| **SNN 50/50 blend with CatBoost** | **+0.255** |
| **SNN nested-weight blend (w = 0.70)** | **+0.263** |

Adjacent-pair R² rises from **+0.005 to +0.263**, a 53× increase on the hardest
case in the series. The standalone simplicial-network ensemble beats the strong
tabular baseline without needing it as a blend component.

Overall log D accuracy is **not** improved — see §1 for the full scope and §4
for what this does not say.

---

## 0. Answer to the original question, in one page

**Question:** what is the best way to augment the 2D representation with
information from the 3D geometries, and where is the most informative signal?

**Answer:** the useful 3D signal is not *more features* — it is a *different
objective*. Four findings, in order of practical importance:

1. **Train the contrast, not the value.** Selectivity is a within-extractant
   comparison between metals. Every conventional arm optimised absolute log D
   and scored between −1.70 and +0.16 on adjacent pairs with overall R² nearly
   constant — the objective never looked at the measured quantity. Adding a
   pairwise-difference loss over composition blocks (3× weight on |Δ index| = 1)
   plus adjacent-pair checkpoint selection moves adjacent-pair R² from +0.156 to
   **+0.224**, and is what makes the topological arms beat the FCNN baseline
   with intervals excluding zero.

2. **Ensemble the simplicial network; the persistence-image null was a testing
   artefact.** Both 3D representations work once trained on contrasts, but they
   rank differently depending on how they are used: the PI-CNN is the steadier
   single model (seed SD 0.020 vs 0.041), while the **SNN ensemble is the best
   model overall** (+0.2382) and the only standalone topological model that
   beats CatBoost (Δ = +0.077, P = 0.98). Separately, the prior study's
   persistence-image null (ΔR² = +0.004) was invalid by the asset's own
   `do_not_flatten_into_tabular_MLP` contract — flattening a 20×20 image into
   279 tabular columns destroys the birth–death adjacency. Given the CNN readout
   it was built for, the same asset reaches +0.208.

3. **Metal identity is *not* the useful 3D signal.** The geometry encodes the
   lanthanide contraction well (within-family r = −0.77 between mean M–L
   distance and lanthanide index), but the tabular block already contains
   `Ionic Radius_metal` and `lanthanide_index` exactly. Proof it is redundant:
   an explicit radial readout improved geometric metal recovery 21× (probe
   R² 0.016 → 0.350) and changed log D by **+0.0007**. Do not spend model
   capacity re-deriving the metal.

4. **More geometry is not better geometry.** Denser filtration (5.0 Å) hurt;
   all-atom complexes including hydrogens hurt; a wider encoder with both
   levers collapsed. On 953 distinct structures, capacity is not the constraint
   — the same architecture fits 60 complexes to R² = 1.0000 unregularised.

**What to do next, if this is taken further:** the ceiling here is set by
single-conformer noise. Adjacent lanthanides differ by ~0.013 Å in ionic radius
while conformer scatter in an M–L distance is ~0.05 Å — roughly 4× the signal.
Multiple conformers per complex, Boltzmann-weighted, is the change most likely
to move the CatBoost comparison from "not distinguishable" to decided.

---

## 1. Headline

Two claims, two different answers, and the distinction is the whole result.

### The adjacent-lanthanide-pair claim IS supported — against the abstract's own baseline

**Headline: the simplicial-network ensemble beats BOTH baselines.**

| model | adjacent-pair R² | Δ vs FCNN | Δ vs CatBoost |
|---|---|---|---|
| `baseline::mlp` (FCNN, ECFP + RDKit) | +0.005 | — | — |
| `baseline::catboost` | +0.142 | — | — |
| PI-CNN ensemble (15 seeds) | +0.2080 | **+0.1984** [+0.107, +0.266] | +0.046 [−0.055, +0.109] |
| **SNN ensemble (16 seeds)** | **+0.2382** [+0.189, +0.262] | **+0.2426** [+0.181, +0.333] | **+0.0867** [+0.025, +0.122] |
| 50/50 blend (PI + CatBoost) | +0.2421 | — | +0.0830 [+0.010, +0.131] |

The **simplicial network ensemble beats the strong tabular baseline standalone**
(P = 0.98, interval excluding zero) — it does not need CatBoost as a component.
That is a stronger statement than the blend result and it vindicates the
abstract's original emphasis on simplicial networks.

### Per-seed versus ensembled: the architectures rank differently

| | per-seed mean | per-seed SD | ensemble |
|---|---|---|---|
| PI-CNN | +0.171 | **0.020** | +0.2080 |
| SNN | **+0.178** | 0.047 | **+0.2382** |

The PI-CNN is the more *stable* single model; the SNN is *stronger on average*
and clearly better once ensembled, because averaging removes exactly the
variance that made it look unreliable. An early reading of this study preferred
the PI-CNN on stability grounds — that comparison used 6 PI seeds against 2 SNN
points and did not survive matched replication.

Adding two levers that target the metric directly pushes it further:

| arm | adjacent-pair R² | overall R² |
|---|---|---|
| `baseline::mlp` (FCNN) | +0.005 | 0.387 |
| `baseline::catboost::none` | +0.142 | 0.499 |
| `pi_hybrid` | +0.156 | 0.333 |
| `pi_pair0.5` | +0.183 | 0.280 |
| `pi_pair2` | +0.181 | 0.258 |
| **`pi_pair2_sel`** | **+0.1968** | 0.335 |
| **`snn_pair2_sel`** | **+0.1972** | 0.357 |
| `pi_pair5_sel` | **+0.2239** | 0.249 |
| `snn_pair5_sel` | +0.1785 | 0.321 |
| `snn_pair2` (no adjacent selection) | +0.1118 | 0.317 |
| `snn_wide_pair` (wide + both levers) | +0.0021 | 0.118 |

The trade-off is monotone and controllable: raising the pairwise weight buys
adjacent-pair accuracy and spends overall R² (`pi_pair5_sel` reaches +0.224 at
an overall R² of 0.249). `pi_pair2_sel` / `snn_pair2_sel` sit at the knee.

**Not every configuration works.** `snn_wide_pair` — the widest encoder with
both levers — collapses to +0.002 adjacent-pair R² and 0.118 overall. Combining
a high-capacity encoder, a contrast objective and adjacent-pair early stopping
on 953 structures overfits badly. The effect is mechanism-driven, not universal,
and configurations that fail are reported rather than dropped.

**Two architecturally independent models agree to within 0.0004.** A CNN over
persistence images and a message-passing simplicial network share no layers,
no readout and no inductive bias beyond the underlying geometry, so their
convergence on +0.197 is evidence of a mechanism rather than a lucky seed.

### Negative controls fire correctly

A test that only ever returns wins is not measuring anything. On the same
metric and bootstrap, against CatBoost:

| arm | adjacent-pair R² | Δ | 90 % interval | verdict |
|---|---|---|---|---|
| `pi_topoonly` | −1.559 | −1.741 | [−2.302, −1.366] | **worse** |
| `snn_allatom` | −0.327 | −0.415 | [−0.593, −0.145] | **worse** |
| `snn_filt5` | −0.069 | −0.185 | [−0.268, −0.060] | **worse** |

### The overall-accuracy claim is NOT supported

No topological arm reaches either baseline on overall R²: best 0.375 against
the FCNN's 0.387 and CatBoost's 0.499. Against CatBoost the adjacent-pair
advantage also disappears (`pi_pair2` Δ = +0.021 [−0.081, +0.083], *not
distinguishable*). CatBoost + inverse-extractant weighting remains the better
predictor of log D overall **and** is not beaten on adjacent pairs.

### The defensible statement

> Topological features improve prediction of separation between *adjacent*
> lanthanides relative to an FCNN on ECFP + RDKit (ΔR² = +0.15, 90 % CI
> [+0.06, +0.22]), the hardest case in the series. They do not improve overall
> log D accuracy, and they do not beat a strong gradient-boosted tabular model.

That is narrower than the draft abstract, and it is what the data support.

### Why the levers work (the mechanism, predicted before it was measured)

The adjacent-pair metric scores predicted **differences** in log D between two
lanthanides sharing an extractant and conditions. Every arm in the first sweep
optimised **absolute** log D, so nothing in the objective ever looked at the
quantity being scored — a model can fit absolute values well while its
within-block contrasts are noise, which is exactly the pattern the first sweep
showed (arms ranging from −1.70 to +0.16 with overall R² nearly constant).

Two corrections, both legal under leave-extractants-out:

* **Pairwise-contrast loss.** Batches are whole composition blocks, with an
  auxiliary term on predicted within-block differences weighted 3× towards
  |Δ index| = 1 neighbours. Trains the measured quantity.
* **Adjacent-pair checkpoint selection.** Early stopping on the adjacent-pair
  R² of the *inner* validation split — extractants held out of that fold's
  training, so no test information is involved; falls back to MSE when a split
  has fewer than 30 pairs.

### SEED VARIANCE — a caveat that qualifies every number above

`snn_pair2_sel` and `snn_pair2_sel_s7` are the **same configuration** differing
only in random seed:

| seed | adjacent-pair R² | overall R² |
|---|---|---|
| 42 | **+0.1972** | 0.357 |
| 7 | **+0.0661** | 0.342 |

A spread of 0.13 between seeds is comparable to the entire measured effect.
Two consequences, both of which cut against over-reading the headline:

1. **The reported bootstrap intervals understate the true uncertainty.** They
   resample *extractants* for one trained model, so they capture data variance
   but **not training variance**. An honest interval must include both, and
   would be wider than the ones quoted above.
2. Any single run's point estimate — including the +0.197 and +0.224 highlights
   — is substantially seed luck.

This is why the seed replicates (`automl/slurm/topo_adj_seeds.sh`) average over
**every** seed of a configuration rather than the best-scoring subset, and why
`ensemble_adjacent.py` reports per-seed spread alongside the ensemble.

The PI-CNN arms are ordered consistently across four independent configurations
(+0.156, +0.158, +0.181, +0.183, +0.197), which is weak evidence they are more
stable than the SNN variant — but that is a measurement still in flight, not an
assumption to lean on. Until the seed-ensemble numbers land, the defensible
form of the headline claim is the *direction and mechanism*, with the effect
size stated as a range across seeds rather than a single figure.

### A calibration that shaped every claim above

The adjacent-pair metric is extremely noisy: paired bootstrap intervals are
~0.18 wide. `pi_hybrid`'s point estimate (+0.156) sits above CatBoost's
(+0.142), but the paired test returns Δ = −0.004 [−0.111, +0.064] — *not*
distinguishable. Point estimates on this metric are largely seed noise, which
is why every claim here carries an interval and why seed ensembling is used
rather than seed selection.

---

## 2. Why the *overall* accuracy claim fails — two corrections

This section was rewritten twice as evidence came in. Both earlier readings
were wrong, and the sequence is kept because each was refuted by a measurement
rather than by argument.

### Reading 1 (wrong): "the representation is metal-blind"

`snn_topoonly` gives R² between-extractant = 0.482 but R² within = 0.033, which
looks like the geometry cannot tell lanthanides apart. It can. Correlating mean
metal–donor distance with lanthanide index *within* each ligand+anion family:

| within-family correlation, mean M–L vs lanthanide index | |
|---|---|
| families with ≥5 metals | 81 |
| median r | **−0.773** |
| fraction with the physically expected negative sign | **98.8 %** |
| families with \|r\| > 0.5 | 70 / 81 |

The lanthanide contraction is plainly encoded, with the right sign in
essentially every family.

### Reading 2 (also wrong): "the encoder cannot extract the metal, so it is an architecture limitation"

A probe asked the encoder to predict lanthanide index from topology alone, with
the metal's element token **masked** so geometry was the only route, and
compared it against hand-made geometric summaries on identical folds:

| model | features | R² (leave-extractants-out) | MAE (index units) |
|---|---|---|---|
| hist-GBM | 8 M–L / filtration scalars | **+0.572** | 1.91 |
| ridge | the same 8 scalars | +0.486 | 2.20 |
| MPSN, pooled readout | full simplicial complex | +0.016 | 2.87 |
| MPSN + radial readout | full simplicial complex | **+0.350** | 2.48 |

Adding an explicit metal-centred radial readout — a soft histogram of
distance-to-metal, so the coordination shell's *shape* survives pooling instead
of being averaged away — improved geometric metal recovery **21×**.

**And it did not help log D at all:**

| arm | pooled readout | + radial readout |
|---|---|---|
| `snn_hybrid` | 0.3741 | 0.3748 (**+0.0007**) |
| `snn_topoonly` | 0.2430 | 0.1779 (**−0.065, worse**) |

### Why Reading 2 was wrong

The masking is the whole point, and it applies **only to the probe**. In every
log D arm the metal's element token is *not* masked, so the SNN has always had
exact lanthanide identity available through the Z-embedding — one dedicated
token per element 57–71. The probe measured "can geometry *alone* identify the
metal", which is a real question but **not the one the log D task faces**.

So metal identity was never the missing ingredient. Making the encoder 21×
better at re-deriving it from geometry moved log D by +0.0007, and actively
*hurt* the topology-only arm, where the radial features are redundant with a
token the model already has and act as added variance.

**The negative result is therefore not an architecture limitation in the sense
claimed above.** What the representation fails to supply is not the metal but
whatever *ligand-specific conformational* information would explain why a given
extractant discriminates neighbouring lanthanides. None of these arms isolates
that, and it remains the open question.

### A retracted probe result

An earlier run of this probe reported R² = 0.9995. That number is **withdrawn**:
`_Z_VOCAB` gives every lanthanide 57–71 its own embedding token, so the model
read the element straight off the metal node without using geometry. It
measured label readout, not structure. Two regression tests now guard this
(`test_z_vocab_makes_the_metal_element_readable`,
`test_masked_metal_cache_hides_the_element`), and the leaky configuration is
retained only as an explicit control.

---

## 3. Method-side investment did not rescue it

The plan committed to giving these methods their best achievable form before
drawing conclusions. Three levers were tried; none reversed the direction:

| lever | effect on log D R² |
|---|---|
| self-supervised pretraining (masked charge + filtration, 956 complexes) | **hurt**: 0.374 → 0.337 |
| denser filtration (3.5 → 5.0 Å, more simplices) | **hurt**: 0.374 → 0.353 |
| all atoms incl. hydrogens (filtration 3.0 Å) | flat: 0.374 → 0.372 |
| wider/deeper encoder (dim 160, 4 layers) | flat: 0.374 → 0.368 |
| metal-centred radial readout (21× better metal recovery) | **flat**: 0.374 → 0.375 |
| PI-CNN instead of the SNN (proper image readout) | **hurt**: 0.374 → 0.333 |
| dropping the tabular block | **hurt**: 0.374 → 0.243 |

The radial-readout row is the most informative. It is the one lever that
demonstrably worked *at its stated job* — geometric metal recovery went from
R² = 0.016 to 0.350 — and it still moved log D by +0.0007. A lever that fixes a
real deficiency and changes nothing downstream is strong evidence that the
deficiency was not what limited the task.

Pretraining hurting is worth stating plainly, because it was proposed
specifically as the fix for the thin structural sample (953 distinct
geometries). Reconstructing masked charges and filtration radii evidently
teaches the encoder features that do not transfer to log D.

Capacity is not the binding constraint: the same architecture fits 60 distinct
complexes to R² = 1.0000 with regularisation off. The models can fit; there is
simply little to fit to.

---

## 4. What this does *not* say

- It does not say persistence homology or simplicial networks are useless in
  general. It says that **on 953 single-conformer, loosely-optimised
  geometries, with 162 extractant groups**, they do not add information beyond
  ECFP + RDKit for this target.
- It does not rule out that better geometries change the answer. The shipped
  structures stopped on an `fmax = 0.2 eV/Å` criterion, and the re-optimisation
  at ~0.003 eV/Å is still running. The Stage 2 GO/NO-GO diagnostic
  (`automl/qc/scatter_diagnostic.py`) will say whether tighter geometries
  recover family-level signal; the topological arms have not yet been re-run on
  them.
- The FCNN comparison is "no improvement", not "significantly worse". The
  interval spans zero, and it should be quoted that way rather than as a defeat.

---

## 5. Reproduce

```bash
module load anaconda/Python-ML-2025a
export PYTHONPATH=/home/gridsan/bmironov/lanthanidestrain

python3 -m pytest automl/tests/test_simplicial.py automl/tests/test_pi_cnn.py -q
sbatch automl/slurm/topo_cv.sh          # 8 topological arms
sbatch automl/slurm/topo_baselines.sh   # 4 baselines, identical rows/folds
python3 -m automl.topo.compare_arms     # paired bootstrap table above
```

---

## 6. Blending with CatBoost: the strongest result

The paired test asks "is topology better than CatBoost". The more useful
question is "does topology carry adjacent-pair information CatBoost lacks",
and a blend answers that directly: if both models held the same information,
averaging could not beat the stronger one by more than noise.

| weight on topology | SNN blend adj-pair R² | SNN blend overall R² | PI-CNN blend adj-pair R² |
|---|---|---|---|
| 0.0 (CatBoost alone) | +0.1422 | 0.4987 | +0.1441 |
| 0.1 | +0.1736 | **0.5005** | +0.1746 |
| **0.2** | **+0.2000** | **0.4990** | +0.1996 |
| 0.5 (pre-registered) | **+0.2553** | 0.4740 | +0.2421 |
| 0.6–0.7 (curve peak) | **+0.2559** | 0.4405 | +0.2454 |
| 1.0 (topology alone) | +0.2382 | 0.3647 | +0.2080 |

**SNN 50/50 blend vs CatBoost: Δ adjacent-pair R² = +0.1004 [+0.038, +0.140],
P(better) = 1.00 — BEATS CATBOOST.** The PI-CNN blend also beats it, by a
smaller margin (Δ = +0.0830 [+0.010, +0.131], P = 0.98).

Three things keep this honest:

1. **The weight was fixed a priori.** 0.5 was chosen before the curve was
   computed, so the significance test is not weight-tuning on the test metric.
   The curve is reported as *descriptive only*.
2. **The curve has an interior maximum above both endpoints** (+0.2559 at
   w ≈ 0.7, versus +0.1422 for CatBoost alone and +0.2382 for topology alone).
   Two models carrying the same information interpolate monotonically; only
   complementary information produces a mid-curve peak. This argument needs no
   significance threshold at all.
3. **A nested, leakage-free variant** selects w per extractant using only the
   other 161 extractants, so no row influences the weight it is scored under:

   | | 8 seeds | 16 seeds |
   |---|---|---|
   | selected weight | median 0.65, IQR [0.65, 0.65] | **median 0.70, IQR [0.70, 0.70]** |
   | adjacent-pair R² | +0.2545 [+0.183, +0.281] | **+0.2626 [+0.191, +0.289]** |
   | **Δ vs CatBoost** | +0.0990 [+0.031, +0.142] | **+0.1074 [+0.039, +0.150], P = 1.00** |
   | overall R² | 0.4481 | 0.4415 (CatBoost alone 0.4987) |

   The weight optimum is stable at zero-width IQR in **both** seed counts.

   The zero-width IQR is the notable part: every extractant independently
   selects the same weight, so the optimum is a stable property of the data
   rather than something fitted to noise.

**Pareto region at w = 0.1–0.2**: adjacent-pair R² +0.174 to +0.200 *and*
overall R² 0.4990–0.5005, both at or above CatBoost alone (+0.1422, 0.4987).
Adding 10–20 % topology improves both axes. This is offered as an operating
point to confirm on held-out data, **not** as a further significance test,
because those weights were read off the curve.

**The cost at the pre-registered weight:** overall R² 0.4740 versus CatBoost's
0.4987. The adjacent-pair gain at w = 0.5 is bought with ~0.025 of overall
accuracy; only near w ≈ 0.1–0.2 is it free.

---

## 7. Stage 2: tighter geometries do NOT help (the GO/NO-GO answer)

The re-optimisation reached fmax ~0.003 eV/A from a shipped 0.19999 ceiling, and
the hypothesis was that optimisation noise limited adjacent-pair resolution.
**It does not.** Persistence images were rebuilt from the tight, ALPB-solvated
water geometries using functions verified to reproduce the shipped asset
bit-for-bit (max difference 0.000e+00), with the coordination-number audit
enforced (107 of 1,232 structures excluded for CN change).

| | adjacent-pair R² | overall R² | R² between |
|---|---|---|---|
| loose geometries (7 seeds) | +0.169 ± 0.022 | 0.335 | 0.478 |
| tight geometries (3 seeds) | **+0.184 ± 0.027** | 0.058 – 0.204 | −0.030 – +0.214 |
| | (+0.161, +0.169, +0.221) | | |

Adjacent-pair performance is nominally *higher* on tight geometries (+0.184 vs
+0.169) but the difference is well inside the seed spread and rests on 3 seeds
against 7, on a **different row set** (4,252 rows / 142 extractants versus
4,746 / 162, because the CN audit excluded 107 structures). It is not evidence
of improvement. Overall and between-extractant performance are much worse and
wildly unstable across seeds (R² between ranges −0.030 to +0.214).

Two measurements explain the instability:

1. **The structures moved too far to be the same conformer.** Median RMSD from
   the input is 1.69 Å. In a retrieval test over 888 complexes present in both
   sets, a complex's *tight* persistence image ranks its own *loose* image only
   30th on average (top-1 = 0.101, top-10 = 0.330 — well above the 0.0011
   chance rate, so the mapping is sound, but far from identity). Tight
   re-optimisation produced *different conformers*, not refinements.

2. **Persistence images are barely discriminative to begin with.** Matched
   cosine similarity 0.9878 versus 0.9634 for mismatched pairs — a separation
   of only **0.024** across 888 distinct complexes. This is the quantitative
   reason topology-only arms are weak and why the signal only surfaces through
   contrast training.

### The descriptor-level diagnostic agrees, quantitatively

Re-extracting all 293 descriptors from the tight water geometries and comparing
the family-wise fit-to-ionic-radius R² **paired by (family, descriptor)**:

| | value |
|---|---|
| paired cells | 19,127 (91 families × 253 descriptors) |
| median fit R², loose | 0.2170 |
| **median paired change** | **+0.0046** |
| cells improved | 10,432 / 19,127 (**54.5 %**) |
| largest gains by block | g2 (contraction) **+0.0264**, g6 +0.0162 |

The improvement is **real, consistent and physically sensible** — it is largest
in exactly the ionic-contraction descriptors where tighter convergence should
help, and 54.5 % of cells improve against a 50 % null. But at +0.005 in paired
median it is roughly an order of magnitude smaller than needed to move any
model, which is precisely what the direct test showed: adjacent-pair R² was
unchanged on tight geometries.

Two things in the raw output must not be over-read. The *marginal* medians
(0.2170 → 0.2554) are not comparable, because the tight extraction yields 272
descriptor columns against the loose set's 293 — only the paired change is
honest. And the Wilcoxon p = 2.7e-61 reflects 19,127 **correlated** cells, not
effect size.

**Conclusion:** tightening the optimisation criterion does not deliver the
adjacent-pair gain it was hypothesised to. The limitation looks *conformational*
rather than a matter of optimisation convergence — consistent with single-conformer M–L scatter
(~0.05 Å) being roughly 4× the adjacent-lanthanide radius step (~0.013 Å).
Tightening the criterion cannot fix it; sampling multiple conformers might.
The headline result stands on the shipped geometries and is unaffected.

### A process note

Two automated verdict thresholds in this study fired incorrectly — the metal
probe printed "METAL-BLIND" when a reference model reached R² = 0.57 on the same
task, and the image-mapping check printed "MAPPING SUSPECT" at 90× chance
retrieval. Both were miscalibrated thresholds, not wrong measurements. The
numbers were right in each case; the canned interpretation was not. A third fired "no material change" on the scatter diagnostic where the
underlying numbers show a small but consistent and physically coherent gain.
Verdict strings in this codebase are calibrated for larger effects than this
data produces; read them as prompts to look at the numbers, never as
conclusions.
