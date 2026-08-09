# Campaign 6: the pre-registered endpoint failed; a threshold was giving away 3D signal; and MAE loss transformed the tabular partner

**Read §10 for the final, corrected result.** The 3D-specific gain is
**+0.011–0.014** (a discarded 3.5 Å threshold); the largest gain in the campaign
is **tabular** (+0.107, a one-line loss change). §8 and §9 are superseded
intermediate positions and §10a records a claim of mine that did not survive its
own matched contrast.

**Bogdan Mironov · 7 August 2026**
Pre-registered in [`C6_PREREGISTRATION.md`](C6_PREREGISTRATION.md), committed
before any screening number was read, with the endpoint configuration and the
bar it had to clear both fixed before the look. 155 screening cells + 96
confirmation runs + 22 CPU partner fits, 0 failures.

---

## 1. The endpoint: null

The pre-declared arm `z1_mphys_f40_w10`, chosen on screen+select only:

| | adjacent-pair R² | vs incumbent D0 |
|---|---|---|
| selection half (604 pairs) | +0.2742 | **+0.0431** |
| **report third, held out (301 pairs)** | **+0.2903** | **−0.0127** |

Paired cluster bootstrap on the report third: **−0.0127, 90 % CI
[−0.0485, +0.0196]**, P(better) = 0.27. A null, not a confirmed regression —
but not an improvement.

**The ranking inverted out of selection.** On screen+select:
z1 > e0 > a7 > z0 > b7 > control. On the report third:

| arm | report third |
|---|---|
| b7_f40_fb64 | **+0.3136** |
| a0_published (the *control*) | +0.3058 |
| D0 incumbent | +0.3030 |
| a7_w10_only | +0.3016 |
| e0_mphys | +0.2986 |
| z0_mphys_f40 | +0.2948 |
| **z1_mphys_f40_w10 (the endpoint)** | **+0.2903** |

The endpoint came **last of six** and the untouched control came second. This is
a textbook winner's curse, visible only because the third partition was set
aside before anything ran. `b7` is top here, but selecting it *now* would be
selecting on the report third, so it is an observation and not a result.

Stacks do not rescue it (nested pair-fitted NNLS, report third): incumbent
3-arm **+0.2901**, new 3-arm **+0.2840**, best mixed **+0.2978** — all below
the best single arm.

**Shrinkage was worse than predicted.** The pre-registration said "expect about
half". Screening → confirmation retained 23–46 % (e0 +0.0853 → +0.0252;
b7 +0.0711 → +0.0162; a7 +0.0519 → +0.0241), and confirmation → held-out went
to zero. Recorded as a failed prediction of my own.

## 2. The positive result: MAE loss for the tabular partner

Not part of the endpoint, selected on screen+select only, and therefore cleanly
testable on the report third.

`CatBoostRegressor(loss_function="MAE")`, everything else at the published
settings:

| | adjacent-pair R² | 90 % cluster bootstrap | log D R² |
|---|---|---|---|
| **full set (905 pairs)** | +0.1422 → **+0.2487** (**+0.1066**) | [+0.0375, +0.1420] **excludes 0** | +0.4987 → **+0.5102** |
| **report third (301 pairs)** | +0.2261 → **+0.2812** (**+0.0552**) | [+0.0177, +0.1009] **excludes 0** | — |

P(better) = 1.000 and 0.992. **It improves both metrics at once** — there is no
tension to trade here, which is unusual in this study and is why it is worth
having. It nearly doubles CatBoost's adjacent-pair R² and lifts it to within
noise of the best 3D encoder (+0.2487 vs D0's +0.2474) while keeping the log D
lead that made CatBoost worth stacking in the first place.

### Mechanism: it is **not** robustness. It is L1 specifically.

My first explanation was that squared error lets a few badly-measured blocks
dominate and MAE bounds each row's influence — i.e. *robustness*. That makes a
prediction: Huber, which is also robust, should also help. It was tested and it
does not.

| CatBoost loss | adjacent-pair R² (selection half) |
|---|---|
| RMSE (published) | +0.1594 |
| Huber δ = 1.0 | +0.1649 |
| Huber δ = 0.3 | +0.1725 |
| **MAE** | **+0.2188** |

Huber recovers almost none of the gain at either δ. **Robustness is not the
mechanism**, and the hypothesis I wrote down is falsified by its own test.

**That explanation is dead too.** I next proposed that MAE works because it
regresses the conditional **median**. `Quantile:alpha=0.5` reproduces MAE exactly
(+0.2188 vs +0.2188), so the test is well-posed — and α = 0.7 then **beats** it
(+0.2384). The median is not optimal. A third explanation — that an upper
quantile down-weights an untrustworthy left tail — predicted the gain would
concentrate in blocks containing low log D, and the gain is in fact **larger
where the left tail is absent** (+0.0964 vs +0.0604). Also falsified.

**Three mechanisms proposed, three falsified.** The effect is real and the
explanation is unknown; see
[`SCIENTIFIC_FINDINGS.md`](SCIENTIFIC_FINDINGS.md) §B2, B3, B6 for each test.

Two facts that any future explanation must accommodate:

- `--pair-loss-kind huber` on the *pair* term did nothing (−0.0161), so the
  leverage is in the **level** fit, not the contrast term;
- the optimum is **family-specific**: CatBoost wants full L1, while the neural
  encoder is *hurt* by it and peaks at partial robustness (§10, F1).

On full data at 16 seeds the α = 0.7 peak did **not** hold either — the ordering
becomes q60 +0.2579 > q65 +0.2430 > q70 +0.2321, and plain MAE with `rsm=0.3`
beats all of them at **+0.2624 / log D +0.5145**. The robust-loss family is real;
the specific α measured on 8 seeds of a smaller partition was not.

## 3. The falsification: aligning the contrast loss to the metric makes it worse

This is what the campaign was built around, and it is dead.

`train.py` builds the contrast term from raw row pairs while
`adjacent_pair_arrays` averages replicates within (block, metal) *before*
differencing. Censused on the modelled rows, the mismatch is large: **61.6 % of
the squared mass inside the 3×-weighted "adjacent emphasis" term is same-metal
replicate pairs**, the loss sees 18,065 adjacent pairs where the metric sees
1,349, and the ten largest blocks take 59.6 % of the gradient.

Repairing it (`--pair-metric-align`) **hurt in all 16 cells that used it**, from
−0.0001 to −0.1497, including the pure test (`a1_align`, −0.0636).

But its two *components*, applied without collapsing replicates, both helped:

| cell | Δ vs control (screening) |
|---|---|
| `a7_w10_only` — adjacent emphasis 3 → 10, replicates kept | **+0.0519** |
| `a6_adjonly_only` — adjacent pairs only, replicates kept | **+0.0365** |
| `a4_align_w10` — emphasis 10 **and** replicates collapsed | −0.0167 |
| `a2_align_adjonly` — adjacent-only **and** replicates collapsed | −0.0180 |

**The diagnosis was right and the prescription was wrong.** Reweighting toward
the pairs the metric scores helps; collapsing the replicate redundancy starves
the term and cancels the gain. The mechanism is already in this repo:
`pair_regressor` failed catastrophically on 905 collapsed pairs, diagnosed as
*"the pair target is too data-poor to learn from."* Alignment does the same
thing inside the loss.

**Portable finding: the row-pair redundancy in the contrast term is not a
defect. It is what makes the term learnable.** A train/eval mismatch is not
automatically worth repairing — this one is real, measurable, and repairing it
costs more than it returns.

## 4. Axes that moved on selection data and did not survive

| axis | screening Δ | report third |
|---|---|---|
| `mphys` — aqueous/f-shell metal constants | +0.0853 | −0.0044 vs D0 |
| receptive field (4.0 Å graph + wider basis) | +0.0711 | +0.0106 vs D0 |
| adjacent emphasis 10 | +0.0519 | −0.0014 vs D0 |

The metal-physics block cleared a free pre-screen designed to avoid exactly this
outcome (`mphys__dG_hyd` correlates with `dy` at 0.215, above the incumbent
ionic radius's 0.171 and above the best of sweep2-A1's geometry columns at
0.183; `c6_prescreen.csv`). **The pre-screen was necessary and not sufficient** —
it correctly predicted the block would not repeat A1's −0.3167 collapse, and it
did not predict that the gain would fail to generalise.

Emphasis is exhausted: 20 (+0.0704) and 40 (+0.0760) both score *below* the
metal block alone, so 10 was already at the plateau.

## 5. Infrastructure that outlives the campaign

- **`automl/topo/build_neighbor_graph.py`** — rebuilds the neighbour graph past
  the shipped asset's 4.0 Å ceiling from the coordinates it already carries.
  The do-no-harm gate rebuilds at 4.0 Å and reproduces the shipped edge set with
  **0 disagreements out of 2,301,232**. Assets built: c50 (4.0 M edges), c60
  (6.1 M), c80 (11.5 M), k24 (a degree-based variant). Result: **nothing past
  4.0 Å helps** (§8), so the shipped ceiling sits at about the right place --
  which was not knowable until this made it testable.
- **`automl/artifacts/c6_split/`** — a fresh three-way extractant split on the
  modelled 162, deterministic and verifiable. The old `pi_sweep` split had been
  scored on by four campaigns.
- **`automl/metal_physics.py`** + `c6_prescreen.py` — the descriptors and the
  zero-GPU, zero-look gate that any candidate must clear before it costs a run.
- **New `train.py` flags**, all default-off and proven so: one deterministic
  configuration run from a pristine HEAD worktree and from the working tree gave
  `max|Δoof| = 0.000e+00` over 4,746 rows.

## 6. What I got wrong

- **Predicted the loss-alignment repair would be the campaign's win.** It was
  the worst axis tested. Falsified by its own 16 cells.
- **Predicted ~50 % shrinkage.** Actual: 23–46 % to confirmation, then zero.
- **Nearly compared the endpoint against the wrong bar.** The report third is an
  *easier* subset (D0 scores +0.3030 there vs +0.2474 on full). Quoting +0.2903
  against the published +0.2474 would have read as a comfortable win while being
  a regression. Caught and fixed in the pre-registration before the look.
- **Left the `mphys__` columns unmasked in the interaction head**, which broke
  the metal-free identity that head exists to enforce. Caught by measuring how
  block-constant its input actually was, not by re-reading the code.
- **`savez_compressed` silently appends `.npz`**, so the asset builder's atomic
  replace failed on a path that never existed.

## 7. Recommendation

Ship **CatBoost-MAE** as the tabular partner: it is a one-line change, it is
validated on extractants that took no part in choosing it, and it improves both
scored quantities.

Do **not** ship `z1_mphys_f40_w10` or any of the C6 encoder arms. The incumbent
D0 remains the best single 3D arm.

The report third is now spent. Any further selection needs a fresh partition —
and on this campaign's evidence, a screening gate well above +0.02, since three
axes cleared +0.02 four times over and none of them survived.

---

## 8. The most successful 3D-derived result

The campaign's one held-out win (§2) is a **tabular** change — it touches no
geometry. Since the point of this project is what the Architector / GFN2-xTB
structures buy, the best result derived *from the 3D structures themselves* is
reported here separately.

**`b7_f40_fb64`** — the distance GNN over the Architector complexes, with two
purely geometric changes and nothing else:

```
--arch dist --pair-loss-weight 2.0 --select-on adjacent \
    --filtration-max 4.0 --rbf-bins 64
```

1. **Stop discarding the outer shell.** Every published run thresholded the
   Vietoris-Rips asset at 3.5 Å, but the asset contains edges out to 4.0 Å.
   Using all of them is free — no new asset, no new code.
2. **Widen the radial basis to match.** With `rbf_max` tracking
   `--filtration-max`, a 32-bin basis over a wider graph puts the new edges in
   its saturated tail; 64 bins resolves them.

| partition | b7 | incumbent D0 | Δ |
|---|---|---|---|
| full set (905 pairs) | **+0.2613** | +0.2474 | **+0.0139** |
| selection half (604) | **+0.2460** | +0.2311 | **+0.0149** |
| **held-out third (301)** | **+0.3136** | +0.3030 | **+0.0106** |

Paired cluster bootstrap on the held-out third: **+0.0106, 90 % CI
[−0.0015, +0.0227], P(better) = 0.929**.

**Why this is the credible 3D result and the endpoint was not.** The effect size
is *stable across all three partitions* — +0.0139, +0.0149, +0.0106 — which is
what a real effect looks like. The pre-registered endpoint, by contrast, scored
+0.0431 on the half it was chosen on and −0.0127 on the half it was not. b7 is
also the **best single arm of any kind on the held-out third** (+0.3136, ahead
of the control's +0.3058 and D0's +0.3030).

**The honest caveats, which matter here.**

- The interval **includes zero** at the lower end (−0.0015). This is suggestive,
  not established.
- b7 was one of 39 screened cells, so it is not free of selection. What it has
  instead of a clean look is *replication of its effect size* on a partition
  that took no part in choosing it.
- It was **not** the pre-declared endpoint. Promoting it now would be selecting
  on the report third, which is exactly the error this design exists to prevent.
  Confirming it needs a fresh partition.

**Boundaries measured around it.** Both axes were then pushed past this point and
both stopped:

| | Δ vs control, selection half |
|---|---|
| 4.0 Å graph (b7) | +0.0711 |
| 6.0 Å rebuilt graph, 6.1 M edges | +0.0718 |
| 5.0 Å rebuilt graph | +0.0551 |
| k-NN graph (degree- not distance-based) | +0.0306 |
| adjacent emphasis 10 | +0.0519 |
| adjacent emphasis 20 / 40 | +0.0704 / +0.0760 |

Nothing beyond 4.0 Å helps. The shipped asset's ceiling turns out to sit at
roughly the right place, which was not knowable before
`build_neighbor_graph.py` made it testable (that rebuild reproduces the shipped
1-skeleton with **0 disagreements out of 2,301,232 edges**).

**Recommended reading of the 3D arm.** The defensible statement is that
**+0.011 to +0.015 of adjacent-pair R² was being thrown away by a threshold**,
recoverable with a two-flag change and no new computation — and that beyond it
the receptive-field axis is exhausted. The best single 3D arm moves from
+0.2474 to **+0.2613** on the full set. That is a real but modest gain, an order
of magnitude smaller than the tabular MAE switch, and it should be described
that way rather than as a new headline.

---

## 9. The strongest secured result: the deployable stack

Combining the campaign's two validated components — the 3D arm from §8 and the
tabular arm from §2 — into the nested pair-fitted stack:

| stack (nested pair-fitted NNLS) | full set (905) | **held-out third (301)** |
|---|---|---|
| published: D0 + fingerprint net + CatBoost(RMSE) | +0.2679 | +0.2901 |
| **new: b7 + fingerprint net + CatBoost(MAE)** | **+0.2936** | **+0.3076** |
| gain | **+0.0257** | **+0.0175** |

Paired cluster bootstrap on the held-out third: **+0.0175, 90 % CI
[+0.0031, +0.0346], P(better) = 0.975 — excludes zero.**

The published reference reproduces (+0.2679 here against the published +0.2672),
so the comparison is like-for-like.

**It is also simpler.** The fingerprint network receives **zero weight** in the
new stack — CatBoost-MAE displaces it entirely. The deployable model is two arms
(`b7` + CatBoost-MAE, weights 0.77 / 0.23 on held-out) where the published one
was three. `b7 + CatBoost(MAE)` alone scores the same +0.2936 / +0.3076.

**This improves on both partitions**, which is what separates it from the
endpoint that failed. The endpoint gained +0.0431 where it was chosen and lost
−0.0127 where it was not; this gains +0.0257 and +0.0175.

**The multiplicity, stated rather than buried.** This is the **third** look at
the report third (after the pre-declared endpoint and the CatBoost contrast),
and it is a *post-hoc combination* — neither the combination nor the arm `b7`
was pre-declared. A 3-look Bonferroni on this interval gives roughly
[−0.003, +0.038], which spans zero. So the honest statement is:

> The deployable stack improves by +0.0175 on extractants that took no part in
> selecting either component, with an uncorrected interval excluding zero and a
> multiplicity-corrected interval that does not. Both components were chosen on
> the other two thirds, and both improve on both partitions. It warrants a
> confirmatory run on a fresh partition, which this dataset can no longer
> supply.

That is stronger than anything else in the campaign and weaker than a
pre-registered result, and it should be quoted as exactly that.

---

## 10. Final result — **corrected**

### 10a. What I first wrote, and why it was wrong

I reported `t1_d02_f40` (4.0 Å graph + Huber δ = 0.2) as the best single arm in
the study at +0.3318 held-out. A direct matched contrast against
`b7_f40_fb64` — identical graph, identical protocol, identical 16 seeds, the
*only* difference being δ — shows the two are **not distinguishable**:

| partition | t1 (δ=0.2) | b7 (δ=1.0) | Δ | 90 % CI |
|---|---|---|---|---|
| held-out (301) | +0.3318 | +0.3136 | +0.0182 | [−0.0072, +0.0397] |
| selection (604) | +0.2527 | +0.2460 | +0.0066 | [−0.0053, +0.0123] |
| full (905) | +0.2704 | +0.2613 | +0.0090 | [−0.0006, +0.0166] |

An independent 12-seed sweep of δ agrees: δ = 0.5–1.0 is the plateau and
δ = 0.2 is *below* the published δ = 1.0. **The robust-loss half of §10 as first
written is withdrawn.**

### 10b. What actually survives — the graph change

| arm | full (905) | held-out (301) |
|---|---|---|
| `b7_f40_fb64` — 4.0 Å graph, 64-bin basis | **+0.2613** | **+0.3136** |
| `d0_dist` incumbent — 3.5 Å, 32-bin | +0.2474 | +0.3030 |
| Δ | **+0.0139** | **+0.0106** [−0.0015, +0.0227] |

Two flags, no new features, no new computation: use the whole shipped 4.0 Å
Vietoris–Rips asset instead of thresholding it at 3.5 Å, and widen the radial
basis to resolve it. Consistent across all three partitions (+0.0139 / +0.0149 /
+0.0106), which is the stability signature. The held-out interval touches zero.

### 10c. The deployable stack

| stack (nested pair-fitted NNLS, held-out third) | adjacent-pair R² |
|---|---|
| published: D0 + fingerprint net + CatBoost(RMSE) | +0.2901 |
| **b7 + CatBoost(MAE, rsm 0.3)** | **+0.3099** |
| gain | **+0.0198** |

Two arms, not three — the fingerprint network takes zero weight. The larger part
of this gain comes from the **tabular** side (CatBoost RMSE→MAE, §2), not from
the geometry.

### 10d. The honest summary

- **3D-specific gain: +0.011–0.014**, from a discarded threshold. Real, modest,
  consistent across partitions, interval touching zero on held-out.
- **Tabular gain: +0.107 full / +0.055 held-out**, from a one-line loss change,
  intervals excluding zero, improving log D as well.
- **Best stack: +0.3099 vs +0.2901 published**, and simpler.
- Everything larger that was reported during this campaign — the metal-physics
  block, the α = 0.7 quantile, the 7.5 % subsampling lever, the δ = 0.2 arm —
  **shrank or vanished** when given adequate seeds or a matched contrast.

