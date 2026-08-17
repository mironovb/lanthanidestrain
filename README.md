# Lanthanide extraction: what limits adjacent-lanthanide selectivity prediction

**As of 16 August 2026.** Predicting the distribution coefficient `log D` of
lanthanide(III) extraction complexes, and in particular the **separation of
*adjacent* lanthanides** — the industrially valuable case, where neighbouring
ionic radii differ by ~0.013 Å.

Dataset: **4,746 measurements · 162 extractants · 14 lanthanides · 956 unique
Architector / GFN2-xTB complexes.** Every number is **leave-extractants-out**
cross-validation (5 folds × 3 repeats), so an extractant never appears in both
train and test.

The project has produced two things: a **modelling result** (3D topology
contributes, by complementarity) and — more recently — an **explanation of why
it contributes so little**, which turns out to be a property of the
electronic-structure method rather than of the models.

---

## 1. The chemistry result: GFN2-xTB gets the lanthanide contraction wrong

**GFN2-xTB underestimates the lanthanide contraction in coordination complexes
by 2.47×. g-xTB reproduces it to within 8 % of experiment.**

Per-ligand compliance `c_L = d⟨M–donor⟩ / d r_Shannon`, where **1.00 is exact
agreement with Shannon (1976) effective ionic radii**. 71 distinct ligands × 15
lanthanides × 2 Hamiltonians, one binary, one protocol, 2,130 optimisations:

| | c_L | vs experiment | t vs 1.0 |
|---|---|---|---|
| **GFN2-xTB** | 0.405 ± 0.145 | **under by 2.47×** | −34.5, p = 1.1e−45 |
| **g-xTB** | 1.078 ± 0.094 | over by 1.08× | +7.0, p = 1.4e−09 |

Improves on **71 of 71 ligands** (paired p = 4.9e−52). Reproduced in solvent and
on an independent 6-ligand pilot. GFN2's per-ligand slope is not merely too
small — it is mostly *noise*: cv 0.358, with only **23 %** of its non-linear
response shared across ligands, against g-xTB's **96 %**.

The cause is documented in GFN2's own parameter file: every lanthanide parameter
from Ce(58) to Lu(71) is **linear interpolation between two fitted anchors**
(max residual 5e−7). Metal identity therefore enters a GFN2 geometry as *one
linear-in-Z scalar*, by construction — no f-shell, no crystal field, no
gadolinium break. That retrodicts the measured **effective rank 1.05 of 8**
across eight independent 3D encoders.

Under g-xTB, which puts the f electrons in the valence, the same complexes show
a **+1.15 eV half-shell (gadolinium) break** against GFN2's +0.012 eV, and 370×
more departure from linear-in-Z, reproducing at r = +0.97 across independent
runs.

**Why this matters beyond this project:** it explains a long run of null 3D
results for lanthanide selectivity. The geometries these models are given barely
encode the contraction, and what they do encode is ligand-inconsistent.

## 2. Fixing the chemistry does *not* improve the score

Better geometry was tested directly and does not help — four independent ways,
any of which could have gone the other way:

| test | result |
|---|---|
| structures rebuilt **in correspondence** (455× cleaner, SNR 0.14 → 0.80) | **−0.0129**, t = −2.67, 2/8 seeds |
| g-xTB geometries, with tabular features | **−0.0150**, 2/5 seeds |
| g-xTB geometries, **geometry-only** | +0.0333 — **an artefact, see below** |
| per-ligand compliance vs measured selectivity (n = 44) | r = +0.11 (GFN2), **−0.02** (g-xTB) |

**96.1 %** of g-xTB's new structure is a *pure function of metal identity*,
which the model already has — metal identity is recoverable at R² = 0.9995. And
g-xTB makes the response **more** uniform across ligands (cv 0.358 → 0.087),
i.e. closer to one universal constant × the tabular ionic radius.

### The one positive arm was calibration, not information

The geometry-only arm cleared its pre-registered bar (Δ ≥ +0.02, ≥6/8 seeds) at
p = 0.020. It is still not real, and the checks that killed it are worth reusing:

- within the same runs `sel_adj_pearson` moved the **other way** (−0.0170, 2/8);
- both arms sit at **negative R²**, where shrinking toward the mean raises R²;
- R² after optimal rescaling (= Pearson²) **reverses**: 0.0072 vs 0.0108;
- **a single scalar applied to the OLD geometry recovers 237 % of the "gain"** —
  the rescaled shipped arm beats g-xTB on **8/8 seeds** (p = 0.0006).

Reporting the pre-registered primary metric alone would have published a
positive 3D result that is an artefact of prediction variance.

## 3. Where the remaining headroom is

**The metric is not noise-limited.** Raw `log_D` scatter within a (block, metal)
group is 0.72–0.95 — larger than the entire spread of adjacent separations
(0.2236), which would imply a negative ceiling. That naive model is wrong:
condition effects are **shared between the two metals of a pair and cancel on
differencing**. Measured on 203 repeated (composition, adjacent-pair) cases, a
separation reproduces to **0.1533** — 6× tighter than the levels it is computed
from.

**Correction (16 Aug 2026, `automl/reports/CEILING_NOTE.md`):** the ≈ +0.53
ceiling previously quoted here came from an estimator that was formally
withdrawn (`CEILING_CLOSED.md`); **no point ceiling is identifiable from this
dataset**. What is defensible: a separation reproduces to ~0.16 log units
across independent condition sets on a spread of ~0.22 — consistent with
substantial but unquantifiable headroom above the current best (+0.326).

Tested and not the answer either: a **direct pair head** predicting the
difference from `[h_i, h_j, h_i − h_j]` gives **+0.0123 (4/6 seeds, p = 0.30,
n.s.)**, and *reconciling* the level head to it is **catastrophic**: −1.16,
0/6 seeds, p = 0.002.

## 3b. The August 2026 campaign (findings I14–I16)

**A single anchored model now beats every stack this project fitted, and the
3D encoder's contribution — confirmed on a never-touched 444-pair
population — flows through its shape channel.**

`pred = anchor + 0.65·shape_tabular + 0.35·shape_encoder`, where the anchor
and tabular shape come from an anchored CatBoost (champion quantile loss for
the level, a second model trained on block-centred targets for the shape) and
the encoder shape is the block-centred c15 distance encoder:
**+0.326 adjacent-pair R² on the legacy 905** (previous best stack +0.313);
pre-declared fresh-444 look: 3D-vs-tabular contrast **+0.0156, PASS**,
uniform ~+0.015 across populations. Full story, killed candidates and named
next tests: [`automl/reports/CAMPAIGN_AUG2026.md`](automl/reports/CAMPAIGN_AUG2026.md).

## 4. The modelling result of the topology campaign (July; superseded as the best system)

The topology campaign's headline stands and is preserved in full in
[`docs/README_2026-07-22.md`](docs/README_2026-07-22.md):

> Adding a simplicial network over a Vietoris–Rips complex to the best
> combination without a 3D model raises adjacent-pair log-SF R² from **+0.2263
> to +0.2672** (+0.0381, 90 % CI [+0.0191, +0.0495]), while also improving
> overall `log D` accuracy.

It earns its place by **complementarity**, not by strength — alone it loses, and
four pre-registered attempts to show otherwise failed.

**One claim in that document is now corrected.** It described the
adjacent-lanthanide geometric contrast as sitting "below the ~0.04 Å
optimisation-noise floor". That number was never measured; it traces to an
asserted conformer scatter and coincides exactly with the `tight` convergence
target in **eV/Å (force)**, not Å (distance). Measured directly by perturbed
restarts: **0.0002 Å**, ~200× smaller, with 0 % basin escape at σ = 0.05 Å. The
contrast is redundant with the tabular ionic radius for the reason in §1, not
because it is below a noise floor.

## 5. Portable findings

Arguably more transferable than any headline.

| finding | evidence |
|---|---|
| **Train the contrast, not the absolute value.** | +0.030 tabular, +0.042 PI-CNN, +0.186 SNN |
| **MAE instead of RMSE nearly doubles the tabular arm's selectivity.** | +0.107 — the largest single gain in the project |
| **Rank transforms destroy separation-factor signal.** A separation factor *is* spacing; `QuantileTransformer` preserves order and destroys spacing. Trees are immune, so it goes unnoticed. | one line: **+0.005 → +0.221** |
| **A metric on a negative-R² baseline needs a scale-free check.** R² can rise purely by shrinking predictions. | a +0.0333 result reversed to −0.0036 once the scale was free |
| **A flag that is silently ignored reads exactly like a real null.** | `--pair-head` was a no-op on `--arch dist` for the whole study while recording `pair_head=True` |
| **A degenerate control must fail loudly.** | a "partial correlation controlling for size and CN" controlled for two constants and returned the raw correlation |
| **Stack predictions, not representations.** | fold identity recoverable from out-of-fold embeddings at 100 % |
| **A cluster bootstrap that collapses duplicates isn't one.** | published intervals were 12–29 % too narrow |
| **Test against the champion, not the convenience baseline.** | four signals vanished or reversed when the baseline was strengthened |

## 6. Reading order

| document | contents |
|---|---|
| [`automl/reports/CAMPAIGN_AUG2026.md`](automl/reports/CAMPAIGN_AUG2026.md) | **start here** — the August campaign: the anchored system, the confirmed 3D shape channel, the kills |
| [`automl/reports/CAMPAIGN_SUMMARY_gxtb.md`](automl/reports/CAMPAIGN_SUMMARY_gxtb.md) | the g-xTB campaign and the null |
| [`automl/reports/SCIENTIFIC_FINDINGS.md`](automl/reports/SCIENTIFIC_FINDINGS.md) | standing register: every claim with its status and its falsifying test |
| [`automl/reports/C8_RESULTS.md`](automl/reports/C8_RESULTS.md) | why the one positive 3D arm was calibration |
| [`automl/reports/NOISE_FLOOR.md`](automl/reports/NOISE_FLOOR.md) | the 200× correction to the noise-floor claim |
| [`automl/reports/SYNTHESIS.md`](automl/reports/SYNTHESIS.md) | the topology campaign, positive and negative |
| [`docs/README_2026-07-22.md`](docs/README_2026-07-22.md) | the archived README for the topology result |

Every confirmatory test has a **pre-registration committed before its data
existed** (`*_PREREGISTRATION.md`), stating the endpoint, the decision rule and
the consequence of each outcome in advance.

## 7. Reproducing

```bash
module load anaconda/Python-ML-2025a
export PYTHONPATH=$PWD

# the modelling result
python3 -m automl.topo.stack_test    --n-boot 400   # the pre-registered result
python3 -m automl.topo.filt_test     --n-boot 400   # replication across radii
python3 -m automl.topo.control_guard --verify       # nothing published moved

# the chemistry result (needs the g-xTB binary; see automl/qc/gxtb_probe.py)
python3 -m automl.qc.gxtb_probe   --anchor <xyz> --method both
python3 -m automl.qc.gxtb_series  --anchors 6 --workers 48
python3 -m automl.qc.compliance_test --tags cf_shard0,cf_shard1
```

`control_guard` pins **324 artefacts** by SHA-256 — every published out-of-fold
parquet, result table and figure — and verifies them byte-identical. `data/` is
read-only throughout: no geometry was regenerated and no shipped table modified.
All derived artefacts live under `automl/artifacts/` and `automl/reports/`.

Any change to `train.py` is proven inert by re-running a published arm under
`--deterministic` and requiring `max |Δoof| = 0`. The most recent such change
(giving `DistanceNet` a working pair head) passed at **0.000e+00** across all
4,746 rows.
