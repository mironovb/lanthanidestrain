# Lanthanide extraction: 3D topology for `log D` and adjacent-lanthanide selectivity

**Result as of 22 July 2026.** Predicting the distribution coefficient `log D`
of lanthanide(III) extraction complexes, and in particular the **separation of
*adjacent* lanthanides** — the hardest and industrially most valuable case,
where neighbouring ionic radii differ by only ~0.013 Å.

Dataset: **4,746 measurements · 162 extractants · 14 lanthanides · 953 unique
Architector / GFN2-xTB complexes.** Every number below is
**leave-extractants-out** cross-validation (5 folds × 3 repeats), so an
extractant never appears in both train and test.

---

## The claim

> **Message passing over a Vietoris–Rips complex of the 3D structure supplies
> adjacent-lanthanide selectivity information that 2D fingerprint models do not
> have.** Adding it to the best no-topology stack raises adjacent-pair
> log-separation-factor R² from **+0.2263 to +0.2672**, while *also* improving
> overall `log D` accuracy.

**Best model — CatBoost + repaired FCNN + simplicial network**, nested
per-extractant weights 0.20 / 0.30 / **0.50**:

| model | adjacent-pair log SF R² | overall `log D` R² |
|---|---|---|
| **stack + simplicial encoder** | **+0.2672** | **+0.4369** |
| stack, no topology (CatBoost + repaired FCNN) | +0.2263 | +0.4328 |
| stack with topology slot given to a matched control | +0.2208 | +0.4288 |
| CatBoost alone | +0.1422 | +0.4987 |
| repaired FCNN alone | +0.2206 | +0.3218 |
| FCNN as originally published | +0.0048 | +0.3872 |

### Significance

Paired cluster bootstrap resampling whole extractants, 400 draws:

| contrast | Δ adjacent-pair R² | 90 % CI | multiplicity-corrected |
|---|---|---|---|
| **drop-in** — add the encoder to the best no-topology stack | **+0.0381** | [+0.0191, +0.0495] | **[+0.0166, +0.0595]** (5-test) |
| **swap** — encoder vs a matched no-topology control, same slot | **+0.0446** | [+0.0298, +0.0544] | **[+0.0272, +0.0621]** (5-test) |

Both also survive a **multiplicity-respecting** cluster bootstrap (one that does
not collapse duplicate clusters) applied *simultaneously* with the Bonferroni
penalty: **[+0.0136, +0.0613]** and **[+0.0225, +0.0651]**.

### Replication

| check | result |
|---|---|
| **Split-half** — two disjoint 8-seed ensembles | both add: **+0.0393** and **+0.0375** |
| **Filtration radius 3.0 Å** | adds: **+0.0382** [+0.0178, +0.0503]; 7-look corrected [+0.0141, +0.0624] |
| **Filtration radius 4.0 Å** | adds: **+0.0327** [+0.0140, +0.0435]; 7-look corrected [+0.0108, +0.0547] |
| Re-run of the whole analysis | reproduces bit-for-bit |

The radii give genuinely different complexes (relative to 3.5 Å: 0.59× and 2.29×
the triangles), so **3.5 Å is not a tuned radius**.

---

## What the claim is *not*

Each limit was measured, not assumed. These are why the result above is
credible.

1. **Not "3D topology helps."** A different topological representation — a CNN
   on persistence images — **fails to replicate** (−0.0041, n.s.) and is as
   redundant with fingerprints as the no-topology control.
2. **Not "topology beats the baseline."** Alone it does not — **four**
   pre-registered attempts failed. It earns its place by *complementarity*.
3. **Not a selectivity signal readable from the geometry.** The
   adjacent-lanthanide contrast is redundant with the tabular ionic radius and
   sits below the ~0.04 Å optimisation-noise floor — four independent null tests.
4. **Not transferable to representations.** Handing a downstream learner the
   *embedding* rather than the *prediction* fails by construction: out-of-fold
   embeddings from *k* folds are *k* different latent bases (fold identity is
   predictable from the embedding at **100 %** accuracy).

### The mechanism, which predicted rather than explained

An arm improves a stack only if it is **both strong on the scored metric and
decorrelated from its partner**:

| arm | adjacent-pair R² | error correlation with baseline | adds? |
|---|---|---|---|
| simplicial, 3.0–4.0 Å | +0.232 – +0.238 | 0.897 – 0.907 | **yes, all radii** |
| persistence-image CNN | +0.210 | 0.933 | no |
| matched tabular control | +0.203 | 0.928 | no |
| CatBoost | +0.144 | 0.880 | no (contributes accuracy instead) |

Stated after the persistence-image failure, this **predicted in advance** that
the filtration variants would add. They did — and the effect declines
monotonically with radius (+0.0382 → +0.0381 → +0.0327) as error correlation
rises (0.898 → 0.897 → 0.907).

---

## Portable findings from the negative results

Arguably more transferable than the headline.

| finding | evidence |
|---|---|
| **Train the contrast, not the absolute value.** Selectivity is a within-block contrast; conventional models optimise absolute `log D`. | +0.030 tabular, +0.042 PI-CNN, +0.186 SNN |
| **Rank transforms destroy separation-factor signal.** `QuantileTransformer` preserves order and destroys *spacing*; a separation factor **is** spacing. Trees are immune, so it goes unnoticed. | one line took the published FCNN from **+0.005 to +0.221** |
| **Baselines need the variance control the arms get.** | a single-seed baseline spanned 0.11 across seed conventions |
| **Model variance and ensembling are substitutes.** Reducing single-model variance cannibalises the ensemble's own gain. | SD −37 %, ensemble *worse*; every lever hurt |
| **Stack predictions, not representations.** | fold identity recoverable from embeddings at 100 % |
| **A cluster bootstrap that collapses duplicates isn't one.** | published intervals were 12–29 % too narrow |
| **Test against the champion, not the convenience baseline.** | four separate signals vanished or reversed when the baseline was strengthened |

---

## Reading order

| document | contents |
|---|---|
| [`automl/reports/SYNTHESIS.md`](automl/reports/SYNTHESIS.md) | **start here** — the whole picture, positive and negative |
| [`automl/reports/STACK_RESULTS.md`](automl/reports/STACK_RESULTS.md) | the positive result, its controls and corrections |
| [`automl/reports/FILT_RESULTS.md`](automl/reports/FILT_RESULTS.md) | replication across filtration radii |
| [`automl/reports/CONTROL_RESULTS.md`](automl/reports/CONTROL_RESULTS.md) | the 2×2 control that reframed the original claim |
| [`S2_RESULTS.md`](automl/reports/S2_RESULTS.md) · [`WO_RESULTS.md`](automl/reports/WO_RESULTS.md) · [`S0X_RESULTS.md`](automl/reports/S0X_RESULTS.md) · [`EMBEDDING_RESULTS.md`](automl/reports/EMBEDDING_RESULTS.md) | the negatives, each with its own diagnosis |
| [`automl/README.md`](automl/README.md) | the AutoML study this grew out of |

Every confirmatory test has a **pre-registration committed before its data
existed** (`*_PREREGISTRATION.md`), stating the endpoint, the decision rule and
the consequence of each outcome in advance.

---

## Reproducing

```bash
module load anaconda/Python-ML-2025a
export PYTHONPATH=$PWD

python3 -m automl.topo.stack_test    --n-boot 400   # the pre-registered result
python3 -m automl.topo.best_stack    --n-boot 400   # the deployable 3-way stack
python3 -m automl.topo.filt_test     --n-boot 400   # replication across radii
python3 -m automl.topo.control_guard --verify       # nothing published moved
```

`control_guard` pins **324 artefacts** by SHA-256 — every published out-of-fold
parquet, result table and figure — and verifies them byte-identical. `data/` is
read-only throughout: no geometry was regenerated and no shipped table modified.
All derived artefacts live under `automl/artifacts/` and `automl/reports/`.
