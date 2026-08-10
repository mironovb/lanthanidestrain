# C8 — the answer: a correct lanthanide contraction does not become a better score

Snapshot arms, 737 matched complexes (identical build ids, node_ptr,
atomic_numbers, is_metal; coordinates differing), 8 paired deterministic seeds.

## The pre-registered contrast: null

| arm | g-xTB | shipped | Δ | seeds up |
|---|---|---|---|---|
| `main` (with tabular block) | +0.1413 | +0.1563 | **−0.0150** | 2/5 |
| `topo` (geometry only) | −0.0582 | −0.0916 | **+0.0333** | 7/8, p = 0.020 |

The bar was Δ ≥ +0.02 with ≥ 6/8 seeds up. `topo` appears to clear it. **It does
not survive inspection, and the reason is instructive.**

## Why the +0.0333 is not a result

Within the same runs, the metrics disagree:

| metric | Δ | seeds up |
|---|---|---|
| `sel_adj_logSF_r2` | +0.0333 | 7/8 |
| `sel_adj_sign_accuracy` | +0.0181 | 6/8 |
| **`sel_adj_pearson`** | **−0.0170** | **2/8** |

R² up, correlation **down**. Both arms sit at *negative* R², where predicting
closer to a constant raises R² without predicting anything better. Measured
directly on the out-of-fold adjacent separations:

| | g-xTB | shipped |
|---|---|---|
| sd of predicted separations | 0.0889 | 0.1102 |
| dispersion vs truth (sd = 0.2729) | **0.33×** | 0.40× |
| Pearson r | +0.0787 | +0.0958 |

Both models predict with about a third of the true spread and correlate at
r ≈ 0.08. The g-xTB arm is simply **19 % more shrunk toward the mean**.

The scale-free test settles it. R² after optimal affine rescaling of the
prediction is exactly Pearson², which removes every calibration effect:

| | as scored | scale-free (r²) |
|---|---|---|
| g-xTB | −0.0582 | **0.00721** |
| shipped | −0.0916 | **0.01078** |
| Δ | +0.0333 (7/8) | **−0.00357** (2/8) |

**Once the scale is free the advantage reverses.** The whole +0.0333 was the
shipped arm being more over-dispersed and therefore more penalised. The
information ceiling of each arm is r² ≈ 0.007 vs 0.011 — geometry alone
explains ~1 % of adjacent-separation variance under either Hamiltonian.

## Verdict

**A Hamiltonian that reproduces the lanthanide contraction to within 8 % of
experiment, replacing one that underestimates it 2.5×, does not improve
adjacent-lanthanide selectivity prediction.** Not with tabular features
(Δ = −0.0150), and not as geometry alone once calibration is accounted for
(Δ = −0.0036).

This is consistent with everything else measured this campaign, and those
independent lines are why the null is credible rather than a power failure:

1. correspondence made the geometry 455× cleaner and predicted **worse**
   (−0.0129, t = −2.67);
2. 96.1 % of g-xTB's new structure is a pure function of metal identity, which
   the model already has at R² = 0.9995;
3. per-ligand compliance does not predict measured selectivity (n = 44,
   r = +0.11 GFN2, −0.02 g-xTB);
4. g-xTB makes the response **more** uniform across ligands (cv 0.358 → 0.087),
   i.e. closer to one universal constant × the tabular ionic radius.

## What this does not say

It does not say g-xTB is not better chemistry — it demonstrably is, by
2.5× against experiment on 71 of 71 ligands (p = 5e−52). It says the geometric
channel is not where adjacent-lanthanide selectivity is limited. The bottleneck
is not geometric fidelity, and further spending on 3D encoders or on better
structures for *this* target is not warranted on the present evidence.

## Methodological note

Had the campaign reported `sel_adj_logSF_r2` alone — the pre-registered primary
metric, clearing its pre-registered bar at p = 0.020 with 7/8 seeds — it would
have published a positive 3D result that is an artefact of prediction variance.
The disagreement between R² and Pearson **within the same runs** is what
exposed it. A single headline metric on a negative-R² baseline cannot be
trusted without a scale-free check.
