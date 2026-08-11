# Campaign summary: a better Hamiltonian, and where the headroom actually is

## 1. The chemistry result — large, externally validated, and new

**GFN2-xTB underestimates the lanthanide contraction in coordination complexes
by 2.47×. g-xTB reproduces it to within 8 % of experiment.**

Per-ligand compliance `c_L = d⟨M–donor⟩ / d r_Shannon`, where 1.00 is exact
agreement with Shannon (1976) effective ionic radii. 71 distinct ligands × 15
lanthanides × 2 Hamiltonians, one binary, one protocol, 2130 optimisations:

| | c_L | vs experiment | t vs 1.0 |
|---|---|---|---|
| **GFN2** | 0.405 ± 0.145 | **under by 2.47×** | −34.5, p = 1.1e−45 |
| **g-xTB** | 1.078 ± 0.094 | over by 1.08× | +7.0, p = 1.4e−09 |

Improves on **71 of 71 ligands** (paired t = +43.0, p = 4.9e−52). Reproduced in
solvent and on an independent 6-ligand pilot. GFN2's per-ligand slope is also
mostly *noise*: cv 0.358, only 23 % of its non-linear response shared across
ligands, against g-xTB's 96 %.

Supporting it electronically, at fixed geometry: after removing the linear-in-Z
trend the HOMO–LUMO gap residual is **0.00075 eV (GFN2) vs 0.278 eV (g-xTB)**,
the latter reproducing at r = +0.97 across independent runs, with a **+1.15 eV
half-shell (gadolinium) break** where GFN2 shows +0.012 eV — a straight ramp
split in the middle, not a break.

**Why this matters beyond us:** it explains a long run of null 3D results, here
and plausibly elsewhere. The geometries these models are given barely encode the
contraction, and what they do encode is ligand-inconsistent.

## 2. It does not become a better score

Four independent lines, three of them able to have gone the other way:

| test | result |
|---|---|
| correspondence (455× cleaner geometry, SNR 0.14→0.80) | **−0.0129**, t = −2.67, 2/8 seeds |
| g-xTB geometry, with tabular features | **−0.0150**, 2/5 seeds |
| g-xTB geometry, **geometry-only** | +0.0333 (7/8, p = 0.020) — **an artefact, see §3** |
| per-ligand compliance vs measured selectivity | r = +0.11 (GFN2), **−0.02** (g-xTB), n = 44 |

96.1 % of g-xTB's new structure is a **pure function of metal identity**, which
the model already has at R² = 0.9995. And g-xTB makes the response *more*
uniform across ligands (cv 0.358 → 0.087) — closer to one universal constant ×
the tabular ionic radius.

## 3. The one positive result was calibration, not information

The geometry-only arm cleared its pre-registered bar (Δ ≥ +0.02, ≥6/8 seeds) at
p = 0.020. It is still not real:

- within the same runs, `sel_adj_pearson` moved the **other way**: −0.0170, 2/8;
- both arms sit at **negative R²**, where shrinking toward the mean raises R²;
- predicted-separation sd 0.0889 (g-xTB) vs 0.1102 (shipped) on a true sd of
  0.2729 — both ~⅓ dispersed, g-xTB 19 % more shrunk;
- R² after optimal rescaling (= Pearson²) **reverses**: 0.00721 vs 0.01078;
- **a single scalar on the OLD geometry recovers 237 % of the "gain"** — the
  rescaled shipped arm beats g-xTB on **8/8 seeds**, t = +5.90, p = 0.0006.

Reporting the pre-registered primary metric alone would have published a
positive 3D result that is an artefact of prediction variance.

## 4. Where the headroom is — and one lever that is now testable

The metric is **not** noise-limited. Raw `log_D` scatter within a (block, metal)
group is 0.72–0.95, but condition effects are shared between the two metals of a
pair and cancel on differencing: the same (composition, adjacent pair) measured
in ≥2 independent strict blocks (n = 203) reproduces to **0.1533**, against a
spread of separations of 0.2236.

**Ceiling ≈ +0.53. Current best +0.183.** We are at ~35 % of what the labels
permit, and that is a lower bound.

Geometry is not where the rest is — it explains ~1 % of adjacent-separation
variance under either Hamiltonian.

## 5. Bugs found, one of which invalidated a lever for the whole study

- **`--pair-head` was a silent no-op on `--arch dist`.** `DistanceNet` never
  accepted the parameter, though its docstring claims signature parity with
  `SimplicialNet`; `--pair-reconcile` then also no-opped. The flag was recorded
  as `pair_head=True` in `results.jsonl`. Caught only because 24 cells returned
  four arms agreeing to six decimals. Fixed; byte-identity with the flag off
  **max |Δoof| = 0.000e+00**.
- Now that it works: auxiliary pair loss **+0.0123 (4/6, p = 0.30, n.s.)**;
  reconciliation **−1.1625 (0/6, p = 0.002)** — catastrophic, because the pair
  head is far worse at levels than the level head and reconcile replaces
  wholesale rather than blending.
- A "partial correlation controlling for ligand size and CN" that **controlled
  for two constants** — `n_atoms` and `cn` never propagated into the records, so
  it returned exactly the raw correlation. `_partial` now fails loudly.
- **Size-correlated missingness**: my sweeper was submitted with `--array` as a
  script argument, died in 0 s, and 52 complexes (median 340 atoms vs 214) went
  unrun. Fixed; re-optimisation completed 956/956.

## 6. Recommendation

Stop spending on 3D encoders and on better structures for adjacent-lanthanide
selectivity. The bottleneck is not geometric fidelity, and that is now shown
four independent ways rather than assumed. The remaining ~0.35 of R² is in the
labels' reach but lives elsewhere — conditions, direct pair modelling done
differently from reconcile, and chemical descriptors of ligand-specific
discrimination.

The chemistry finding (§1) stands on its own and is publishable independent of
any modelling outcome.
