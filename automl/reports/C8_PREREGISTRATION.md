# C8 — does a Hamiltonian that gets the lanthanide contraction right move the score?

Written before any C8 training number exists. The geometry facts below are
already measured; the modelling claim is not.

## What is already established

**GFN2-xTB under-predicts the lanthanide contraction by ~2.5×, in gas and in
solvent.** Slope of the computed mean M–donor distance against Shannon (1976)
ionic radii, 6 ligands × 15 lanthanides, 270/270 optimisations converged:

| arm | gas | water |
|---|---|---|
| GFN2 | **0.386** (per-ligand 0.12–0.73) | 0.398 (0.22–0.73) |
| g-xTB high spin | **1.142** (1.03–1.20) | 1.313 (1.01–1.64) |

GFN2 is not merely too small — its per-ligand slope scatters six-fold, i.e. it
is mostly noise (23 % of its non-linear response is shared across ligands,
against g-xTB's 96 %; cross-ligand residual profile r = +0.16 vs **+0.963**).

At fixed geometry the same split appears electronically: the HOMO–LUMO gap
residual from linear-in-Z is 0.00075 eV under GFN2 and **0.278 eV** under
g-xTB, the latter reproducing at r = +0.97 across independent runs, with a
**+1.15 eV** half-shell (gadolinium) break where GFN2 shows +0.012 eV.

## What is already known to argue AGAINST a score gain

Stated here so a null result cannot be re-narrated afterwards as expected all
along, and a positive one cannot be over-read.

1. **Correspondence bought nothing.** The serial geometries were 455× cleaner
   (adjacent-pair RMSD 5.46 → 0.0120 Å, SNR 0.14 → 0.799) and predicted
   **worse**: −0.0129 on `sel_adj_logSF_r2`, t = −2.67, 2/8 seeds up. Better
   geometry has already failed to become a better score once.
2. **96.1 % of g-xTB's new structure is a pure function of metal identity**,
   which the model already has — metal identity is recoverable at R² = 0.9995.
3. **The per-ligand compliance does not predict measured selectivity.** On 36
   ligands (23 matched), partial correlation controlling for ligand size and CN:
   **+0.024** (GFN2), **+0.056** (g-xTB). The raw GFN2 value (+0.221) was almost
   entirely size confounding.
4. **The g-xTB minima sit a median 1.24 Å from the shipped ones**, and 21 % of
   complexes change their donor set outright — a conformer hop, not physics.

## The C8 contrast

Two arms over an identical complex set, identical order, identical build ids;
the only difference is where the atoms are:

| arm | coordinates |
|---|---|
| `gxtb` | the g-xTB minimum, reached from the shipped coordinates |
| `ship` | the shipped coordinates — what models are given today |

Config fixed at the C7 setting, **not re-tuned** (C-I says architecture is not
the variable): `--arch dist --pair-loss-weight 2.0 --select-on adjacent
--filtration-max 4.0 --rbf-bins 64 --folds 5 --repeats 3 --deterministic`,
8 paired seeds per arm.

`build_vr_gxtb --verify` must show `build_ids`, `node_ptr`, `atomic_numbers`
and `is_metal` identical between arms and coordinates differing. Anything else
means the contrast compares datasets, not geometries.

**Basin hops are dropped from BOTH arms together**, never from one. A hop is a
changed donor set or CN; the rate is reported, not hidden.

## Declared in advance

- **Primary metric:** `sel_adj_logSF_r2`, paired by seed, reported as mean Δ
  with the paired t and the up/down seed count. The seed count matters: the
  serial result was −0.0129 with 2/8 up, and a mean without the count hides
  whether one seed carried it.
- **Effect I would call real:** Δ ≥ +0.02 with ≥ 6/8 seeds up. That is above
  the ±0.0136 paired sd seen in the serial contrast.
- **Null:** |Δ| < 0.02, or an inconsistent seed split.
- **A negative result is publishable and will be reported as such.** Three
  independent lines already predict it; if it lands, the conclusion is that
  adjacent-lanthanide selectivity is not limited by geometric fidelity at all,
  and the recommendation is to stop spending on 3D encoders for this problem.

## What a positive result would NOT license

If Δ is positive, it must still be shown that the gain is not a re-expression
of the tabular ionic radius. The masked-metal diagnostic and the
leave-metal-block-out split ({11,12,13} Ho/Er/Tm held out of training rows)
decide that, and both must be run before any claim that geometry carries
transferable information.
