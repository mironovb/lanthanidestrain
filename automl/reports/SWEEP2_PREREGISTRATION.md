# Pre-registration: can angular information, or an auxiliary target, extract more from the complexes?

**Written and committed before the first run of any cell below exists.**

---

## 1. The gap, established by inventory rather than intuition

Across **all 662 topological runs on disk**, `preset` is `baseline_2d` in
**662/662**. The tabular side-input to every neural encoder this study has ever
trained is `rdkit + ecfp + metal + cond + plan`.

What that means concretely, from `automl/topo/simplicial_data.py`:

- **node inputs are 5 scalars** — xTB charge, a charge-missing flag, is-metal,
  is-donor, distance-to-metal — plus an element embedding;
- **edge input is 1 scalar**, the VR filtration radius, which for a 1-simplex *is*
  the interatomic distance;
- **triangle input is 1 scalar**, its filtration radius.

There is **no angular, directional or three-body quantity anywhere in any neural
encoder in this repository.** `snn.py` and `dist_gnn.py` both say so explicitly:
invariance is achieved by *refusing to admit a coordinate*.

Meanwhile the hand-crafted blocks contain **119 angular/polyhedral columns** —
donor–M–donor angles (`p3d_poly`), continuous shape measures against ideal
polyhedra (`g3`), %V_bur and solid-angle fraction (`g4`), bite angles (`g8`) —
and they have **only ever been tested as tree features**, where they lose
(`inner_sphere` = +0.004 [−0.018, +0.020]).

A coordination polyhedron is an angular object. Every encoder in this study has
been blind to it.

**And the coordinates are already there.** `Complex.coords` is loaded from the
shipped asset and used only to compute `dist_to_metal` before being discarded.

Second gap: **no auxiliary-target / multi-task setup has ever been run.** xTB
charge, `E_int`, coordination number and CShM are all *inputs* somewhere; none has
ever been a training *target*.

## 2. Why this is the first sweep in this study that can select

The persistence-image sweep could not rank 25 configurations: re-running one cell
changed Stage A's winner, because the 8-seed run-to-run spread was **0.0092**.
`--deterministic` now makes runs **bit-identical** (`DETERMINISM_RESULTS.md`), so
4 seeds give an exactly reproducible ranking and there is nothing left to average
away. That is why fewer seeds are correct here, not a corner cut.

## 3. Fixed configuration

Base is **G0**, the graph encoder: `--arch snn --no-triangles`. Chosen on
evidence, not preference — it beats the published simplicial arm (+0.2459 vs
+0.2382 binned, +0.1795 vs +0.1683 strict), runs faster (538 s vs 676 s median),
and is much cheaper to make deterministic because the float64 sorted scatter is
dominated by the 9.3 M triangles it does not have.

Fixed in every cell: `--pair-loss-weight 2.0 --select-on adjacent --deterministic
--folds 5 --repeats 3`, dim 96, layers 3, dropout 0.15, filtration 3.5,
heavy-only, `--level-weight` unset, seeds **{7, 11, 23, 37}**.

**Capacity is deliberately not swept.** `snn_wide` (dim 160, layers 4) was flat
and `snn_wide_pair` **collapsed to +0.0021** when capacity met the contrast
objective. Re-finding a known failure is not worth four cells.

## 4. The cells — 11 × 4 seeds = 44 runs

| cell | change | axis |
|---|---|---|
| **A0** | anchor: G0 as above | — |
| **A1** | `--preset baseline_2d_shape` — 119 angular/polyhedral columns into the hybrid head | angular |
| **A2** | `--node-angular` — per-node cosine histogram of the angles its neighbours subtend | angular |
| **A3** | `--angular-readout` — a readout block from the donor–M–donor angle distribution | angular |
| **B1** | `--aux-target cshm` — second head predicts distance to the nearest ideal polyhedron | multi-task |
| **B2** | `--aux-target eint` — second head predicts `E_int` and `dG_transfer` | multi-task |
| **B3** | `--aux-target qtransfer` — second head predicts metal charge transfer | multi-task |
| **C1** | `--radial-bins 64 --radial-max 10.0` (hardcoded 32 / 8.0 in all 662 runs) | readout |
| **C2** | `--attn-pool` — attention pooling, metal embedding as query | readout |
| **C3** | `--lr 5e-4` (2e-3 in all 662 runs) | optimisation |
| **C4** | `--weight-decay 1e-3` (1e-4 in all 662 runs) | optimisation |

**Axis B is the one to watch.** It reuses the energy campaign that *failed*: as
**inputs** the energies destroyed the adjacent-pair metric (−0.2993 strict) by
substituting a proxy at SNR 0.25 for the exact ionic-radius lookup. As **targets**
they cannot enter the prediction path at all and can only shape the encoder. That
is a genuinely different use of the same 953 calculations.

Invariance is preserved and tested: cosines are invariant to rotation,
translation and reflection, so admitting them costs none of the invariance the
"no coordinates" rule was protecting. `automl/tests/test_angular.py` asserts all
three plus permutation equivariance — and already caught a subsampling shortcut
that silently broke equivariance on 12.5 % of nodes.

## 5. Analysis, fixed now

**Screening is selection, not inference.** Every cell is scored on the **84 tune
extractants only**, against A0. No confirmatory language is used about the screen
and no multiplicity penalty is claimed for it.

**One confirmatory look.** The single best cell is then run at **16 seeds**,
matched against A0 at 16 seeds — *both sides replicated*, the rule
`PI_SWEEP_PRECISION.md` paid 25 runs to learn — and scored **once** on the 78
confirm extractants, under **both block keys**, with the multiplicity-respecting
cluster bootstrap. Look count rises to **≥ 21**.

Main effects are reported per axis wherever an axis has ≥ 2 cells, not a cell
ranking.

## 6. Decision rule

| outcome | consequence |
|---|---|
| a cell beats A0 on tune by > 0.005 **and** its confirmatory contrast excludes zero after correction | **Angular information (or an auxiliary target) is a real addition.** This would be the study's first genuine improvement to the headline metric rather than another control, and it would say every encoder to date was blind to the coordination polyhedron. |
| a cell wins on tune but its confirmatory contrast spans zero | Screening noise. Report the null and say the screen selected on 84 extractants and did not replicate on 78. |
| **no cell beats A0 by > 0.005 on the tune half** | Report the null and **do not spend the confirmatory run** — looking twice at nothing is how a winner gets manufactured. It would also be a strong statement: the encoder is not limited by its blindness to angles, which is surprising given a coordination polyhedron is angular. |
| A1 wins but A2/A3 do not | The information helps *through the tabular head*, not through message passing — i.e. it is a feature-engineering result, not a representation result. Say which. |

**Every one of these is reportable.** Writing them down now is the point.

## 7. Guards

`control_guard --verify` before and after. Runs write to
`automl/artifacts/topo_sweep2/`, which no published test reads. Every new flag
defaults off and `test_angular.py` pins that the encoder with all flags off is
parameter-identical to the published one. `node_feat[:, :5]` is unchanged and the
angular columns are appended after it, so `snn.py`'s reads of `dist_to_metal`
(index 4) and `is_donor` (index 3) are untouched. No writes to `data/`. Existing
reports append-only.

---

**Bogdan Mironov · 30 July 2026**
