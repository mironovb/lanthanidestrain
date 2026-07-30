# The conformer lever is closed, with a number instead of a hope

**Bogdan Mironov · 29 July 2026**
Data: `conformer_diagnostic.csv`, `automl/artifacts/mtd_ensemble/`.
Jobs 5278400 (smoke), 5278467 (pilot, 120 complexes), 5278631 (diagnostic).

---

## Why this was run

Multi-conformer sampling is named as the untested physics lever in three
documents — `S2_RESULTS.md` §5, `WO_RESULTS.md` §4, `PI_EMAIL.md` §9 — always
qualitatively: *"a proper CREST/metadynamics ensemble is the real experiment"*.

`ENERGY_RESULTS.md` turned it into a target. Reference xTB energetics destroy the
adjacent-pair metric because within a ligand family they carry the lanthanide
trend at **SNR ≈ 0.25**: a 0.17 eV per-step signal buried in 0.73 eV of scatter.
If that scatter is conformational, an ensemble removes it, and the scatter has to
fall **≥ 3.9×** for the trend to dominate.

CREST is not installed and this cluster has no outbound network, so the search is
built from xtb's own metadynamics: GFN-FF sampling, GFN-FF relaxation of
snapshots, **GFN2 single point for every energy**, RMSD + energy deduplication,
Boltzmann weights at 298 K. Called **CREST-lite** throughout, because it is: no
structure-space clustering, no torsional pre-screening, and the geometries are
GFN-FF minima.

**120 complexes across 9 whole ligand families**, 14 members each — whole
families, because the question is entirely about *within-family* scatter and a
family of one has none.

## Result 1 — the search works, and Boltzmann weighting cannot use it

| | |
|---|---|
| unique conformers per complex | **median 16**, 0 % returned only one |
| **effective ensemble size** `1/Σw²` | **median 1.17** |
| conformer energy spread | median ~1 eV |
| kT at 298 K | 0.0257 eV |

Conformer energy gaps are **~40× kT**, so `exp(−ΔE/kT)` puts essentially all the
weight on one structure. **A Boltzmann average of this ensemble simply is its
minimum.** It cannot reduce scatter by √n, because n_effective is 1.

That falsifies the remedy I assumed when the pilot was designed. It is recorded
as a falsification rather than quietly replaced, because the assumption was
explicit in the module that ran it.

## Result 2 — the shipped geometries are arbitrary, and it is measurable

The surviving hypothesis was different and better posed: the scatter is not
thermal breadth but **arbitrariness** — each dataset geometry is one Architector
local minimum, and *which* minimum differs between complexes for reasons
unrelated to chemistry.

Relaxing the shipped geometry alongside the search structures gives

```
gap = E(shipped, relaxed) − E(global minimum found)
```

| | |
|---|---|
| gap | median **0.317 eV**, IQR [0.051, 0.798], max 3.73 eV |
| shipped geometry is **not** the global minimum | **79 % of complexes** |
| **within-family SD of the gap** | **median 0.434 eV** |

So the hypothesis is *confirmed*: the shipped geometries really are arbitrary
local minima, and the arbitrariness varies within a family by 0.434 eV —
**59 % of the 0.73 eV scatter** that has to be removed.

## Result 3 — and it is still not enough

This is the part that closes the lever.

Removing 59 % of the *variance* leaves √(0.731² − 0.434²) = **0.588 eV** of
scatter, against a per-step signal of 0.170 eV:

> **SNR after a perfect conformer campaign: 0.29.** Still far below 1.

So even a full CREST campaign over all 953 complexes — succeeding completely at
what it is for — **cannot rescue these features**. The remaining 0.588 eV is not
an artefact of which minimum was found. The most likely reading is genuine
sensitivity of GFN2 energies to ligand conformation, which no amount of searching
removes because it is not an error.

**Verdict: not worth running.** The three levers the reports keep naming —
better geometries, better energetics, more conformers — are now all measured, and
none of them reaches the adjacent-pair problem.

## What this does *not* say

- **Not that conformer ensembles are useless in general.** It says they cannot
  lift *these* energy features above the ionic radius on *this* metric. S2 already
  showed conformer augmentation does not help the topological arm either
  (−0.0195), by a different route.
- **Not that GFN2 is wrong.** `metal_probe.csv` showed it separates adjacent
  lanthanides at 17× the relevant scale in a *frozen* cage. The method resolves
  the metals; the geometries do not hold still.
- **Not that DFT would fix it.** DFT on single conformers inherits the same
  problem, because the problem is the conformer, not the Hamiltonian. That
  matters: "use DFT" is the expensive next step and this is evidence against it.

## Limits, stated

- **GFN-FF geometries.** The snapshots are relaxed at GFN-FF and scored at GFN2.
  Acceptable for "can the scatter be cut ~4×", and it would not be acceptable if
  the geometries themselves were the deliverable. A GFN2 relaxation of every
  snapshot was the original design and was abandoned on measurement: 15+ minutes
  per 300-atom structure, ~340 CPU-hours for the pilot alone, against a two-node
  cap — for a pilot whose job is to decide whether a larger campaign is worth it.
- **One diverged calculation**, a "conformer" 1340 eV above the minimum (an SCF
  failure, not chemistry), was filtered by an explicit 20 eV physical bound. The
  filter reports what it drops. The verdict is median-based and was unchanged by
  it — 0.434 eV either way.
- **9 families, 120 complexes**, not the full 953. The pilot was sized to answer
  its own question, and its answer is that the campaign it would have justified
  should not be run.
- The energy spread is sensitive to the metadynamics settings (10 ps, 400 K,
  xtb's own kpush/alpha defaults, deliberately untuned). A more aggressive search
  would find *more* conformers and a *larger* gap, which moves the verdict
  further toward "not enough", not toward the opposite.

---

**Reproduce**

```bash
sbatch --array=0-3 automl/slurm/... mtd_ensemble --pilot 120 --time-ps 10 --max-snapshots 16
python3 -m automl.qc.mtd_ensemble --collect
python3 -m automl.qc.conformer_diagnostic
```

---

## Correction, 30 July 2026 — the shipped-geometry index bug, and the corrected numbers

`AUDIT_2026-07-30.md` E2. `mtd_ensemble.py` recorded `relaxed[0]` as the shipped
geometry's relaxed energy, but `relaxed` receives an entry **only when an
optimisation succeeds**. When the shipped structure's own relaxation failed,
`relaxed[0]` was a metadynamics snapshot and the gap for that complex was
meaningless.

Fixed by carrying an explicit `is_shipped` flag and recording `None` when it
failed. The pilot was re-run with `--overwrite` so every complex carries it.
**4 of 120 complexes had the shipped relaxation fail** and had a wrong gap.

| | as published | corrected |
|---|---|---|
| shipped geometry is not the global minimum | 79 % | **82 %** |
| median gap | 0.321 eV | **0.305 eV** |
| max gap | 3.73 eV | **2.01 eV** |
| **median within-family SD of the gap** | 0.434 eV | **0.503 eV** |
| share of the 0.731 eV scatter it accounts for | 59 % | **69 %** |
| residual scatter after removing it | 0.588 eV | **0.531 eV** |
| **SNR a perfect conformer campaign would reach** | **0.29** | **0.32** |

The audit predicted the verdict would survive because it is a median over nine
families, and it does: **SNR 0.32 is still far below 1**, so a full CREST campaign
over all 953 complexes, succeeding completely at what it is for, still cannot
rescue these features. **Not worth running** stands.

The bug made the study's own case *understated*: arbitrary local minimisation
explains more of the scatter than reported (69 % rather than 59 %), and the
irreducible remainder is correspondingly smaller. Neither is enough.
