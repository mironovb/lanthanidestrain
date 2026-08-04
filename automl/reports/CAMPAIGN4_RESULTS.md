# The neutral extracted species can be built, and the metric does not use it

**Bogdan Mironov · 4 August 2026**
Pre-registered in `CAMPAIGN4_PREREGISTRATION.md` (+ Amendments 1 and 2), each
committed before the step it governs.
Data: `c4_cells.csv`, `neutralize_audit.csv`, `automl/artifacts/vr_neutral/`.
~1,370 GFN2 optimisations plus 12 GPU runs.

---

## Summary

| stage | outcome |
|---|---|
| build the neutral species | **succeeded** — 91.9 % of structures pass every gate |
| does it change the chemistry? | **yes** — 36 % of placed nitrates coordinate inner-sphere |
| does it help the metric? | **no** — `neutral − control` = **+0.0040**, gate is +0.005 |

No confirmatory look was spent. The 29-look budget is intact.

## 1. Why this was run

Three campaigns and 15 cells had all been null, and every one of them changed
*how* the complex is represented. None changed **what molecule is represented**:
953 of 956 modelled complexes were still cationic, while the species that
partitions into kerosene is neutral. Charge neutralisation is the physics of
solvent extraction and it was absent from every structure in the study.

## 2. The structures

683 HNO₃ complexes, two arms each — `neutral` with counter-ions added, `control`
re-optimised through the identical ladder with nothing added. The control is the
comparator, not a diagnostic: this repo had already measured that re-optimising
the same molecule at a new level moves it by median 1.87 Å, so without it
"the neutral structure differs from the shipped one" carries no information.

| | |
|---|---|
| records | 682 of 683 (one structure exceeded the wall clock) |
| **accepted** | **627 (91.9 %)**, above the pre-registered 80 % floor |
| family-correlated failure | none — no REVIEW banner |
| structures carrying per-atom Mulliken charges | 679 |

Rejections, which vindicate the gates rather than merely passing them:

| code | n | what it caught |
|---|---|---|
| `LIGAND_MOVED` | 43 | wholesale conformational change past 2.5 Å |
| `CN_CHANGED` | 4 | displacement beyond what a bound nitrate can occupy |
| **`NITRATE_PROTONATED`** | **3** | **HNO₃ + deprotonated ligand: same formula, same total charge, converged, and invisible to every composition or charge check** |
| `NITRATE_PYRAMIDAL` | 1 | |
| **`CONNECTIVITY_CHANGED`** | **1** | a bond formed or broken inside the ligand |

## 3. Two pre-registered premises of mine were falsified by the data

**Amendment 1 — nitrate does reach the inner sphere.** The pre-registration
asserted that a saturated coordination sphere forces nitrate outward, and forbade
any inner-sphere claim. The supporting analysis had tested *rigid* insertion;
relaxation opens the sphere. Seeded at 5.75–6.50 Å, nitrates relax to Ln–O of
2.13–2.53 Å against a corpus mean of 2.130 Å. At full scale, **614 of 1,728
placed nitrates (36 %) coordinate inner-sphere.**

**Amendment 2 — the CN gate used the wrong reference and forbade the effect.**
It compared the *shipped* CN against the *relaxed neutral* CN, and re-relaxation
alone changes the ligand CN in **49 %** of structures — so the gate was charging
the anion for changes it did not cause. It also required CN to be unchanged,
which a bidentate nitrate entering a saturated sphere cannot satisfy. The revised
bound is `−1 ≤ displaced ≤ 2 × n_inner_nitrate`, derived from the binding mode
rather than fitted; the pilot's own null confirms it, since **with zero
inner-sphere nitrates exactly zero donors are displaced.**

## 4. The screen — 605 adjacent pairs, 130 extractants, tune half only

All three arms share an identical row set: 3,845 rows over 627 complexes.

| arm | binned adj-R² | vs control | strict adj-R² | vs control | overall R² |
|---|---|---|---|---|---|
| shipped, as published | +0.2263 | +0.0046 | +0.1730 | +0.0054 | +0.3912 |
| **control**, re-optimised, no anion | +0.2216 | — | +0.1675 | — | +0.4392 |
| **neutral** | +0.2256 | **+0.0040** | **+0.1863** | **+0.0188** | +0.4565 |

> **Verdict: null.** `neutral − control` = **+0.0040** against a +0.005 gate. The
> confirmatory run is not spent.

**The strict-key number is +0.0188 and I am not claiming it.** The
pre-registration fixed the +0.005 threshold but did not say which block key the
*screen* gates on. I resolved that by precedent, not by outcome: sweep2 and
campaign 3 both gated on the binned key, so binned is the established convention
and applying it here is consistent rather than chosen after seeing the data.
Recording the ambiguity is the honest course, and the strict-key value is
reported as an observation that did not gate. It is a candidate for a future
pre-registration with its own held-out half, not a result of this one.

## 5. What is established regardless of the null

- **The neutral species can be built at scale and validated.** 91.9 % acceptance,
  no family-correlated failure, three proton-transfer structures caught that
  would otherwise have entered training as the wrong molecule.
- **The chemistry genuinely changed.** 36 % inner-sphere coordination is not a
  cosmetic difference; the neutral arm carries 147,564 nodes against 140,652,
  a difference of exactly 1,728 × 4 atoms.
- **The coordination-number measure is fragile where the representation is not.**
  49 % of structures change their measured CN under re-optimisation alone, yet
  the metric moves only **+0.0046** between shipped and control. That bears on
  every re-optimisation in this project, and it means the control arm is a close
  match to shipped, so the comparison it anchors is a fair one.
- **The adjacent-pair metric does not use charge state.** Adding the counter-ions
  that make the complex neutral, with a third of them entering the first
  coordination sphere, moves the metric by less than its screening threshold.

## 6. Limits, stated

- 605 pairs over 130 extractants, **67 % of the published 905**. The price of
  restricting to HNO₃ and to structures passing every gate. No number here may be
  compared to a 905-pair number.
- Nitrate only. Testing the cation-exchange versus solvating axis needs data with
  acidic extractants in it, which this table essentially lacks (97.5 % of the 162
  extractants are neutral solvating agents).
- GFN2/ALPB water throughout. The organic phase is not modelled; ALPB offers no
  kerosene or dodecane, and the partition itself remains unrepresented.
- 4 seeds per arm, legitimate only because `--deterministic` makes runs
  bit-identical.
- One structure exceeded the wall clock and is absent from all three arms
  equally, so no arm is advantaged.

---

**Reproduce**

```bash
MODE=pilot PILOT=40 sbatch automl/slurm/neutralize.sh
sbatch automl/slurm/analysis.sh automl.qc.neutralize_report
NSHARDS=8 sbatch --array=0-7 automl/slurm/neutralize.sh
NSHARDS=2 sbatch --array=0-1 --export=ALL,MODE=charges,NSHARDS=2 automl/slurm/neutralize.sh
python3 -m automl.topo.build_vr_neutral --verify-against-shipped
for a in shipped control neutral; do
  sbatch automl/slurm/analysis.sh automl.topo.build_vr_neutral --arm $a; done
automl/slurm/campaign_driver.sh automl/slurm/campaign4.sh 12 8 30
python3 -m automl.topo.c4_test --n-boot 400
```

**Errors caught before they became results**

| what | how it would have surfaced |
|---|---|
| the `shipped` arm loaded the 956-complex published asset, not the 627 subset | two arms trained on different datasets, compared as one, output entirely normal |
| `--geometry` alone would have misindexed every row | `_cplx` from the shipped asset slicing a different asset: every row on the wrong molecule |
| the HNO₃ filter keyed on the inner-sphere *fill ligand*, not the acid | the wrong 618 structures, silently |
| generated structures carried no per-atom charges | "charge missing" becomes a marker for which structures were generated |
| the pose objective maximised clearance | the counter-ion placed in vacuum at 10.5 Å instead of a contact pocket at 7.0 Å |
| a test helper wrote both arms from the same coordinates | its own connectivity gate passed vacuously |
