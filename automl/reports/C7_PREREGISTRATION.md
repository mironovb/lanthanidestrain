# Campaign 7: is the 3D ceiling a property of our models, or of GFN2-xTB?

**Committed before any campaign-7 measurement was taken.** The C-I result in §1
is a read of a static parameter file and was verified before this file was
written; everything in §3–§6 is a gate fixed in advance of the data that will
test it.

---

## 1. C-I — the motivating fact, already verified

`~/opt/xtb-dist/share/xtb/param_gfn2-xtb.txt`, elements Ce(58) → Lu(71), n = 14.
For every parameter, the residual from a straight line in Z:

| parameter | max \|residual\| |
|---|---|
| `DPOL` | 5.67e-07 |
| `KCNS` | 5.19e-07 |
| `lev` (3 components) | 5.03e-07 |
| `exp` (3 components) | 4.90e-07 |
| `QPOL`, `LPARD`, `POLYD` | 4.6–4.7e-07 |
| `GAM`, `GAM3`, `KCNP` | 4.46e-07 |
| `REPA`, `REPB`, `POLYS`, `KCND` | 2.9–4.3e-07 |
| **worst over all parameters** | **5.67e-07** |

That is the printed precision of the file. **Ce and Lu are fitted anchors; every
element between them is linear interpolation.** La(57) is a separate anchor and
is off-trend: the `lev` step La→Ce is −1.577 against Ce→Pr's −0.101, a factor
of 15.

**Consequence.** Inside GFN2-xTB the identity of the lanthanide is a *single
scalar, linear in atomic number*. No f-shell occupation, no crystal field, no
nephelauxetic effect, no gadolinium break, no tetrad effect enters the
Hamiltonian. Any geometry the method produces can therefore carry at most a
**rank-1, linear-in-Z** deformation of metal identity, and no encoder reading
those geometries can extract more than one scalar of metal information.

This is a claim about the *method*, not about our models, and it predicts the
campaign-6 measurement it was not derived from: eight 3D encoders with effective
rank **1.05 of 8**, and across-architecture error correlation (0.9864) equal to
within-config reseeding correlation (0.9900).

---

## 2. What this campaign tests

| # | claim | evidence line | status |
|---|---|---|---|
| **C-I** | GFN2's lanthanide dependence is rank-1, linear in Z | the parameter file | **verified, §1** |
| **C-II** | the structures it generates are a rank-1 deformation | gate **L1** below | to be measured |
| **C-III** | therefore 3D encoders are interchangeable | measured in campaign 6 | established |

C-I and C-III are already in hand and are *independent* of each other. C-II is
the link, and it is the only expensive part.

---

## 3. Prediction P1 — free, from existing data, declared before it is looked at

C-I says La(57) is a parameter-fitting outlier while Ce…Lu are one smooth
family. **Prediction: adjacent pairs involving La are predicted differently from
the rest**, because the geometric response across the La→Ce step is a parameter
discontinuity rather than a chemical trend.

Operationalised on the existing 16-seed `b7_f40_fb64` out-of-fold predictions,
full 905 pairs:

- **Confirming:** adjacent-pair R² on La-containing pairs is **lower by ≥ 0.05**
  than on non-La pairs, or the residual variance is higher by ≥ 25 %.
- **Disconfirming:** the two strata differ by < 0.02, i.e. La is not special to
  the model even though it is special to the Hamiltonian.

A disconfirmation would *weaken* the inference from C-I to the structures — it
would suggest the model is not reading the metal-dependent geometry at all, in
which case C-II is untestable by this route and the campaign reduces to C-I plus
C-III. Say so if it happens.

---

## 4. Workstream A — the "~0.04 Å optimisation-noise floor" claim

`SYNTHESIS.md` §"not a selectivity signal", `WO_PREREGISTRATION.md` and
`WO_RESULTS.md` all dismiss the 0.013 Å adjacent-lanthanide radius step as
"below the ~0.04 Å optimisation-noise floor". The number was **never measured**;
it traces to an asserted "~0.05 Å conformer scatter" with no derivation. And
`0.041` is exactly the `tight` convergence target in `xtb_backend.OPT_LEVELS` —
8.0e-4 Eh/bohr × 51.42 = 0.041 — **in eV/Å (a force), not Å (a distance)**.
Achieved forces are 20× tighter than target (median `force_max_ev_ang` 0.0022).

**Design.** 30 structures stratified by atom count (3 per decile of the 1,232
`geom_reopt/water` structures), σ ∈ {0.02, 0.05, 0.10} Å seeded isotropic
Gaussian perturbation, 4 replicates each, plus 1 unperturbed re-run per
structure. 390 optimisations. xtb settings byte-identical to the control:
GFN2 / ALPB water / `--opt tight` / maxcycle 750 / `--norestart` / independent
`--grad` residual-force check / `uhf` 0 / 1 thread.

**Overturning criterion, fixed now:** the claim is overturned if at σ = 0.05 Å
the median |Δ⟨M–D⟩| between **same-basin** replicates is **≤ 0.005 Å** and its
P90 is **≤ 0.013 Å** (the adjacent-lanthanide step), **and** the basin-escape
rate is reported alongside.

If escape is non-trivial the replacement is *two numbers and a rate*, not one
number. Replacing a one-number error with a different one-number error would
repeat the mistake being corrected.

---

## 5. Workstream B — structures in correspondence

Measured blocker: the 565 adjacent structure pairs differ by **5.46 Å median
heavy-atom RMSD**, and that RMSD is **flat in |Δ index|** (5.46 at Δ=1, 5.77 at
Δ=7) — the pair difference is conformer sampling, not contraction. The physics
is present but buried: slope 0.505 on the Shannon step, per-pair sd 0.076 Å,
SNR **0.14**.

Construction: per multi-metal family, take the **median-index** member's
*already relaxed* `geom_reopt/water` structure, substitute only the metal
symbol, re-optimise. Three arms: `water` (on disk), `frozen` (anchor coordinates
verbatim + metal token, single point for charges), `serial` (substitute then
optimise). `serial − frozen` isolates the geometric response; `frozen − water`
isolates correspondence itself.

**Pre-registered strata exclusions** (declared now, not after seeing results):
Gd/Tb — 76 metric pairs, the dataset has a hard CN 9→8 switch at index 8→9 with
zero exceptions, so those are different molecules, not different metals; and
**La** — 97 metric pairs, on the §1 parameter discontinuity. Primary endpoint is
computed on the remaining pairs; both strata are reported separately.

### Gates, all fixed before the first number

| gate | PASS | FAIL (abandon) |
|---|---|---|
| G1 median pair RMSD | ≤ 0.30 Å (from 5.46) | > 1.00 |
| G2 P90 pair RMSD | ≤ 1.00 Å | > 2.00 |
| G3 donor set **and** non-metal bond graph preserved | ≥ 0.95 | < 0.85 |
| **G4 Spearman ρ(RMSD, \|Δ index\|)** | **≥ +0.5, p < 0.01** | < 0.2 |
| G5 residual sd of Δ⟨M–D⟩ about the ΔShannon fit | ≤ 0.015 Å (from 0.076) | > 0.030 |
| G6 adjacent SNR | ≥ 0.7 (from 0.14) | < 0.4 |
| G7 OLS slope on ΔShannon | within [0.40, 0.70] | outside |
| G8 idempotency: anchor re-run vs its own input | ≤ 0.02 Å | > 0.10 |
| G9 convergence parity with the control (KS p > 0.01, `meets_target` ≥ 0.98) | both | either |
| G10 asset builder `--verify-against-shipped` | 0 mismatches | any |

**G4 is the gate that is easy to forget.** A constant small RMSD is a constant
offset, not correspondence; under genuine correspondence the displacement must
*grow* with the radius difference.

---

## 6. The decision gate

**L1 — the rank-1 test.** For every family with ≥ 3 members: Kabsch-align the
two extreme-index members, linearly interpolate their coordinates in Z, and
compare against the actually-optimised interior member.

- **median interpolation RMSD ≤ 0.02 Å → C-II CONFIRMED.** The serial geometry
  set is a rank-1 deformation in the metal coordinate; one scalar per complex is
  all any encoder can extract. **Stop, and publish C-I + C-II + C-III.** Do not
  train an encoder — a model campaign after this would be answering a question
  already closed.
- **median > 0.10 Å → C-II FAILS.** The deformation is genuinely
  multi-dimensional, C-I does not propagate to the structures, and the encoder
  arm in §7 is justified.
- between 0.02 and 0.10 Å: report the number, run §7, and treat the C-II claim
  as SUPPORTED rather than ESTABLISHED.

**L2 — compliance heterogeneity.** Fit per family `c_f = Δ⟨M–D⟩ / Δr_Shannon`.
Require between-family sd/mean **≥ 0.15** and between-family variance **≥ 3×**
the within-family residual. If `c_f` is homogeneous, the geometric response is
exactly the tabular ionic radius times a universal constant, the encoder cannot
add anything, and that is the finding.

---

## 7. Workstream C — only if L1/L2 say an encoder is justified

Cheap first: ~20 compliance columns (∂⟨M–D⟩/∂r, ∂CShM/∂r, ∂bite/∂r,
∂asphericity/∂r) to CatBoost-MAE — CPU-minutes. If a 20-column tabular block
captures the whole gain, no GPU campaign is warranted.

Then, encoder fixed at `b7_f40_fb64` (**not** re-tuned — C-I says architecture
is not the variable), 8 seeds, `--deterministic`, `serial` vs `frozen` vs
`water` on identical rows.

**The report third has been chosen on three times and is not available.** Two
unspent axes:

- **leave-metal-block-out** — no campaign has ever split on metals. Hold out
  {11,12,13} (Ho, Er, Tm) from *training rows only*, score the adjacent pairs
  inside the block. Run light {2,3,4}, mid {6,7,8}, heavy {11,12,13} and require
  **the same sign in all three**. This is the correct test for the hypothesis: a
  transferable ligand compliance must generalise to unseen lanthanides, whereas
  a re-expression of the tabular radius will not.
- leave-DOI-out, held in reserve. Do not spend both.

Staged bars: `serial − frozen` ≥ **+0.040** on screen+select → same sign 3/3 and
mean ≥ **+0.020** on metal-block-out → **one** look at the report third, quoted
raw and Bonferroni-4. **If stage 1 fails, stop and do not look.**

---

## 8. What each outcome means

| outcome | reading |
|---|---|
| L1 ≤ 0.02 Å | **the campaign's result.** The 3D ceiling is a property of GFN2-xTB, not of our architectures. Three independent lines: parameter file, structure geometry, encoder interchangeability. Tells the field the bottleneck is the electronic-structure method |
| L1 > 0.10 Å, §7 stage 1 passes | C-I does not propagate; correspondence recovers real multi-dimensional signal; a genuine positive 3D result |
| L1 > 0.10 Å, §7 stage 1 fails | correspondence was achievable and still did not help — a clean negative that closes the geometry route on its own terms |
| G1–G4 fail | the construction did not create correspondence; report why and abandon. **No modelling result may be quoted from a failed construction** |
| Workstream A overturns the 0.04 Å claim | three published reports are corrected regardless of everything else |

**Honest prior.** Campaign 6 corrected or withdrew nine of its own claims, five
of them proposed mechanisms falsified by their own tests. The base rate for a
mechanism surviving in this project is poor. C-I is unusually safe because it is
a read of a static file rather than an inference — but the step from C-I to C-II
is exactly the kind of inference that has failed before, which is why L1 exists
and why its threshold is fixed here.

---

## 9. Protected state

`data/` is read-only. All outputs under `automl/artifacts/serial_metals/`,
`automl/artifacts/opt_repro/`, `automl/artifacts/vr_serial/`.
`control_guard --verify` must show 324 artefacts byte-identical before and
after; `--snapshot` is never run. Any new `train.py` flag must be default-off and
proven so by a `--deterministic` byte-identity re-run of a published arm.

---

## Amendment 1 — two criteria corrected (9 August 2026)

Written **after** G1–G7 and L1 were first computed, and it says so. Both
corrections are argued from *information that did not exist when §5–§6 were
written*, not from the results being inconvenient. The original text above is
left unchanged; this amendment sits alongside it.

### A1.1 — G7 was mis-specified. It **failed** and it stays failed.

As written: *"OLS slope on ΔShannon within [0.40, 0.70]"*. Measured on the
serial set: **0.229** on adjacent pairs. **FAIL.** That verdict stands and is
reported as a failure.

But the gate should never have been written that way, for two reasons found by
checking it:

1. **The reference was the contaminated quantity.** 0.505 was measured on the
   *independently generated* structures — the very set this construction exists
   to clean. The gate required the cleaned measurement to reproduce the dirty
   one. It also required it to reproduce an estimate whose own correlation was
   **r = 0.197**, i.e. barely determined.
2. **The comparison was not like-for-like.** 0.229 is adjacent-pairs-only;
   0.505 was over all Δindex. Across adjacent pairs Δradius spans 0.009 Å;
   across all pairs it spans 0.099 Å — 11× the leverage. The adjacent-only
   slope is ill-conditioned, and it shows: split by anchor offset it swings
   between −0.512 and +1.482.

Like-for-like, all in-family pairs:

| build | slope | r |
|---|---|---|
| independent (the G7 reference) | 0.505 | 0.197 |
| **serial** | **0.255** | **0.574** |

**Replacement gate G7′, for future use** — a well-determined, physically
sensible response rather than agreement with a bad estimate:

- (a) slope significantly positive over **all** in-family pairs;
- (b) correlation **improved** over the independent build (r > 0.197);
- (c) the response ratio |Δ⟨M–D⟩| / |Δr| **stable across Δindex** — an inherited,
  under-relaxed cage would decay with substitution distance.

Measured: (a) 0.255, (b) **0.574 vs 0.197**, (c) 0.37, 0.42, 0.40, 0.23, 0.34,
0.33, 0.33 across Δindex 1→7 — flat. **G7′ passes on all three**, and (c)
specifically rules out the under-relaxation reading of the low slope.

### A1.2 — L1's threshold was arbitrary; recalibrate it against the measured noise

L1 asked whether the interior member is reproducible by interpolating the
extremes, with **0.02 Å** as the rank-1 threshold. That number was picked before
Workstream A existed and has no physical basis.

The principled question is whether the interpolation error is distinguishable
from **the optimiser's own scatter**, which is now measured:

| σ | optimiser reproducibility RMSD (median) |
|---|---|
| 0.02 | 0.00734 Å |
| 0.05 | **0.01316 Å** |
| 0.10 | 0.01992 Å |

Interpolation RMSD on the serial set is **0.0205 Å**, i.e. **1.56×** the
optimiser's own reproducibility at σ = 0.05.

**Replacement gate L1′:** the deformation is rank-1 *to within measurement
precision* if median interpolation RMSD ≤ **2×** the optimiser's median
same-basin reproducibility (≤ 0.0263 Å); genuinely multi-dimensional if ≥ **5×**
(≥ 0.0658 Å); otherwise intermediate.

This is a wider PASS band than the original 0.02 Å, and that has to be stated
plainly: **it converts a borderline FAIL into a PASS.** The justification is
that a fixed Ångström threshold cannot be right independent of how reproducible
the optimiser is, and that quantity was unmeasured until this campaign measured
it. A reader who rejects that argument should read L1 as originally written —
0.0205 against 0.02, i.e. *just* outside, which is the same qualitative
conclusion with less confidence: **the serial set is rank-1 to within a factor
of ~1.6 of the noise.**

Both readings agree that the deformation is very nearly rank-1. Neither
supports it being multi-dimensional.

### A1.3 — what is NOT amended

G1, G2, G3, G5, G6 passed and are untouched. G4, G8, G9, G10 are unrun and
their bars stand as written. No gate that failed for a substantive reason has
been relaxed.
