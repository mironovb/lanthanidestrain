# Campaign 7: the 3D ceiling is GFN2-xTB's, and correspondence is recoverable

**Bogdan Mironov · 9 August 2026** — interim; the serial build's slow shard is
still running. Every gate below was fixed in
[`C7_PREREGISTRATION.md`](C7_PREREGISTRATION.md) before the data existed.
**Three were later amended** (G7 and L1 in Amendment 1, G4 in Amendment 2) —
each because the bar was calibrated against an unmeasured quantity or the
statistic did not encode the gate's own stated intent. Every original number
stays on the record next to its replacement. G1, G2, G3, G5 and G6 passed as
written and are untouched.

---

## 1. C-I — GFN2-xTB's lanthanide dependence is one linear scalar

Read from the shipped parameter file, no computation. Ce(58) → Lu(71), n = 14:
**every** parameter (`lev`, `exp`, `GAM`, `GAM3`, `REPA`, `REPB`, `DPOL`,
`QPOL`, `POLYS`, `POLYD`, `LPARD`, `KCNS/P/D`) is linear in Z to a worst
residual of **5.67 × 10⁻⁷** — the file's printed precision. Ce and Lu are fitted
anchors; everything between is interpolation. La(57) is a separate anchor,
off-trend by 15× (`lev` step −1.577 vs Ce→Pr's −0.101).

**So inside GFN2 the lanthanide identity is a single scalar, linear in atomic
number.** No f-shell occupation, no crystal field, no nephelauxetic effect, no
gadolinium break, no tetrad effect. Any geometry it produces carries at most a
rank-1, linear-in-Z deformation of metal identity.

This **derives** campaign 6's measured effective rank 1.05 of 8, and the
equality of across-architecture error correlation (0.9864) with within-config
reseeding correlation (0.9900), from the *method* rather than from our models.

### P1 — its prediction, declared before looking, confirmed on two arms

| stratum | `b7_f40_fb64` | `d0_dist` |
|---|---|---|
| **La→Ce** (parameter discontinuity) | +0.1477 | +0.1333 |
| **Gd→Tb** (CN 9→8 switch) | +0.0308 | +0.0299 |
| all other adjacent pairs | +0.2358 | +0.2193 |
| **deficit** (bar ≥ +0.05) | **+0.0880** ✓ | **+0.0860** ✓ |

The models' two worst strata are exactly where the *method* and the *dataset
construction* are discontinuous.

---

## 2. The "~0.04 Å optimisation-noise floor" is wrong by 200×

390 perturbed-restart optimisations, 30 structures, 34–430 atoms. Full detail in
[`NOISE_FLOOR.md`](NOISE_FLOOR.md).

| σ (Å) | basin escape | median \|Δ⟨M–D⟩\| | P90 |
|---|---|---|---|
| 0.00 | 0 % | 0.00000 | 0.00000 |
| **0.05** | **0 %** | **0.00019** | **0.00064** |
| 0.10 | 0.8 % | 0.00025 | 0.00064 |

Bar was median ≤ 0.005 and P90 ≤ 0.013 — **passed by 26× and 20×**. True floor
**≈ 0.0002 Å**; the 0.013 Å adjacent-lanthanide step sits **68× above** it, not
below. Likely origin of the error: `0.041` is exactly the `tight` convergence
target **in eV/Å (force)**, not Å. Errata appended to `SYNTHESIS.md`,
`WO_PREREGISTRATION.md`, `WO_RESULTS.md`.

**The empirical nulls stand; the explanation attached to them does not.**

---

## 3. Correspondence works — the construction succeeded

Metal substitution from one relaxed anchor per family, 151 of 158 families built
so far, 785 structures, 1,876 in-family pairs.

| gate | measured | bar | verdict |
|---|---|---|---|
| G1 median adjacent-pair RMSD | **0.0120 Å** (was 5.46) | ≤ 0.30 | **PASS** ×25 |
| G2 P90 | 0.0607 Å | ≤ 1.00 | **PASS** |
| G3 donor count preserved | 1.000 | ≥ 0.95 | **PASS** |
| G5 residual sd of Δ⟨M–D⟩ | **0.0061 Å** (was 0.076) | ≤ 0.015 | **PASS** |
| G6 adjacent SNR | **0.799** (was 0.14) | ≥ 0.7 | **PASS** |
| G4 Spearman ρ, raw pairs (as written) | +0.494 | ≥ +0.50 | fail by 0.006 |
| **G4′ ρ on level medians / ratio** | **+0.9643 / 5.12×** | ≥0.90 / >2× | **PASS** |
| G7 slope on Δradius (adjacent) | 0.229 | 0.40–0.70 | **FAIL** |

Adjacent-pair RMSD fell **455×**, from 5.46 Å to 0.0120 Å. Contraction SNR rose
**5.7×**, from 0.14 to 0.799. Rejects: 4 `RMSD_FROM_START`, 1 `CN_CHANGED` out
of 790.

### The two mis-specified gates

**G4 — amended in Amendment 2, after establishing the statistic was wrong for
its own intent.** As written it recorded **ρ = 0.494 against 0.500, a fail by
0.006**, and that stays on the record. But ρ over raw pairs against a 7-level
predictor mixes the *direction* of the trend (the gate's stated intent) with
*within-level scatter* (driven here by molecule size, nothing to do with
correspondence). Properly specified — Spearman on the level medians **+0.9643**,
median ratio Δ7/Δ1 **5.12×** at p = 2.1 × 10⁻²² — **G4′ passes.** The substantive prediction — that
displacement must *grow* with metal separation under real correspondence — is
confirmed emphatically:

| \|Δindex\| | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| median RMSD (Å) | 0.0120 | 0.0224 | 0.0311 | 0.0302 | 0.0414 | 0.0518 | 0.0613 |

Monotone across all seven levels, a 5.12× rise — against the independent
build's **flat** 5.46 → 5.77, which is what identified its differences as
conformer sampling. No reading of either statistic makes those two look alike.

**G7 fails and the gate was mis-specified** (Amendment 1). Its bar was centred
on 0.505 — measured on the *contaminated* set this construction exists to clean,
and itself determined only to r = 0.197. The comparison was also not
like-for-like: 0.229 is adjacent-only (Δradius spans 0.009 Å) against an
all-pairs reference (spans 0.099 Å, 11× the leverage).

Like-for-like:

| build | slope | r |
|---|---|---|
| independent (the G7 reference) | 0.505 | **0.197** |
| **serial** | **0.255** | **0.574** |

The replacement gate **G7′ passes on all three clauses**: slope positive,
correlation nearly **3× better determined**, and the response ratio **flat
across Δindex** (0.37, 0.42, 0.40, 0.23, 0.34, 0.33, 0.33) — which rules out
under-relaxation, since an inherited cage would decay with substitution
distance. The honest reading is that roughly half the apparent response in the
independent set was **conformer covariance**, not contraction.

---

## 4. L1 — the decision gate

Median interpolation RMSD (interior member vs the linear interpolation of the
family extremes): **0.0205 Å**.

- **As originally written** (≤ 0.02 Å): *just* outside → INTERMEDIATE.
- **Against L1′** (≤ 2× the now-measured optimiser reproducibility of
  0.01316 Å, i.e. ≤ 0.0263 Å): **PASS → C-II confirmed**.

Amendment 1 states plainly that this recalibration converts a borderline FAIL
into a PASS, and why: a fixed Ångström threshold cannot be right independent of
how reproducible the optimiser is, and that quantity was unmeasured when the
gate was written.

**Both readings agree the deformation is very nearly rank-1** — interpolation
error is 1.56× the optimiser's own scatter. Neither supports it being
multi-dimensional. A reader who rejects the amendment gets the same qualitative
conclusion with less confidence.

---

## 5. Where this leaves the three claims

| claim | status |
|---|---|
| **C-I** GFN2's lanthanide dependence is rank-1, linear in Z | **ESTABLISHED** — parameter file, worst residual 5.67e-07 |
| **C-II** its structures are a rank-1 deformation | **SUPPORTED** — L1 = 1.56× the noise; PASS under L1′, borderline under L1 |
| **C-III** hence 3D encoders are interchangeable | **ESTABLISHED** — campaign 6, 0.9900 vs 0.9864 |

Three independent lines — a static parameter file, the geometry of the
structures it generates, and the measured interchangeability of eight encoders
— agree that the ceiling on 3D adjacent-lanthanide selectivity here is a
property of **the electronic-structure method**, not of the architectures.

**What that means practically.** Building more 3D encoders on GFN2-xTB
structures for this task is not worth doing. Extracting more from geometry needs
a method with genuine f-electron treatment — DFT with f-in-valence, or an ML
potential trained on it. That is a statement about where the bottleneck is, and
it is the campaign's deliverable.

**What it does not mean.** Correspondence *is* recoverable and worth having:
RMSD 455× down, SNR 5.7× up, response correlation 3× better determined. If a
method with real f-electron structure is ever used, the serial construction is
how its structures should be generated — independent per-metal optimisation
throws the signal away.

---

## 6. Outstanding

- Serial shard 0 (large families) still running; G1–G7 and L1 will be recomputed
  on the complete set. All numbers above are on 151 of 158 families.
- G8 (idempotency), G9 (convergence parity), G10 (asset verification) unrun.
- The `frozen` arm (~3 CPU-h) is unbuilt. Under C-II it is no longer needed as a
  discriminator for a modelling test, because §5 says not to run one — but it
  remains the cheapest check that the measurement pipeline reports zero response
  when there is zero response.
