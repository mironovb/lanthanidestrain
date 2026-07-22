# Pre-registration: does topology add to the *strongest* baseline in a stack?

**Written and committed before the contrasts are computed.** No new training —
every out-of-fold vector already exists, so nothing here can be tuned by
re-running.

---

## 1. What is already established

Positive, pre-registered, significant topology results **already exist**:

| test | Δ adjacent-pair R² | 90 % CI |
|---|---|---|
| SNN ensemble vs FCNN | +0.2426 | [+0.181, +0.333] |
| SNN ensemble vs CatBoost | +0.0867 | [+0.025, +0.122] |
| SNN ensemble vs **matched tabular control** | +0.0485 | [+0.009, +0.106] |

What has **not** been shown is topology beating the **repaired** FCNN baseline
(`StandardScaler`, 16 seeds, +0.2206): S0 − repaired = +0.0261 [−0.005, +0.076],
n.s.; S2 = +0.0066, n.s.

**Correction to an earlier overclaim.** The geometry information audit tested the
**89-column tabular 3D summary**, not the raw geometry. The SNN message-passes
over the raw complex and *does* beat the matched control, so it extracts
something the tabular summary destroys. "The signal is not in the geometry" was
too strong; "not in the tabular 3D summary" is what was shown.

---

## 2. The question this asks

The practically meaningful claim is not "topology alone beats the best tabular
model" but **"topology earns a place in the best model."** A stack is how anyone
would actually deploy this, and two models of different families with similar
accuracy usually have complementary errors.

S0 (+0.2382) and the repaired baseline (+0.2206) are close in accuracy and share
no architecture: simplicial message passing over 3D complexes vs an sklearn MLP
on ECFP + RDKit.

---

## 3. Contrasts, fixed now

Blend weight chosen by **nested leave-one-extractant-out** selection (the
machinery already used and verified in `blend_test.py`): for each extractant the
weight is picked on the other 148 only, so no row influences the weight it is
scored under. Metric: adjacent-pair log-SF R². Paired cluster bootstrap over
extractants, 400 draws, seed 0, 90 % interval.

| # | contrast | question |
|---|---|---|
| **1 (primary)** | blend(S0, repaired) − repaired | does topology add to the strongest baseline? |
| **2 (control)** | blend(T0w, repaired) − repaired | does *any* second neural model add? |
| **3 (decisive)** | blend(S0, repaired) − blend(T0w, repaired) | does topology add **specifically**? |

Contrast 3 is the one that matters. Contrast 1 alone could be satisfied by any
model with decorrelated errors — the earlier blend analysis showed exactly that
trap, where an "interior maximum" attributed to topology reproduced for a plain
tabular MLP. T0w is the matched tabular control: same harness, folds, seeds and
objective as S0, encoder removed. If contrast 1 is positive but contrast 3 is
not, the gain is generic ensembling and **must be reported as such**.

**Secondary, descriptive:** the same three with S2 in place of S0; the full
stack (CatBoost + repaired + S0) with a leave-S0-out ablation; overall log D
alongside adjacent-pair for every blend.

**Fixed:** nested weight grid 0–1 step 0.05; 400 draws seed 0; the arms exactly
as they exist on disk; no re-running, no seed selection, no post-hoc arm swaps.

| outcome | consequence |
|---|---|
| contrasts 1 **and** 3 exclude 0 positive | **topology adds to the strongest available model, specifically** — the result the study has been chasing |
| 1 positive, 3 spans 0 | generic ensembling gain, not topology; report as such |
| 1 spans 0 | topology does not add to the repaired baseline even in a stack; report the null |

**Multiplicity, disclosed:** this is the third confirmatory attempt at "topology
beats/adds to the repaired baseline" (after S0 and S2). The 90 % interval is the
headline; the three-test corrected interval is reported beside it every time.

---

## 4. Guards

`control_guard.py --verify` must pass. No training, no `data/` writes; outputs
to `automl/reports/stack_*` only. Existing reports append-only.

---

*Signed off before the contrasts were computed — B. Mironov, 21 July 2026.*
