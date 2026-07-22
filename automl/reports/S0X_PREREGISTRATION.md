# Pre-registration: does a converged S0 ensemble clear the repaired baseline?

**Written and committed before any of the 32 extra seeds finished.**

---

## 1. The idea, and why it is not S2 again

S2 changed the winning configuration and every lever hurt — the leave-one-out
ablation is unambiguous (drop conformers +0.184, drop block-centring +0.189,
drop pretraining +0.174, against full S2's +0.163).

**This changes nothing about the model.** It runs 32 more seeds of the
*unchanged* S0 configuration (`--arch snn --pair-loss-weight 2.0 --select-on
adjacent`, dim 96, layers 3, 5×3 folds). That is not a new arm; it is a better
estimate of the ensemble S0 already defines.

The motivation is a measured fact: S0's ensemble sits **+0.060 above its own
per-seed mean** (+0.178 → +0.238) and that curve had not visibly flattened at
16 seeds. Seed-ensemble means converge as ~1/√n, so 48 seeds should land above
16 — and unlike S2's levers, averaging more replicates of the same model cannot
*hurt* the expected ensemble.

S0 = +0.2382 (16 seeds); repaired baseline = +0.2206; gap +0.0261
[−0.005, +0.076], n.s. A converged ensemble is the cleanest remaining shot at
the claim.

---

## 2. Arms and endpoint

**S0X** = all available seeds of the unchanged S0 config: the published 16
(`topo_adj_seeds`, `topo_adjacent`) **plus** the 32 in `topo_s0_extra` = up to
48, all ensembled, no subset selection.

Output went to a separate directory deliberately, so the published 16-seed S0
is untouched and `control_factorial` keeps reproducing **+0.2382** exactly —
verified as a precondition of reporting.

**Primary endpoint.** S0X − repaired baseline
(`oof_fcnn_std_scaler_ens16.parquet`, +0.2206), adjacent-pair log-SF R², paired
cluster bootstrap over extractants, 400 draws, seed 0, 90 % interval, via
`control_factorial.paired_adjacent_fast`.

**Secondary, descriptive.** S0X − S0(16) — how much did convergence buy?
The seed-count curve (ensemble R² at n = 4, 8, 16, 24, 32, 48) to show whether
it has flattened; overall log D alongside.

**Fixed now:** every completed seed is included, none dropped or selected; 400
draws seed 0; no configuration change; no re-running a seed after seeing it.

| outcome | consequence |
|---|---|
| S0X − repaired excludes 0, positive | **topology beats the strongest baseline** — the pre-registered claim, achieved by convergence rather than by a new lever |
| spans 0, S0X > S0 | convergence helped but not enough; report the improved point estimate and the null together |
| spans 0, S0X ≈ S0 | the ensemble had already converged at 16 seeds; report it |

---

## 3. Multiplicity — this is the fourth attempt

Attempts at "topology beats/adds to the repaired baseline": S0 (+0.0261, n.s.),
S2 (+0.0066, n.s.), the stack test (running), and this. **The 90 % interval is
the headline; the four-test corrected interval is reported beside it every
time.** Four shots at one claim is exactly the situation where a nominal 90 %
interval overstates, and a reader must not have to work that out.

Stated plainly: with SE ≈ 0.025 the estimate must reach ≈ +0.041 uncorrected and
≈ +0.052 corrected for four tests, against the current +0.026. Convergence
plausibly buys a few thousandths, not two centi-units. **The honest prior is
that this does not clear**, and it is run because it is cheap, it cannot make
the model worse, and it is the last lever that requires no new assumptions.

---

## 4. Guards

`control_guard --verify` must pass, and S0(16) must still re-ensemble to
+0.2382, before any S0X number is reported. New outputs in
`automl/artifacts/topo_s0_extra/` and `automl/reports/s0x_*` only.

---

*Signed off before any extra seed completed — B. Mironov, 22 July 2026.*
