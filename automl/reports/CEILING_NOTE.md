# What may honestly be said about the adjacent-pair ceiling (August 2026)

**Bogdan Mironov · 14 August 2026** · supersedes the ≈ +0.53 quoted in
`README.md` §3, `SCIENTIFIC_FINDINGS.md` I7 and the Vogiatzis deck slide 17,
and reconciles them with `CEILING_CLOSED.md` (31 July), which withdrew the
estimator those numbers came from.

## The contradiction being resolved

`ceiling_test.py` was withdrawn on 30 July ("this module's answer was wrong"):
its E2 estimator divided the noise variance of a 203-pair replicated subset by
the variance of the full 905-pair population — two populations whose spreads
differ by 2.6×.  Inside the subset itself the same arithmetic gives +0.173,
*below* the best model's score there — an impossibility for a real ceiling.
`CEILING_CLOSED.md` then showed split-half reliability estimators return
negative values on this data and declared no ceiling quotable.  Despite this,
+0.53 continued to be quoted as ESTABLISHED.  That stops here.

## What is defensible (recomputed 14 Aug, both populations)

Per (extractant, adjacent-pair) identity measured in ≥ 2 *strict* condition
blocks — replication across genuinely independent condition sets:

| population | strict adjacent separations | repeated identities | repeat sd | spread | naive bound 1 − σ²ₙ/σ²ₜ |
|---|---|---|---|---|---|
| legacy `ok_only` | 1,417 | 113 (999 rows) | **0.165** | 0.224 | 0.454 |
| expanded `has3d` | 2,098 | 174 (1,519 rows) | **0.162** | 0.227 | 0.492 |

1. **A separation reproduces to ~0.16 log units** across independent condition
   sets, 4–6× tighter than the 0.72–0.95 scatter of the levels it is computed
   from.  The noise-cancellation-on-differencing mechanism is real and robust
   (it replicates on the expanded population).
2. **The naive R² bound from that subset is ~0.45–0.49 — but it is not a
   ceiling**, for the reasons `CEILING_CLOSED.md` established: the replicated
   identities are a non-random ~8 % of pairs (a cell gets a replicate when
   sources disagree, biasing the noise estimate up); and part of the 0.16 is
   *real condition dependence* of selectivity, not noise (biasing it up
   further), while single-source correlated errors bias it down.  The two
   biases do not cancel by any argument anyone has made.
3. **Therefore: no point ceiling should be quoted.**  The defensible sentence
   is: *current best is +0.313; the label-reproducibility evidence is
   consistent with substantial remaining headroom (indicatively +0.1 to +0.2),
   but the exact ceiling is not identifiable from this dataset because 70 % of
   scored pairs come from cells with no replicate.*
4. Part of the apparent noise is conditions, and a conditions-aware pair model
   can convert it into signal — that is a modelling opportunity, not a wall;
   `cond__extractant_concentration_M` carries the largest measured within-block
   correlation with the pair target (|r| ≈ 0.30–0.36).

## Actions

- README §3 and the deck slide must replace "ceiling ≈ +0.53, we are at 35 %"
  with the sentence in (3).  The g8/f7 figures' 0.53 line should be relabelled
  "indicative, not identifiable" or dropped.
- Any future ceiling claim must come from *designed* replication (same
  composition re-measured across independent labs/DOIs), not from the
  opportunistic replicate structure of SAFE.
