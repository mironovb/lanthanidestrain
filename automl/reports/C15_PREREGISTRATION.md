# C15 — a night sized by power, not by habit

Written before any C15 cell runs.

## Why the design changed

Three positives dissolved in one session, all for the same reason: **8 paired
seeds cannot resolve the effects being claimed.**

| claim | at n = 4–8 | properly tested |
|---|---|---|
| C8 geometry-only | +0.0333, p = 0.020 | +0.0041 on the full asset; scale-free −0.0036 |
| C10 `q0.7` | +0.0168, 4/4 seeds | +0.0019 at n = 8 |
| C12 `plw4` | +0.0270, p = 0.021 | **+0.0145, p = 0.079** pooled over 14 |

Per-seed sd of a paired Δ is **0.0285**. So:

| to detect | paired seeds needed (80 % power) |
|---|---|
| +0.040 | 4 |
| +0.030 | 8 |
| **+0.020** | **16** |
| **+0.0145** | **31** |
| +0.010 | 64 |

Every n = 8 result in this project resolves ~0.028 at best. The remaining
plausible effects are 0.010–0.020. **That is the whole reason the last three
positives failed, and it is fixable by spending seeds instead of configurations.**

## The design

**One question, enough seeds to answer it.** Not a 19-config sweep — those are
what generate the winner's curse in the first place.

| arm | seeds | rationale |
|---|---|---|
| `--pair-loss-weight 2.0` (published) | 32 | control |
| `--pair-loss-weight 3.0` | 32 | between published and the C10 optimum |
| `--pair-loss-weight 4.0` | 32 | the C10 optimum; 11/14 up so far |

96 cells. Powered to resolve **+0.0145 at 80 %**, which is the effect actually
in question. Seeds 201–... , disjoint from every seed used in C10, C12 and C14,
so nothing here is contaminated by the runs that chose these values.

## Declared in advance

- **Endpoint:** paired Δ on `sel_adj_logSF_r2`, `plw4 − plw2` over 32 seeds.
- **Real:** p < 0.05 paired **and** the scale-free (Pearson²) contrast agrees in
  sign. Both, because the first distinguishes existence and the second
  distinguishes information from calibration — C8 passed one and failed the
  other.
- **Null:** anything else, including a positive point estimate that misses
  significance. Reported as a null.
- `plw3` is *not* a second test of the same hypothesis: it exists to say whether
  the response is monotone, and is quoted descriptively.

## What a null would mean

That the contrast weight is already at its optimum and the 11/14 sign count was
chance. That is a publishable, useful statement: it would close the objective
axis the way the geometry axis has been closed, and the honest recommendation
would become "the remaining 0.35 of R² is not in the objective either".

I cannot guarantee a positive result. What this design guarantees is that
whichever way it lands, **the answer will be trustworthy at the scale of effect
that is actually plausible** — which the last three campaigns were not.
