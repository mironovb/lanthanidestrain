# Where 3D helps, and where it does not: the whole picture

**Bogdan Mironov · 22 July 2026**
One document tying the positive result to the negatives that make it credible.
Every claim below links to a pre-registered test and its own results file.

---

## The result

> **Message passing over a Vietoris–Rips complex** of the 3D structure carries
> adjacent-pair selectivity information that fingerprint models do not, and
> adding it to the best no-topology stack raises adjacent-pair R² from
> **+0.2263 to +0.2672** while *also* improving overall R².

**Replicated across filtration radii** (added 22 July, after this document was
first written): the same gain appears at 3.0 Å (+0.0382) and 4.0 Å (+0.0327) as
at the published 3.5 Å (+0.0381), all three surviving Bonferroni for every one
of the seven confirmatory looks taken. So the result is **not** an artefact of a
tuned radius — which was the live alternative until that test ran.
([`FILT_RESULTS.md`](FILT_RESULTS.md))

| contrast | Δ | corrected resampling + 5-test multiplicity |
|---|---|---|
| remove the encoder from the stack | +0.0381 | **[+0.0136, +0.0613] excludes 0** |
| swap it for a matched no-topology control, same slot | +0.0446 | **[+0.0225, +0.0651] excludes 0** |

Best model: **CatBoost + repaired FCNN + simplicial encoder**, nested weights
0.20 / 0.30 / **0.50** — the topological arm carries the largest share.

**Robustness.** Pre-registered before computing ([`STACK_PREREGISTRATION.md`](STACK_PREREGISTRATION.md));
survives a multiplicity-respecting cluster bootstrap *and* Bonferroni for five
looks, simultaneously; and **both independent 8-seed halves of the ensemble
reproduce it** (+0.0393 and +0.0375), so it is not a lucky ensemble.

---

## What the result is *not*

Four honest limits, each measured rather than assumed:

1. **Not "3D topology helps."** A different topological encoder — the
   persistence-image CNN — **fails to replicate** (−0.0041) and is as redundant
   with fingerprints as the tabular control (corr 0.933 vs 0.928). The claim is
   about *message passing over the Vietoris–Rips complex*, not 3D in general.
2. **Not "topology beats the baseline."** Alone it does not: four attempts
   failed ([`CONTROL_RESULTS.md`](CONTROL_RESULTS.md),
   [`S2_RESULTS.md`](S2_RESULTS.md), [`WO_RESULTS.md`](WO_RESULTS.md)). It earns
   its place by **complementarity**, not superiority.
3. **Not a selectivity signal in the geometry.** The adjacent-lanthanide contrast
   is redundant with the tabular ionic radius and sits below the ~0.04 Å
   optimisation-noise floor — four independent tests, all null.
4. **Not transferable to representations.** Handing the *embedding* rather than
   the *prediction* to a downstream learner fails by construction
   ([`EMBEDDING_RESULTS.md`](EMBEDDING_RESULTS.md)).

---

## The mechanism, and the tight condition it imposes

An arm helps a stack only if it is **both** strong on the scored metric **and**
decorrelated from its partner. Only one arm is both:

| arm | adj R² (strong?) | corr with baseline error | stack gain |
|---|---|---|---|
| **S0 simplicial** | **+0.241** | **+0.896** | **+0.0381** |
| P0 persistence CNN | +0.210 | +0.933 | −0.0041 |
| T0w tabular control | +0.203 | +0.928 | −0.0066 |
| CatBoost | +0.144 | +0.880 | ~0 (adds accuracy instead) |

P0 fails *both* conditions at once; CatBoost is the most decorrelated of all and
still adds nothing to selectivity because it is far too weak. This is why the
replication failed, and it is the most useful thing the positive result has to
say: it tells you *what to look for* in a candidate representation.

---

## Findings worth keeping regardless of the headline

These came out of the negatives and are, arguably, the more portable results.

| finding | evidence |
|---|---|
| **Train the contrast, not the absolute value.** Selectivity is a within-block contrast while every conventional model optimises absolute log D. | +0.030 (tabular), +0.042 (PI-CNN), +0.186 (SNN) |
| **Rank transforms destroy separation-factor signal.** `QuantileTransformer` preserves order and destroys spacing; a separation factor *is* spacing. Trees are immune, which is why it went unnoticed. | one line took the published FCNN from +0.005 to +0.221 |
| **Baselines need the variance control the arms get.** A single-seed baseline spanning 0.11 across seed conventions cannot anchor a claim about a 0.05 effect. | −0.042 / +0.005 / +0.068 across three seed schemes |
| **Model variance and ensembling are substitutes.** Reducing single-model variance cannibalises the ensemble's own gain. | S2: SD −37 %, ensemble *worse*; every lever hurt |
| **Stack predictions, not representations.** Out-of-fold embeddings from *k* folds are *k* coordinate systems. | fold identity predictable from embedding at **100 %** accuracy |
| **A "cluster bootstrap" that collapses duplicates isn't one.** | published intervals 12–29 % too narrow |
| **Test against the champion, not the convenience baseline.** | four separate signals vanished or reversed when the baseline was strengthened |

---

## What I got wrong, and how it was caught

Recorded because the pattern is the point: in every case the *measurement* was
right and my *interpretation* was wrong, and a check caught it.

- Predicted the FCNN's weakness was non-group-aware early stopping — **falsified**
  (−0.0045, worse).
- Predicted the collapsed bootstrap made intervals *conservative* — **falsified**
  (0.71–0.88×, too narrow). I had already written "therefore conservative" into a
  docstring.
- Claimed "the signal is not in the geometry" — **too strong**; it tested the
  89-column tabular summary, not the raw geometry the encoder uses.
- Called the stack result a "topology result" in the PI report — **too broad**;
  corrected to "simplicial encoder" after the replication failed.
- Estimated the pretraining saving at 16 h — **4× overstated**, corrected in the
  pre-registration it appeared in.
- Built the embedding test without checking basis coherence — **invalid by
  construction**, diagnosed after the fact.

Two bugs were caught by verification before they could fake a result: a Kabsch
correspondence error (2.25 Å shell RMSD, physically impossible) and a donor rule
that would have marked every augmented structure.

---

## Recommendation

**Frame the paper around complementarity, not superiority.** The defensible
headline is that a simplicial encoder supplies adjacent-pair information
orthogonal to fingerprints, worth +0.038–0.045 in the best stack, with the
mechanism (strong **and** decorrelated) stated as the condition another
representation would have to meet.

The negatives are not filler — they are what makes it credible. Without the
control, the S2 null, the geometry audit and the failed replication, this is a
fourth-try positive with no story. With them, it is a narrow claim with a
mechanism and a boundary.

**Scope is now bounded on both sides.** Varying the *filtration radius*
(3.0/4.0 A) reproduces the effect; varying the *representation class*
(persistence images) does not. So "message passing over a VR complex" is
supported and "3D topology" is not.

**Next, if the work continues:** a plain distance-based 3D GNN with no simplicial
structure. That varies the representation class in the other direction and would
say whether "simplicial" or "3D message passing" is the operative word -- the
one boundary the filtration test could not probe.

---

*All results reproduce from `automl/reports/*.csv`; `control_guard --verify`
confirms 324 pinned artefacts byte-identical across the entire study.*

---

## Postscript: the last two pre-registered tests (22 July)

| test | outcome | effect on the claim |
|---|---|---|
| **Extended S0** - 48 seeds of the unchanged config | **negative**: +0.0244 [-0.006, +0.073], n.s.; convergence bought -0.0017 | none. The seed curve shows the ensemble was already at its asymptote by 16 seeds, so the premise that motivated the test was wrong. Fourth failed attempt to beat the baseline *alone*. ([`S0X_RESULTS.md`](S0X_RESULTS.md)) |
| **Filtration 3.0 / 4.0 A** | **both add**: +0.0382 and +0.0327, surviving 7-look correction | **broadens** it from "the simplicial encoder at 3.5 A" to "message passing over a VR complex, across radii". ([`FILT_RESULTS.md`](FILT_RESULTS.md)) |

Both were pre-registered with their outcomes' meanings fixed in advance, and both
analyses were committed before their data existed. The extended-S0
pre-registration explicitly predicted its own failure - *"the honest prior is
that this does not clear"* - which is the right thing for a pre-registration to
do when the prior is genuinely poor.

**The mechanism now has a track record rather than a rationale.** Stated after
the PI-CNN failure, it predicted *before* the filtration runs that those variants
would add, because they are both strong and decorrelated. They did. The effect
even declines monotonically with radius (+0.0382 -> +0.0381 -> +0.0327) as the
error correlation rises (0.898 -> 0.897 -> 0.907), which is the same mechanism
operating inside the family.
