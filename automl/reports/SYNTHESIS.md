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

1. **Demonstrated for one family of topological representation, not all.** The
   effect replicates across filtration radii (3.0/4.0 Å), so it is a property of
   the Vietoris–Rips complex rather than of a tuned radius. A persistence-image
   CNN did **not** add (−0.0041). When this was written I called that weak
   evidence, because the images had never been tuned. **They have now been —
   57 constructions** ([`PI_SWEEP_RESULTS.md`](PI_SWEEP_RESULTS.md)) — and the
   verdict is no longer untested. The pre-registered endpoint, scored on
   extractants no selection decision touched and behind a passing positive
   control, returns **+0.0171 [−0.0143, +0.0327]** — positive but spanning zero.
   Resolution, Gaussian spread, birth–death window and the H0/H1 channel split
   are inert; the one axis that matters is feature weighting, where *equal* beats
   weighting-by-persistence (that ordering replicates out of selection at 3.3 σ,
   though the gain over the shipped setting is only **+0.0080**, n.s.).
   **The comparison is now a fair one, and it still fails** — a stronger
   statement than the one it replaces.
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
still adds nothing to selectivity because it is far too weak. This tells you
*what to look for* in a candidate representation.

**Tuning has since been done** (57 constructions, with a held-out confirmatory
test). It moves P0 on the *strength* axis only, and by less than the selection
half suggested — **+0.0080 out of selection, n.s.** — while leaving the
correlation axis where it was. So the mechanism's diagnosis survives its own
strongest test: P0's problem was never that it was untuned. Construction buys a
little accuracy and cannot buy independence from the fingerprint baseline.

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

**Scope.** Varying the *filtration radius* (3.0/4.0 Å) reproduces the effect, so
it is not a tuned-radius artefact. The one alternative representation tried
(persistence images, untuned) did not add — which bounds what has been
*demonstrated*, not what is *possible*. "Message passing over a VR complex
helps" is supported; "other topological representations do not" is **not
established**, because only one was tried and it was never tuned.

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

---

## Erratum, 29 July 2026 — the headline is a *binned-key* result

The adjacent-pair metric blocks by `composition_key`, which **bins** the
condition columns. `strict_composition_key` — every numeric condition matched,
so the only thing varying inside a block is the lanthanide — has been in
`dataset.py` since the beginning, with a comment saying the binned key "turns a
real log D difference into label noise". The metric had never been computed with
it.

It has now been, under a pre-registered decision rule
([`DUALKEY_PREREGISTRATION.md`](DUALKEY_PREREGISTRATION.md), committed before the
contrasts existed). Both published contrasts **add under the binned key and are
not distinguishable from zero under the strict key**:

| contrast | binned | strict |
|---|---|---|
| drop-in | +0.0375 [+0.0173, +0.0510] **adds** | +0.0177 [−0.0023, +0.0367] **n.s.** |
| swap | +0.0438 [+0.0253, +0.0554] **adds** | +0.0177 [−0.0023, +0.0367] **n.s.** |

Every number in this document is therefore a **binned-key** number and should be
read as one. The effect is a weakening rather than a reversal — the strict-key
estimate is positive at P=0.93 and about half the size — and the encoder
comparison against the matched control actually *strengthens* under the strict
key (+0.0376 → +0.0716). Full account, including which key is defensible and why
neither is clean: [`DUALKEY_RESULTS.md`](DUALKEY_RESULTS.md).

---

## Erratum 2, 29 July 2026 — it is not "simplicial", it is 3D message passing

The study's oldest open question — named here, in `PI_EMAIL.md` §9 and in
`STACK_RESULTS.md` §8 — has been answered, and it broadens the claim rather than
supporting it.

Two new encoders, both over the **same** Vietoris–Rips edges, same node inputs,
same readout, same 16 seeds
([`ENCODER_PREREGISTRATION.md`](ENCODER_PREREGISTRATION.md), committed before
either was run):

- **G0** — the same network with the **triangles removed**;
- **D0** — a continuous-filter distance network with **no simplices at all**.

Both earn the stack slot, and the decisive contrast is a null:

| contrast (binned key) | Δ | 13-look Bonferroni |
|---|---|---|
| add G0 (no triangles) | **+0.0343** | [+0.0117, +0.0569] **adds** |
| add D0 (no simplices) | **+0.0284** | [+0.0025, +0.0543] **adds** |
| **S0 vs D0, same slot** | +0.0091 | [−0.0292, +0.0474] **not distinguishable** |

As single arms both *outscore* the published simplicial one: G0 +0.2459, D0
+0.2474, S0 +0.2382.

**So the simplicial structure is not the operative ingredient.** The mechanism
rule stated in this document — an arm earns a slot only if it is both accurate on
the scored metric and decorrelated from its partner — survives intact and is the
transferable contribution. The claim about the *complex* does not. Full account:
[`ENCODER_RESULTS.md`](ENCODER_RESULTS.md).
