**Subject:** 3D structure does improve neighbouring-lanthanide separation — result, controls, and what it does and doesn't show

Dear Dr. Vogiatzis,

The open question from my last report — whether the 3D structural model was
actually contributing anything, or whether the apparent gain came from the
training objective and a weak baseline — is now settled. **It contributes, and
the effect survives every control I could design.** Below is the argument step by
step, the five figures, and an honest statement of the limits.

---

## 1. The problem, and how it is scored

We predict `log D` for lanthanide(III) extraction complexes. The number that
matters industrially is not overall accuracy but **the separation factor between
*neighbouring* lanthanides** — Nd/Pm, Eu/Gd and so on — where the ionic radii
differ by only about 0.013 Å. That is the hard case and the valuable one.

So the headline metric is the R² for the **predicted difference in `log D`
between two adjacent lanthanides measured under identical conditions with the
same extractant**. An R² of 0 means the model does no better than assuming every
such pair separates by the average amount.

All numbers below use **leave-extractants-out** cross-validation (5 folds ×
3 repeats): an extractant never appears in both training and test, so the model
is always predicting a ligand it has never seen. Dataset: **4,746 measurements,
162 extractants, 14 lanthanides, 953 unique GFN2-xTB complexes.**

---

## 2. First, a correction to the baseline — this reframes the old numbers

My earlier report compared against a fingerprint neural network scoring **+0.005**
on this metric, which made the 3D model look enormously better. That comparison
was unfair, and the reason is worth knowing in its own right.

The shared preprocessing applied a **quantile (rank) transform** to the features.
A rank transform preserves *ordering* but destroys *spacing* — and a separation
factor **is** spacing. Neighbouring lanthanides differ by 0.013 Å in radius, and
a rank transform maps the 14 distinct radii to roughly equal intervals regardless
of how close together they really are. Gradient-boosted trees are unaffected
(they only ever compare values), which is why this went unnoticed for so long. A
neural network asked to predict a *difference* is badly affected.

Changing that one preprocessing step, and nothing else, took the same fingerprint
network from **+0.005 to +0.221**.

**Everything below is measured against that corrected baseline**, not the
original one. It is a much harder comparison, and I think the only honest one.

---

## 3. The result

The best model is a weighted combination of three: gradient-boosted trees
(CatBoost), the scaling-corrected fingerprint network, and the **3D simplicial
network** that passes messages over a Vietoris–Rips complex built from the xTB
geometry. Weights are fitted on held-out extractants only.

**Figure 1 — `pi_fig1_headline.png`**

| model | neighbouring-lanthanide separation R² | overall `log D` R² |
|---|---|---|
| **combination including the 3D model** | **+0.267** | **+0.437** |
| best combination without any 3D model | +0.226 | +0.433 |
| same combination, 3D model swapped for a matched 2D control | +0.221 | +0.429 |
| fingerprint network alone (scaling corrected) | +0.221 | +0.322 |
| CatBoost alone | +0.142 | +0.499 |
| fingerprint network as originally configured | +0.005 | +0.387 |

Two things to note. First, the combination **improves overall accuracy as well**
(+0.433 → +0.437) — earlier 3D models bought selectivity by sacrificing accuracy,
and this one does not. Second, the fitted weights put the **largest share (0.50)
on the 3D model**, with 0.30 on the fingerprint network and 0.20 on CatBoost.

---

## 4. Why I believe it — the control that matters

A combination of two models will often beat either one simply because their
errors differ. So the decisive test is not "does adding a third model help" but
**"does adding *this* model help more than adding an equally good model without
3D information?"**

I built exactly that control: the same architecture, same training objective,
same folds, same random seeds, with **the 3D encoder removed** and only the 2D
descriptors kept. Then I put it in the identical slot in the identical
combination.

**Figure 2 — `pi_fig2_significance.png`**

| test | improvement | 90 % interval |
|---|---|---|
| add the 3D model to the best 2D-only combination | **+0.038** | [+0.019, +0.050] |
| **3D model vs the matched 2D control, same slot** | **+0.045** | [+0.030, +0.054] |

Intervals come from resampling whole extractants (400 draws), which is the
correct unit here because measurements on the same ligand are not independent.

Both intervals exclude zero, and both still exclude zero after a **Bonferroni
penalty for every comparison made in this study** — [+0.017, +0.060] and
[+0.027, +0.062] — and after correcting a flaw I found in our own resampling
code, which had been making all intervals about 12–29 % too narrow.

**This test was written down and committed before the numbers were computed**,
including what each possible outcome would mean. I did the same for every
confirmatory test in this work.

---

## 5. Four independent replications

**Figure 2** also shows these:

- **Split-half.** I split the 16 random seeds into two disjoint sets of 8 and
  rebuilt everything twice. Both halves improve: **+0.039** and **+0.038**.
- **Different neighbourhood radius.** The Vietoris–Rips complex is built by
  connecting atoms within a cutoff. The main result uses 3.5 Å; rebuilding at
  **3.0 Å** and **4.0 Å** gives **+0.038** and **+0.033**, both significant.
  These are genuinely different complexes — the 3.0 Å version has 0.59× the
  triangles, the 4.0 Å version 2.29× — so **3.5 Å is not a tuned parameter.**
- The entire analysis reproduces bit-for-bit on re-running.

---

## 6. Why it works — a rule that made a correct prediction

**Figure 3 — `pi_fig3_mechanism.png`**

A model earns a place in the combination only if it is **both** (a) accurate
enough on the separation metric and (b) making **different errors** from the
models already there. Plotting every candidate on those two axes, only the 3D
simplicial models sit in the useful corner — accurate *and* complementary — and
only they improve the combination. CatBoost, for instance, is the most
complementary of all but far too weak on selectivity (+0.142), so it contributes
overall accuracy instead.

I formulated this rule partway through, and it then **correctly predicted, before
those runs existed**, that the 3.0 Å and 4.0 Å variants would help. It also
predicts the small monotone decline with radius that we observe (+0.038 → +0.038
→ +0.033) as the larger complexes become slightly more redundant with the
fingerprints.

---

## 7. What the model actually predicts

**Figure 4 — `pi_fig4_parity.png`** shows predicted against measured separation
factor for all 905 neighbouring-lanthanide pairs, with and without the 3D model.

An honest observation that belongs in the paper: **both versions compress toward
zero.** Measured separations span roughly ±2 log units; predictions span about
±0.5. The models get *direction* and *ranking* substantially better than
*magnitude*. So this predicts which pairs separate and in which direction, not
yet how much — and we should say so plainly.

**Figure 5 — `pi_fig5_per_extractant.png`** shows the change in squared error for
each held-out extractant. The 3D model improves **45 of 67** extractants with
enough pairs to score, and its wins are larger than its losses. The gain is
spread across the set rather than coming from one fortunate ligand.

---

## 8. What this does *not* show

- **It does not show that 3D beats 2D on its own.** It does not — four separate
  attempts failed. The 3D model's value is *complementarity*: it supplies
  information the fingerprint models lack, and is useful in combination.
- **It does not show that overall accuracy needs 3D.** CatBoost on 2D descriptors
  remains the best single model for overall `log D` (+0.499).
- **The demonstration is for the Vietoris–Rips / simplicial representation.** A
  persistence-image CNN did not reproduce the gain. When I first wrote this I
  flagged that as weak evidence, because our images used the shipped fixed
  settings and had never been tuned. **I have since tuned them — 57
  constructions — and can now say something much more specific.** See §10.

---

## 9. What I suggest next

1. ~~**Tune the persistence-image construction** and re-test.~~ **Done — see
   §10.** It did not change the conclusion, but it produced one result worth
   having in its own right.
2. **A plain distance-based 3D network** with no simplicial structure, to
   determine whether "simplicial" or simply "3D message passing" is the operative
   ingredient.
3. **Multi-conformer sampling** remains the physics lever I could not properly
   test. I tried it with the two re-optimised geometries per complex and it did
   not help, but those are two arbitrary local minima rather than a
   thermally-weighted ensemble; a proper CREST/metadynamics ensemble is the real
   experiment.

For the paper, I suggest framing the contribution as **complementarity rather
than superiority**: a 3D topological representation supplies neighbouring-
lanthanide selectivity information that fingerprints do not carry, worth about
+0.04 R² in the best combined model, with a stated rule for what any candidate
representation must satisfy. The supporting negative results — the preprocessing
correction especially — are a substantial part of the value.

Everything is in the repository, with each test's pre-registration committed
before its data existed. Happy to walk through any of it.

---

## 10. Addendum — the persistence images have now been tuned

Thank you for the guidance on resolution and spread. I ran the benchmark you
suggested, plus three further construction axes, and replicated the result. **57
constructions in total**, all on the same folds and the same readout, so any
difference is attributable to the image rather than the model.

### What matters, and what does not

| construction axis | effect on neighbouring-lanthanide R² |
|---|---|
| **feature weighting** | **the only axis that matters** — see below |
| image resolution (20 → 128) | none detectable |
| Gaussian spread (0.5 → 4 pixels) | none detectable |
| birth–death window | shipped (0, 2.5) is the **best** of four; widening it *hurts* |
| H₀ and H₁ as separate channels vs summed | none detectable |

**The weighting result is the interesting one, and it runs against usual
practice.** Weighting each topological feature by how long it persists — the
standard choice, and what our images used — is *worse* than treating every
feature equally. The effect is monotone in how strongly persistence is
emphasised:

| weighting | neighbouring-lanthanide R² |
|---|---|
| **constant (equal weight)** | **+0.161** |
| arctan | +0.148 |
| linear (persistence) — *what we used* | +0.148 |
| persistence squared | +0.128 |

On this dataset, long-lived topological features are not the informative ones.
That ordering holds up on extractants I had set aside and never used while
choosing settings, so I am confident in it. The size of the gain over our
original setting is more modest there — about **+0.008**, which on its own is
within the noise — so I would describe it as a real effect of uncertain size
rather than a reliable improvement.

### It does not change the conclusion

The improvement makes the persistence-image model *more accurate* but **no less
redundant** with the fingerprint models — its errors still correlate with theirs
at 0.96, where the simplicial model sits at 0.93. Under the rule in §6 a model
needs both. Tested on the held-out extractants, with a check confirming the
comparison had the power to detect an effect of this size, the tuned model adds
**+0.017** to the combination with a 90 % interval of **[−0.014, +0.033]** — a
positive central value that includes zero. I would not call that a contribution.

For scale, the untuned persistence-image model scores **+0.210** against the
simplicial model's **+0.238** (§3), and tuning moves it to about +0.215. The
difference between the two models is not a tuning gap.

**So the honest revision of §8 is:** the persistence-image result was *not*
materially handicapped by its settings. One setting was mildly wrong, fixing it
helps a little, and the gap to the simplicial model is not a tuning gap.

### One caution about this dataset

With 953 complexes, **re-running the identical configuration moves the score by
about 0.005–0.009** — GPU training is not reproducible at fixed seed. That is
larger than most of the differences between constructions, and it caught me out:
my first reading of this sweep showed a +0.018 gain from tuning that turned out
to be **zero** once I replicated both the tuned and the baseline configuration.
Ensembling more seeds does not fix it, because the noise is shared across seeds
within a run.

If your group tunes persistence images on datasets of this size, that is the
number I would most want to have known at the start: **replicate both sides of
any comparison before believing it**, because a search over many constructions
will reliably manufacture an apparent winner.


Best regards,
Bogdan
