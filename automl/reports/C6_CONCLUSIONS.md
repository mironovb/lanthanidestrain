# Campaign 6 — conclusions in plain language

The task: predict how well an extractant separates two **neighbouring**
lanthanides — the hardest and most valuable case, where the two ions differ in
radius by about 0.013 Å. Score is R² on the log separation factor, with whole
extractants held out.

Full detail: [`C6_RESULTS.md`](C6_RESULTS.md) · standing register:
[`SCIENTIFIC_FINDINGS.md`](SCIENTIFIC_FINDINGS.md) · job ledger:
[`C6_LOG.md`](C6_LOG.md).

---

## 1. What got better

**The model improved from +0.2901 to +0.3099** on extractants that took no part
in building it — and it now uses **two components instead of three**.

Two changes did the work:

**a) Change one line in the tabular model.** CatBoost was fitting squared error.
Fitting absolute error instead nearly doubles its selectivity score
(+0.1422 → +0.2487) *and* makes it slightly better at the ordinary log D task
(+0.4987 → +0.5102). Normally these two goals trade against each other. Here
they do not. This is the single biggest improvement in the campaign, and it is
one keyword.

**b) Stop throwing away part of the 3D structure.** The stored molecular graph
contains contacts out to 4.0 Å, but every previous run discarded everything past
3.5 Å. Using all of it, and widening the matching distance filter so the model
can actually resolve the extra contacts, is worth **+0.0139**. Two flags, no new
computation, no new data.

The fingerprint neural network, which was in the published model, is now
**dropped entirely** — the improved CatBoost does its job better.

---

## 2. The most important thing we learned about 3D

**All our 3D models are the same model.**

We built eight of them: two different network architectures, four different
definitions of which atoms count as neighbours (3.5 Å up to 8 Å, plus a
nearest-neighbour rule), four different loss functions, two different feature
sets. On this task they make **the same mistakes on the same molecule pairs**.

The cleanest way to see it: take one single configuration, train it twice with
different random number seeds, and compare. Then take two *completely different*
architectures and compare those.

| comparison | how similar the errors are |
|---|---|
| same model, different random seed | 0.9900 |
| totally different architecture, graph, and loss | 0.9864 |

**Changing the architecture changes the answer less than changing the random
seed does.**

This explains something that had been puzzling for seven campaigns: every time
someone built a new 3D encoder and added it to the model, it contributed
nothing. Not because the encoders were bad — because they were all measuring the
same thing. When you give the combiner three 3D models, it puts 99 % of the
weight on one and ignores the rest.

We tried to break this with eleven different representations, including
persistence images (a completely different mathematical object). None escaped.

**The useful flip side:** the 3D models *are* different from the
fingerprint/tree models (error similarity 0.877 versus 0.918 among the tree
models themselves). Geometry genuinely contributes something the 2D methods
cannot. There is simply **exactly one** such contribution, and we already have
it.

**Practical consequence:** building more 3D encoders for this problem is not
worth doing. Getting more out of the structures would require a genuinely
different kind of representation — and we now have a concrete test for whether a
candidate qualifies, before spending compute on it.

---

## 3. What did not work, and why that is useful

**The idea the campaign was built on was wrong.** The training loss compares
molecules in a way that does not match how the score is computed — 62 % of the
relevant training signal is spent on comparisons the scoring throws away. Fixing
that mismatch made things **worse**, in all 16 attempts. The apparent redundancy
turns out to be what makes the training signal learnable. *A measurable
mismatch between training and evaluation is not automatically worth fixing.*

**Adding chemistry knowledge did not help.** We added hydration energies, f-shell
electron counts and other genuine solvent-extraction physics for each
lanthanide. They passed a screening test designed to catch bad features, looked
strong (+0.085), and then contributed **nothing** on held-out extractants.

**Re-optimising the structures in solvent did not help.** Water- and
octanol-relaxed structures already existed but had never been tried. Compared
fairly on the same molecules: gas-phase +0.1866, water +0.1788, octanol +0.1060.

---

## 4. The methodological result, which may be the most transferable

**Choosing among ~40 candidate models on the same data inverts the ranking.**

We set aside a third of the extractants at the very start and never looked at
them until the end. Ranked on the data used for choosing, our chosen model was
best. On the untouched extractants it came **last of six**, and the
do-nothing control came second.

Related, and quantified: **small-scale screening systematically overstates.**
Testing a configuration with 4 random seeds instead of 12 inflates its apparent
benefit by about 0.04 — and inflates it *most* for whatever happens to look
best. Two results I reported during this campaign lost half their size when
given more seeds.

**In this campaign I stated nine conclusions that later had to be corrected or
withdrawn**, including five proposed explanations that each failed their own
test. All are listed in [`C6_LOG.md`](C6_LOG.md). That record is deliberate: the
improvements in section 1 are believable *because* the things that did not
survive were reported the same way.

---

## 5. Honest limits

- The 3D improvement (+0.0139) is **real but small**. Its held-out confidence
  interval just touches zero.
- The tabular improvement (+0.1066) is **large and solid** — but it is a loss
  function change, not chemistry, and we tested three explanations for *why* it
  works and **falsified all three**. The effect is reliable; the reason is
  unknown.
- The held-out extractants have now been used several times. A further
  confirmation would need data this dataset cannot supply.
- None of this says how good the model *could* be. That ceiling was shown in
  earlier work to be unmeasurable from this data.

---

## 6. If someone continues this

1. **Use MAE for the tabular model.** One keyword, largest effect, improves both
   goals at once.
2. **Use the full 4.0 Å graph with a matching filter.** Two flags, free.
3. **Do not build another 3D encoder** unless it first passes the independence
   test in §2 — it will otherwise duplicate what exists.
4. **Screen with at least 12 seeds**, and keep a partition genuinely untouched.
   Both failures in §4 were avoidable at that price.
5. The open question worth real effort is **why absolute-error fitting helps so
   much** — three explanations are already eliminated, so a fourth would be
   informative either way.
