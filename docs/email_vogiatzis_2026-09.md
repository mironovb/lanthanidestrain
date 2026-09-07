# Draft — update for Prof. Vogiatzis after our call (September 2026)

Dear Kostas,

Thank you again for the call. I went through your suggestions one by one;
here is what each one showed, plus two new results.

**1. "Use all pairs, not only adjacent ones, as more training data."**
Already in place, in two forms: the graph network's contrast loss trains on
all 18,065 within-block pairs (adjacent ones weighted 3×), and I also
built a model trained directly on all 6,389 within-block pair differences.
That standalone pair model reaches R² +0.19 on adjacent pairs (the old
adjacent-only version got +0.03), but it adds nothing when combined with
the row-level models — its errors duplicate theirs. So the augmentation
works as training signal but does not carry new information for this
target.

**2. "Give the lanthanide identity a larger weight."**
CatBoost supports per-feature weights, so I tested it literally on the
three metal columns (Z, series index, Shannon radius) in the shape model:
×2 is neutral (+0.321 vs +0.319, within seed noise), ×5 and ×10 hurt
(≈ +0.25 and lower). The structural version of your idea is what actually
worked: predicting the block level and the within-block shape with
separate models (+0.268 → +0.318 with the same learner and features),
which forces the second model to spend all its capacity on the metal
dependence.

**3. "If the Vietoris–Rips version equals a plain graph, it is a plain graph."**
Agreed, and now measured: the simplicial and the distance-graph encoders
make the same predictions (r = 0.963 at the pair level), triangles are no
better than edges at matched seeds, and persistence descriptors actively
damage the shape model (78 % of the damage traced to their within-block
variation by a block-mean control). I have dropped the topological
framing and call it a graph network.

**4. MLIPs / second coordination sphere.** Not pursued, for the reason you
gave — it would need its own lanthanide benchmark first.

**Two new things.**

*Decision quality.* R² undersells what the model is for. Ranking held-out
extractants for a given adjacent separation, it reaches a mean Spearman of
0.46 across the 12 positions and a top-quartile hit rate of 0.50 against
0.25 by chance; conformal intervals are calibrated (79 % / 90 % coverage at
nominal 80 / 90 %). That is the practically useful statement for
screening.

*Americium.* Taking your "more data" point where it matters industrially:
the SAFE database holds 1,815 usable Am(III) rows on 110 ligands, 95 % of
them ligands with lanthanide data, giving 312 clean Am/Eu separation
factors under identical conditions on 82 extractants — the SANEX
chemistry (DGA, BTBP, BTPhen, CyMe4 families on both sides). Two tests,
extractants held out throughout: (a) zero-shot, treating Am as a
lanthanide at its radius, the Am/Eu separation is NOT predicted (R² < 0,
rank correlation ≈ 0) — which is the chemically right answer, since
An/Ln selectivity comes from soft-donor covalency, not radius; (b) joint
training with Am rows and a 5f flag: [RESULT PENDING — Am/Eu R² and effect
on the lanthanide target]. Details in the repository
(automl/an_ln/, findings I19–I20).

**One request.** The obvious third Hamiltonian for the contraction
benchmark is Ln-xTB (Zhang, J. Comput. Chem. 2026, 10.1002/jcc.70321).
The parameter files are only in the Wiley supporting information, which
I cannot reach from the cluster. If you have access, could you forward the
SI, or would you be comfortable with me emailing the author?

Best regards,
Bogdan
