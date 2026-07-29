# Pre-registration: is it *simplicial*, or merely *3D message passing*?

**Written and committed before any run of either new arm exists.**

---

## 1. The question, and why it is the one left open

This is the study's oldest unanswered question. It is named in three places:

- `PI_EMAIL.md` §9, next step 2 — "a plain distance-based 3D network with no
  simplicial structure, to determine whether *simplicial* or simply *3D message
  passing* is the operative ingredient";
- `SYNTHESIS.md` — "the one boundary the filtration test could not probe", with
  the explicit note that *"other topological representations do not"* is **not
  established**;
- `STACK_RESULTS.md` §8 — "what would settle it" is a **third encoder**.

The published claim is that message passing over a Vietoris–Rips complex carries
adjacent-lanthanide selectivity that fingerprint models do not, worth +0.038 to
+0.045 in the best stack. The filtration replication (3.0 Å / 4.0 Å) showed the
effect is not an artefact of a tuned radius. But **every arm that has ever shown
the effect was simplicial**, and the only other topological representation tried
— a persistence-image CNN — did not reproduce it even after 57 constructions.

Two readings survive, and no existing run separates them:

1. **Topology specifically** does the work.
2. **Any learned encoder over the 3D structure** would do, and the simplicial
   complex is one arbitrary route to it.

## 2. The two arms

Both reuse the shipped Vietoris–Rips asset, so the *neighbourhood definition* is
identical and only the algebra over it differs.

| arm | what it is | what it isolates |
|---|---|---|
| **G0** `--arch snn --no-triangles` | the same `SimplicialNet` with the 2-simplex level removed | simplicial **vs graph**, with literally everything else — node features, readout, head, edges, folds, seeds — held fixed |
| **D0** `--arch dist` | SchNet-style continuous-filter network over the same edges, no boundary maps, no filtration | 3D message passing **vs** the simplicial construction |

**Held fixed by construction, and asserted in `automl/tests/test_encoders.py`:**
same node inputs (xTB partial charge, metal/donor flags, distance to metal), the
same 27-element Z vocabulary, the same metal-centred readout and radial shell
histogram, and the **same embedding width** (9 blocks of `dim`) so the head sees
an identically shaped vector and no capacity difference is smuggled in.

The readout is copied deliberately rather than reinvented. It exists because
pooling over a 300-atom complex averages away exactly the sub-0.1 Å shell shifts
the lanthanide contraction lives in — a control lacking it would lose for a
reason having nothing to do with simplicial structure. Both arms also pass the
permutation- and edge-direction-invariance tests the SNN passes.

**Configuration:** the published S0 settings, unchanged — `--pair-loss-weight
2.0 --select-on adjacent --folds 5 --repeats 3`, the same 16 seeds, ensembled
over all 16, `--deterministic`. Selection of nothing: these are not swept.

## 3. Endpoints, fixed now

Scored exactly as the published contrasts are, under **both** block keys (see
`DUALKEY_PREREGISTRATION.md`), with the multiplicity-respecting cluster
bootstrap, `n_boot=400`, seed 0, 90 % interval.

| # | contrast | question |
|---|---|---|
| **1** | stack(CatBoost, repaired, **G0**) − stack(CatBoost, repaired) | does the graph-only ablation add? |
| **2** | stack(CatBoost, repaired, **D0**) − stack(CatBoost, repaired) | does a non-simplicial 3D encoder add? |
| **3** | stack(CatBoost, repaired, **S0**) − stack(CatBoost, repaired, **D0**) | is the simplicial arm better *in the slot*? |

Contrast 3 is the decisive one. Contrasts 1 and 2 alone could be satisfied by any
decorrelated model — the study has already been caught by exactly that trap once,
when a blend "interior maximum" attributed to topology reproduced *larger* for a
plain tabular MLP (`CONTROL_RESULTS.md` §4.4).

Also reported, descriptive: single-arm adjacent-pair R², error correlation with
the repaired baseline, and each arm's fitted stack weight — the two axes of the
mechanism rule in `SYNTHESIS.md` (accurate **and** decorrelated).

## 4. Decision rule

| outcome | consequence |
|---|---|
| **D0 adds and contrast 3 spans zero** | The claim **broadens**: the effect is "a learned 3D representation", not "a simplicial complex". The paper's contribution becomes the *rule* for what any candidate representation must satisfy, and the VR complex is one instance. This is a **larger** result than the current one, and it must be reported as such rather than defended against. |
| **D0 does not add, S0 does** | The claim is **bounded to simplicial message passing** — sharper and more surprising than the present statement, and the persistence-image null stops being the only evidence for specificity. |
| **G0 adds but D0 does not** | The operative ingredient is the *complex*, not the triangles: a graph over VR edges suffices. Report the 2-simplex level as unnecessary and simplify the model. |
| **neither adds, and S0 still does** | Strongest form of the specificity claim. Also the outcome most in need of a "why", which the mechanism plot must supply rather than assert. |
| **S0 stops adding under this protocol** | Something has moved that should not have. Treat as a bug and stop; the standing precondition (S0 re-ensembles to +0.2382) is checked first for this reason. |

**Every one of these is publishable.** Writing that down now is the point: there
is no outcome here that I need to be true, and the analysis is committed before
the runs so it cannot be reshaped by what comes back.

**Multiplicity, disclosed:** three new contrasts. The topology claim's look count
goes from 10 (after the dual-key re-analysis) to **13**. Uncorrected and 13-look
Bonferroni intervals are reported side by side for every contrast.

## 5. Guards

`control_guard --verify` before and after. New arms write to
`automl/artifacts/topo_encoder/`, a directory no published test reads. No writes
to `data/`. The `--no-triangles` runs carry a `_notri` suffix in their OOF
filename so they cannot overwrite the full model's, and
`control_factorial._matches` rejects both new flags so a new run can never be
swept into a published cell. Existing reports append-only.

---

**Bogdan Mironov · 29 July 2026**
