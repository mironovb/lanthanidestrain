# Regeneration strategy improvements

## Executive recommendation

The next regeneration engine should rank **joint coordination hypotheses**, not
retry one frozen complex specification.  A hypothesis must contain the donor
set, coordination number, ligand stoichiometry, fill occupancy, placement
template, and provenance.  The engine should generate a small, chemically
diverse candidate set, score it before expensive assembly, build the highest
confidence hypothesis, and use donor-aware QC to decide whether a different
hypothesis is warranted.

A focused first implementation is now in `scripts/build_unique_geometries.py`:

- accepted rows are durable across pilot/full shard-cardinality changes and are
  copied into the current shard report when resumed;
- merge and family status reconcile accepted rows across all shard
  cardinalities, while filtering them to the current build-ID queue;
- two consecutive compact `BORDERLINE_AMBIGUOUS_SHELL` outcomes stop further
  seed/profile retries;
- `prepare-adaptive-regeneration` emits a bounded, versioned adjacent-CN
  hypothesis only for repeated ambiguous nitrate-filled shells when the change
  removes the odd open site that would otherwise require secondary water;
- the alternative preserves donors and ligand count, receives a new build ID,
  records its source build ID and QC diagnosis, and never overwrites the source
  geometry.

A topology-only prototype that replaced transitive pocket grouping with strict
all-pairs donor cliques was also tested and rejected: it fixed a real
multi-pocket failure mode, but incorrectly reduced the established donor sets
of HEDTA, CDTA, a PBI-like ligand, and a phosphoryl phenanthroline.  This
demonstrates that graph distance alone cannot separate a flexible ligand that
wraps around one metal from a ligand containing independent coordination
pockets.  Shipping that rule would lower chemical correctness despite making
pocket logic look cleaner.

The safest high-value next implementation is therefore:

1. introduce a versioned `CoordinationHypothesis` model and retain the current
   family-aware result as the first candidate;
2. add generic alternative-pocket generation without changing the selected
   candidate;
3. rank candidates using ligand-only 3D preorganisation and strain metrics;
4. add donor-fidelity QC and let its failure diagnosis control whether the next
   donor/CN/template hypothesis is tried.

This is algorithmic diversification, not seed or retry-count diversification.

## Evidence from the current checkout

The local `main` branch and `origin/main` were synchronized at commit `3bbdd62`
at the start of this review.  A later fetch found upstream commit `1b8450f`, but
it was not applied because it deletes or replaces a large tracked research-data
surface and conflicts with pre-existing local files; see "Upstream candidate
API review" below.  The current local reports contain 152 still-failed rows
spanning 52 unique ligand SMILES and 14 metals.  The family planner routes them
as:

| Route | Rows |
|---|---:|
| template replan | 128 |
| fixed-template borderline conformer | 13 |
| fixed-template deep conformer | 4 |
| no-XYZ UFF route | 4 |
| accepted-sibling conformer route | 2 |
| no-XYZ ligType route | 1 |
| manual review | 0 |

Among the 152 rows, 11 have no best XYZ.  For rows with a measured
`best_coreCN_max_dist`, 24 lie in `(3.1, 3.3]` angstrom, 28 in `(3.3, 3.6]`, and
78 above `3.6` angstrom.  The remaining problem is therefore not mainly an
execution-coverage problem.  It is dominated by hypotheses that can produce a
structure but do not place the intended coordination sphere correctly.

### Family-run log diagnosis

The later pincer-family artifacts contain 66 planned rows.  The canonical union
of the 8-shard pilot and 16-shard full reports is 39 accepted and 27 still
failed, with no genuinely pending rows.  The old merged summary reported only
33 accepted because six pilot acceptances were skipped by the full run but were
not materialized into its shard reports.  That is a resume/merge accounting bug,
not a chemistry failure, and is fixed by the durable-acceptance changes above.

The chemistry outcomes are more informative:

| Template | Accepted / planned |
|---|---:|
| BTP N3 + nitrate | 25 / 25 |
| PyTri N3 | 2 / 2 |
| BTTP N5 | 1 / 1 |
| BTBP/BTPhen N4 + water | 5 / 5 |
| BTBP/BTPhen N4 + nitrate, CN8 | 6 / 13 |
| BTBP/BTPhen N4 + nitrate, CN9 | 0 / 20 |

The 30 first-attempt failures were all `BORDERLINE_AMBIGUOUS_SHELL`; a second
profile rescued only three, leaving 27 with compact first spheres
(`coreCN_max_dist` 2.448--2.659 angstrom) but gaps of only 0.0008--0.0925
angstrom.  There were no build failures and no long-bond failures in this run.
This is direct evidence against more seeds or larger retry counts.  For one N4
ligand at CN9, five open sites imply two bidentate nitrates plus one secondary
water; the implemented adaptive queue tests the adjacent CN8 hypothesis, which
has four open sites and can be filled by two bidentate nitrates without that
extra ambiguous neighbor.  This remains a hypothesis until geometry QC accepts
it; the original CN9 row is retained unchanged.

## Upstream candidate API review

Unapplied upstream commit `1b8450f` adds `rank_donor_sets()` to
`src/chemistry/coordination.py`.  Read-only review found that it is a reasonable
audit foundation but does **not** yet improve regeneration probability:

- no planner or regeneration worker calls it, so the selected `COORDLIST`,
  build ID, queue, placement, and QC policy remain unchanged;
- it correctly preserves the validated family-aware donor set at rank zero;
- alternatives are center-radius graph neighborhoods and disconnected
  compatibility components, ranked only by denticity and graph distances;
- neighborhoods are not guaranteed to be mutually compatible, and candidates
  above the denticity ceiling are still truncated by atom index;
- molecular automorphisms, flexibility, preorganisation, charge/protonation,
  donor directionality, CN, stoichiometry, and placement are not part of the
  score;
- the two focused tests prove deterministic enumeration on `OCCOCCO`, but not
  chemical correctness on wrapping versus multi-pocket chelators;
- no candidate score, confidence, symmetry class, hypothesis ID, or QC trigger
  is recorded.

The API should therefore remain audit-only until it is extended with the model
and validation gates below.  In particular, automatic use should not be based
only on repeated `FAIL_LONG_BOND`: current QC cannot yet prove that the planned
donor atoms, rather than other nearby heteroatoms, caused the failure.

## Architectural findings

### 1. Donor detection returns one answer too early

`detect_donors()` returns a single `DonorSet`.  Curated recognizers for the
known difficult families are a strong improvement, but the generic fallback
still selects one largest graph pocket and discards alternatives.  This forces
later stages to interpret a donor-selection mistake as a conformer-generation
failure.

The fallback also uses transitive connectivity: if donor A is close to B and B
is close to C, all three can enter one component even when A and C cannot form a
reasonable common chelate.  Conversely, replacing this with a strict pairwise
distance condition is also wrong because valid wrapping and rigid chelators can
have long graph distances between donors.  Pocket detection needs a 2D
candidate generator plus a 3D/preorganisation scorer, not one graph cutoff.

Finally, a donor set above `MAX_DENTICITY` is truncated by atom index.  Atom
order is deterministic but not chemical evidence; any capped selection should
be ranked by pocket quality.

### 2. Donor choice, CN, stoichiometry, and placement are coupled but planned separately

The planner first selects donors, then applies an aqua-ion CN8/CN9 baseline with
specific family exceptions, then derives ligand count mostly by integer mass
balance.  This is understandable and auditable, but a wrong early choice
propagates into all later fields.  A tetradentate ligand at CN8 with two copies,
for example, is a different steric hypothesis from one copy plus fill ligands;
it is not merely a different parameterization of the same build.

The correct unit of ranking is therefore the tuple:

```text
(donor set, donor protonation/charge state, core CN, n_ligs,
 fill identity/occupancy, placement template)
```

Every distinct tuple must have a distinct build ID and retain its parent/source
build ID.  Existing geometry and frozen metadata must remain untouched.

### 3. Current QC cannot diagnose why the hypothesis failed

Regeneration acceptance is based on nearest-`coreCN` distances and the shell
gap.  This is useful for detecting a bad first sphere, but it does not establish
that the atoms near the metal are the **specified donors**, that every ligand
copy contributes its intended denticity, or that fill ligands occupy the
planned sites.  A geometry may pass a distance-shell test through unintended
heteroatoms, while a chemically correct lower-CN structure may fail because the
planner requested too many neighbors.

The attempt ranking consequently prefers the smallest maximum first-shell
distance, regardless of donor identity, ligand coverage, or strain.  QC cannot
currently tell the policy whether to change the donor set, CN, stoichiometry,
or only the placement template.

### 4. Operational family groups are mixed with chemical inference

Family labels currently determine both chemical templates and queue/resource
groups.  This is practical for the current rescue, but the next engine should
keep these concepts separate:

- chemical hypothesis: what complex is being proposed and why;
- placement class: which geometric construction algorithm should be used;
- operational group: how the cluster work is sharded and resourced.

That separation will allow unseen ligands to reuse a placement class without
being hardcoded as a new named chemical family.

## Proposed next-generation engine

### A. Candidate donor-pocket generation

Keep the current family-aware donor set as a high-priority, high-provenance
candidate.  In parallel, generate family-independent alternatives from the
ligand graph:

- enumerate chemically available donor atoms with the existing exclusions;
- detect ring systems, conjugated donor motifs, carbonyl/ether/amine motifs,
  articulation bonds, and repeated/symmetric subgraphs;
- propose connected donor subgraphs rather than only one connected component;
- include maximal pairwise-compatible pockets, but do not assume they are
  automatically better than wrapping pockets;
- collapse automorphism-equivalent donor sets using RDKit atom symmetry ranks;
- retain only a bounded Pareto frontier, for example candidates that are not
  simultaneously worse in denticity, topological compactness, flexibility, and
  donor plausibility.

Candidate records should include `candidate_rank`, `coord_list`, donor types,
generation rule, symmetry class, topological diameter, ring fraction, linker
rotatable bonds, formal charge, and an explicit uncertainty/confidence value.

This generalizes to unseen ligands while allowing curated family rules to act
as priors rather than an exclusive dispatch table.

### B. Ligand-only 3D preorganisation scoring

Before Architector assembly, embed a small deterministic ligand-only conformer
ensemble with RDKit ETKDG and minimize it with MMFF when supported, otherwise
UFF.  This is much cheaper than rebuilding the full metal complex and directly
answers the ambiguity exposed by the rejected topology-only prototype.

For each donor candidate and conformer, compute:

- donor centroid radius and maximum donor-to-centroid distance;
- donor-donor distance matrix and feasible chelate-ring spans;
- ability of donor lone-pair directions to point toward a common metal region;
- conformational energy above the ligand minimum;
- number of torsions that must leave their low-energy state;
- pocket opening/solid-angle proxy and steric occlusion around the centroid;
- planarity or pincer curvature for rigid aromatic systems.

Rank by a robust aggregate over conformers, not the single best extreme.  A
useful score is the fraction of low-energy conformers in which a common metal
position is geometrically feasible, with a strain penalty for the best feasible
conformer.  This estimates preorganisation and flexibility without fabricating
condition-driven geometry.

### C. Joint adaptive CN and stoichiometry hypotheses

Generate a small allowed CN set instead of one unconditional value.  The metal
radius remains the prior: light Ln favors CN9 and heavy Ln favors CN8.  Candidate
CN values should then be filtered/ranked by:

- donor capacity of the selected pocket and number of ligand copies;
- steric footprint and pocket solid angle from the ligand conformers;
- charge balance and plausible inner-sphere fill count;
- known family constraint when it is chemically strong;
- evidence from accepted sibling complexes of the same ligand/scaffold across
  nearby lanthanides.

Do not search a broad CN range.  For ordinary Ln(III) extraction complexes the
automatic alternative should normally be the adjacent CN8/CN9 hypothesis, and
only when donor/steric evidence makes the baseline uncertain.  Family-imposed
CN values remain authoritative unless QC contradicts donor occupancy.

Likewise, rank `(n_ligs, n_fill)` alternatives jointly.  Reject candidates with
negative fill, implausible over-coordination, duplicate giant ligands whose
estimated solid angles cannot fit, or fill counts inconsistent with the chosen
CN.  This replaces integer division plus exceptions with a bounded constrained
optimization problem.

### D. Symmetry-aware placement and template selection

Placement should be selected from the hypothesis geometry, not only denticity.
Represent common CN8/CN9 polyhedra as idealized site graphs and match ligand
donor graphs to compatible subsets of sites.  Score assignments by bite
distance/angle fit, steric overlap, and ligand-copy symmetry.

Use molecular automorphisms to eliminate equivalent donor permutations and
polyhedron automorphisms to eliminate equivalent site assignments.  The saved
compute should be spent on genuinely different placements: cis versus trans
site occupation, alternative pincer curvature, or different ligand-copy
arrangements.  This is symmetry diversification, not increasing a generic
`n_symmetries` count.

For unseen ligands, choose the placement class using features such as pocket
denticity, donor graph shape, rigidity, planarity, and estimated bite geometry.
Named-family templates can override this when validated.

### E. Donor-aware, diagnostic QC

Extend QC with atom-mapped checks derived from the proposed hypothesis:

1. intended donor-to-metal distances by donor element and metal radius;
2. recall: fraction of specified donor atoms in the first sphere;
3. precision: fraction of first-sphere ligand heteroatoms that were specified;
4. per-ligand-copy donor coverage and loss of an entire ligand copy;
5. unintended donor invasion and fill-ligand displacement;
6. observed CN and shell-gap stability over reasonable distance perturbations;
7. bite-distance/bite-angle strain relative to the ligand-only feasible range;
8. ligand-ligand and ligand-fill clashes;
9. connectivity/fragment integrity and metal detachment.

QC should emit a structured diagnosis, not only pass/fail:

| Diagnosis | Next policy action |
|---|---|
| intended donors absent, alternative donors occupy shell | try next donor-pocket hypothesis |
| correct donors present but observed CN is adjacent | try adjacent CN/fill hypothesis |
| correct donors and CN, severe ligand-ligand clash | try different stoichiometry/site assignment |
| correct chemistry, high bite/strain mismatch | try next placement template or donor candidate |
| all intended donors long, no coherent shell | placement/build failure; do not infer donor change from distance alone |
| no XYZ / ligType failure | change placement adapter, not chemistry automatically |

Acceptance for clean 3D features should require both shell quality and donor
fidelity.  Borderline structures should retain all diagnostic scalars for
auditing and later threshold calibration.

### F. QC-guided bounded search policy

For each source row:

1. build the current curated hypothesis;
2. if it passes donor-aware QC, stop;
3. if QC identifies a donor mismatch, advance to the next donor candidate;
4. if QC identifies CN/fill mismatch, advance to the adjacent CN hypothesis;
5. if chemistry is correct but placement is strained, change only the site
   assignment/template;
6. stop at a small hypothesis budget or when the remaining candidates have low
   confidence; route unresolved ambiguity to review.

The budget counts distinct chemical/geometric hypotheses, not random attempts.
Each transition must record the QC diagnosis that authorized it.  This keeps
the search explainable, resumable, and scientifically auditable.

## Integration into the current pipeline

### Data model

Add a frozen, versioned model near `DonorSet` in
`src/chemistry/coordination.py`:

```text
CoordinationHypothesis
  hypothesis_version
  source_build_id
  hypothesis_id / candidate_rank
  coord_list / donor_types / donor_provenance
  core_cn / n_ligs / fill_ligand / n_fill
  placement_class / site_assignment
  graph_features / preorganisation_features
  score / confidence / parent_diagnosis
```

The hypothesis ID should hash all chemistry- and placement-defining fields.
The existing `build_id` semantics should remain the identity of the physical
complex specification; if placement variants share the same physical spec,
give them a separate `placement_id` rather than overloading `build_id`.

### Planning

Extend `prepare-family-regeneration` internally instead of creating a second
pipeline.  Write a candidate-manifest report under the existing family plan
directory, with one-to-many hypotheses per source row.  Continue to write the
current one-row-per-source family queues until offline validation proves the
new ranking.  Do not modify the existing 152-row plan in place.

The first safe deliverable is an audit-only command that produces candidates
and scores but cannot submit or build them.  It should report top-1/top-k donor
sets, confidence margins, symmetry collapses, and which current templates would
change.

### Regeneration worker

Once validated, let the existing regeneration subcommand consume one explicit
hypothesis row.  Preserve child-process isolation, timeout, atomic output,
immediate report flushing, resume behavior, and isolated output roots.  A
failed hypothesis remains visible; advancing to the next hypothesis creates a
new auditable attempt record rather than overwriting it.

### Reports

Add these fields to attempt/still-failed outputs:

- `source_build_id`, `hypothesis_id`, `candidate_rank`, `hypothesis_version`;
- donor provenance and confidence;
- intended-donor recall/precision and per-copy coverage;
- observed CN, unintended donors, fill occupancy, clash/strain metrics;
- structured `qc_diagnosis` and `next_policy_action`.

The merged summary should distinguish:

- built and chemically accepted;
- built but wrong donor set;
- built but wrong CN/fill;
- built but placement/strain failure;
- no structure returned;
- unresolved low-confidence chemistry.

## Validation plan and promotion gates

### Offline, no geometry generation

1. Build a golden donor/CN/template set from the current family regression
   tests plus representative accepted structures for every existing family.
2. Add adversarial generic ligands: repeated pockets, flexible linked pockets,
   rigid wrapping pockets, symmetric pockets, protonated donors, and ligands
   with many non-coordinating heteroatoms.
3. Require 100% preservation of current curated-family top-1 results.
4. Measure top-1 and top-3 donor-set recall, candidate count, symmetry
   deduplication, and confidence calibration on the golden set.
5. Compare predicted donor sets with atom-mapped donor occupancy in already
   accepted XYZ files.  This is read-only analysis and should not relabel data.
6. Require that every proposed template/CN change has explicit provenance and a
   new ID; no silent mutation of frozen specs.

### Controlled pilot after explicit authorization

Select a small stratified pilot from the unresolved rows, not the easiest rows
from one family.  Include multi-pocket, flexible, rigid, CN8, CN9, no-XYZ, and
wrong-donor-shell examples.  Compare the current family-aware hypothesis with
the new ranked policy using equal walltime and an equal number of **distinct
hypotheses**.

Primary endpoint: donor-aware chemically accepted geometry.  Secondary
endpoints: build success, current nearest-coreCN QC, correct donor-set rank,
number of hypotheses evaluated, walltime, and unresolved/manual-review rate.
Do not promote based only on more XYZ files or a lower metal-neighbor distance.

### Required focused tests

- curated family donor selections remain unchanged;
- graph alternatives retain symmetry-equivalent sites only once;
- linked repeated pockets are separated without breaking HEDTA/CDTA wrapping;
- candidate ranking is invariant to equivalent SMILES atom ordering;
- max-denticity handling is score-based, never atom-index truncation;
- CN8/CN9 hypotheses obey metal/family constraints and mass balance;
- donor-aware QC rejects an unintended-heteroatom shell;
- QC diagnosis selects the intended next hypothesis class;
- resume/merge preserves multiple hypotheses and incomplete work;
- original geometries and frozen specs are never overwritten.

## Recommended implementation order

1. **Donor-aware QC first.**  Without a diagnostic target, a better candidate
   generator cannot be evaluated honestly.
2. **Audit-only `CoordinationHypothesis` generation.**  Preserve the current
   selected template while exposing ranked alternatives and uncertainty.
3. **Ligand-only preorganisation scoring.**  Resolve flexible multi-pocket
   versus wrapping-chelator ambiguity before changing top-1 selection.
4. **Joint CN/stoichiometry candidates.**  Add only bounded, adjacent physical
   alternatives and learn from accepted siblings.
5. **Symmetry-aware polyhedral site matching.**  Diversify meaningful
   placements and eliminate redundant ones.
6. **QC-guided bounded policy.**  Enable automatic transitions only after the
   preceding components pass offline and pilot gates.

The key design principle is to move uncertainty forward explicitly.  The
current engine commits early to one donor/template answer and later spends
compute trying to realize it.  The proposed engine retains a small set of
chemically distinct explanations, uses inexpensive structural evidence to rank
them, and spends full-complex generation only when QC provides a reason.

## Implemented validation and operational boundary

The focused recovery, family-planning, adaptive-hypothesis, coordination-family,
schema, ligType, missing-geometry, and stage-E tests all pass: 51 tests total.
`scripts/build_unique_geometries.py` also passes `py_compile` and
`git diff --check`.  The tests use temporary synthetic reports and do not generate
Architector structures.

No file under `data/` was created, modified, renamed, or deleted.  No geometry
job was launched or submitted.  The new command prepares an auditable queue;
running that queue remains a separate, explicit operation.
