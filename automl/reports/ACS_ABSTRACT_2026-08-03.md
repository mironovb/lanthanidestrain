# ACS Fall abstract, rewritten against measured results

**Title**

What three-dimensional structure adds to machine learning prediction of lanthanide separation

*Alternatives, same content, different emphasis:*
- Learned 3D structure of lanthanide complexes improves separation prediction where topological descriptors do not
- Testing whether 3D coordination geometry improves prediction of rare earth separation

---

**Abstract**

Separating individual lanthanides remains a bottleneck in rare earth processing
because the ions are chemically alike across the series. Machine learning models
can predict distribution ratios (log D) for an extractant and lanthanide pair,
but published models use two dimensional ligand representations that encode
connectivity and leave out the coordination geometry of the metal complex. We
asked whether adding the 3D structure helps, and scored it on the quantity that
decides a separation rather than on overall accuracy.

The target is the R2 of the predicted difference in log D between two
neighbouring lanthanides measured with the same extractant under the same
conditions. Zero means no better than assuming every adjacent pair separates by
the series average. The dataset is 4,746 measurements covering 162 extractants,
14 lanthanides and 110 papers, with one Architector complex per metal, ligand and
inner sphere anion, optimized at GFN2-xTB. Cross validation leaves whole
extractants out, so no ligand appears in both training and test.

A neural encoder over the 3D complex supplies adjacent lanthanide information
that fingerprint models do not have. Added to the strongest model without 3D
input it is worth +0.035 R2 (interval +0.017 to +0.065), and the combination of
gradient boosting, a fingerprint network and the 3D encoder reaches 0.267 against
0.226 without it. A control makes this structural rather than an ensembling
artifact: a matched model with the same architecture, objective, folds and seeds
and only the 3D encoder removed is worth +0.006 (-0.017 to +0.018), and the 3D
arm beats that control directly by +0.030. Each contrast and its decision rule
was fixed before the numbers existed, and the bootstrap resamples whole
extractants.

The effect is not topological. We built the complexes as Vietoris-Rips
filtrations, but deleting the triangles costs nothing and a plain distance
network with no simplicial structure earns the same slot, the two differing by
+0.009 with an interval spanning zero. Persistence images add nothing either,
their errors correlating with the fingerprint model about as strongly as a 2D
control does. What helps is a learned 3D representation, not the simplicial
machinery.

Two limits belong with the result rather than in an appendix. The size depends on
how strictly identical conditions are defined: under strict blocking the estimate
is +0.018 and no longer clear of zero. And we cannot say what fraction of the
attainable signal this is, because the noise floor of the metric is not
identifiable from this data, by any of three methods we tried. We also report
what failed, including angular and polyhedral descriptors delivered through three
separate routes and a head trained to predict the adjacent pair difference
directly, since the pattern in those failures is informative about what this
metric can and cannot use.

---

**Short version (about 250 words, for a tighter character limit)**

Separating individual lanthanides remains a bottleneck in rare earth processing
because the ions are chemically alike across the series. Machine learning models
predict distribution ratios (log D) for an extractant and lanthanide pair, but
they use two dimensional ligand representations that encode connectivity and omit
the coordination geometry of the metal complex. We tested whether adding the 3D
structure helps, scored on the quantity that decides a separation: the R2 of the
predicted difference in log D between neighbouring lanthanides measured with the
same extractant under the same conditions. The dataset is 4,746 measurements over
162 extractants, 14 lanthanides and 110 papers, one GFN2-xTB optimized complex
per metal, ligand and inner sphere anion, with whole extractants held out.

A neural encoder over the 3D complex adds +0.035 R2 (interval +0.017 to +0.065)
to the strongest model without 3D input, and the full combination reaches 0.267
against 0.226. A matched control with the same architecture and only the 3D
encoder removed adds +0.006 (-0.017 to +0.018), so the gain is structural rather
than an ensembling artifact. The effect is not topological: deleting the
triangles from the Vietoris-Rips complexes costs nothing and a plain distance
network earns the same slot. Under strict condition blocking the estimate falls
to +0.018 and is no longer clear of zero, and the noise floor of the metric is
not identifiable from this data. We report the negative results alongside, since
the pattern in them constrains what this metric can use.

---

## What changed from the earlier draft, and why

Every number below is read from the result CSVs.

| earlier draft | measured | source |
|---|---|---|
| "1,202 measurements, 109 extractants" | **4,746 measurements, 162 extractants**, 110 papers | row table |
| "topological descriptors improve log D predictions" | 3D helps; **topology specifically does not** | `encoder_test.csv` |
| "simplicial networks provide the largest gains" | simplicial vs plain distance net = **+0.009, interval spans zero** | `encoder_test.csv` |
| persistence images presented as a co-equal result | **add nothing**; errors correlate with the fingerprint model about as strongly as a 2D control | `compare_arms` |
| no control mentioned | matched 2D control **+0.006 (-0.017, +0.018)** is what makes the claim structural | `stack_test.csv` |
| no blocking caveat | strict blocking **+0.018**, not clear of zero | `dualkey_test.csv` |
| "compare when fixed descriptors suffice vs end to end learning" | that framing presumes a topological result we do not have | -- |

Two additions the earlier draft had no way to include: the noise floor is not
identifiable (three methods, `CEILING_CLOSED.md`), and 15 subsequent cells across
two pre-registered campaigns were null or negative, which is what justifies
reporting the failures rather than omitting them.
