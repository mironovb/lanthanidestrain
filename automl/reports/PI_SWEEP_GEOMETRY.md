# The shipped persistence images have an effective dimension of 2.7

**Bogdan Mironov · 22 July 2026**
Measured from the rendered images alone — **no training, no `log D`** — and
written while Stage A was still running, with only 2 of 25 configurations
complete. The prediction in §4 is therefore made *before* the data that tests it.

---

## 1. Why measure this

The likely outcome of this sweep is that every configuration ties at stack weight
**0.00**, because the published persistence-image arm already does. Two
completely different explanations would produce that tie, and the training runs
cannot separate them:

* **The images barely changed.** Then no readout could tell the configurations
  apart, and a tie says nothing about persistence homology.
* **The images changed substantially and it did not help.** Then the tie is a
  real result about the representation.

Pixel-wise comparison is impossible across resolutions, so the comparison is on
the **geometry each configuration induces over the 953 complexes**: the matrix of
pairwise distances between complexes in image space, correlated Mantel-style
against the shipped anchor (`rho`), plus the **participation ratio** of each
image set's covariance spectrum (`eff_dim`) — how many directions the
representation actually varies in.

Images are compared after the same `log1p` and per-channel standardisation
`PersistenceImages` applies, so this describes what the network is handed.

## 2. The headline number

> **The shipped persistence images vary in ~2.7 effective directions across 953
> complexes**, from 400 pixels.

That is an extremely impoverished representation — close to a three-parameter
family — and it is a concrete, training-free explanation for why the
persistence-image arm has never been competitive with the simplicial one. It also
bounds what any readout could extract: a CNN cannot recover structure the input
does not contain.

**Tuning raises it by 7.4×**, to 20.3.

## 3. The grid is monotone in both axes

Effective dimension, by resolution × spread (spread in pixels, so it means the
same thing at every resolution). The shipped anchor is the 0.6-pixel column:

| resolution | 0.5 px | **0.6 px (shipped)** | 1.0 px | 2.0 px | 4.0 px |
|---|---|---|---|---|---|
| 20 | 3.2 | **2.7** | 2.1 | 1.7 | 1.2 |
| 32 | 5.1 | | 2.7 | 1.9 | 1.5 |
| 48 | 8.5 | | 3.8 | 2.3 | 1.8 |
| 64 | 12.1 | | 5.3 | 2.7 | 2.0 |
| 96 | 17.3 | | 8.9 | 3.8 | 2.3 |
| 128 | **20.3** | | 13.0 | 5.3 | 2.7 |

Perfectly monotone in both directions: **resolution adds information, smoothing
destroys it.**

`rho` against the anchor falls as low as **0.681** (median 0.965), so the
configurations genuinely reorder which complexes look alike — they are not
re-expressions of one another.

**A tie in the training results would therefore be informative.**

## 4. This corrects my own opening diagnosis — and makes a prediction

The sweep was motivated partly by my claim that the shipped spread of 0.61 pixels
left the images **under**-smoothed, "a sparse histogram rather than a smooth
image, so a CNN has almost nothing to convolve over". Two parts of that are
wrong, and the correction matters because it points the opposite way:

* **The images are not sparse.** 65 % of pixels carry mass (and 90 % are nonzero
  to floating point). ~493 diagram points cover a 400-pixel plane regardless of
  how narrow each kernel is. I inferred sparsity from the spread-to-pixel ratio
  without checking the occupancy that was already in front of me.
* **More smoothing makes the representation strictly poorer**, monotonically, on
  every row of the table. If anything the shipped setting is over-smoothed for
  information content, not under-smoothed.

What survives is the *resolution* half of the diagnosis, and it survives much
more strongly than I argued: at 20 pixels the representation has ~2.7 effective
dimensions no matter how it is smoothed.

### The prediction, stated before the runs that test it

If effective dimensionality is the binding constraint, then across Stage A the
tune-half adjacent-pair R² should **increase with resolution** and **decrease
with spread**, with `128 px / 0.5 px spread` the strongest arm and
`20 px / 4.0 px spread` the weakest.

**It is not a safe prediction, and the opposing force is already documented.**
`PersistenceCNN` has a fixed 7×7 receptive field and a global pool, so at 128
pixels each unit sees only 5.5 % of the plane against 35 % at 20 pixels
(`PI_SWEEP_PREREGISTRATION.md` §5). Information rises with resolution while the
readout's ability to integrate it falls. Which dominates is exactly what Stage A
measures.

| outcome | reading |
|---|---|
| adj R² rises with resolution | information was the binding constraint; the shipped 20 px was the problem |
| adj R² falls with resolution | the fixed receptive field dominates; the constraint is the **readout**, not the representation — and the honest next step is sweeping the CNN, not the images |
| flat | neither is binding; the target cannot use this representation at any construction |

All three are informative, which is the point of writing it down now.

---

*Reproduce: `python3 -m automl.qc.pi_sweep_geometry --stage a`
(`automl/reports/pi_sweep_geometry_a.csv`). Uses no target and no trained model,
so it cannot influence the confirmatory endpoint.*
