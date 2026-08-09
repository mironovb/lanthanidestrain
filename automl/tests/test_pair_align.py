"""The contrast term's new shape, and the census that motivated changing it.

Two things are checked here and they are different claims:

1. With every CAMPAIGN6 flag at its default, the new loss block computes the
   *same number* as the published one.  That is what makes the change
   default-off rather than merely default-ish, and it is the only reason the
   published arms can still be compared against new ones.

2. With ``--pair-metric-align`` on, the pairs the loss differences are the
   pairs ``evaluation.adjacent_pair_arrays`` scores -- same count, same values.
   Asserted against the evaluator itself rather than against a second
   hand-rolled enumeration, because a second enumeration is exactly what
   evaluation.py:181-192 records going wrong once already.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import automl.evaluation as ev
from automl.topo.train import block_means


def _published_term(pred, tgt, cb, li, pair_adj_w=3.0):
    """The contrast term exactly as it stood before CAMPAIGN6."""
    same = torch.as_tensor(cb[:, None] == cb[None, :])
    pi, pj = torch.nonzero(torch.triu(same, diagonal=1), as_tuple=True)
    dl = torch.as_tensor(np.abs(li[:, None] - li[None, :]),
                         dtype=torch.float32)[pi, pj]
    w = torch.where(dl <= 1.0, pair_adj_w, 1.0)
    dp, dt = pred[pi] - pred[pj], tgt[pi] - tgt[pj]
    return (w * (dp - dt) ** 2).mean()


def _new_term(pred, tgt, cb, li, *, align=False, adj_only=False,
              pair_adj_w=3.0, kind="sq", delta=1.0):
    """The block from train.py, lifted verbatim so it can run without a fold."""
    same = torch.as_tensor(cb[:, None] == cb[None, :])
    pi, pj = torch.nonzero(torch.triu(same, diagonal=1), as_tuple=True)
    dl = torch.as_tensor(np.abs(li[:, None] - li[None, :]),
                         dtype=torch.float32)[pi, pj]
    if align:
        ccodes, _ = pd.factorize(pd.MultiIndex.from_arrays([cb, li]))
        nC = int(ccodes.max()) + 1
        cidx = torch.as_tensor(ccodes, dtype=torch.long)
        P, T = block_means(pred, tgt, cidx, nC)
        first = np.unique(ccodes, return_index=True)[1]
        cbk, clx = cb[first], li[first]
        cs = torch.as_tensor(cbk[:, None] == cbk[None, :])
        qi, qj = torch.nonzero(torch.triu(cs, diagonal=1), as_tuple=True)
        dql = torch.as_tensor(np.abs(clx[:, None] - clx[None, :]),
                              dtype=torch.float32)[qi, qj]
    else:
        P, T, qi, qj, dql = pred, tgt, pi, pj, dl
    if adj_only:
        keep = dql <= 1.0
        qi, qj, dql = qi[keep], qj[keep], dql[keep]
    w = torch.where(dql <= 1.0, pair_adj_w, 1.0)
    dp, dt = P[qi] - P[qj], T[qi] - T[qj]
    err = ((dp - dt) ** 2 if kind == "sq"
           else torch.nn.functional.huber_loss(dp, dt, delta=delta,
                                               reduction="none"))
    return (w * err).mean(), qi, qj


def _toy():
    """Two blocks, replicates in both, one adjacent pair and one gap."""
    cb = np.array(["A", "A", "A", "A", "B", "B", "B"])
    li = np.array([3.0, 3.0, 4.0, 6.0, 8.0, 9.0, 9.0])
    rng = np.random.default_rng(0)
    pred = torch.tensor(rng.normal(size=7), dtype=torch.float32)
    tgt = torch.tensor(rng.normal(size=7), dtype=torch.float32)
    return pred, tgt, cb, li


def test_defaults_reproduce_the_published_term():
    pred, tgt, cb, li = _toy()
    new, _, _ = _new_term(pred, tgt, cb, li)
    old = _published_term(pred, tgt, cb, li)
    assert torch.equal(new, old), f"{new.item()} != {old.item()}"


def test_alignment_removes_same_metal_pairs():
    pred, tgt, cb, li = _toy()
    _, qi, qj = _new_term(pred, tgt, cb, li, align=True)
    # Cell indices, so a same-metal pair cannot exist by construction.
    ccodes, _ = pd.factorize(pd.MultiIndex.from_arrays([cb, li]))
    first = np.unique(ccodes, return_index=True)[1]
    clx = li[first]
    assert not np.any(clx[qi.numpy()] == clx[qj.numpy()]), \
        "aligned term still contains a same-metal pair"
    # Unaligned it certainly does: rows 0,1 are both li==3 in block A.
    _, ui, uj = _new_term(pred, tgt, cb, li, align=False)
    assert np.any(li[ui.numpy()] == li[uj.numpy()])


def test_aligned_pairs_match_the_evaluator():
    """The aligned adjacent differences ARE the metric's, value for value."""
    pred, tgt, cb, li = _toy()
    _, qi, qj = _new_term(pred, tgt, cb, li, align=True, adj_only=True)
    ccodes, _ = pd.factorize(pd.MultiIndex.from_arrays([cb, li]))
    nC = int(ccodes.max()) + 1
    P, T = block_means(pred, tgt, torch.as_tensor(ccodes, dtype=torch.long), nC)
    got = np.sort(np.abs((T[qi] - T[qj]).numpy()))

    dy, _ = ev.adjacent_pair_arrays(tgt.numpy().astype(float),
                                    pred.numpy().astype(float), cb, li)
    want = np.sort(np.abs(dy))
    assert len(got) == len(want), f"{len(got)} pairs vs evaluator's {len(want)}"
    assert np.allclose(got, want, atol=1e-6)


def test_adj_weight_and_huber_are_reachable():
    pred, tgt, cb, li = _toy()
    base, _, _ = _new_term(pred, tgt, cb, li)
    hi, _, _ = _new_term(pred, tgt, cb, li, pair_adj_w=10.0)
    hub, _, _ = _new_term(pred, tgt, cb, li, kind="huber")
    assert not torch.equal(base, hi)
    assert not torch.equal(base, hub)


def test_same_metal_pairs_dominate_the_emphasis_term_on_real_data():
    """The census in the plan, re-measured, so it cannot silently drift.

    If this ever fails the motivation for --pair-metric-align has changed and
    the flag should be re-argued, not re-tuned.
    """
    from automl.matrix_cache import load_cache
    df, _, _ = load_cache()
    d = df.dropna(subset=["log_D"])
    if "geometry_feature_build_id" in d:
        d = d[d["geometry_feature_build_id"].notna()]
    y = d["log_D"].to_numpy(float)
    blk = d["composition_key"].to_numpy()
    li = d["lanthanide_index"].to_numpy(float)

    sq = {"same": 0.0, "adj": 0.0}
    for _, idx in pd.Series(range(len(d))).groupby(blk).groups.items():
        idx = np.asarray(list(idx))
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        dl = np.abs(li[idx][i] - li[idx][j])
        dy = y[idx][i] - y[idx][j]
        sq["same"] += float((dy[dl == 0] ** 2).sum())
        sq["adj"] += float((dy[dl == 1] ** 2).sum())
    share = sq["same"] / (sq["same"] + sq["adj"])
    assert share > 0.5, (
        f"same-metal share of the 3x-weighted term is {share:.3f}; the "
        f"mismatch --pair-metric-align exists to fix has changed")
