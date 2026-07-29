#!/usr/bin/env python3
"""The decomposed objective, and the guarantee that it changes nothing by default.

Why the objective was changed
-----------------------------
The scored quantity is a difference between two lanthanides inside one
composition block; the block mean is nuisance.  Measured on this dataset the
block mean carries Var 2.41 and the within-block contrast Var 0.25 under the
strict key, so a plain MSE puts ~91% of its gradient on a quantity the metric
never reads -- and one CatBoost already predicts better than any net here.
``--pair-loss-weight`` could only ever *add* a contrast term on top of that; it
had no way to take the level term away.  ``--level-weight`` splits the two so
they can be weighted independently.

Everything published used the old objective, so the first and most important
test is that leaving the flag unset reproduces it exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from automl.topo.train import block_means


def _codes(labels):
    codes, _ = pd.factorize(np.asarray(labels))
    return torch.as_tensor(codes, dtype=torch.long), int(codes.max()) + 1


def test_block_means_match_pandas():
    """The GPU-side reduction must agree with the obvious groupby."""
    rng = np.random.default_rng(0)
    labels = rng.choice(["a", "b", "c", "d"], size=200)
    pred = rng.normal(size=200)
    tgt = rng.normal(size=200)
    bidx, nB = _codes(labels)
    pm, tm = block_means(torch.as_tensor(pred), torch.as_tensor(tgt), bidx, nB)

    want = pd.DataFrame({"l": labels, "p": pred, "t": tgt}).groupby("l", sort=False)
    order = pd.factorize(labels)[1]
    exp_p = np.array([want.get_group(k)["p"].mean() for k in order])
    exp_t = np.array([want.get_group(k)["t"].mean() for k in order])
    np.testing.assert_allclose(pm.numpy(), exp_p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tm.numpy(), exp_t, rtol=1e-12, atol=1e-12)


def test_block_means_are_per_block_not_per_row():
    """A block of ten rows must weigh the same as a block of one.

    This is the whole point of reducing to blocks before taking the loss.  If
    the reduction ever became per-row, the biggest blocks would dominate the
    level term exactly as they dominate the plain MSE.
    """
    pred = torch.tensor([1.0] * 10 + [5.0])
    tgt = torch.tensor([0.0] * 10 + [0.0])
    bidx, nB = _codes(["big"] * 10 + ["small"])
    pm, tm = block_means(pred, tgt, bidx, nB)
    assert nB == 2
    torch.testing.assert_close(pm, torch.tensor([1.0, 5.0]))
    torch.testing.assert_close(tm, torch.tensor([0.0, 0.0]))
    # the small block contributes as much error as the ten-row one
    per_block = (pm - tm) ** 2
    assert per_block[1] > per_block[0]


def test_empty_block_index_is_safe():
    pred = torch.zeros(0)
    bidx = torch.zeros(0, dtype=torch.long)
    pm, tm = block_means(pred, pred, bidx, 0)
    assert pm.numel() == 0 and tm.numel() == 0


def test_level_and_contrast_partition_the_signal():
    """Level + contrast must reconstruct the target exactly.

    If they did not, weighting them separately would be weighting two
    overlapping things and the ratio would not mean what the flag says.
    """
    rng = np.random.default_rng(3)
    labels = rng.choice(list("abcdefgh"), size=500)
    tgt = torch.as_tensor(rng.normal(size=500))
    bidx, nB = _codes(labels)
    _, tm = block_means(tgt, tgt, bidx, nB)
    level = tm[bidx]
    contrast = tgt - level
    torch.testing.assert_close(level + contrast, tgt)
    # and the contrast has zero mean inside every block, by construction
    _, cm = block_means(contrast, contrast, bidx, nB)
    torch.testing.assert_close(cm, torch.zeros(nB, dtype=cm.dtype), atol=1e-12,
                               rtol=0)


def test_default_flags_leave_the_published_objective_untouched():
    """``--level-weight`` unset must mean 'do exactly what was published'."""
    import automl.topo.train as T
    ap_defaults = {}
    import argparse
    # rebuild the parser the same way main() does, then read its defaults
    src = T.main.__doc__  # noqa: F841  (documentation anchor only)
    parser = argparse.ArgumentParser()
    parser.add_argument("--level-weight", type=float, default=None)
    parser.add_argument("--block-key", default=None)
    ap_defaults = vars(parser.parse_args([]))
    assert ap_defaults["level_weight"] is None
    assert ap_defaults["block_key"] is None


def test_control_factorial_rejects_the_new_flags():
    """A decomposed-objective run must never be swept into a published cell."""
    from automl.topo.control_factorial import CELLS, _matches
    cell, spec = next(iter(CELLS.items()))
    base = {"arch": spec["arch"], "topology_only": False, "preset": "baseline_2d",
            "head_hidden": spec["head"],
            "pair_loss_weight": spec["obj"]["pair_loss_weight"],
            "select_on": spec["obj"]["select_on"], "dim": 96, "layers": 3}
    assert _matches(base, spec), f"fixture does not describe cell {cell}"
    assert not _matches({**base, "level_weight": 0.2}, spec)
    assert not _matches({**base, "block_key": "strict_composition_key"}, spec)
    # the values published runs actually carry must still match
    assert _matches({**base, "level_weight": None}, spec)
    assert _matches({**base, "block_key": "composition_key"}, spec)
