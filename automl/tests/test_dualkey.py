#!/usr/bin/env python3
"""The dual-key re-analysis must not move a single published number.

``pairs_by_cluster``, ``_assert_fast_matches``, ``paired_adjacent_fast``,
``nested_stack`` and ``_score`` all gained a ``key_col`` argument.  Every one
defaults to ``composition_key``, which is what every published result used.
These tests pin that: called without ``key_col``, each function must behave
exactly as it did before, and the corrected bootstrap must reduce to the
published statistic on a draw with no duplicates.

Synthetic fixtures only -- never the generated research tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl import evaluation as ev
from automl.topo import best_stack, control_factorial as cf
from automl.topo.adjacent_test import adj_r2
from automl.topo.dualkey_test import (BINNED, STRICT, assert_nested,
                                      paired_adjacent_corrected)


def _frame(seed: int = 0, n_ext: int = 12) -> pd.DataFrame:
    """A miniature study: extractants > conditions > lanthanides.

    Built so the two keys genuinely differ -- each (extractant, binned
    condition) block holds two distinct strict conditions -- which is the whole
    situation under test.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_ext):
        for cond_bin in range(2):
            for strict in range(2):
                base = rng.normal(0, 1)
                for li in range(3):
                    rows.append({
                        "safe_exp_id": f"e{e}_c{cond_bin}_s{strict}_m{li}",
                        "extractant_group": f"ext{e}",
                        "composition_key": f"ext{e}||cond{cond_bin}",
                        STRICT: f"ext{e}||cond{cond_bin}||strict{strict}",
                        "lanthanide_index": li,
                        "metal": f"M{li}",
                        "y": base + 0.4 * li + rng.normal(0, 0.2),
                    })
    d = pd.DataFrame(rows).set_index("safe_exp_id")
    d["oof"] = d["y"] + rng.normal(0, 0.5, len(d))
    return d


def test_two_keys_actually_differ():
    """Guard the fixture: a test that both keys agree proves nothing."""
    d = _frame()
    assert d[BINNED].nunique() < d[STRICT].nunique()
    a = adj_r2(d["y"].to_numpy(), d["oof"].to_numpy(),
               d[BINNED].to_numpy(), d["lanthanide_index"].to_numpy())
    b = adj_r2(d["y"].to_numpy(), d["oof"].to_numpy(),
               d[STRICT].to_numpy(), d["lanthanide_index"].to_numpy())
    assert a != pytest.approx(b)


def test_pairs_by_cluster_default_unchanged():
    d = _frame()
    g = d["extractant_group"].to_numpy()
    old = cf.pairs_by_cluster(d, g)
    new = cf.pairs_by_cluster(d, g, BINNED)
    assert len(old) == len(new)
    for (dy0, dp0), (dy1, dp1) in zip(old, new):
        np.testing.assert_array_equal(dy0, dy1)
        np.testing.assert_array_equal(dp0, dp1)


def test_fast_path_matches_shared_metric_under_both_keys():
    """The speedup must agree with ``adjacent_pair_metrics`` for either key."""
    d = _frame()
    g = d["extractant_group"].to_numpy()
    for key in (BINNED, STRICT):
        cf._assert_fast_matches(d, g, n_checks=15, key_col=key)


def test_paired_adjacent_fast_default_unchanged():
    d, e = _frame(0), _frame(0)
    e["oof"] = e["oof"] + 0.1
    a = cf.paired_adjacent_fast(d, e, n_boot=60, seed=3)
    b = cf.paired_adjacent_fast(d, e, n_boot=60, seed=3, key_col=BINNED)
    assert a == b


def test_nested_stack_and_score_default_unchanged():
    d, e = _frame(0), _frame(1)
    e = e.reindex(d.index)
    e["oof"] = d["y"] + 0.3
    frames = {"a": d, "b": e}
    f0, w0 = best_stack.nested_stack(frames, ["a", "b"])
    f1, w1 = best_stack.nested_stack(frames, ["a", "b"], key_col=BINNED)
    np.testing.assert_allclose(f0["oof"].to_numpy(), f1["oof"].to_numpy())
    np.testing.assert_allclose(w0, w1)
    assert best_stack._score(f0) == best_stack._score(f0, BINNED)


def test_nested_stack_differs_between_keys():
    """Fitting and scoring under the strict key must be a real change.

    Both arms must be *imperfect*: ``y + const`` predicts every within-block
    difference exactly, scores 1.0 under either key, and would hide a real
    disagreement behind a saturated metric.
    """
    d, e = _frame(0), _frame(1)
    e = e.reindex(d.index)
    rng = np.random.default_rng(11)
    e["oof"] = d["y"].to_numpy() + rng.normal(0, 0.6, len(d))
    frames = {"a": d, "b": e}
    f_bin, _ = best_stack.nested_stack(frames, ["a", "b"], key_col=BINNED)
    f_str, _ = best_stack.nested_stack(frames, ["a", "b"], key_col=STRICT)
    assert (best_stack._score(f_bin, BINNED)[0]
            != pytest.approx(best_stack._score(f_str, STRICT)[0]))


def test_corrected_bootstrap_reduces_to_published_without_duplicates():
    """With every cluster drawn exactly once, the correction is a no-op.

    The correction only bites when a cluster is drawn twice; suffixing the copy
    index onto a block key that is already unique cannot change the statistic.
    """
    d = _frame()
    y = d["y"].to_numpy(float); p = d["oof"].to_numpy(float)
    comp = d[BINNED].to_numpy().astype(str)
    li = d["lanthanide_index"].to_numpy()
    plain = adj_r2(y, p, comp, li)
    tagged = adj_r2(y, p, np.char.add(comp, "#0"), li)
    assert plain == pytest.approx(tagged)


def test_corrected_bootstrap_is_wider_than_collapsing_one():
    """The published draw collapses duplicates; the corrected one must not.

    ``bootstrap_check.py`` measured the published intervals at 0.71-0.88x the
    corrected width on the real arms.  The direction of that inequality is the
    part that must hold universally, so that is what is asserted.
    """
    d, e = _frame(0), _frame(0)
    rng = np.random.default_rng(7)
    e["oof"] = d["y"] + rng.normal(0, 0.5, len(d))
    fast = cf.paired_adjacent_fast(d, e, n_boot=300, seed=0)
    corr = paired_adjacent_corrected(d, e, n_boot=300, seed=0, key_col=BINNED)
    assert fast is not None and corr is not None
    assert (corr["hi"] - corr["lo"]) > (fast["hi"] - fast["lo"])


def test_assert_nested_rejects_a_block_spanning_two_extractants():
    d = _frame()
    bad = d.copy()
    bad.iloc[0, bad.columns.get_loc("extractant_group")] = "somewhere_else"
    assert_nested(d, BINNED)                      # the good case passes
    with pytest.raises(RuntimeError, match="span more than one extractant"):
        assert_nested(bad, BINNED)


def test_adjacent_pair_arrays_is_linear_in_the_prediction():
    """``nested_stack`` exploits this; if it ever stops holding, stacking breaks."""
    d = _frame()
    y = d["y"].to_numpy(float)
    comp = d[STRICT].to_numpy(); li = d["lanthanide_index"].to_numpy()
    p1 = d["oof"].to_numpy(float)
    p2 = np.roll(p1, 5)
    _, dp1 = ev.adjacent_pair_arrays(y, p1, comp, li)
    _, dp2 = ev.adjacent_pair_arrays(y, p2, comp, li)
    _, dpm = ev.adjacent_pair_arrays(y, 0.3 * p1 + 0.7 * p2, comp, li)
    np.testing.assert_allclose(dpm, 0.3 * dp1 + 0.7 * dp2, atol=1e-12)
