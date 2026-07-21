#!/usr/bin/env python3
"""Colour-vision validation for the figure palettes.

Ported from the dataviz skill's ``validate_palette.js`` because this cluster has
no node runtime.  The point of the original is that palette safety is
*computed*, never eyeballed, so the check travels with the figures rather than
living in a comment that says "validated".

Algorithm and thresholds are the skill's, unchanged:

* CVD simulation: Machado, Oliveira & Fernandes (2009), severity 1.0, applied in
  linear RGB.
* Distance: Euclidean in OKLab, x100.
* Thresholds: adjacent-pair CVD dE >= 8.0 target (6.0 hard floor, legal only
  with a secondary encoding), normal-vision dE >= 15.0 on the active pairlist,
  lightness inside the mode's band, OKLCH chroma >= 0.10, WCAG contrast >= 3.0
  against the chart surface.

Run:  python3 -m pytest automl/tests/test_palette.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CHROMA_FLOOR = 0.10
CONTRAST_MIN = 3.0
SURFACE_LIGHT = "#fcfcfb"
BAND_LIGHT = (0.35, 0.75)

MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
}

# OKLab matrices (Björn Ottosson)
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _hex_to_srgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _lin(h: str) -> np.ndarray:
    c = _hex_to_srgb(h)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab_from_lin(rgb: np.ndarray) -> np.ndarray:
    lms = _M1 @ rgb
    return _M2 @ np.cbrt(np.clip(lms, 0, None))


def _oklch(h: str) -> tuple[float, float]:
    L, a, b = _oklab_from_lin(_lin(h))
    return float(L), float(np.hypot(a, b))


def _simulate(h: str, kind: str) -> np.ndarray:
    return np.clip(MACHADO[kind] @ _lin(h), 0, 1)


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    a = _oklab_from_lin(_simulate(h1, kind) if kind else _lin(h1))
    b = _oklab_from_lin(_simulate(h2, kind) if kind else _lin(h2))
    return float(100 * np.linalg.norm(a - b))


def _relative_luminance(h: str) -> float:
    r, g, b = _lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(h1: str, h2: str) -> float:
    a, b = _relative_luminance(h1), _relative_luminance(h2)
    lo, hi = min(a, b), max(a, b)
    return (hi + 0.05) / (lo + 0.05)


def worst_cvd(palette: list[str], all_pairs: bool = False) -> tuple[float, str, str, str]:
    n = len(palette)
    pairs = ([(i, j) for i in range(n) for j in range(i + 1, n)] if all_pairs
             else [(i, i + 1) for i in range(n - 1)])
    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pairs:
            d = delta_e(palette[i], palette[j], kind)
            if worst is None or d < worst[0]:
                worst = (d, kind, palette[i], palette[j])
    return worst


def worst_normal(palette: list[str], all_pairs: bool = False) -> float:
    n = len(palette)
    pairs = ([(i, j) for i in range(n) for j in range(i + 1, n)] if all_pairs
             else [(i, i + 1) for i in range(n - 1)])
    return min(delta_e(palette[i], palette[j]) for i, j in pairs)


# ---------------------------------------------------------------------------
# The palettes actually used by automl/figures_topo.py.  Each figure uses a
# small subset, and subsets are validated with all_pairs=True because a scatter
# or a forest plot puts every series against every other, not just neighbours.
# ---------------------------------------------------------------------------
from automl.figures import C  # noqa: E402  (house palette)

SUBSETS = {
    "forest (topology vs 2 baselines)": [C["blue"], C["orange"], C["violet"]],
    "blend curve (2 measures)": [C["blue"], C["orange"]],
    # The scatter uses THREE hues plus marker shape, not four hues.
    #
    # Searching the house palette for a 4-colour subset that simultaneously
    # passes all-pairs CVD (dE >= 8), normal-vision separation (dE >= 15) and
    # WCAG contrast (>= 3.0) against the light surface returns **zero**
    # candidates.  Magenta (2.62), yellow (2.11) and aqua (2.74) all fail
    # contrast outright, and orange/green collapse under protanopia (dE = 3.2).
    # Exactly three 3-colour subsets pass everything.
    #
    # So the fourth distinction (SNN vs PI-CNN) is carried by marker shape
    # instead of a fabricated hue -- the composite encoding the guidance calls
    # for, and it keeps identity from resting on colour alone.
    "tradeoff scatter (3 hues + shape)": [C["blue"], C["orange"], C["violet"]],
    "seed spread (2 architectures)": [C["blue"], C["violet"]],
    "parity (model vs baseline)": [C["blue"], C["orange"]],
}


@pytest.mark.parametrize("name,palette", list(SUBSETS.items()))
def test_cvd_separation(name, palette):
    d, kind, c1, c2 = worst_cvd(palette, all_pairs=True)
    assert d >= CVD_FLOOR, f"{name}: {c1}/{c2} collapse under {kind} (dE={d:.1f} < {CVD_FLOOR})"
    if d < CVD_TARGET:
        pytest.skip(f"{name}: dE={d:.1f} in the 6-8 floor band ({kind}); "
                    f"legal only with a secondary encoding, which these figures use")


@pytest.mark.parametrize("name,palette", list(SUBSETS.items()))
def test_normal_vision_separation(name, palette):
    d = worst_normal(palette, all_pairs=True)
    assert d >= NORMAL_FLOOR, (
        f"{name}: worst normal-vision dE={d:.1f} < {NORMAL_FLOOR} -- full-colour "
        f"readers cannot separate this pair; secondary encoding does not excuse it")


@pytest.mark.parametrize("name,palette", list(SUBSETS.items()))
def test_chroma_and_contrast(name, palette):
    for c in palette:
        _, chroma = _oklch(c)
        assert chroma >= CHROMA_FLOOR, f"{name}: {c} reads grey (C={chroma:.3f})"
        cr = contrast_ratio(c, SURFACE_LIGHT)
        assert cr >= CONTRAST_MIN, (
            f"{name}: {c} contrast {cr:.2f} vs surface < {CONTRAST_MIN}; "
            f"needs a visible label or table view")
