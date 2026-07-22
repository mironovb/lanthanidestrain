#!/usr/bin/env python3
"""Render cached persistence diagrams into images under a swept configuration.

Why the shipped configuration needed examining at all
-----------------------------------------------------
Measured over 120 complexes / 59,171 persistence points, the shipped settings
have three properties that between them explain why a CNN on these images
carries so little signal:

* **spread 0.08 against a pixel spacing of 0.132** -- ratio 0.61.  Each diagram
  point deposits essentially all of its mass in one pixel, so the "image" is a
  sparse histogram rather than a smooth surface.  There is very little spatial
  structure for a convolution to exploit.
* **the (0, 2.5) birth/death window discards 13.5 % of all points.**  Deaths
  reach 20.6 with a p95 of 3.24, so the entire upper tail is being clipped.
* **H0 and H1 are summed into a single channel** despite occupying disjoint
  regions (H0 deaths median 0.30, H1 deaths median 1.98).

The PI (Kostas) independently named resolution and spread as the two axes to
benchmark, noting that spread "becomes important when we have many points in the
persistence diagrams".  These diagrams carry ~493 points per complex, so that
applies directly.

Rendering, and why it is a matrix product
-----------------------------------------
The Gaussian kernel is separable:

    exp(-((x-b)^2 + (y-d)^2) / 2s^2) = exp(-(x-b)^2 / 2s^2) * exp(-(y-d)^2 / 2s^2)

so an image accumulating P weighted Gaussians is

    image = (V * w).T @ U            U: (P, R_birth), V: (P, R_death)

one BLAS call instead of P separate ``resolution x resolution`` meshgrid
evaluations.  At 128 x 128 with ~493 points the naive loop is ~7.7e9 operations
per configuration; this is milliseconds.  The axis convention matches the
shipped ``persistence_image`` exactly: it uses ``meshgrid(..., indexing="xy")``,
so ``image[i, j]`` is (death_grid[i], birth_grid[j]) -- rows are death, columns
are birth.

Output uses the shipped ``{images, build_ids}`` schema, so ``PersistenceImages``
loads a swept configuration through the existing ``PI_IMAGES_PATH`` environment
override with no code change.

``data/`` is never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "automl/artifacts/pi_sweep/images"

# Weighting of a diagram point by its persistence.  The shipped choice is
# "linear", which is why 24.7 % of kept points -- those whose persistence is
# under one pixel -- are close to invisible in the rendered image.
WEIGHTS = ("linear", "constant", "squared", "arctan")


def weight_fn(kind: str, pers: np.ndarray) -> np.ndarray:
    if kind == "linear":
        return pers                      # shipped: death - birth
    if kind == "constant":
        return np.ones_like(pers)
    if kind == "squared":
        return pers ** 2
    if kind == "arctan":
        # The standard Adams et al. weighting: saturating, so a few very
        # long-lived features cannot dominate the whole image.
        return np.arctan(pers)
    raise ValueError(f"unknown weight {kind!r}")


@dataclass(frozen=True)
class PIConfig:
    resolution: int = 20
    spread: float = 0.08
    lo: float = 0.0
    hi: float = 2.5
    weight: str = "linear"
    channels: str = "sum"        # "sum" = H0+H1 in one image; "split" = 2 channels
    dims: tuple[int, ...] = (0, 1)

    def key(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def label(self) -> str:
        return (f"r{self.resolution}_s{self.spread:g}_"
                f"{self.lo:g}-{self.hi:g}_{self.weight}_{self.channels}")

    @property
    def pixel(self) -> float:
        return (self.hi - self.lo) / (self.resolution - 1)


# Exactly the shipped asset's settings.  Rendering the cache with this must
# reproduce complex_gfn2xtb_pi_images.npz; that is the gate in pi_sweep_build.
SHIPPED_CONFIG = PIConfig()


def _render_one(pts: np.ndarray, cfg: PIConfig) -> np.ndarray:
    """One channel: (resolution, resolution), rows death, columns birth."""
    R = cfg.resolution
    if len(pts) == 0:
        return np.zeros((R, R), dtype=np.float64)
    b, d = pts[:, 0], pts[:, 1]
    # Same admission rule as the shipped persistence_image: both coordinates
    # inside the window and strictly positive persistence.
    keep = ((b >= cfg.lo) & (b <= cfg.hi) &
            (d >= cfg.lo) & (d <= cfg.hi) & (d > b))
    if not keep.any():
        return np.zeros((R, R), dtype=np.float64)
    b, d = b[keep], d[keep]
    w = weight_fn(cfg.weight, d - b)

    grid = np.linspace(cfg.lo, cfg.hi, R)
    s2 = 2.0 * cfg.spread ** 2
    U = np.exp(-((grid[None, :] - b[:, None]) ** 2) / s2)   # (P, R) birth
    V = np.exp(-((grid[None, :] - d[:, None]) ** 2) / s2)   # (P, R) death
    return (V * w[:, None]).T @ U


def render_all(cache: dict, cfg: PIConfig) -> tuple[np.ndarray, list[str]]:
    """(N, C, R, R) float32 images for every complex in the cache."""
    points, dims, ptr = cache["points"], cache["dims"], cache["ptr"]
    ids = cache["build_ids"]
    if cfg.channels == "sum":
        groups: list[tuple[int, ...]] = [tuple(cfg.dims)]
    elif cfg.channels == "split":
        groups = [(dim,) for dim in cfg.dims]
    else:
        raise ValueError(f"unknown channels {cfg.channels!r}")

    out = np.zeros((len(ids), len(groups), cfg.resolution, cfg.resolution),
                   dtype=np.float32)
    for i in range(len(ids)):
        p = points[ptr[i]:ptr[i + 1]]
        dm = dims[ptr[i]:ptr[i + 1]]
        for c, want in enumerate(groups):
            out[i, c] = _render_one(p[np.isin(dm, want)], cfg).astype(np.float32)
    return out, ids


def path_for(cfg: PIConfig) -> Path:
    return OUT_DIR / f"img_{cfg.key()}.npz"


def write(cfg: PIConfig, images: np.ndarray, ids: list[str]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = path_for(cfg)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, images=images.astype(np.float32),
                        build_ids=np.asarray(ids, dtype="U32"),
                        config=json.dumps(asdict(cfg)))
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# The swept grid
# ---------------------------------------------------------------------------

# Stage A -- the benchmark Kostas asked for.  Resolution upward from the shipped
# 20; 150 is excluded because he advises against it and because 953 images would
# not support that many free pixels.  Spread is expressed as a multiple of the
# pixel spacing so it means the same thing at every resolution -- the shipped
# 0.08 sits at 0.61 pixels, i.e. below the smallest multiple swept here, which
# is the point.
STAGE_A_RESOLUTIONS = (20, 32, 48, 64, 96, 128)
STAGE_A_SPREAD_PIXELS = (0.5, 1.0, 2.0, 4.0)

# Stage B -- the axes the diagnostics implicate, at Stage A's best resolution
# and spread.  "auto" resolves to the p99 of all cached deaths.
STAGE_B_RANGES = ("shipped", "wide", "wider", "auto")
STAGE_B_CHANNELS = ("sum", "split")
STAGE_B_WEIGHTS = ("linear", "constant", "squared", "arctan")


def auto_hi(cache: dict, q: float = 99.0) -> float:
    """Upper bound covering all but the extreme tail of observed deaths."""
    dims, pts = cache["dims"], cache["points"]
    sel = np.isin(dims, (0, 1))
    return float(np.round(np.percentile(pts[sel, 1], q), 2))


def stage_a_configs() -> list[PIConfig]:
    # The shipped settings, rendered from the same cache as everything else.
    # This is the **reproduction anchor**, and it is swept rather than assumed:
    # 18 of the 953 complexes cannot be reproduced bit-for-bit from the shipped
    # asset in this environment (a gudhi/CGAL version difference, attributed by
    # the gate in pi_sweep_build).  Comparing a tuned configuration against the
    # shipped *asset* would therefore confound tuning with that discrepancy.
    # Comparing it against this anchor does not, because both are rendered the
    # same way here.  It doubles as a reproduction check on the published P0
    # arm: the anchor should land near +0.2101 adjacent-pair R2.
    out = [SHIPPED_CONFIG]
    for res in STAGE_A_RESOLUTIONS:
        px = (SHIPPED_CONFIG.hi - SHIPPED_CONFIG.lo) / (res - 1)
        for mult in STAGE_A_SPREAD_PIXELS:
            out.append(PIConfig(resolution=res, spread=round(px * mult, 5)))
    return out


def stage_b_configs(res: int, spread: float, cache: dict) -> list[PIConfig]:
    hi_of = {"shipped": 2.5, "wide": 4.0, "wider": 6.0, "auto": auto_hi(cache)}
    out = []
    for rng in STAGE_B_RANGES:
        for ch in STAGE_B_CHANNELS:
            for w in STAGE_B_WEIGHTS:
                out.append(PIConfig(resolution=res, spread=spread,
                                    hi=hi_of[rng], weight=w, channels=ch))
    # A wider window at fixed resolution silently coarsens the grid, so hold the
    # spread at the same multiple of pixel spacing the winner of Stage A used;
    # otherwise "wider range" and "more smoothing" are confounded.
    ref_px = (SHIPPED_CONFIG.hi - SHIPPED_CONFIG.lo) / (res - 1)
    mult = spread / ref_px
    return [PIConfig(resolution=c.resolution,
                     spread=round(c.pixel * mult, 5),
                     lo=c.lo, hi=c.hi, weight=c.weight, channels=c.channels,
                     dims=c.dims) for c in out]


def main() -> int:
    from automl.qc.pi_sweep_build import load

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("a", "b"), required=True)
    ap.add_argument("--res", type=int, help="stage b: Stage A's winning resolution")
    ap.add_argument("--spread", type=float, help="stage b: Stage A's winning spread")
    ap.add_argument("--manifest", default=None,
                    help="where to write the config -> path manifest")
    args = ap.parse_args()

    cache = load()
    if args.stage == "a":
        cfgs = stage_a_configs()
    else:
        if args.res is None or args.spread is None:
            raise SystemExit("stage b needs --res and --spread from Stage A")
        cfgs = stage_b_configs(args.res, args.spread, cache)

    seen: dict[str, PIConfig] = {}
    rows = []
    for cfg in cfgs:
        if cfg.key() in seen:            # e.g. a Stage B cell equal to Stage A's
            continue
        seen[cfg.key()] = cfg
        out = path_for(cfg)
        if out.exists():
            print(f"[pi-render] {cfg.label():44s} cached")
        else:
            imgs, ids = render_all(cache, cfg)
            write(cfg, imgs, ids)
            nz = float(np.mean(imgs != 0))
            print(f"[pi-render] {cfg.label():44s} {imgs.shape}  "
                  f"nonzero {100*nz:5.1f}%  spread/pixel {cfg.spread/cfg.pixel:.2f}")
        rows.append({"key": cfg.key(), "path": str(out), **asdict(cfg)})

    man = Path(args.manifest) if args.manifest else \
        OUT_DIR.parent / f"manifest_stage_{args.stage}.json"
    man.write_text(json.dumps(rows, indent=2, default=list) + "\n")
    print(f"[pi-render] {len(rows)} configurations -> {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
