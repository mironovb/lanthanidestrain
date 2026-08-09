#!/usr/bin/env python3
"""Build once, load everywhere: cached feature matrix + block map."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from automl.dataset import Blocks, build_matrix

CACHE_DIR = Path(__file__).resolve().parents[1] / "automl/artifacts/matrix"
MATRIX_PATH = CACHE_DIR / "matrix.parquet"
BLOCKS_PATH = CACHE_DIR / "blocks.json"


def build_cache() -> tuple[pd.DataFrame, Blocks, dict]:
    df, blocks, info = build_matrix(require_3d=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic replace: a sweep shard starting while the cache is rewritten must
    # never read a half-written parquet.
    tmp = MATRIX_PATH.with_suffix(".parquet.tmp")
    df.copy().to_parquet(tmp, index=False)
    os.replace(tmp, MATRIX_PATH)
    tmp_json = BLOCKS_PATH.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps({"blocks": blocks.mapping, "info": info}, indent=2))
    os.replace(tmp_json, BLOCKS_PATH)
    return df, blocks, info


def load_cache(rebuild: bool = False) -> tuple[pd.DataFrame, Blocks, dict]:
    if rebuild or not MATRIX_PATH.exists() or not BLOCKS_PATH.exists():
        return build_cache()
    df = pd.read_parquet(MATRIX_PATH)
    payload = json.loads(BLOCKS_PATH.read_text())
    blocks = Blocks(mapping=payload["blocks"])
    _attach_late_blocks(df, blocks, payload["info"])
    return df, blocks, payload["info"]


def _attach_late_blocks(df: pd.DataFrame, blocks: Blocks, info: dict) -> None:
    """Blocks added after the cache on disk was written.

    Derived purely from columns already in the cache, so they can be rebuilt in
    memory in milliseconds.  Doing it here rather than rewriting
    matrix.parquet keeps a file that a dozen other analyses read byte-identical
    -- a cache rebuild is not a safe way to ship a new feature block when the
    published arms have to stay reproducible from the same file.

    Inert for every existing preset: BLOCK_PRESETS selects by block NAME, and
    baseline_2d does not name mphys.
    """
    if "mphys" in blocks.mapping:
        return
    from automl.metal_physics import attach as _attach_mphys
    cols = _attach_mphys(df)
    blocks.add("mphys", cols)
    info["mphys_columns"] = len(cols)


if __name__ == "__main__":
    _df, _b, _i = build_cache()
    print(json.dumps(_i, indent=2))
