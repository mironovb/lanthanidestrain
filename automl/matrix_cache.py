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
    return df, blocks, payload["info"]


if __name__ == "__main__":
    _df, _b, _i = build_cache()
    print(json.dumps(_i, indent=2))
