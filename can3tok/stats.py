"""Dataset-level xyz normalisation statistics.

center/scale are derived from robust quantiles so that a few far outliers cannot
shrink the useful range. The formula is kept identical to the previous pipeline
(center = midpoint of the quantile box, scale = 1.05 * half of its longest side)
so xyz_rmse_norm values remain comparable across code bases.
"""

from __future__ import annotations

import glob
import os
from typing import Dict

import numpy as np

from .io_utils import load_json, load_npz_state, save_json

QUANTILE_MARGIN = 1.05


def compute_global_stats(root: str, stride: int = 50, quantile: float = 0.02) -> Dict:
    files = sorted(glob.glob(os.path.join(root, "step_*.npz")))
    if not files:
        raise FileNotFoundError(f"No step_*.npz found under {root}")
    stride = max(int(stride), 1)
    chunks = []
    for path in files[::stride]:
        gs = load_npz_state(path)
        if gs["xyz"].size:
            chunks.append(gs["xyz"])
    xyz = np.concatenate(chunks, axis=0)
    lo = np.quantile(xyz, quantile, axis=0).astype(np.float32)
    hi = np.quantile(xyz, 1.0 - quantile, axis=0).astype(np.float32)
    center = ((lo + hi) * 0.5).astype(np.float32)
    scale = float(np.max((hi - lo) * 0.5)) * QUANTILE_MARGIN
    scale = max(scale, 1e-6)
    return {
        "root": os.path.abspath(root),
        "num_files": len(files),
        "sample_stride": stride,
        "quantile": float(quantile),
        "center": center.tolist(),
        "scale": scale,
        "lo": lo.tolist(),
        "hi": hi.tolist(),
    }


def load_or_create_stats(root: str, stats_path: str = None, stride: int = 50, quantile: float = 0.02) -> Dict:
    if stats_path and os.path.exists(stats_path):
        return load_json(stats_path)
    stats = compute_global_stats(root, stride=stride, quantile=quantile)
    if stats_path:
        save_json(stats_path, stats)
    return stats
