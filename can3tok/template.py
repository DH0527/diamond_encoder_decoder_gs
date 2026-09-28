"""The fixed within-group point template, shared by the packer and the decoder.

Both sides must agree bit-for-bit: ``data.py`` orders every group's points by an
optimal assignment to this template, and the folding decoder emits ``R diag(s)
(template + residual)``. If the two ever drift apart the residual head has to
spend its rank undoing a permutation instead of describing shape.

Why a filled ball and not a shell, and why the residual is defined in the
whitened local frame -- measured on val, radius-normalised within-group chamfer
(GT's own nearest-neighbour spacing is 0.191, i.e. 1.0x):

    template x envelope                       chamfer   x spacing
    fibonacci shell  x axis-aligned scale       0.367      1.92     <- first try
    fibonacci shell  x rotated eigenframe       0.320      1.68
    filled ball      x axis-aligned scale       0.336      1.76
    filled ball      x rotated eigenframe       0.301      1.58     <- this file

and with the residual on top, target ordered by assignment to the template:

    + rank-8  residual                          0.248      1.30
    + rank-22 residual                          0.225      1.18     <- budget
    + rank-22 residual, target NOT aligned      0.261      1.37

For reference the best number this codebase had ever reached before was 0.345
(1.81x), from a plain ``shape_xyz`` MLP after 6000 steps.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["fibonacci_ball", "fibonacci_ball_torch", "fibonacci_prefix",
           "prefix_center_template", "SCALE"]

SCALE = math.sqrt(3.0)


def fibonacci_ball(n: int) -> np.ndarray:
    """``(n, 3)`` deterministic low-discrepancy points filling the unit ball.

    Fibonacci spiral for the directions, ``((i + 0.5) / n) ** (1/3)`` for the
    radii so the points are volume-uniform rather than concentrated on the
    surface.

    ``SCALE`` is not the value that makes the per-axis std match the whitened
    target (that would be sqrt(5)). Measured envelope-only chamfer against the
    template scale, and the same with a rank-22 residual on top:

        scale   per-axis std   envelope only   + rank-22
        1.000          0.443           0.365       0.226
        1.732          0.767           0.299       0.225   <- used
        2.236          0.990           0.331       0.224

    The converged number is flat -- the residual absorbs the scale -- but the
    *initial* one is not, and the whole point of the fixed template is to start
    above the previous best (0.345) instead of climbing to it. Real groups are
    denser in the middle than a uniform ball, so a slightly shrunken ball
    balances the two chamfer directions; sqrt(5) overshoots into g->p.
    """
    i = np.arange(n, dtype=np.float64)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / float(n)
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = phi * i
    dirs = np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=-1)
    pts = dirs * (((i + 0.5) / float(n)) ** (1.0 / 3.0))[:, None]
    pts = pts - pts.mean(axis=0, keepdims=True)
    return pts * SCALE


def fibonacci_ball_torch(n: int, dtype=None):
    import torch

    return torch.from_numpy(fibonacci_ball(n)).to(dtype or torch.float32)


def fibonacci_prefix(n: int, k: int) -> np.ndarray:
    """First ``k`` slots of the n-point ball, re-centered so the live prefix is 0-mean.

    The full 256-ball is centered, but any prefix is not. Partial cells that keep
    the leading k slots therefore start from a biased template unless this
    recentering is applied on both the packer and the decoder.
    """
    k = int(max(0, min(int(k), int(n))))
    pts = fibonacci_ball(int(n))
    if k <= 0:
        return pts[:0]
    live = pts[:k].copy()
    live = live - live.mean(axis=0, keepdims=True)
    return live


def prefix_center_template(pts: np.ndarray, k: int) -> np.ndarray:
    """Return a copy of ``pts`` whose first ``k`` rows are mean-centered."""
    out = np.asarray(pts, dtype=np.float64).copy()
    k = int(max(0, min(int(k), out.shape[0])))
    if k > 0:
        out[:k] = out[:k] - out[:k].mean(axis=0, keepdims=True)
    return out.astype(np.float32)
