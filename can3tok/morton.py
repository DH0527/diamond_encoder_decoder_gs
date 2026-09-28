"""Morton (Z-order) utilities for 3D points and 2D latent grids.

Two distinct uses:
  * ``morton3d_np``  : orders points so that spatially close Gaussians land in the
    same patch/group. Same convention as the original CoordFirstGSAE pipeline.
  * ``grid_zorder_permutation`` : maps a *sequential* cell index (cells are already
    ordered by 3D Morton code) onto a 2D grid position so that 2D neighbourhoods
    approximate 3D neighbourhoods. Row-major flattening breaks this: vertical
    neighbours are ``W`` cells apart in Morton order. A latent map whose 2D
    locality matches 3D locality is much easier for a conv/UNet diffusion model
    to denoise (see "Improving the Diffusability of Autoencoders", 2502.14831).
"""

from __future__ import annotations

import numpy as np
import torch


def _part1by2_np(n: np.ndarray) -> np.ndarray:
    """Insert two zero bits before each of the 21 low bits of n."""
    n = n.astype(np.uint64) & np.uint64(0x1FFFFF)
    n = (n | (n << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    n = (n | (n << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    n = (n | (n << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    n = (n | (n << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    n = (n | (n << np.uint64(2))) & np.uint64(0x1249249249249249)
    return n


def morton3d_np(xyz_norm: np.ndarray, bits: int = 10) -> np.ndarray:
    """Morton code for xyz normalized to roughly [-1, 1]."""
    scale = float(int(1) << int(bits))
    q = np.clip((xyz_norm + 1.0) * 0.5, 0.0, 1.0) * (scale - 1.0)
    q = np.floor(q).astype(np.uint64)
    return (
        _part1by2_np(q[:, 0])
        | (_part1by2_np(q[:, 1]) << np.uint64(1))
        | (_part1by2_np(q[:, 2]) << np.uint64(2))
    )


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def grid_zorder_permutation(h: int, w: int) -> torch.Tensor:
    """Sequential cell index -> flat index in an (h, w) grid, Z-order interleaved.

    Returns a LongTensor ``perm`` of length ``h * w`` where ``perm[i]`` is the
    row-major position in the grid that sequential cell ``i`` should occupy.

    Bits of ``i`` are handed out alternately to the x and y axes (starting with
    x). When one axis runs out of bits the remaining bits go to the other axis,
    so non-square grids such as 256x128 are handled as well. Falls back to the
    identity permutation when the grid is not a power of two, which keeps the
    code safe for arbitrary configs.
    """
    h = int(h)
    w = int(w)
    cells = h * w
    if not (_is_pow2(h) and _is_pow2(w)):
        return torch.arange(cells, dtype=torch.long)

    bits_x = int(np.log2(w))
    bits_y = int(np.log2(h))
    idx = np.arange(cells, dtype=np.int64)
    x = np.zeros_like(idx)
    y = np.zeros_like(idx)
    bx = 0
    by = 0
    bit = 0
    turn_x = True
    while bx < bits_x or by < bits_y:
        take_x = turn_x and bx < bits_x
        if not take_x and by >= bits_y:
            take_x = True
        if take_x:
            x |= ((idx >> bit) & 1) << bx
            bx += 1
        else:
            y |= ((idx >> bit) & 1) << by
            by += 1
        bit += 1
        turn_x = not turn_x
    flat = y * w + x
    return torch.from_numpy(flat).long()


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv


def kd_equal_groups(xyz: np.ndarray, num_groups: int) -> list:
    """Balanced kd-tree split of ``xyz`` into ``num_groups`` index arrays.

    Group sizes differ by at most 1. Returns a list of length ``num_groups``;
    each entry is an ``np.ndarray`` of indices into ``xyz``.
    """
    n = int(xyz.shape[0])
    ng = int(num_groups)
    if ng <= 0:
        raise ValueError("num_groups must be positive")
    if n == 0:
        return [np.zeros(0, dtype=np.int64) for _ in range(ng)]
    # Target sizes: first (n % ng) groups get floor(n/ng)+1.
    base, rem = divmod(n, ng)
    targets = [base + (1 if i < rem else 0) for i in range(ng)]

    out: list = [None] * ng  # type: ignore[list-item]
    # stack entries: (index_array_into_xyz, group_id_start, n_groups_in_this_node)
    stack = [(np.arange(n, dtype=np.int64), 0, ng)]
    while stack:
        idx, g0, ng_here = stack.pop()
        if ng_here == 1:
            out[g0] = idx
            continue
        # Split group count as evenly as possible; match point counts to targets.
        left_g = ng_here // 2
        right_g = ng_here - left_g
        n_left = int(sum(targets[g0 : g0 + left_g]))
        if n_left <= 0:
            out[g0] = idx[:0]
            stack.append((idx, g0 + left_g, right_g))
            continue
        if n_left >= len(idx):
            stack.append((idx, g0, left_g))
            for j in range(g0 + left_g, g0 + ng_here):
                out[j] = idx[:0]
            continue
        pts = xyz[idx]
        ax = int(np.argmax(pts.max(0) - pts.min(0)))
        order = np.argpartition(pts[:, ax], n_left)
        left_idx = idx[order[:n_left]]
        right_idx = idx[order[n_left:]]
        # Stable-ish: sort each side along the split axis for locality.
        left_idx = left_idx[np.argsort(xyz[left_idx, ax], kind="stable")]
        right_idx = right_idx[np.argsort(xyz[right_idx, ax], kind="stable")]
        stack.append((right_idx, g0 + left_g, right_g))
        stack.append((left_idx, g0, left_g))
    for i, g in enumerate(out):
        if g is None:
            out[i] = np.zeros(0, dtype=np.int64)
    return out


def morton_equal_groups(n: int, num_groups: int) -> list:
    """Contiguous equal-count split of a length-``n`` Morton-ordered sequence."""
    ng = int(num_groups)
    base, rem = divmod(int(n), ng)
    out = []
    offset = 0
    for g in range(ng):
        sz = base + (1 if g < rem else 0)
        out.append(np.arange(offset, offset + sz, dtype=np.int64))
        offset += sz
    return out
