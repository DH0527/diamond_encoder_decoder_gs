"""Hungarian-matched scale/opacity L1: identity 0, collapse high, permute ~0."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from can3tok.losses import intra_group_hungarian_attr_loss

LAY = {"scale": 3, "rot": 6, "opacity": 10, "color": 11}


def _batch(seed=0, b=1, gsz=32, ng=4):
    torch.manual_seed(seed)
    n = gsz * ng
    x = torch.randn(b, n, 14)
    x[..., 6:10] = torch.nn.functional.normalize(x[..., 6:10], dim=-1)
    m = torch.ones(b, n)
    return x, m, gsz


def test_identity_is_zero():
    tgt, m, gsz = _batch()
    sc, op, rt = intra_group_hungarian_attr_loss(tgt, tgt, m, gsz, LAY)
    assert float(sc) < 1e-6 and float(op) < 1e-6 and float(rt) < 1e-6


def test_collapse_is_large():
    tgt, m, gsz = _batch()
    pr = tgt.clone()
    g = pr.reshape(-1, gsz, 14)
    g[..., 3:6] = g[..., 3:6].mean(1, keepdim=True)
    g[..., 10:11] = g[..., 10:11].mean(1, keepdim=True)
    pr = g.reshape_as(pr)
    sc, op, _ = intra_group_hungarian_attr_loss(pr, tgt, m, gsz, LAY)
    assert float(sc) > 0.2 and float(op) > 0.05


def test_joint_permutation_is_near_zero():
    tgt, m, gsz = _batch()
    pr = tgt.clone().reshape(-1, gsz, 14)
    perm = torch.randperm(gsz)
    pr = pr[:, perm, :].reshape_as(tgt)
    sc, op, rt = intra_group_hungarian_attr_loss(pr, tgt, m, gsz, LAY)
    assert float(sc) < 1e-5 and float(op) < 1e-5 and float(rt) < 1e-5


def test_scale_has_gradient():
    tgt, m, gsz = _batch()
    pr = torch.nn.Parameter(tgt + 0.3 * torch.randn_like(tgt))
    sc, op, rt = intra_group_hungarian_attr_loss(pr, tgt.detach(), m, gsz, LAY)
    (sc + op + rt).backward()
    assert pr.grad is not None
    assert float(pr.grad[..., 3:6].abs().sum()) > 0.0


if __name__ == "__main__":
    test_identity_is_zero()
    test_collapse_is_large()
    test_joint_permutation_is_near_zero()
    test_scale_has_gradient()
    print("ok")
