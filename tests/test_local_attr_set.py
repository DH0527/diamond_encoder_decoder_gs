"""Local attr_set: identity 0, collapse high, slot permute ~0, not k=1 copy."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from can3tok.losses import intra_group_local_attr_set_loss

LAY = {"scale": 3, "rot": 6, "opacity": 10, "color": 11}


def _batch(seed=0, b=1, gsz=32, ng=4):
    torch.manual_seed(seed)
    n = gsz * ng
    x = torch.randn(b, n, 14)
    x[..., 0:3] = torch.randn(b, n, 3) * 0.05
    x[..., 6:10] = torch.nn.functional.normalize(x[..., 6:10], dim=-1)
    m = torch.ones(b, n)
    return x, m, gsz


def test_identity_is_near_zero():
    tgt, m, gsz = _batch()
    v = intra_group_local_attr_set_loss(tgt, tgt, m, gsz, LAY, k=8)
    assert float(v) < 1e-5, float(v)


def test_collapse_is_large():
    tgt, m, gsz = _batch()
    pr = tgt.clone()
    g = pr.reshape(-1, gsz, 14)
    g[..., 3:6] = g[..., 3:6].mean(1, keepdim=True)
    g[..., 6:10] = torch.nn.functional.normalize(g[..., 6:10].mean(1, keepdim=True), dim=-1)
    pr = g.reshape_as(pr)
    v = intra_group_local_attr_set_loss(pr, tgt, m, gsz, LAY, k=8)
    assert float(v) > 0.15, float(v)


def test_joint_permutation_is_near_zero():
    tgt, m, gsz = _batch()
    pr = tgt.clone().reshape(-1, gsz, 14)
    perm = torch.randperm(gsz)
    pr = pr[:, perm, :].reshape_as(tgt)
    v = intra_group_local_attr_set_loss(pr, tgt, m, gsz, LAY, k=8)
    assert float(v) < 1e-5, float(v)


def test_scale_has_gradient_xyz_does_not():
    tgt, m, gsz = _batch()
    pr = torch.nn.Parameter(tgt + 0.2 * torch.randn_like(tgt))
    v = intra_group_local_attr_set_loss(pr, tgt.detach(), m, gsz, LAY, k=8)
    v.backward()
    assert pr.grad is not None
    assert float(pr.grad[..., 3:6].abs().sum()) > 0.0
    assert float(pr.grad[..., 0:3].abs().sum()) == 0.0


def test_k1_is_not_k8_on_mixed_cell():
    tgt, m, gsz = _batch(seed=1)
    pr = tgt.clone()
    pr[..., 3:6] = pr[..., 3:6].mean(dim=1, keepdim=True)
    a = float(intra_group_local_attr_set_loss(pr, tgt, m, gsz, LAY, k=1))
    b = float(intra_group_local_attr_set_loss(pr, tgt, m, gsz, LAY, k=8))
    assert abs(a - b) > 1e-4, (a, b)


if __name__ == "__main__":
    test_identity_is_near_zero()
    test_collapse_is_large()
    test_joint_permutation_is_near_zero()
    test_scale_has_gradient_xyz_does_not()
    test_k1_is_not_k8_on_mixed_cell()
    print("ok")
