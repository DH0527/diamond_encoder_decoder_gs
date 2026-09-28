"""What does equivariance_loss cost?

losses.equivariance_loss asks the *shape* channels to be unchanged when the whole
cloud is rotated. Within-group offsets rotate with the cloud, so satisfying that
means the shape code has to be rotation invariant, and the shape_xyz decoder has
no rotation input to put the orientation back. This reproduces the constraint on
the oracle autoencoder and reports what it costs at the same code width.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_bottleneck import GroupAE, load_groups


def rot(deg: float, dev) -> torch.Tensor:
    a = np.deg2rad(deg)
    ax = np.random.default_rng().normal(size=3)
    ax /= np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    return torch.tensor(R, dtype=torch.float32, device=dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--code", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    gsz = 32

    scenes = sorted({f.split("_codec_gt.ply")[0] for f in os.listdir(a.eval_dir) if f.endswith("_codec_gt.ply")})
    g = load_groups(a.eval_dir, scenes, gsz)
    off = g - g.mean(1, keepdims=True)
    scale = np.maximum(np.abs(off).max(axis=(1, 2), keepdims=True), 1e-6)
    x = (off / scale).reshape(len(g), -1).astype(np.float32)
    sc = scale.reshape(-1, 1).astype(np.float32)

    dev = torch.device(a.device)
    X, S = torch.from_numpy(x).to(dev), torch.from_numpy(sc).to(dev)
    LS = torch.log(S / 0.02) / 3.0
    perm = np.random.default_rng(0).permutation(len(x))
    tr = torch.from_numpy(perm[: int(0.9 * len(x))]).to(dev)
    te = torch.from_numpy(perm[int(0.9 * len(x)) :][:8000]).to(dev)

    print(f"code={a.code}  groups={len(g)}")
    for w_inv in (0.0, 0.25, 1.0):
        torch.manual_seed(0)
        net = GroupAE(gsz, a.code).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        for it in range(a.steps):
            i = tr[torch.randint(0, len(tr), (2048,), device=dev)]
            xb, sb, lb = X[i], S[i], LS[i]
            rec, z = net(xb, lb)
            loss = ((rec - xb) * sb).abs().mean()
            if w_inv > 0 and it % 4 == 0:
                R = rot(10.0, dev)
                xr = (xb.reshape(-1, gsz, 3) @ R.T).reshape(-1, gsz * 3)
                _, zr = net(xr, lb)
                loss = loss + w_inv * (zr - z).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            rec, z = net(X[te], LS[te])
            d = ((rec - X[te]) * S[te]).reshape(-1, gsz, 3)
            e = d.pow(2).sum(-1).sqrt()
            R = rot(10.0, dev)
            xr = (X[te].reshape(-1, gsz, 3) @ R.T).reshape(-1, gsz * 3)
            _, zr = net(xr, LS[te])
            inv = float((zr - z).abs().mean() / z.abs().mean())
        print(f"  w_shape_invariance={w_inv:4.2f}  rmse {float(d.pow(2).sum(-1).mean().sqrt()):.6f}  "
              f"p50 {float(e.median()):.6f}  p90 {float(e.quantile(0.9)):.6f}  "
              f"code std {float(z.std()):.3f}  rel code change under 10deg {inv:.3f}")


if __name__ == "__main__":
    main()
