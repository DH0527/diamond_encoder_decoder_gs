"""Same 12-channel budget, different objective.

The codec is trained to put GT point *i* into slot *i*. Under a bottleneck that
target is unreachable, so the L1/L2 optimum is the conditional mean and the
group contracts -- which is what "the shape is right but the detail is mush"
looks like. A set objective has no such optimum: it is minimised by putting the
points *somewhere on the right structure*.

This trains the identical autoencoder with three objectives and reports the
metrics that actually track perceived detail (within-group chamfer, spread
ratio), not just the metric the loss happens to optimise.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_bottleneck import GroupAE, load_groups


def group_chamfer(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """(B, G, 3) vs (B, G, 3) symmetric chamfer, mean L2."""
    d = torch.cdist(p, g)
    return 0.5 * (d.min(2).values.mean() + d.min(1).values.mean())


def evaluate(net, X, S, LS, idx, gsz):
    with torch.no_grad():
        rec, _ = net(X[idx], LS[idx])
        p = (rec.reshape(-1, gsz, 3)) * S[idx].unsqueeze(-1)
        t = (X[idx].reshape(-1, gsz, 3)) * S[idx].unsqueeze(-1)
        rmse = float((p - t).pow(2).sum(-1).mean().sqrt())
        d = torch.cdist(p, t)
        ch = float(0.5 * (d.min(2).values.mean() + d.min(1).values.mean()))
        sp = float((p.std(dim=(1, 2)) / t.std(dim=(1, 2)).clamp(min=1e-9)).median())
        # nn spacing inside the group: does it clump?
        dd = torch.cdist(p, p) + torch.eye(gsz, device=p.device) * 1e3
        nnp = float(dd.min(-1).values.median())
        dt = torch.cdist(t, t) + torch.eye(gsz, device=t.device) * 1e3
        nnt = float(dt.min(-1).values.median())
    return rmse, ch, sp, nnp, nnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--code", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    scenes = sorted({f.split("_codec_gt.ply")[0] for f in os.listdir(a.eval_dir) if f.endswith("_codec_gt.ply")})
    g = load_groups(a.eval_dir, scenes, a.group_size)
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
    gsz = a.group_size

    print(f"groups={len(g)} code={a.code} group_size={gsz}")
    print(f"{'objective':<22}{'pointwise rmse':>16}{'group chamfer':>16}{'spread p50':>13}"
          f"{'nn pred':>11}{'nn gt':>11}")

    for name in ("pointwise L1", "group chamfer", "L1 + chamfer"):
        torch.manual_seed(0)
        net = GroupAE(gsz, a.code).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        for it in range(a.steps):
            i = tr[torch.randint(0, len(tr), (2048,), device=dev)]
            rec, _ = net(X[i], LS[i])
            p = rec.reshape(-1, gsz, 3) * S[i].unsqueeze(-1)
            t = X[i].reshape(-1, gsz, 3) * S[i].unsqueeze(-1)
            if name == "pointwise L1":
                loss = (p - t).abs().mean()
            elif name == "group chamfer":
                loss = group_chamfer(p, t)
            else:
                loss = (p - t).abs().mean() + group_chamfer(p, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        r, c, s, np_, nt = evaluate(net, X, S, LS, te, gsz)
        print(f"{name:<22}{r:>16.6f}{c:>16.6f}{s:>13.3f}{np_:>11.6f}{nt:>11.6f}")


if __name__ == "__main__":
    main()
