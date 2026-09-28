"""How good can 12 channels per 32-point group possibly be?

Trains a plain per-group MLP autoencoder on the *same* offsets the real model has
to code, with the *same* bottleneck width, and no other job to do: no attention,
no neighbours, no attributes, no presence, no latent regularisation, no diffusion
friendliness. Whatever it reaches is a practical upper bound on the codec branch.

If the real model is far above this number the problem is the network / training
loop. If it sits on top of it the problem is the channel budget and no loss
re-weighting will ever help.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_detail import read_ply, pca_bound


def load_groups(eval_dir: str, scenes, group_size: int, branch: str = "codec"):
    out = []
    for sc in scenes:
        mp = os.path.join(eval_dir, f"{sc}_{branch}_metrics.json")
        gp = os.path.join(eval_dir, f"{sc}_{branch}_gt.ply")
        if not os.path.exists(gp):
            continue
        with open(mp) as f:
            met = json.load(f)
        s = met["xyz_rmse_metric"] / met["xyz_rmse_norm"]
        g = read_ply(gp) / s
        n = len(g) // group_size * group_size
        out.append(g[:n].reshape(-1, group_size, 3))
    return np.concatenate(out, 0)


class GroupAE(nn.Module):
    def __init__(self, gsz: int, code: int, width: int = 512, depth: int = 3):
        super().__init__()
        din = gsz * 3

        def stack(a, b):
            layers, prev = [], a
            for _ in range(depth):
                layers += [nn.Linear(prev, width), nn.LayerNorm(width), nn.GELU()]
                prev = width
            layers += [nn.Linear(prev, b)]
            return nn.Sequential(*layers)

        self.enc = stack(din + 1, code)
        self.dec = stack(code + 1, din)

    def forward(self, x, logs):
        z = self.enc(torch.cat([x, logs], -1))
        return self.dec(torch.cat([z, logs], -1)), z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--code", type=int, default=12)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    scenes = sorted({f.split("_codec_gt.ply")[0] for f in os.listdir(a.eval_dir) if f.endswith("_codec_gt.ply")})
    g = load_groups(a.eval_dir, scenes, a.group_size)
    print(f"scenes={len(scenes)} groups={len(g)} group_size={a.group_size} code={a.code}")

    cen = g.mean(1, keepdims=True)
    off = g - cen
    scale = np.maximum(np.abs(off).max(axis=(1, 2), keepdims=True), 1e-6)  # same as decode_scale role
    x = (off / scale).reshape(len(g), -1).astype(np.float32)
    sc = scale.reshape(-1, 1).astype(np.float32)

    perm = np.random.default_rng(0).permutation(len(x))
    ntr = int(0.9 * len(x))
    tr, te = perm[:ntr], perm[ntr:]

    print(f"PCA-{a.code} on normalised offsets, absolute rmse: "
          f"{pca_bound(x[tr][:20000], a.code) * float(np.sqrt((sc[tr]**2).mean())):.6f}")
    print(f"PCA-{a.code} on raw offsets,        absolute rmse: "
          f"{pca_bound(off.reshape(len(g), -1)[tr][:20000], a.code):.6f}")

    dev = torch.device(a.device)
    X = torch.from_numpy(x).to(dev)
    S = torch.from_numpy(sc).to(dev)
    LS = torch.log(S / 0.02) / 3.0
    net = GroupAE(a.group_size, a.code).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    tr_t = torch.from_numpy(tr).to(dev)
    te_t = torch.from_numpy(te).to(dev)
    bs = 4096

    for it in range(a.steps):
        i = tr_t[torch.randint(0, len(tr_t), (bs,), device=dev)]
        xb, sb, lb = X[i], S[i], LS[i]
        rec, _ = net(xb, lb)
        # absolute-units L1, exactly what the codec loss sees
        loss = ((rec - xb) * sb).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % 1500 == 0:
            with torch.no_grad():
                rec, _ = net(X[te_t], LS[te_t])
                d = ((rec - X[te_t]) * S[te_t]).reshape(len(te_t), a.group_size, 3)
                e = d.pow(2).sum(-1).sqrt()
                print(f"  step {it+1:5d}  test offset rmse {d.pow(2).sum(-1).mean().sqrt():.6f}  "
                      f"median {e.median():.6f}  p90 {e.quantile(0.9):.6f}")

    with torch.no_grad():
        rec, _ = net(X[te_t], LS[te_t])
        d = ((rec - X[te_t]) * S[te_t]).reshape(len(te_t), a.group_size, 3)
        e = d.pow(2).sum(-1).sqrt().flatten().cpu().numpy()
    print(f"FINAL oracle  rmse {np.sqrt((e**2).mean()):.6f}  "
          f"p50 {np.percentile(e,50):.6f}  p90 {np.percentile(e,90):.6f}  p99 {np.percentile(e,99):.6f}")


if __name__ == "__main__":
    main()
