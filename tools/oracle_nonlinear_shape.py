"""Can a nonlinear decoder beat the linear bound at the same 28-channel budget?

`oracle_shape_rank.py` put the model exactly on the rank-28 *linear* ceiling
(nn_unique 0.529, ich 0.236). Two readings of that, with opposite consequences:

  * the budget is the wall, geometry is finished; or
  * the budget's *linear* projection is the wall, the decoder happens to be doing
    no better than linear, and a nonlinear map from the same 28 numbers could do
    more.

This settles it by training the smallest possible thing that could beat PCA: an
MLP autoencoder straight on the 192-dimensional group-offset vectors with a
28-dimensional bottleneck. No attention, no context, no latent grid -- just the
question "how much of a 64-point group survives 28 numbers".

Two corrections to the earlier measurement are folded in:

  * **the PCA number was in-sample.** It was fit on the same scene's 4096 groups
    it was then scored on, so 0.529 is optimistic. Here both methods are fit on
    train scenes and scored on held-out scenes.
  * the comparison uses the identical held-out set and the identical metric, so
    the difference is the nonlinearity and nothing else.

If the autoencoder ties PCA, geometry at this budget is closed and the remaining
work is all in attributes. If it wins clearly, the decoder architecture is leaving
something on the table and the shape->offset map is worth revisiting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.train import make_datasets  # noqa: E402


def offsets_of(item, g: int, dev: str):
    """Radius-normalised, centred group offsets: exactly what the shape code must emit."""
    m = item["mask"].to(dev) > 0.5
    xyz = item["target"].to(dev).float()[m][:, 0:3]
    ng = xyz.shape[0] // g
    grp = xyz[: ng * g].reshape(ng, g, 3)
    off = grp - grp.mean(1, keepdim=True)
    rad = off.norm(dim=-1).mean(-1).clamp(min=1e-9)
    return (off / rad[:, None, None]).reshape(ng, g * 3)


@torch.no_grad()
def measure(pred: torch.Tensor, true: torch.Tensor, g: int, chunk: int = 512):
    p = pred.reshape(-1, g, 3)
    t = true.reshape(-1, g, 3)
    p = p - p.mean(1, keepdim=True)
    rad = t.norm(dim=-1).mean(-1).clamp(min=1e-9)
    uniq, ich = [], []
    for i in range(0, p.shape[0], chunk):
        d = torch.cdist(p[i : i + chunk], t[i : i + chunk])
        nn_ = d.argmin(dim=2)
        hit = torch.zeros_like(nn_, dtype=torch.bool).scatter_(1, nn_, True)
        uniq.append(hit.float().mean(1))
        ich.append(0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1)) / rad[i : i + chunk])
    return float(torch.cat(uniq).mean()), float(torch.cat(ich).median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--train_scenes", type=str, default="40,63,74,120,138")
    ap.add_argument("--test_scenes", type=str, default="160,200,240")
    ap.add_argument("--dim", type=int, default=28, help="bottleneck = the shape channel count")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val_ds = make_datasets(targs)
    g, dev = int(targs.group_size), "cuda"
    D = g * 3

    tr = torch.cat([offsets_of(val_ds[int(i)], g, dev)
                    for i in a.train_scenes.split(",") if i], dim=0)
    te = torch.cat([offsets_of(val_ds[int(i)], g, dev)
                    for i in a.test_scenes.split(",") if i], dim=0)
    print(f"offset dim {D}  bottleneck {a.dim}  train groups {tr.shape[0]}  "
          f"held-out groups {te.shape[0]}")

    # --- linear baseline, fit on train only (the earlier 0.529 was in-sample) ---
    mu = tr.mean(0, keepdim=True).double()
    _, _, vh = torch.linalg.svd((tr.double() - mu), full_matrices=False)
    b = vh[: a.dim]
    lin = (((te.double() - mu) @ b.T @ b) + mu).float()
    u_l, i_l = measure(lin, te, g)

    mu_f, b_f = mu.float(), b.float()
    lin_tr = (((tr - mu_f) @ b_f.T @ b_f) + mu_f)
    u_lt, i_lt = measure(lin_tr, tr, g)

    # --- nonlinear autoencoder, same bottleneck ---
    h = a.hidden
    net = nn.Sequential(
        nn.Linear(D, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, a.dim),
        nn.Linear(a.dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, D),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.02)
    gen = torch.Generator(device=dev).manual_seed(0)
    for it in range(a.steps):
        idx = torch.randint(0, tr.shape[0], (a.batch,), generator=gen, device=dev)
        x = tr[idx]
        opt.zero_grad(set_to_none=True)
        loss = (net(x) - x).abs().mean()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % max(a.steps // 4, 1) == 0:
            net.eval()
            with torch.no_grad():
                u, i = measure(net(te), te, g)
            net.train()
            print(f"  step {it+1:5d}  train L1 {float(loss):.5f}   "
                  f"held-out nn_unique {u:.3f}  ich {i:.4f}")
    net.eval()
    with torch.no_grad():
        u_n, i_n = measure(net(te), te, g)
        u_nt, i_nt = measure(net(tr), tr, g)

    print(f"\n{'method':>28} {'nn_unique':>10} {'ich':>8}   set")
    print(f"{'PCA-' + str(a.dim):>28} {u_lt:10.3f} {i_lt:8.4f}   train (in-sample)")
    print(f"{'PCA-' + str(a.dim):>28} {u_l:10.3f} {i_l:8.4f}   HELD-OUT")
    print(f"{'MLP autoencoder-' + str(a.dim):>28} {u_nt:10.3f} {i_nt:8.4f}   train (in-sample)")
    print(f"{'MLP autoencoder-' + str(a.dim):>28} {u_n:10.3f} {i_n:8.4f}   HELD-OUT")
    print(f"\nnonlinearity buys {u_n - u_l:+.3f} nn_unique and {i_n - i_l:+.4f} ich on "
          f"held-out groups.")
    print("model today (whole pipeline, 28 shape channels): nn_unique 0.51  ich 0.251")


if __name__ == "__main__":
    main()
