"""Where does the reconstruction stop being one-to-one?

Rendering the best checkpoint showed 0.540 distinct nearest-GT points per
prediction against 0.991 for the GT subset itself: 46% of the point budget lands
on a surface patch another predicted point already covers, and the corresponding
patches go empty. That single number explains most of the gap between the
point-space score (1.23x spacing, respectable) and the render (17.9 dB, poor),
because a symmetric chamfer is only weakly sensitive to many-to-one
correspondence -- the duplicate scores a perfect p->g and the uncovered GT point
is paid for once, averaged over 64 slots.

The decoder builds its output in stages, and only some of them are supervised
one-to-one:

    z_raw_hat --unpack--> coarse --refine(10 layers)--> xyz

``w_z_residual`` (weight 40) pins the *packed* representation slot-by-slot
against the template-aligned target. Everything after it -- the identity unpack,
the folding path and the refine stack -- is supervised only by set losses
(``w_chamfer``, ``w_intra_chamfer``), whose sensitivity to many-to-one
correspondence is far too weak to hold it.
``w_xyz_residual``, the extent-normalised one-to-one term that *would* see it,
is set to 0.

So the question is which stage loses injectivity. Measuring the coarse output
(refine gated off) against the full output separates "the pack already
duplicates" from "the refine stack collapses it".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.model import build_model  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402


def stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, g: int):
    """Per-group distinct-NN fraction and the intra-group chamfer beside it."""
    m = mask > 0.5
    p, t = pred[m][:, 0:3].float(), target[m][:, 0:3].float()
    ng = p.shape[0] // g
    p, t = p[: ng * g].reshape(ng, g, 3), t[: ng * g].reshape(ng, g, 3)
    op, ot = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    rad = ot.norm(dim=-1).mean(-1).clamp(min=1e-9)
    uniq, ich = [], []
    for i in range(0, ng, 512):
        d = torch.cdist(op[i : i + 512], ot[i : i + 512])
        nn = d.argmin(dim=2)
        hit = torch.zeros_like(nn, dtype=torch.bool).scatter_(1, nn, True)
        uniq.append(hit.float().mean(1))
        ich.append(0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1)) / rad[i : i + 512])
    return float(torch.cat(uniq).mean()), float(torch.cat(ich).median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--indices", default="40,120,200")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    targs = Namespace(**ck["args"])
    targs.out_dir = os.path.dirname(os.path.abspath(a.ckpt))
    train_ds, val_ds = make_datasets(targs)
    cfg = build_config(targs, train_ds.target_dim, train_ds.sh_dim)
    model = build_model(cfg, allow_padded_tokens=getattr(targs, "allow_padded_tokens", False))
    model.load_state_dict(ck["model"])
    model.to(a.device).eval()
    g = int(cfg.group_size)
    print(f"ckpt {a.ckpt}  step {ck.get('step')}  group_size {g}")
    print(f"{'scene':>10}  {'stage':<22} {'nn_unique':>10} {'intra_chamfer':>14}")

    agg = {}
    for i in [int(x) for x in a.indices.split(",") if x != ""]:
        item = val_ds[i]
        x = item["input"].unsqueeze(0).to(a.device).float()
        m = item["mask"].unsqueeze(0).to(a.device).float()
        tgt = item["target"].unsqueeze(0).to(a.device).float()[..., 0:3]
        name = os.path.splitext(item["name"])[0]

        rows = []
        for label, refine in (("coarse (refine off)", 0.0), ("full (refine on)", 1.0)):
            model.cfg.decoder_refine_alpha = refine
            with torch.no_grad():
                out = model(x, m, run_decode=True, run_gen=False, gen_noise_std=0.0)
            rows.append((label, out["pred"][0], m[0]))
            if "gen_pred" in out and refine == 1.0:
                rows.append(("gen", out["gen_pred"][0], m[0]))
        # The template-aligned target itself: the ceiling this decomposition can
        # reach, and the control for the metric.
        rows.append(("target (control)", tgt[0], m[0]))

        for label, pr, mk in rows:
            u, c = stats(pr, tgt[0], mk, g)
            agg.setdefault(label, []).append((u, c))
            print(f"{name:>10}  {label:<22} {u:10.3f} {c:14.3f}")

    print(f"\n{'mean':>10}  {'stage':<22} {'nn_unique':>10} {'intra_chamfer':>14}")
    for label, vals in agg.items():
        u = float(np.mean([v[0] for v in vals]))
        c = float(np.mean([v[1] for v in vals]))
        print(f"{'':>10}  {label:<22} {u:10.3f} {c:14.3f}")


if __name__ == "__main__":
    main()
