"""Is 16 channels a limit of the latent, or of the decoder that reads it?

Every appearance ceiling quoted in this project -- 13.16, 21.43, 23.06, 25.29 --
comes from the same oracle: attributes are `base + code @ B`, a rank-k affine
subspace shared across groups. That is linear in the code and, crucially, has no
per-point input at all. It maps 16 numbers to 704 numbers and nothing else.

The decoder being trained is neither of those things. `AttributeDecoder` is
nonlinear, and it reads each point's own xyz through a Fourier encoding, so it can
answer "what does a Gaussian *here* look like" rather than having to spend code
capacity saying where the variation goes. The oracle therefore does not bound it;
it bounds a strictly weaker model, and quoting it as "what the latent can carry"
was wrong.

Four decoders, same 16-channel code, same fitted-per-scene generosity:

    linear        base + code @ B                    the old oracle
    mlp           MLP(code) -> 64x11                 nonlinear, still no xyz
    mlp_xyz       MLP(code, xyz_pe) -> 11 per point  what the model actually is
    free          11 numbers per point, unconstrained   upper bound, no code at all

`free` is the ceiling any decoder could reach with these positions -- it has one
parameter per output. The gap between `mlp_xyz` and `free` is what the 16-channel
code costs; the gap between `linear` and `mlp_xyz` is what the old oracle was
mistaking for that cost.
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

from can3tok.io_utils import camera_from_vector  # noqa: E402
from can3tok.layers import FourierFeatures  # noqa: E402
from can3tok.render import (Camera, perturb_camera_vector, photometric_loss,  # noqa: E402
                            psnr, render_gaussians, sh_dc_to_rgb, ssim)
from can3tok.train import make_datasets  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_attr_code_size import A_DIM, rank_k_positions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--shape", type=int, default=12)
    ap.add_argument("--code", type=int, default=16)
    ap.add_argument("--kinds", type=str, default="linear,mlp,mlp_xyz,free")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--net_lr", type=float, default=3e-4)
    ap.add_argument("--fit_views", type=int, default=4)
    ap.add_argument("--test_views", type=int, default=3)
    ap.add_argument("--downscale", type=int, default=2)
    a = ap.parse_args()

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val_ds = make_datasets(targs)
    item = val_ds[a.scene]
    g, dev = int(targs.group_size), "cuda"

    m = item["mask"].to(dev) > 0.5
    t = item["target"].to(dev).float()[m]
    ng = t.shape[0] // g
    t = t[: ng * g]
    sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)
    gt_grp = t[:, 0:3].reshape(ng, g, 3)
    A0 = t[:, 3:14].contiguous()

    cv = item["camera"].numpy()
    rf, rt = np.random.default_rng(0), np.random.default_rng(9999)
    fit = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
    fit += [Camera(camera_from_vector(perturb_camera_vector(cv, rf)), device=dev,
                   downscale=a.downscale) for _ in range(a.fit_views - 1)]
    test = [Camera(camera_from_vector(perturb_camera_vector(cv, rt)), device=dev,
                   downscale=a.downscale) for _ in range(a.test_views)]

    lo = float(A0[:, 0:3].min()) - 0.5
    hi = float(A0[:, 0:3].max()) + 0.5

    def gaussians(xyz, attr):
        return (xyz, torch.exp(attr[:, 0:3].clamp(lo, hi)) * sc, attr[:, 3:7],
                torch.sigmoid(attr[:, 7:8]), sh_dc_to_rgb(attr[:, 8:11]))

    gt_xyz = gt_grp.reshape(-1, 3) * sc + cen
    with torch.no_grad():
        fit_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in fit]
        test_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in test]

    p_norm = rank_k_positions(gt_grp, a.shape)
    xyz_k = (p_norm * sc + cen).detach()
    sd = A0.std(0, keepdim=True).clamp(min=1e-6)
    base_g = A0.reshape(ng, g, A_DIM).mean(1, keepdim=True)          # (ng,1,11)
    k, W = a.code, a.width

    # the decoder sees the positions it is decoding, exactly as the model does
    pe = FourierFeatures(3, int(getattr(targs, "num_freqs_xyz", 6)), include_input=True).to(dev)
    with torch.no_grad():
        loc = (p_norm.reshape(ng, g, 3) - p_norm.reshape(ng, g, 3).mean(1, keepdim=True))
        loc = loc / loc.norm(dim=-1).mean(1, keepdim=True).clamp(min=1e-6)[..., None]
        PE = pe(loc)                                                  # (ng,g,pe)

    print(f"scene {item['name']}  groups {ng}  shape {a.shape} / appearance {a.code}")
    print(f"  fit {len(fit)} views, held-out {len(test)} views\n")
    print(f"{'decoder':>10} {'params':>10} {'held-out dB':>12} {'SSIM':>7}")

    for kind in [s for s in a.kinds.split(",") if s]:
        torch.manual_seed(0)
        C = torch.zeros(ng, k, device=dev, requires_grad=True)
        net, extra = None, []
        if kind == "linear":
            B = torch.empty(k, g * A_DIM, device=dev)
            nn.init.normal_(B, std=1.0 / (g * A_DIM) ** 0.5)
            B.requires_grad_(True)
            extra = [B]
        elif kind == "mlp":
            net = nn.Sequential(nn.Linear(k, W), nn.SiLU(), nn.Linear(W, W), nn.SiLU(),
                                nn.Linear(W, g * A_DIM)).to(dev)
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
        elif kind == "mlp_xyz":
            net = nn.Sequential(nn.Linear(k + PE.shape[-1], W), nn.SiLU(),
                                nn.Linear(W, W), nn.SiLU(), nn.Linear(W, A_DIM)).to(dev)
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
        elif kind == "free":
            F_ = torch.zeros(ng, g, A_DIM, device=dev, requires_grad=True)
            extra = [F_]
        else:
            continue

        groups = [{"params": [C] + extra, "lr": a.lr}]
        if net is not None:
            groups.append({"params": list(net.parameters()), "lr": a.net_lr})
        opt = torch.optim.Adam(groups)

        def build():
            if kind == "linear":
                return (base_g + (C @ extra[0]).reshape(ng, g, A_DIM) * sd).reshape(-1, A_DIM)
            if kind == "mlp":
                return (base_g + net(C).reshape(ng, g, A_DIM) * sd).reshape(-1, A_DIM)
            if kind == "mlp_xyz":
                h = torch.cat([C[:, None].expand(ng, g, k), PE], -1)
                return (base_g + net(h) * sd).reshape(-1, A_DIM)
            return (base_g + extra[0] * sd).reshape(-1, A_DIM)

        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            attr = build()
            loss = 0.0
            for r, c in zip(fit_ref, fit):
                l, _, _ = photometric_loss(render_gaussians(*gaussians(xyz_k, attr), c), r, 0.2)
                loss = loss + l / len(fit)
            loss.backward()
            if net is not None:
                torch.nn.utils.clip_grad_norm_(list(net.parameters()), 1.0)
            opt.step()

        with torch.no_grad():
            attr = build()
            p = float(np.mean([psnr(r, render_gaussians(*gaussians(xyz_k, attr), c))
                               for r, c in zip(test_ref, test)]))
            ss = float(np.mean([ssim(r, render_gaussians(*gaussians(xyz_k, attr), c))
                                for r, c in zip(test_ref, test)]))
        npar = sum(x.numel() for x in extra) + (
            sum(x.numel() for x in net.parameters()) if net is not None else 0)
        print(f"{kind:>10} {npar/1e6:9.2f}M {p:12.2f} {ss:7.3f}")

    print("\n모든 행이 같은 16채널 코드를 씁니다. 차이는 그 코드를 읽는 함수뿐입니다.")
    print("'free' 는 코드 없이 점마다 11개 값을 자유롭게 둔 상한입니다.")


if __name__ == "__main__":
    main()
