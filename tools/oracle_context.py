"""Is appearance short of information, or short of context?

Every appearance number this project has produced assumes each group is decoded
from its own code alone. `oracle_attr_code_size.py` fits one code per group,
`oracle_budget_split.py` does the same, and the model matches them: the
decompressor publishes `appearance` as a raw slice of z_compact
(`pg[..., a0:a0+c_app]`), deliberately bypassing the window attention that the
geometry path gets two layers of. So the attribute decoder's receptive field is
one group -- 64 points -- and every measurement so far inherited that.

That makes "16 channels is not enough information" an unproven claim. A group is
not independent of its neighbours: a wall spans dozens of them, and its colour is
the same in all. An image VAE reaches 48:1 precisely because its decoder is
convolutional and each output pixel sees a wide neighbourhood of latent. If the
same holds here, the missing capacity is not in the latent at all -- it is in the
decoder's access to it.

This measures the difference and nothing else. Same budget, same fitted rank-k
code per group, same everything -- except that before decoding, each group's code
is replaced by a learned mixture of the codes in its neighbourhood on the 64x64
latent grid. Groups are laid out in z-order, so grid neighbours are spatial
neighbours.

    none    code_g decoded alone                    (what the model does today)
    3x3     one conv layer over the code grid       (9 groups, 576 points)
    5x5     two layers                              (25 groups)
    9x9     four layers                             (81 groups)

If the numbers barely move, the latent really is the bottleneck and the honest
answer is to spend channels. If they move a lot, the channels were always
sufficient and the decoder was reading them through a keyhole.
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
from can3tok.render import (Camera, perturb_camera_vector, photometric_loss,  # noqa: E402
                            psnr, render_gaussians, sh_dc_to_rgb, ssim)
from can3tok.train import make_datasets  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_attr_code_size import A_DIM, rank_k_positions  # noqa: E402


class CodeMixer(nn.Module):
    """`layers` 3x3 convolutions over the (k, H, W) grid of per-group codes."""

    def __init__(self, k: int, layers: int):
        super().__init__()
        self.layers = layers
        self.net = nn.Sequential(*[
            m for i in range(layers)
            for m in ((nn.Conv2d(k, k, 3, padding=1),) if i == layers - 1
                      else (nn.Conv2d(k, k, 3, padding=1), nn.SiLU()))
        ]) if layers > 0 else None
        if self.net is not None:
            for m in self.net:
                if isinstance(m, nn.Conv2d):       # start at identity
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                    with torch.no_grad():
                        m.weight[:, :, 1, 1] = torch.eye(k)

    def forward(self, c, hw, ng):
        if self.net is None:
            return c
        h, w = hw
        g = c.new_zeros(h * w, c.shape[1])
        g[:ng] = c
        g = g.T.reshape(1, -1, h, w)
        return self.net(g).reshape(c.shape[1], h * w).T[:ng]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--shape", type=int, default=12)
    ap.add_argument("--code", type=int, default=16)
    ap.add_argument("--contexts", type=str, default="0,1,2,4",
                    help="conv layers; 0=none, 1=3x3, 2=5x5, 4=9x9")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mix_lr", type=float, default=1e-3)
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
    hw = (int(targs.latent_hw[0]) if targs.latent_hw[0] else 64, 64)
    hw = (64, 64)

    cv = item["camera"].numpy()
    rf, rt = np.random.default_rng(0), np.random.default_rng(9999)
    fit = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
    fit += [Camera(camera_from_vector(perturb_camera_vector(cv, rf)), device=dev,
                   downscale=a.downscale) for _ in range(a.fit_views - 1)]
    test = [Camera(camera_from_vector(perturb_camera_vector(cv, rt)), device=dev,
                   downscale=a.downscale) for _ in range(a.test_views)]

    # Bounded to the data's own log-scale range. Left free, the fit grows a splat
    # to cover a hole -- it is the cheapest way to reduce a photometric loss -- and
    # the rasteriser dies with an illegal memory access before the run finishes.
    lo = float(A0[:, 0:3].min()) - 0.5
    hi = float(A0[:, 0:3].max()) + 0.5

    def gaussians(xyz, attr):
        return (xyz, torch.exp(attr[:, 0:3].clamp(lo, hi)) * sc, attr[:, 3:7],
                torch.sigmoid(attr[:, 7:8]), sh_dc_to_rgb(attr[:, 8:11]))

    gt_xyz = gt_grp.reshape(-1, 3) * sc + cen
    with torch.no_grad():
        fit_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in fit]
        test_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in test]
    xyz_k = (rank_k_positions(gt_grp, a.shape) * sc + cen).detach()
    sd = A0.std(0, keepdim=True).clamp(min=1e-6)
    sd_g = sd.repeat(1, g).reshape(1, g * A_DIM)
    base = A0.reshape(ng, g, A_DIM).mean(1, keepdim=True).expand(ng, g, A_DIM)
    base = base.reshape(ng, g * A_DIM).clone()

    print(f"scene {item['name']}  groups {ng}  shape {a.shape} / appearance {a.code}")
    print(f"  fit {len(fit)} views, held-out {len(test)} views. "
          f"코드 예산은 모든 행에서 동일합니다 -- 바뀌는 것은 수용 영역뿐\n")
    print(f"{'context':>10} {'groups seen':>12} {'points seen':>12} "
          f"{'held-out dB':>12} {'SSIM':>7}")

    k = a.code
    for L in [int(x) for x in a.contexts.split(",") if x != ""]:
        torch.manual_seed(0)
        B = torch.empty(k, g * A_DIM, device=dev)
        nn.init.normal_(B, std=1.0 / (g * A_DIM) ** 0.5)
        C = torch.zeros(ng, k, device=dev)
        B.requires_grad_(True)
        C.requires_grad_(True)
        mix = CodeMixer(k, L).to(dev)
        # The mixer needs its own, much smaller step. It is initialised to the
        # identity so every row starts exactly at the `none` baseline; at the code
        # vectors' learning rate its 2304-weight convolutions diverge instead of
        # departing from that baseline (measured: 23.02 -> 4.85 dB, SSIM 0.09).
        opt = torch.optim.Adam(
            [{"params": [B, C], "lr": a.lr},
             {"params": list(mix.parameters()), "lr": a.mix_lr}])
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            attr = (base + (mix(C, hw, ng) @ B) * sd_g).reshape(-1, A_DIM)
            loss = 0.0
            for r, c in zip(fit_ref, fit):
                l, _, _ = photometric_loss(render_gaussians(*gaussians(xyz_k, attr), c), r, 0.2)
                loss = loss + l / len(fit)
            loss.backward()
            if L > 0:
                torch.nn.utils.clip_grad_norm_(list(mix.parameters()), 1.0)
            opt.step()
        with torch.no_grad():
            attr = (base + (mix(C, hw, ng) @ B) * sd_g).reshape(-1, A_DIM)
            p = float(np.mean([psnr(r, render_gaussians(*gaussians(xyz_k, attr), c))
                               for r, c in zip(test_ref, test)]))
            ss = float(np.mean([ssim(r, render_gaussians(*gaussians(xyz_k, attr), c))
                                for r, c in zip(test_ref, test)]))
        side = 2 * L + 1
        seen = side * side
        tag = "  <- 현재 구조" if L == 0 else ""
        print(f"{('none' if L == 0 else f'{side}x{side}'):>10} {seen:12d} "
              f"{seen*g:12d} {p:12.2f} {ss:7.3f}{tag}")

    print("\n같은 채널 수인데 수치가 오르면, 부족했던 것은 정보가 아니라")
    print("디코더가 그 정보에 닿는 범위입니다.")


if __name__ == "__main__":
    main()
