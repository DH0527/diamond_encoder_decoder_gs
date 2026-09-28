"""Which predicted attribute is destroying the render?

`codec_psnr_own` is 14.94 dB against 17.35 for the same geometry with GT
attributes, and the image is washed out rather than merely blurred -- that is a
signature, not generic error. Washed out means one of three things: opacity too
low so the background shows through, scale too large so every Gaussian smears,
or colour collapsed toward grey.

Aggregate metrics cannot separate those. This substitutes GT for one attribute
at a time, keeping everything else as the model predicted, and reads the PSNR
back. The attribute whose substitution recovers the most is the one that is
broken, and the distribution table underneath says in which direction.

Run against the same held-out scenes and the same camera the render comparison
uses, so the numbers line up with `render_compare.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.io_utils import load_npz_state, load_replay_dict  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.render import Camera, psnr, render_gaussians, sh_dc_to_rgb, ssim  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402

# name -> (slice in the 14-channel target, how the rasteriser wants it)
PARTS = (("scale", slice(3, 6)), ("rot", slice(6, 10)),
         ("opacity", slice(10, 11)), ("color", slice(11, 14)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--indices", default="160,240,183")
    ap.add_argument("--downscale", type=int, default=2)
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
    model.cfg.decoder_refine_alpha = 1.0
    dev = a.device
    print(f"ckpt {a.ckpt}  step {ck.get('step')}\n")

    rows, dist = {}, {n: [[], []] for n, _ in PARTS}
    for i in [int(x) for x in a.indices.split(",") if x]:
        item = val_ds[i]
        raw = load_replay_dict(val_ds.files[i])
        cam = Camera(raw["camera"], device=dev, downscale=a.downscale)
        gs = load_npz_state(val_ds.files[i])
        T = lambda k, d: torch.from_numpy(np.asarray(gs[k], np.float32)).to(dev).reshape(-1, d)
        gt_full = {"xyz": T("xyz", 3), "scaling": T("scaling", 3), "rot": T("rot", 4),
                   "opacity": T("opacity", 1), "color": sh_dc_to_rgb(T("color", 3))}

        m = item["mask"].to(dev) > 0.5
        tgt = item["target"].to(dev).float()[m]
        sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)
        with torch.no_grad():
            out = model(item["input"].unsqueeze(0).to(dev).float(),
                        item["mask"].unsqueeze(0).to(dev).float(),
                        run_decode=True, run_gen=False, gen_noise_std=0.0)
        pr = out.get("attr_pred", out["pred"])[0][m].float()

        def build(src):
            """src[name] = 'own' or 'gt'; xyz always from the model."""
            t = lambda n: pr if src[n] == "own" else tgt
            return (pr[:, 0:3] * sc + cen,
                    torch.exp(t("scale")[:, 3:6].clamp(-15, 5)) * sc,
                    t("rot")[:, 6:10],
                    torch.sigmoid(t("opacity")[:, 10:11]),
                    sh_dc_to_rgb(t("color")[:, 11:14]))

        with torch.no_grad():
            ref = render_gaussians(gt_full["xyz"], gt_full["scaling"], gt_full["rot"],
                                   gt_full["opacity"], gt_full["color"], cam)
            all_own = {n: "own" for n, _ in PARTS}
            all_gt = {n: "gt" for n, _ in PARTS}
            rows.setdefault("own 전부", []).append(psnr(ref, render_gaussians(*build(all_own), cam)))
            rows.setdefault("GT 전부", []).append(psnr(ref, render_gaussians(*build(all_gt), cam)))
            for n, _ in PARTS:
                s = dict(all_own); s[n] = "gt"
                rows.setdefault(f"{n} 만 GT", []).append(
                    psnr(ref, render_gaussians(*build(s), cam)))

        for n, sl in PARTS:
            dist[n][0].append(pr[:, sl].float().cpu())
            dist[n][1].append(tgt[:, sl].float().cpu())

    base = float(np.mean(rows["own 전부"]))
    top = float(np.mean(rows["GT 전부"]))
    print(f"{'설정':>14} {'PSNR':>8} {'회복':>8}")
    print(f"{'own 전부':>14} {base:8.2f} {'-':>8}   <- 현재")
    for n, _ in PARTS:
        v = float(np.mean(rows[f"{n} 만 GT"]))
        print(f"{n + ' 만 GT':>14} {v:8.2f} {v - base:+8.2f}")
    print(f"{'GT 전부':>14} {top:8.2f} {top - base:+8.2f}   <- 상한(기하만의 오차)")

    print(f"\n{'채널':>10} {'pred 평균':>10} {'GT 평균':>10} {'pred std':>9} {'GT std':>8}")
    for n, _ in PARTS:
        p, g = torch.cat(dist[n][0]), torch.cat(dist[n][1])
        if n == "opacity":
            p, g = torch.sigmoid(p), torch.sigmoid(g)
        elif n == "scale":
            p, g = p.exp(), g.exp()
        print(f"{n:>10} {float(p.mean()):10.4f} {float(g.mean()):10.4f} "
              f"{float(p.std()):9.4f} {float(g.std()):8.4f}")


if __name__ == "__main__":
    main()
