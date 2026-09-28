#!/usr/bin/env python3
"""How loud is each attribute term at the attribute decoder's own weights?

The previous `w_render_attr = 22` was calibrated when the attribute heads hung off
the geometry decoder's refine token. They now live in a separate module with its
own trunk, its own initialisation and roughly a tenth of the parameters, so that
number carries over only by luck. This measures it again where it now applies.

Each term is backpropagated alone, with every other weight zeroed and a fresh
graph, and the gradient norm is read off `attr_decoder.*` only. The share column
is what decides the balance -- not the weight, since a rasteriser gradient and a
smooth-l1-on-log-scale gradient have no reason to arrive at the same scale.

Geometry comes from a real checkpoint so the positions the attribute decoder is
conditioned on are the ones it will actually see; the attribute decoder itself is
at init, which is the point at which the weights have to be chosen.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from can3tok.config import channel_budget, patch_layout  # noqa: E402
from can3tok.losses import _attr_losses, render_loss, responsibility_target  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.schedule import effective_weights  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402

PARAM_TERMS = ["w_scale", "w_rot", "w_opacity", "w_color", "w_sh"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", type=int, default=8000)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--extra_str", type=str, default="")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = Namespace(**ck["args"])
    args.out_dir = str(Path(a.ckpt).resolve().parent)
    for tok in a.extra_str.split():
        if tok.startswith("--"):
            key = tok[2:]
        else:
            cur = getattr(args, key, None)
            setattr(args, key, type(cur)(tok) if isinstance(cur, (int, float)) else tok)

    train_ds, _ = make_datasets(args)
    cfg = build_config(args, train_ds.target_dim, train_ds.sh_dim)
    model = build_model(cfg)
    missing, _ = model.load_state_dict(ck["model"], strict=False)
    print(f"geometry from {Path(a.ckpt).name}; {len(missing)} attribute tensors at init")
    model.cuda().train()

    lay, bud = patch_layout(cfg), channel_budget(cfg)
    model_layout = dict(lay)
    model_layout.update({"per_group": bud["per_group"], "c_centroid": bud["centroid"],
                         "c_occupancy": bud["occupancy"], "c_shape": bud["shape"]})
    layout = {"xyz": 0, "scale": 3, "rot": 6, "opacity": 10, "color": 11,
              "sh": 14, "sh_dim": int(train_ds.sh_dim)}

    item = train_ds[a.scene]
    xin = item["input"][None].cuda().float()
    x = item["target"][None].cuda().float()
    m = item["mask"][None].cuda().float()
    params = [p for n, p in model.named_parameters() if n.startswith("attr_decoder.")]
    if not params:
        raise SystemExit("no attribute decoder -- pass --extra_str '--attr_decoder_layers 4'")
    print(f"{sum(p.numel() for p in params)/1e6:.3f}M parameters on the attribute path\n")

    base = effective_weights(a.step, args)
    mags = {}

    def grad_norm() -> float:
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                       for p in params]).float()
        return float(g.norm())

    # The attribute decoder consumes only detached geometry, so the geometry half
    # needs no graph at all -- and keeping one costs 31 GB at 262k points.
    with torch.no_grad():
        z_raw, anchors, _ = model.encode(xin, m)
        z_compact, _ = model.compress(z_raw, anchors)
        z_hat, ctx = model.decompressor(z_compact, shortcut_alpha=float(cfg.shortcut_alpha_eval))
        pred, _ = model.decoder(z_hat, ctx["cell_vec"], ctx["scale"],
                                centroid=ctx["centroid"], count=ctx["count"],
                                group_vec=ctx.get("group_vec"),
                                appear=ctx.get("appearance"))
        g_xyz = pred[..., 0:3].clone()
        g_app = ctx["appearance"].clone() if ctx.get("appearance") is not None else None
        g_scale = ctx["scale"].clone()
    del z_raw, anchors, z_compact, z_hat, ctx, pred
    torch.cuda.empty_cache()

    def attr_forward():
        return model.attr_decoder(g_xyz, g_app, g_scale)["pred"]

    tgt = responsibility_target(g_xyz, x, m, layout, int(cfg.group_size))
    for name in PARAM_TERMS:
        w = float(base.get(name, 0.0))
        if w == 0.0:
            continue
        model.zero_grad(set_to_none=True)
        parts = _attr_losses(attr_forward(), tgt, m, layout, base, int(train_ds.sh_dim))
        key = name[2:]
        if key not in parts:
            continue
        (w * parts[key]).backward()
        mags[name] = grad_norm()

    w_ra = float(base.get("w_render_attr", 0.0)) or 1.0
    model.zero_grad(set_to_none=True)
    r = render_loss(attr_forward().float(), x, m,
                    item["camera"][None], item["center"][None], item["scale"][None],
                    layout, lam_dssim=float(getattr(args, "render_lam_dssim", 0.2)),
                    downscale=int(getattr(args, "render_downscale", 2)),
                    min_coverage=float(getattr(args, "render_min_coverage", 0.25)),
                    views=int(getattr(args, "render_views", 1)),
                    detach_xyz=True)
    (w_ra * r["render"]).backward()
    mags["w_render_attr"] = grad_norm()

    tot = sum(mags.values()) or 1.0
    print(f"{'term':18s} {'weight':>9s} {'|w*g|':>11s} {'share':>8s}")
    for k in sorted(mags, key=lambda k: -mags[k]):
        w = w_ra if k == "w_render_attr" else float(base[k])
        print(f"{k:18s} {w:9.3f} {mags[k]:11.3e} {mags[k]/tot:7.1%}")

    share = mags["w_render_attr"] / tot
    others = tot - mags["w_render_attr"]
    print(f"\nstep {a.step}, w_render_attr = {w_ra:g} -> render is {share:.1%} of the "
          f"attribute gradient")
    for want in (0.5, 0.7, 0.8):
        print(f"  for {want:.0%}: w_render_attr = "
              f"{w_ra * (want / (1 - want)) * others / mags['w_render_attr']:.2f}")
    print("\n렌더가 주 신호여야 합니다: 대응 기반 타깃은 16.9 dB에서 포화하고 "
          "렌더 최적화는 26.9 dB에 도달합니다.")


if __name__ == "__main__":
    main()
