"""Does the appearance path carry information the latent did not have before?

This is the pre-committed gate for the first run that trains anything except xyz,
and it has to be readable without restarting the job -- the run was already going
when it turned out that neither the log line nor the eval json reported a single
attribute number.

The gate is R^2 of the decoded attributes against the truth, on **held-out**
scenes. The bar is 0, and 0 is meaningful rather than arbitrary: predicting the
dataset mean scores exactly 0, and a geometry-only latent measurably scores
*below* it -- log_scale 0.227, rotation -0.248, opacity -0.993, colour -0.234
(tools/oracle_attr_from_geometry.py). So R^2 > 0 on rotation, opacity and colour
is direct evidence that appearance reached z_compact, which no previous run could
have produced.

Also reports a channel-ablation control: shuffle the appearance channels of
z_compact across groups and re-decode. If the attribute output barely moves, the
heads are ignoring the code and any R^2 gain came from geometry correlation
instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.config import channel_budget  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402

ATTRS = (("log_scale", slice(3, 6)), ("rot", slice(6, 10)),
         ("opacity", slice(10, 11)), ("color_DC", slice(11, 14)))


def r2(pred: torch.Tensor, true: torch.Tensor) -> float:
    ss_res = ((pred - true) ** 2).sum(0)
    ss_tot = ((true - true.mean(0, keepdim=True)) ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - ss_res / ss_tot).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--indices", default="40,120,160,200,240")
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
    bud = channel_budget(cfg)
    c_app = int(bud.get("appearance", 0))
    a0 = bud["centroid"] + bud["occupancy"] + bud["shape"]

    print(f"ckpt {a.ckpt}  step {ck.get('step')}")
    print(f"  target_dim={cfg.target_dim}  appearance channels={c_app}  "
          f"attr_pack_dim={int(getattr(cfg, 'attr_pack_dim', 0))}")
    if cfg.target_dim <= 3:
        raise SystemExit("  stage=xyz -- no attribute heads exist in this checkpoint")
    print(f"  held-out scenes: {a.indices}\n")

    acc = {n: [[], []] for n, _ in ATTRS}
    shuffled_delta = []
    for i in [int(x) for x in a.indices.split(",") if x]:
        item = val_ds[i]
        x = item["input"].unsqueeze(0).to(a.device).float()
        m = item["mask"].unsqueeze(0).to(a.device).float()
        tgt = item["target"].unsqueeze(0).to(a.device).float()
        with torch.no_grad():
            out = model(x, m, run_decode=True, run_gen=False, gen_noise_std=0.0)
            # `pred` carries the geometry decoder's own attribute head, which is
            # dead weight once a separate AttributeDecoder exists -- nothing trains
            # it. Reading it scored every attribute negative and made the
            # appearance-shuffle control look inert (1.8x instead of 277x), which
            # is a property of the tool, not the model.
            pred = out.get("attr_pred", out["pred"])
            sel = m[0] > 0.5
            for n, sl in ATTRS:
                if pred.shape[-1] < sl.stop:
                    continue
                acc[n][0].append(pred[0][sel][:, sl].float().cpu())
                acc[n][1].append(tgt[0][sel][:, sl].float().cpu())

            # Control: does the appearance code actually drive the heads?
            if c_app > 0:
                z = out["z_compact"]
                zs = z.clone()
                b, c, h, w = z.shape
                per = bud["per_group"]
                idx = (torch.arange(c, device=z.device) % per) >= a0
                flat = zs[:, idx].reshape(b, -1, h * w)
                perm = torch.randperm(h * w, device=z.device)
                zs[:, idx] = flat[:, :, perm].reshape(b, -1, h, w)
                zr, ctx = model.decompressor(zs, shortcut_alpha=0.0)
                p2, _ = model.decoder(zr, ctx["cell_vec"], ctx["scale"],
                                      centroid=ctx["centroid"], count=ctx["count"],
                                      group_vec=ctx.get("group_vec"),
                                      appear=ctx.get("appearance"))
                if getattr(model, "attr_decoder", None) is not None:
                    ap2 = ctx.get("appearance")
                    p2 = model.attr_decoder(p2[..., 0:3], ap2, ctx["scale"])["pred"]
                d_attr = (p2[0][sel][:, 3:] - pred[0][sel][:, 3:]).abs().mean()
                d_xyz = (p2[0][sel][:, :3] - pred[0][sel][:, :3]).abs().mean()
                shuffled_delta.append((float(d_xyz), float(d_attr)))

    print(f"{'attribute':>10} {'held-out R^2':>13} {'기하만(참고)':>14}  판정")
    geom_only = {"log_scale": 0.227, "rot": -0.248, "opacity": -0.993, "color_DC": -0.234}
    passed = []
    for n, _ in ATTRS:
        if not acc[n][0]:
            continue
        s = r2(torch.cat(acc[n][0]), torch.cat(acc[n][1]))
        ok = s > 0.0
        passed.append(ok)
        print(f"{n:>10} {s:13.3f} {geom_only[n]:14.3f}  {'통과' if ok else '미달'}")

    if shuffled_delta:
        dx = float(np.mean([d[0] for d in shuffled_delta]))
        da = float(np.mean([d[1] for d in shuffled_delta]))
        print(f"\nappearance 채널 셔플 대조군: xyz 변화 {dx:.5f}, attribute 변화 {da:.5f}")
        print("  attribute 변화가 xyz 변화보다 훨씬 커야 헤드가 코드를 실제로 읽는 것입니다.")

    print(f"\n게이트: rotation / opacity / colour 의 R^2 > 0")
    print(f"결과  : {'통과' if all(passed[1:]) else '미달'}")


if __name__ == "__main__":
    main()
