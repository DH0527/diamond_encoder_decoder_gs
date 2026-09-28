"""Where in the compression chain does appearance information die?

`exp_code_learnability.py` asks whether the code the decoder needs is a function
of what the *pack* carries. It answers that by regressing straight from the
group's ground-truth attributes, which walks past the encoder, the compressor and
the decompressor entirely. So a "yes" from it leaves two very different states of
the world indistinguishable:

  * the information survives compression and the decoder simply is not using it,
  * the information is destroyed somewhere between the pack and z_compact, and no
    decoder could use it.

The second is an architecture problem, not a training one, and this measures it
directly. Appearance passes through four representations:

    A  the group's own attributes            64 x 11 = 704 numbers
    B  the pack's aux block                  64        (learned attribute encoder)
    C  z_compact's appearance channels       16        (after the compressor)
    D  ctx["appearance"] after decompression 16

A -> B is the learned per-group attribute encoder, B -> C the compressor, C -> D
the decompressor. Fitting a regressor from each stage to the *group's mean
attributes* (train groups, scored on held-out groups) gives the share of
appearance that is still linearly-or-better recoverable at that point. The stage
where R^2 falls off is the stage that is losing it.

The control that makes the numbers readable is stage A: it is the same target
predicted from the same target's own raw form, so it bounds what any later stage
could reach given a perfect chain.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.config import channel_budget  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402


def fit_r2(xtr, ytr, xte, yte, steps=4000, hid=512, dev="cuda", seed=0):
    """Held-out R^2 of an MLP from x to y. Standardised both sides."""
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp(min=1e-6)
    ym, ys = ytr.mean(0, keepdim=True), ytr.std(0, keepdim=True).clamp(min=1e-6)
    xtr_n, xte_n = (xtr - mu) / sd, (xte - mu) / sd
    net = nn.Sequential(nn.Linear(xtr.shape[1], hid), nn.GELU(),
                        nn.Linear(hid, hid), nn.GELU(),
                        nn.Linear(hid, ytr.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, xtr_n.shape[0], (min(4096, xtr_n.shape[0]),),
                            generator=gen, device=dev)
        opt.zero_grad(set_to_none=True)
        ((net(xtr_n[idx]) - (ytr[idx] - ym) / ys) ** 2).mean().backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        p = net(xte_n) * ys + ym
    ss_res = ((p - yte) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0, keepdim=True)) ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - ss_res / ss_tot).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_scenes", default="40,63,74,120,138")
    ap.add_argument("--test_scenes", default="160,200,240")
    ap.add_argument("--steps", type=int, default=4000)
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
    dev, g = a.device, int(cfg.group_size)
    bud = channel_budget(cfg)
    a0 = bud["centroid"] + bud["occupancy"] + bud["shape"]
    c_app = int(bud.get("appearance", 0))
    print(f"ckpt {a.ckpt}  step {ck.get('step')}", flush=True)
    print(f"  appearance {c_app}ch  attr_pack_dim {int(getattr(cfg,'attr_pack_dim',0))}\n", flush=True)

    def collect(spec):
        A, B, C, D, Y = [], [], [], [], []
        for i in [int(s) for s in spec.split(",") if s]:
            item = val_ds[i]
            x = item["input"].unsqueeze(0).to(dev).float()
            mk = item["mask"].unsqueeze(0).to(dev).float()
            with torch.no_grad():
                z_raw, anchors, patch = model.encoder(x, mk)
                z_compact, _ = model.compressor(z_raw, anchors)
                _, ctx = model.decompressor(z_compact, shortcut_alpha=0.0)
            m = item["mask"].to(dev) > 0.5
            tgt = item["target"].to(dev).float()[m]
            ng = min(tgt.shape[0] // g, int(model.encoder.num_groups))
            at = tgt[: ng * g, 3:14].reshape(ng, g, 11)

            A.append(at.reshape(ng, g * 11).cpu())                      # 704, 원본
            B.append(patch[0, :ng, g * 3:].float().cpu())               # 64, pack aux
            zc = z_compact[0].reshape(c_app + a0 if False else z_compact.shape[1], -1)
            per = bud["per_group"]
            sel = (torch.arange(z_compact.shape[1], device=dev) % per) >= a0
            cells = z_compact[0][sel].reshape(c_app, -1).transpose(0, 1)  # (cells, 16)
            C.append(cells[:ng].float().cpu())
            D.append(ctx["appearance"][0, :ng].float().cpu())            # 16, 복원 후
            Y.append(at.mean(1).cpu())                                   # 11, 그룹 평균 속성
            print(f"  수집 {item['name']}  groups {ng}", flush=True)
        cat = lambda L: torch.cat(L).to(dev)
        return cat(A), cat(B), cat(C), cat(D), cat(Y)

    Atr, Btr, Ctr, Dtr, Ytr = collect(a.train_scenes)
    Ate, Bte, Cte, Dte, Yte = collect(a.test_scenes)
    print(f"\n  train groups {Atr.shape[0]}  held-out groups {Ate.shape[0]}", flush=True)
    print(f"  단계별 폭: A {Atr.shape[1]}  B {Btr.shape[1]}  C {Ctr.shape[1]}  D {Dtr.shape[1]}\n",
          flush=True)

    stages = (("A 원본 속성 (704)", Atr, Ate),
              ("B pack aux (64)", Btr, Bte),
              ("C z_compact appearance (16)", Ctr, Cte),
              ("D 복원 후 appearance (16)", Dtr, Dte))
    print(f"{'단계':>30} {'held-out R^2':>13} {'직전 대비':>10}")
    prev = None
    for name, xtr, xte in stages:
        r = fit_r2(xtr, Ytr, xte, Yte, steps=a.steps, dev=dev)
        delta = "" if prev is None else f"{r - prev:+10.3f}"
        print(f"{name:>30} {r:13.3f} {delta:>10}", flush=True)
        prev = r
    print("\n각 단계에서 '그룹 평균 속성 11채널' 을 얼마나 복원할 수 있는가.")
    print("A->B 에서 크게 떨어지면 attribute pack 인코더가 병목,")
    print("B->C 면 compressor, C->D 면 decompressor 가 병목입니다.")


if __name__ == "__main__":
    main()
