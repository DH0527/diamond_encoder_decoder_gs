"""How much of each Gaussian attribute is predictable from geometry alone?

The encoder packs ``x[..., 0:3]`` and a mask -- xyz and nothing else. Scale,
rotation, opacity, colour and SH never enter ``z_raw``, so they never reach
``z_compact``. Every attribute the decoder emits is therefore a function of
geometry, and two scenes that differ only in colour receive the *identical*
latent.

`oracle_attr_compensation.py` showed that attributes exist which lift a
rank-limited point set from 17.05 to 26.69 dB on held-out views. That proved the
attributes exist. It did not prove the latent can carry them -- and with this
encoder it demonstrably cannot carry any attribute information at all.

This puts a number on what is left: fit a network from a group's *geometry*
(its 64 offsets, centroid and extent) to that group's attributes, on train
scenes, and score it on held-out scenes. R^2 = 0 means the current encoder caps
that attribute at the dataset mean. R^2 near 1 would mean geometry already
determines it and the encoder's blindness costs nothing.

This is an upper bound on the current architecture, not a lower bound on the
fixed one: it is the best any decoder could do reading a geometry-only code.
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

# name -> (slice into the 59-channel target, whether it is per-point or per-group)
ATTRS = [("log_scale", slice(3, 6)), ("rot", slice(6, 10)),
         ("opacity", slice(10, 11)), ("color_DC", slice(11, 14)),
         ("sh_rest", slice(14, 59))]


def features_and_targets(item, g: int, dev: str):
    """Per-group geometry features, and the per-group attribute means to predict."""
    m = item["mask"].to(dev) > 0.5
    t = item["target"].to(dev).float()[m]
    ng = t.shape[0] // g
    t = t[: ng * g]
    xyz = t[:, 0:3].reshape(ng, g, 3)
    cen = xyz.mean(1)
    off = xyz - cen[:, None]
    rad = off.norm(dim=-1).mean(-1).clamp(min=1e-9)
    unit = (off / rad[:, None, None]).reshape(ng, g * 3)
    # Everything the geometry-only latent could possibly know about a group.
    feat = torch.cat([unit, cen, rad[:, None].log()], dim=-1)
    tgt = {n: t[:, s].reshape(ng, g, -1).mean(1) for n, s in ATTRS}
    return feat, tgt


def r2(pred, true):
    ss_res = ((pred - true) ** 2).sum(0)
    ss_tot = ((true - true.mean(0, keepdim=True)) ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - ss_res / ss_tot).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--train_scenes", type=str, default="40,63,74,120,138")
    ap.add_argument("--test_scenes", type=str, default="160,200,240")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val_ds = make_datasets(targs)
    g, dev = int(targs.group_size), "cuda"

    def gather(spec):
        fs, ts = [], {n: [] for n, _ in ATTRS}
        for i in spec.split(","):
            if not i:
                continue
            f, t = features_and_targets(val_ds[int(i)], g, dev)
            fs.append(f)
            for n in ts:
                ts[n].append(t[n])
        return torch.cat(fs), {n: torch.cat(v) for n, v in ts.items()}

    ftr, ttr = gather(a.train_scenes)
    fte, tte = gather(a.test_scenes)
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True).clamp(min=1e-6)
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    print(f"train groups {ftr.shape[0]}  held-out groups {fte.shape[0]}  "
          f"geometry feature dim {ftr.shape[1]}\n")
    print(f"{'attribute':>10} {'dim':>4} {'held-out R^2':>13} {'구조적으로 예측 가능한가':>26}")

    for name, _ in ATTRS:
        y_tr, y_te = ttr[name], tte[name]
        if y_tr.shape[1] == 0 or float(y_tr.std()) < 1e-8:
            print(f"{name:>10} {y_tr.shape[1]:4d} {'(상수)':>13}")
            continue
        ym, ys = y_tr.mean(0, keepdim=True), y_tr.std(0, keepdim=True).clamp(min=1e-6)
        net = nn.Sequential(
            nn.Linear(ftr.shape[1], a.hidden), nn.GELU(),
            nn.Linear(a.hidden, a.hidden), nn.GELU(),
            nn.Linear(a.hidden, y_tr.shape[1]),
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        gen = torch.Generator(device=dev).manual_seed(0)
        for _ in range(a.steps):
            idx = torch.randint(0, ftr.shape[0], (2048,), generator=gen, device=dev)
            opt.zero_grad(set_to_none=True)
            loss = ((net(ftr[idx]) - (y_tr[idx] - ym) / ys) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            score = r2(net(fte) * ys + ym, y_te)
        verdict = "예" if score > 0.5 else ("부분적" if score > 0.15 else "아니오")
        print(f"{name:>10} {y_tr.shape[1]:4d} {score:13.3f} {verdict:>26}")

    print("\nR^2 <= 0 이면 기하만으로는 데이터셋 평균보다 나을 수 없다는 뜻입니다.")
    print("현재 인코더는 xyz + mask 만 pack 하므로 이것이 attribute 재구성의 상한입니다.")


if __name__ == "__main__":
    main()
