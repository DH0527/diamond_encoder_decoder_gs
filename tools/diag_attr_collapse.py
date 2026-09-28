"""Does the attribute target -- or the prediction -- collapse toward the mean?

Averaging the ground-truth attributes a prediction is responsible for is only
safe if the average still spreads. Two separate ways it can fail, and they need
different fixes, so they are measured apart:

  target collapse      the aggregation itself destroys spread. Each prediction
                       averages the GT points it owns, so if it owns many the
                       target regresses to the group mean and no decoder, however
                       large, could reproduce the original variation.

  prediction collapse  the target is fine but the decoder emits something close to
                       constant. This is the expected failure of an L1/L2 loss
                       under uncertainty -- the optimum is the conditional mean --
                       and it is what a group budget of 8 appearance channels
                       against 64 slots invites. A previous run's student emitted
                       colour with 19% of the ground truth's spread and rendered
                       as a near-uniform plate.

The number reported for each is std(x) / std(GT), per channel group, over the
valid points of a scene. 1.0 means the spread is preserved; 0 means everything
sits on one value. Also split into between-group and within-group parts, because
those have different budgets: the group code sees 8 channels, and anything that
varies *inside* a group has to come out of the per-slot basis.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.losses import responsibility_target  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402

CH = (("log_scale", slice(3, 6)), ("rot", slice(6, 10)),
      ("opacity", slice(10, 11)), ("color", slice(11, 14)))


def spread(v, g, gsz):
    """std(v)/std(gt), split into the between-group and within-group parts."""
    n = (v.shape[0] // gsz) * gsz
    vg = v[:n].reshape(-1, gsz, v.shape[-1])
    gg = g[:n].reshape(-1, gsz, g.shape[-1])
    tot = g.std().clamp(min=1e-9)
    btw = gg.mean(1).std().clamp(min=1e-9)
    wit = (gg - gg.mean(1, keepdim=True)).std().clamp(min=1e-9)
    return (float(v.std() / tot),
            float(vg.mean(1).std() / btw),
            float((vg - vg.mean(1, keepdim=True)).std() / wit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", type=int, default=160)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = Namespace(**ck["args"])
    args.out_dir = os.path.dirname(os.path.abspath(a.ckpt))
    if int(getattr(args, "attr_decoder_layers", 0)) <= 0:
        args.attr_decoder_layers = 4          # build it at init just to read positions
    tr, val = make_datasets(args)
    cfg = build_config(args, tr.target_dim, tr.sh_dim)
    model = build_model(cfg)
    model.load_state_dict(ck["model"], strict=False)
    model.cuda().eval()
    gsz, dev = int(cfg.group_size), "cuda"

    item = val[a.scene]
    x = item["input"].unsqueeze(0).to(dev).float()
    m = item["mask"].unsqueeze(0).to(dev).float()
    t = item["target"].unsqueeze(0).to(dev).float()
    with torch.no_grad():
        out = model(x, m, run_decode=True, run_gen=False)
    ap_t = out.get("attr_pred")
    if ap_t is None:
        raise SystemExit("no attribute decoder in this checkpoint")
    lay = getattr(val, "layout", None) or {"xyz": 0, "scale": 3, "rot": 6,
                                           "opacity": 10, "color": 11, "sh": 14}
    tgt = responsibility_target(ap_t[..., 0:3].float(), t, m, lay, gsz)

    v = m[0] > 0.5
    print(f"scene {item['name']}  step {ck.get('step')}  {int(v.sum())} points\n")
    print(f"{'channel':10s} {'':>6} {'all':>7} {'between-group':>15} {'within-group':>14}")
    for name, sl in CH:
        g = t[0][v][:, sl]
        for what, arr in (("target", tgt[0][v][:, sl]), ("pred", ap_t[0][v][:, sl].float())):
            r = spread(arr, g, gsz)
            print(f"{name if what=='target' else '':10s} {what:>6} "
                  f"{r[0]:7.3f} {r[1]:15.3f} {r[2]:14.3f}")
    print("\n1.0 = GT와 같은 분산. target 행이 낮으면 집계가 뭉갠 것이고,")
    print("pred 행만 낮으면 디코더가 평균으로 수렴한 것입니다.")


if __name__ == "__main__":
    main()
