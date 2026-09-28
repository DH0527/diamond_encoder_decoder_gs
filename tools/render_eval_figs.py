"""Regenerate the GT/PRED projection figures for a checkpoint.

`run_eval` writes these during training, but only at eval milestones and only for
whatever the running process had compiled in. This produces them for any
checkpoint after the fact -- which is how a figure format change reaches a run
that is already going, without restarting it.

Same output names and layout as the in-training path, so the files drop straight
into `eval/step*/val/` alongside the ones the run produced itself.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.eval_utils import compute_eval_metrics, save_eval_outputs  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.schedule import apply_eval_schedule  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--indices", default="",
                    help="default: the run's own --eval_val_indices")
    ap.add_argument("--branches", default="codec,gen")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--no_ply", action="store_true")
    ap.add_argument("--chamfer_samples", type=int, default=8192,
                    help="smaller than the run's own: this usually shares a GPU with the "
                         "training job, and the chamfer cdist is what OOMs first")
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
    apply_eval_schedule(model, int(ck.get("step", 0)), targs)

    step = int(ck.get("step", 0))
    out_dir = a.out_dir or os.path.join(targs.out_dir, "eval", f"step{step:08d}", "val")
    idx_src = a.indices or str(targs.eval_val_indices)
    indices = [int(i) for i in idx_src.split(",") if i != ""]
    indices = [i for i in indices if i < len(val_ds)]
    branches = [b for b in a.branches.split(",") if b]
    print(f"ckpt {a.ckpt}  step {step}  -> {out_dir}")

    for i in indices:
        item = val_ds[i]
        x = item["input"].unsqueeze(0).to(a.device).float()
        m = item["mask"].unsqueeze(0).to(a.device).float()
        tgt = item["target"].unsqueeze(0).to(a.device).float()[..., : cfg.target_dim]
        ex, em, ga = item.get("enc_input"), item.get("enc_mask"), item.get("group_anchor")
        with torch.no_grad():
            out = model(x, m, run_decode=True,
                        run_gen=bool(cfg.use_gen_branch) and "gen" in branches,
                        gen_noise_std=0.0,
                        enc_x=None if ex is None else ex.unsqueeze(0).to(a.device).float(),
                        enc_mask=None if em is None else em.unsqueeze(0).to(a.device).float(),
                        group_anchor=None if ga is None else ga.unsqueeze(0).to(a.device).float())
        raw_name = os.path.splitext(item["name"])[0]
        sk = item.get("scene_key") or f"scene{int(item.get('scene', 0))}"
        name = f"{str(sk).replace('/', '_')}_{raw_name}"
        for br, key, pkey, akey in (
            ("codec", "pred", "presence", "attr_pred"),
            ("gen", "gen_pred", "gen_presence", "gen_attr_pred"),
        ):
            if br not in branches or key not in out:
                continue
            deploy = out[akey] if akey in out else out[key]
            met = compute_eval_metrics(out[key].float(), out[pkey].float(), tgt, m,
                                       float(item["scale"]), int(a.chamfer_samples),
                                       attr_pred=None if akey not in out else out[akey].float())
            save_eval_outputs(out_dir, name, br, deploy.float()[0], tgt[0], m[0],
                              item["center"], float(item["scale"]), met,
                              write_ply_files=not a.no_ply,
                              pred_mask=out[pkey].float()[0])
            print(f"  {name}_{br}: rmse {met['xyz_rmse_norm']:.5f}" +
                  (f"  attr_color {met['attr_color_nrmse']:.3f}"
                   if "attr_color_nrmse" in met else ""))


if __name__ == "__main__":
    main()
