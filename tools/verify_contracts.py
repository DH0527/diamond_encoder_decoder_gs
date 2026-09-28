"""Measure the A/B/C contracts on existing S/T/E1 checkpoints.

Does not train. Uses GPU only for decode + held-out renders.
Writes JSON and a few PNG pairs under --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from can3tok.eval_utils import (  # noqa: E402
    EvalView,
    eval_fingerprint,
    filter_eval_views,
    render_eval_metrics,
    scene_ids_match,
)
from can3tok.model import build_model  # noqa: E402
from can3tok.render import Camera, psnr, render_gaussians, sh_dc_to_rgb  # noqa: E402
from can3tok.schedule import apply_eval_schedule  # noqa: E402
from can3tok.train import build_config, load_eval_views, make_datasets  # noqa: E402
from can3tok.io_utils import camera_from_vector  # noqa: E402
from PIL import Image  # noqa: E402


RUNS = [
    dict(name="T16k_70k",
         args="runs/T16k_20260913_093332/args.json",
         ckpt="runs/T16k_20260913_093332/ckpt_step00070000.pt"),
    dict(name="S16k_62k",
         args="runs/S16k_20260910_012456/args.json",
         ckpt="runs/S16k_20260910_012456/ckpt_step00062000.pt"),
    dict(name="E1_16k",
         args="runs/E1_20260915_041813/args.json",
         ckpt="runs/E1_20260915_041813/ckpt_step00016000.pt"),
]
VAL_INDICES = (166, 382)  # train 028630, truck 028750 — same as the 09-15 audit


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _to_img(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().clamp(0, 1).cpu().numpy()
    x = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def _side_by_side(a: torch.Tensor, b: torch.Tensor, path: Path) -> None:
    ia, ib = _to_img(a), _to_img(b)
    if ia.size != ib.size:
        ib = ib.resize(ia.size, Image.BILINEAR)
    out = Image.new("RGB", (ia.size[0] * 2, ia.size[1]))
    out.paste(ia, (0, 0))
    out.paste(ib, (ia.size[0], 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def _cfg_flags(cfg) -> dict:
    return {
        "normalize_pooler_xyz": bool(getattr(cfg, "normalize_pooler_xyz", False)),
        "count_aware_template": bool(getattr(cfg, "count_aware_template", False)),
        "attr_slot_mask": bool(getattr(cfg, "attr_slot_mask", False)),
        "use_fixed_anchor_center": bool(getattr(cfg, "use_fixed_anchor_center", False)),
    }


def _unpack(t, mask, scale, center, layout, device):
    idx = torch.nonzero(mask > 0.5, as_tuple=False).reshape(-1)
    t = t[idx].float()
    c = center.to(device).reshape(1, 3).float()
    s = float(scale)
    a = layout
    return (
        t[:, a["xyz"]: a["xyz"] + 3] * s + c,
        torch.exp(t[:, a["scale"]: a["scale"] + 3].clamp(-15.0, 5.0)) * s,
        t[:, a["rot"]: a["rot"] + 4],
        torch.sigmoid(t[:, a["opacity"]: a["opacity"] + 1]),
        sh_dc_to_rgb(t[:, a["color"]: a["color"] + 3]),
    )


def verify_one(run, args, val, views, device, out_dir: Path, save_figs: bool) -> dict:
    ckpt = ROOT / run["ckpt"]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = build_config(args, val.target_dim, val.sh_dim)
    model = build_model(cfg, allow_padded_tokens=getattr(args, "allow_padded_tokens", False))
    missing = model.load_state_dict(ck["model"], strict=False)
    step = int(ck.get("step", -1))
    flags = apply_eval_schedule(model, step, args)
    model.to(device).eval()
    layout = getattr(val, "layout", None) or {
        "xyz": 0, "scale": 3, "rot": 6, "opacity": 10, "color": 11, "sh": 14}
    fp_now = eval_fingerprint(args, getattr(args, "_held_idx", set()))
    row = {
        "name": run["name"],
        "checkpoint": str(ckpt),
        "step": step,
        "saved_best_score": ck.get("best_score"),
        "saved_score_metric": (ck.get("args") or {}).get("score_metric"),
        "saved_eval_fingerprint": ck.get("eval_fingerprint"),
        "current_eval_fingerprint": fp_now,
        "best_score_would_reset": ck.get("eval_fingerprint") != fp_now,
        "cfg_flags": _cfg_flags(cfg),
        "dataset_flags": {
            "shared_cell_owner": bool(getattr(val, "shared_cell_owner", False)),
            "holdout_own_photo": bool(getattr(val, "holdout_own_photo", False)),
            "keep_extra_fullres": bool(getattr(val, "keep_extra_fullres", False)),
        },
        "eval_schedule": {k: flags[k] for k in
                          ("decoder_refine_alpha", "folding_res_gain", "attr_detach_geometry")},
        "load_missing": list(missing.missing_keys)[:8],
        "snapshots": [],
    }
    del ck

    for vi in VAL_INDICES:
        if vi >= len(val):
            continue
        item = val[vi]
        scene_key = item.get("scene_key") or f"scene{int(item.get('scene', 0))}"
        x = item["input"].unsqueeze(0).to(device).float()
        mask = item["mask"].unsqueeze(0).to(device).float()
        ex = item.get("enc_input")
        em = item.get("enc_mask")
        ga = item.get("group_anchor")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                x, mask, run_decode=True, run_gen=False,
                enc_x=None if ex is None else ex.unsqueeze(0).to(device).float(),
                enc_mask=None if em is None else em.unsqueeze(0).to(device).float(),
                group_anchor=None if ga is None else ga.unsqueeze(0).to(device).float(),
            )
            rec = model.decode_compact(out["z_compact"])
            z_n = model.encode_compact(
                x if ex is None else ex.unsqueeze(0).to(device).float(),
                mask if em is None else em.unsqueeze(0).to(device).float(),
                normalized=True)
            rec_n = model.decode_compact(z_n, normalized=True)
            rec_wrong = model.decode_compact(z_n, normalized=False)

        ap = out.get("attr_pred", out["pred"])
        rec_ap = rec.get("attr_pred", rec["pred"])
        decode = {
            "attr_pred_in_decode": "attr_pred" in rec,
            "xyz_max_vs_forward": float((rec["pred"][..., :3] - out["pred"][..., :3]).abs().max()),
            "attr_mae_vs_forward": float((rec_ap[..., 3:] - ap[..., 3:]).abs().mean()),
            "attr_max_vs_forward": float((rec_ap - ap).abs().max()),
            "normalized_xyz_max_vs_forward": float(
                (rec_n["pred"][..., :3] - out["pred"][..., :3]).abs().max()),
            "unnormalized_encode_xyz_max": float(
                (rec_wrong["pred"][..., :3] - out["pred"][..., :3]).abs().max()),
        }

        own = filter_eval_views(views, scene_key)
        foreign = [v for v in views if isinstance(v, EvalView) and not scene_ids_match(v.scene_id, scene_key)]
        if not own:
            raise RuntimeError(f"{run['name']} val[{vi}] scene {scene_key!r} has no eval views")
        target = item["target"].to(device).float()[..., : cfg.target_dim]
        pred = ap.float()[0]
        gt_mask = mask[0]
        pred_mask = out["presence"].float()[0]
        down = int(getattr(args, "render_downscale", 2))

        def _metrics(vs, pm=None):
            m = render_eval_metrics(
                pred, target, gt_mask, item["center"], float(item["scale"]),
                vs, layout, downscale=down, pred_mask=pm)
            m.pop("_eval_view_ids", None)
            return m

        own_m = _metrics(own, None)
        foreign_m = _metrics(foreign, None)
        pred_m = _metrics(own, pred_mask)
        mixed_m = {}
        n_own, n_for = len(own), len(foreign)
        n_mix = n_own + n_for
        if n_mix:
            for k in set(own_m) | set(foreign_m):
                if k.startswith("_"):
                    continue
                mixed_m[k] = (
                    own_m.get(k, 0.0) * n_own + foreign_m.get(k, 0.0) * n_for
                ) / n_mix
        render = {
            "own_gtmask": own_m,
            "own_predmask": pred_m,
            "foreign_gtmask": foreign_m,
            "mixed_gtmask": mixed_m,
            "n_own_views": n_own,
            "n_foreign_views": n_for,
            "n_mixed_views": n_mix,
            "own_view_ids": [v.image_id if isinstance(v, EvalView) else "" for v in own][:4],
        }

        snap = {
            "val_index": vi,
            "scene_key": scene_key,
            "name": item.get("name"),
            "path": val.files[vi] if hasattr(val, "files") else None,
            "decode": decode,
            "render": render,
        }
        row["snapshots"].append(snap)

        if save_figs and own:
            gs = _unpack(pred, pred_mask, float(item["scale"]), item["center"], layout, device)
            for tag, view in (("own", own[0]), ("foreign", foreign[0] if foreign else None)):
                if view is None:
                    continue
                cam_vec = view.camera if isinstance(view, EvalView) else view[0]
                photo = view.photo if isinstance(view, EvalView) else view[1]
                cam = Camera(camera_from_vector(cam_vec), device=device, downscale=down)
                img = render_gaussians(*gs, cam)
                ref = photo.to(device).float()
                if ref.shape[-2:] != img.shape[-2:]:
                    ref = torch.nn.functional.interpolate(
                        ref[None], size=img.shape[-2:], mode="bilinear", align_corners=False)[0]
                stem = f"{run['name']}_{os.path.splitext(str(item['name']))[0]}_{tag}"
                _side_by_side(ref, img, out_dir / "figs" / f"{stem}_photo_pred.png")
                snap.setdefault("figs", {})[tag] = str(out_dir / "figs" / f"{stem}_photo_pred.png")
                snap.setdefault("fig_psnr", {})[tag] = float(psnr(img, ref))

        del out, rec, rec_n, rec_wrong, x, mask, item
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return row


def holdout_audit(args, views, held) -> dict:
    pm_path = str(getattr(args, "photo_map", "") or "")
    if not pm_path or not os.path.exists(pm_path):
        return {}
    pm = json.load(open(pm_path))
    vp = json.load(open(args.view_pool)) if getattr(args, "view_pool", "") else {}
    held_images = set()
    if isinstance(vp, dict):
        for pool in vp.values():
            for i in held:
                if 0 <= i < len(pool):
                    held_images.add(os.path.normpath(os.path.abspath(pool[i]["image"])))
                    held_images.add(os.path.basename(pool[i]["image"]))
    leak = 0
    by_scene = defaultdict(int)
    n = 0
    for path, ent in pm.items():
        n += 1
        ip = ent.get("image") if isinstance(ent, dict) else ent
        scene = ent.get("scene") if isinstance(ent, dict) else ""
        if not ip:
            continue
        if os.path.normpath(os.path.abspath(ip)) in held_images or os.path.basename(ip) in held_images:
            leak += 1
            by_scene[str(scene)] += 1
    return {
        "held_pool_indices": sorted(int(i) for i in held),
        "n_photo_map": n,
        "own_photo_would_leak": leak,
        "leak_by_scene": dict(by_scene),
        "n_eval_views": len(views),
        "eval_scenes": sorted({v.scene_id for v in views if isinstance(v, EvalView)}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="mdmd/verify_20260916")
    ap.add_argument("--no_figs", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)

    report = {"device": str(device), "val_indices": list(VAL_INDICES), "runs": []}
    first_args = Namespace(**json.load(open(ROOT / RUNS[0]["args"])))
    first_args.out_dir = str((ROOT / RUNS[0]["args"]).parent)
    views, held = load_eval_views(first_args)
    first_args._held_idx = held
    report["eval_views"] = {
        "n": len(views),
        "held": sorted(held),
        "tagged": all(isinstance(v, EvalView) for v in views),
        "scenes": sorted({v.scene_id for v in views if isinstance(v, EvalView)}),
    }
    report["holdout_own_photo_leak_if_flag_off"] = holdout_audit(first_args, views, held)
    _save_json(out_dir / "progress.json", report)
    print(json.dumps(report["eval_views"], ensure_ascii=False), flush=True)
    print("holdout leak", report["holdout_own_photo_leak_if_flag_off"].get("own_photo_would_leak"), flush=True)

    _, val = make_datasets(first_args)
    val.extra_real_views = 0
    print(f"val={len(val)} scene_names={getattr(val, 'scene_names', None)}", flush=True)

    for i, run in enumerate(RUNS):
        args = Namespace(**json.load(open(ROOT / run["args"])))
        args.out_dir = str((ROOT / run["args"]).parent)
        args._held_idx = held
        # Old args.json has no contract keys -> getattr 0. Keep it that way.
        if i == 0:
            ds = val
        else:
            _, ds = make_datasets(args)
            ds.extra_real_views = 0
        print(f"== {run['name']} ==", flush=True)
        row = verify_one(run, args, ds, views, device, out_dir, save_figs=not a.no_figs)
        report["runs"].append(row)
        _save_json(out_dir / "verify.json", report)
        for snap in row["snapshots"]:
            r = snap["render"]
            d = snap["decode"]
            print(
                f"  {snap['name']} scene={os.path.basename(str(snap['scene_key']).rstrip('/'))} "
                f"own={r['own_gtmask'].get('psnr_photo', float('nan')):.3f} "
                f"foreign={r['foreign_gtmask'].get('psnr_photo', float('nan')):.3f} "
                f"mixed={r['mixed_gtmask'].get('psnr_photo', float('nan')):.3f} "
                f"attr_mae={d['attr_mae_vs_forward']:.4g} "
                f"xyz_max={d['xyz_max_vs_forward']:.4g}",
                flush=True,
            )
        if i:
            del ds
    _save_json(out_dir / "verify.json", report)
    print("wrote", out_dir / "verify.json", flush=True)


if __name__ == "__main__":
    main()
