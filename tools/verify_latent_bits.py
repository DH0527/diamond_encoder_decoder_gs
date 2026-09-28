#!/usr/bin/env python3
"""Channel-wise effective-bit probe via uniform quantization of z_compact.

Loads a checkpoint, encodes val scenes, quantizes centroid / scale / shape
(or occupancy) channel blocks to k bits, decodes, and reports perm-invariant
within-group chamfer / radius. Answers: where does reconstruction collapse?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from can3tok.config import Can3TokConfig  # noqa: E402
from can3tok.data import ReplayGaussianDataset  # noqa: E402
from can3tok.model import Can3TokAE  # noqa: E402


def quantize(z: torch.Tensor, bits: int, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    if bits >= 16:
        return z
    levels = (1 << bits) - 1
    # per-channel range
    zz = (z - lo) / (hi - lo).clamp(min=1e-6)
    zz = zz.clamp(0, 1)
    q = torch.round(zz * levels) / levels
    return q * (hi - lo) + lo


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--root",
        default="/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg",
    )
    p.add_argument("--stats", default=str(ROOT / "assets/stats_replay_seg.json"))
    p.add_argument("--scenes", default="0,74,138,288")
    p.add_argument("--bits", default="16,8,6,4,3,2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(ROOT / "runs/verify_latent_bits.json"))
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ck.get("cfg") or ck.get("model_config") or {}
    # rebuild config loosely
    fields = {f.name for f in Can3TokConfig.__dataclass_fields__.values()}  # type: ignore
    kw = {k: v for k, v in cfg_dict.items() if k in fields}
    for k in ("latent_hw", "compact_latent_hw"):
        if k in kw and isinstance(kw[k], list):
            kw[k] = tuple(kw[k])
    cfg = Can3TokConfig(**kw) if kw else Can3TokConfig()
    model = Can3TokAE(cfg).to(args.device).eval()
    model.load_state_dict(ck["model"], strict=False)

    gsz = int(cfg.group_size)
    c_cen = int(cfg.budget_centroid)
    c_occ = int(cfg.budget_occupancy)
    c_shp = int(cfg.budget_shape)
    scenes = [int(x) for x in args.scenes.split(",") if x.strip()]
    bit_list = [int(x) for x in args.bits.split(",")]

    ds_kw = dict(
        root=args.root,
        max_points=cfg.max_points,
        group_size=gsz,
        drop_outside=True,
        sample_mode="stratified",
        crop_prob=0.0,
        stats_path=args.stats,
        seed=0,
        slot_redistribute=False,
        partition_mode="morton",
        partition_block=32,
        slot_sort="morton",
    )

    # collect z_compact stats for range
    zs = []
    batches = []
    for sid in scenes:
        ds = ReplayGaussianDataset(**ds_kw, file_indices=[sid])
        it = ds[0]
        x = it["target"][None, :, :3].to(args.device)
        m = it["mask"][None].to(args.device)
        out = model(x, m, run_decode=True, run_gen=False)
        z = out["z_compact"].float()
        zs.append(z)
        batches.append((x, m, out))
    zcat = torch.cat(zs, 0)
    # per-channel lo/hi over batch
    lo = zcat.amin(dim=(0, 2, 3), keepdim=True)
    hi = zcat.amax(dim=(0, 2, 3), keepdim=True)

    def chamfer_rel(pred, tgt, mask):
        # group-wise symmetric NN, averaged, / group radius
        b, n, _ = pred.shape
        ng = n // gsz
        pr = pred.reshape(b, ng, gsz, 3)
        tg = tgt.reshape(b, ng, gsz, 3)
        mm = mask.reshape(b, ng, gsz)
        vals = []
        for bi in range(b):
            for g in range(ng):
                k = int(mm[bi, g].sum().item())
                if k < 2:
                    continue
                p = pr[bi, g, :k]
                t = tg[bi, g, :k]
                # use first k slots of pred too (prefix); for redistributed may mismatch count
                d12 = torch.cdist(p, t).min(-1).values.mean()
                d21 = torch.cdist(t, p).min(-1).values.mean()
                r = (t - t.mean(0)).norm(dim=-1).mean().clamp(min=1e-6)
                vals.append(float(0.5 * (d12 + d21) / r))
        return float(np.mean(vals)) if vals else float("nan")

    results = {}
    for bits in bit_list:
        # quantize all / shape-only / centroid-only
        for mode in ("all", "shape", "centroid"):
            scores = []
            for (x, m, out), z in zip(batches, zs):
                zq = z.clone()
                if mode == "all":
                    zq = quantize(zq, bits, lo, hi)
                elif mode == "shape":
                    # channels [c_cen+c_occ : ]
                    sl = slice(c_cen + c_occ, c_cen + c_occ + c_shp)
                    zq[:, sl] = quantize(zq[:, sl], bits, lo[:, sl], hi[:, sl])
                else:
                    sl = slice(0, c_cen)
                    zq[:, sl] = quantize(zq[:, sl], bits, lo[:, sl], hi[:, sl])
                z_raw_hat, ctx = model.decompressor(zq, shortcut_alpha=0.0)
                pred, _ = model.decoder(
                    z_raw_hat,
                    ctx["cell_vec"],
                    ctx["scale"],
                    centroid=ctx["centroid"],
                    count=ctx["count"],
                    group_vec=ctx.get("group_vec"),
                )
                scores.append(chamfer_rel(pred[:, : x.shape[1], :3], x, m))
            results[f"bits{bits}_{mode}"] = float(np.mean(scores))
            print(f"bits={bits:2d} mode={mode:8s}  mean_group_chamfer_rel={results[f'bits{bits}_{mode}']:.4f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
