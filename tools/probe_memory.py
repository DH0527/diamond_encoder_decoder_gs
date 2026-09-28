"""Peak memory of one real training step, at the real point count.

An 8k-point smoke cannot answer this. It ran at 1.0 GB while the 262k run needed
31.4 of 31.47 GiB and died -- twice, both times on the exact step the decoder
turns on, because that is when activation memory jumps. Scaling a smoke's memory
by 32x is not a measurement, and treating it as one cost two runs.

So run the actual configuration, one full forward+backward, at the schedule steps
where the memory profile changes:

    decode on          the first jump, and where both crashes landed
    gen ramping        second decoder's activations join
    render on          rasteriser forward + backward on top

Single GPU, so DDP's gradient buckets are not included: add roughly
`params x 4 bytes` (0.46 GB for 114M) plus reduction scratch on top of what this
prints, and leave margin above that.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from can3tok.config import channel_budget, patch_layout  # noqa: E402
from can3tok.data import ReplayGaussianDataset  # noqa: E402
from can3tok.losses import render_loss, total_loss  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.schedule import effective_weights, schedule_flags  # noqa: E402
from can3tok.train import build_config, build_parser, resolve_shape_args  # noqa: E402


def argv_from_launcher(path: Path) -> list:
    txt = path.read_text()
    stub = txt.replace('"$TORCHRUN"', "echo").replace("$TORCHRUN", "echo")
    stub = re.sub(r"^\s*nohup.*$", "", stub, flags=re.M)
    out = subprocess.run(["bash", "-c", stub], capture_output=True, text=True,
                         cwd=str(ROOT), env={"PATH": "/usr/bin:/bin"})
    for line in (out.stdout or "").splitlines():
        if "--root" in line and "--out_dir" in line:
            parts = line.split()
            return parts[parts.index("train.py") + 1:]
    raise SystemExit(f"could not extract argv:\n{out.stdout[-800:]}{out.stderr[-800:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", default="scripts/launch_attr_262k_ddp.sh")
    ap.add_argument("--steps", default="2500,5500,7000")
    ap.add_argument("--scene", type=int, default=0)
    a = ap.parse_args()

    args = build_parser().parse_args(argv_from_launcher(ROOT / a.launcher))
    resolve_shape_args(args)
    ds = ReplayGaussianDataset(
        root=args.root, max_points=args.max_points, group_size=args.group_size,
        drop_outside=True, sample_mode=args.sample_mode, crop_prob=0.0,
        stats_path=args.stats_path, seed=0, slot_redistribute=False,
        partition_mode=args.partition_mode, partition_block=args.partition_block,
        slot_sort=args.slot_sort, file_indices=[a.scene],
    )
    cfg = build_config(args, ds.target_dim, ds.sh_dim)
    model = build_model(cfg).cuda().train()
    n_par = sum(p.numel() for p in model.parameters())
    lay = dict(patch_layout(cfg)); bud = channel_budget(cfg)
    lay.update(per_group=bud["per_group"], c_centroid=bud["centroid"],
               c_occupancy=bud["occupancy"],
               geom_pack_dim=(cfg.group_size * 3 if cfg.attr_pack_dim > 0 else 0))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)

    item = ds[0]
    x = item["input"].unsqueeze(0).cuda().float()
    m = item["mask"].unsqueeze(0).cuda().float()
    tgt_full = item["target"].unsqueeze(0).cuda().float()
    tgt = tgt_full[..., : cfg.target_dim]
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30

    print(f"config from {a.launcher}")
    print(f"  max_points={cfg.max_points} group_size={cfg.group_size} "
          f"patch_chunk={cfg.patch_chunk} gen_region_chunk={cfg.gen_region_chunk}")
    print(f"  params {n_par/1e6:.2f}M   amp={args.amp}   "
          f"ckpt_decode={cfg.checkpoint_decode} ckpt_gen={cfg.checkpoint_gen} "
          f"teacher_cycle={cfg.teacher_cycle}")
    print(f"  GPU total {total_gb:.2f} GiB\n")
    print(f"{'step':>6} {'phase':>7} {'decode':>7} {'gen':>5} {'render':>7} "
          f"{'peak GiB':>9} {'+DDP est':>9} {'헤드룸':>8}")

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp == "bf16" else None
    for step in [int(s) for s in a.steps.split(",") if s]:
        f = schedule_flags(step, args)
        w = effective_weights(step, args); w["residual_pack"] = bool(cfg.residual_pack)
        model.cfg.shortcut_alpha = float(f["shortcut_alpha"])
        model.cfg.folding_res_gain = float(f["folding_res_gain"])
        model.cfg.decoder_refine_alpha = float(f["decoder_refine_alpha"])
        model.cfg.attr_teacher_prob = float(f["attr_teacher_prob"])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        try:
            ctx = amp if amp is not None else torch.autocast("cuda", enabled=False)
            with ctx:
                out = model(x, m, run_decode=f["run_decode"], run_gen=f["run_gen"],
                            gen_noise_std=args.denoise_std,
                            attr_xyz=tgt_full[..., 0:3] if cfg.target_dim > 3 else None)
                loss, _ = total_loss(out, tgt, m, w, ds.layout, lay, cfg.sh_dim,
                                     with_attrs=f["with_attrs"], run_decode=f["run_decode"],
                                     run_gen=f["run_gen"])
            wr = float(w.get("w_render", 0.0))
            if wr > 0 and f["run_decode"] and step >= args.render_start:
                r = render_loss(out["pred"].float(), tgt_full, m, item["camera"].unsqueeze(0),
                                item["center"].unsqueeze(0), item["scale"].reshape(1), ds.layout,
                                lam_dssim=args.render_lam_dssim, downscale=args.render_downscale,
                                min_coverage=args.render_min_coverage, views=args.render_views)
                loss = loss + wr * r["render"]
            loss.backward()
            opt.step()
            peak = torch.cuda.max_memory_allocated() / 2**30
            ddp = peak + n_par * 4 / 2**30 * 1.5      # buckets + reduction scratch
            room = total_gb - ddp
            flag = "" if room > 2.0 else ("  <-- 위험" if room > 0 else "  <-- 안 들어감")
            print(f"{step:6d} {f['phase']:>7} {str(f['run_decode']):>7} {str(f['run_gen']):>5} "
                  f"{str(wr > 0 and step >= args.render_start):>7} "
                  f"{peak:9.2f} {ddp:9.2f} {room:8.2f}{flag}")
        except torch.OutOfMemoryError:
            print(f"{step:6d} {f['phase']:>7} {str(f['run_decode']):>7} {str(f['run_gen']):>5} "
                  f"{'':>7} {'OOM':>9}")
            torch.cuda.empty_cache()

    print("\n+DDP est = peak + params*4B*1.5 (gradient buckets + reduction scratch).")
    print("헤드룸 2 GiB 미만이면 다른 장면/증강에서 터질 수 있습니다.")


if __name__ == "__main__":
    main()
