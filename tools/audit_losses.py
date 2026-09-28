#!/usr/bin/env python3
"""Which loss terms actually pull in the same direction, and which cancel?

For each active weight, run one backward with *only* that weight on and record
the gradient on the parameters that decide within-group geometry (the folding
frame head, the folding residual head, and the decompressor token path). Then
report, per term:

  * ``|w*g|``     weighted gradient norm -- how loud the term is
  * ``cos`` matrix  pairwise cosine between term gradients

Two terms with cos ~ +1 are the same constraint counted twice (one can go).
Two terms with cos ~ -1 are fighting, and the sum of their weights decides the
winner rather than any geometric argument.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from can3tok.data import ReplayGaussianDataset  # noqa: E402
from can3tok.losses import total_loss  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.schedule import effective_weights, schedule_flags  # noqa: E402
from can3tok.train import build_config, build_parser  # noqa: E402
from can3tok.config import channel_budget, patch_layout  # noqa: E402

# every weight the codec path can switch on, grouped by what it is supposed to say
TERMS = [
    ("w_z_raw", "latent"), ("w_z_raw_mse", "latent"), ("w_z_hard", "latent"),
    ("w_z_token_var", "latent"), ("w_z_std_ratio", "latent"),
    ("w_z_residual", "latent"), ("w_z_intra_chamfer", "set-local"), ("w_z_residual_hard", "latent"), ("w_shape_direct", "latent"),
    ("w_latent_std", "reg"), ("w_latent_decorr", "reg"), ("w_res_ratio", "reg"),
    ("w_xyz", "ordered"), ("w_xyz_mse", "ordered"), ("w_xyz_hard", "ordered"),
    ("w_xyz_residual", "ordered"), ("w_xyz_residual_hard", "ordered"),
    ("w_intra_chamfer", "set-local"),
    ("w_gen_chamfer", "gen-global"), ("w_gen_intra_chamfer", "gen-local"),
    ("w_gen_xyz_residual", "gen-ordered"), ("w_gen_presence", "gen-mask"),
    ("w_distill", "gen-distill"),
    # Every active weight MUST appear here. The list is used twice -- once to
    # choose which term to measure and once to zero all the others -- so a term
    # that is missing is never zeroed and silently contaminates every row. That
    # is not hypothetical: w_latent_decorr was absent once and made all four
    # measured terms report the identical magnitude at cosine 1.00.
    ("w_gen_basis", "gen-distill"), ("w_gen_p2g", "gen-local"),
    ("w_gen_radius", "gen-local"), ("w_teacher_cycle", "latent"),
    ("w_chamfer", "set-global"), ("w_coverage", "set-global"),
    ("w_plane_chamfer", "set-global"), ("w_proj_hist", "set-global"),
    ("w_voxel_occ", "set-global"), ("w_presence", "mask"),
]

# Not a total_loss term -- the trainer adds it outside the autocast block -- so it
# has to be measured separately. Worth the special case: a rasteriser gradient has
# no reason to arrive at the same scale as a point-space one, and the raw weight
# says nothing about the balance.
RENDER_TERM = ("w_render", "render")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5000, help="which schedule point to audit")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--launcher", type=str, default="scripts/launch_fix2_262k_ddp.sh")
    ap.add_argument("--extra", type=str, nargs="*", default=[])
    # nargs="*" stops at the first token beginning with "-", so launcher overrides
    # have to arrive as one quoted string.
    ap.add_argument("--extra_str", type=str, default="",
                    help="e.g. --extra_str '--max_points 65536 --patch_chunk 128'")
    ap.add_argument("--params", type=str, default="local",
                    choices=["local", "all", "compressor", "decoder", "gen"],
                    help="local = within-group geometry path; compressor = the encode side "
                         "that owns pose/centroid; all = everything")
    a = ap.parse_args()

    argv = _args_from_launcher(ROOT / a.launcher) + list(a.extra) + a.extra_str.split()
    args = build_parser().parse_args(argv)
    if int(args.latent_hw[0]) * int(args.latent_hw[1]) == 0:
        # train.py's main resolves this; build_config alone does not
        need = (cfgless := int(args.max_points) // int(args.group_size)) * int(args.local_tokens_per_group)
        side = 1
        while side * side < need:
            side *= 2
        args.latent_hw = [side, max(1, need // side)]
    cfg = build_config(args, 3, 0)
    model = build_model(cfg).cuda().train()
    lay = patch_layout(cfg)
    bud = channel_budget(cfg)
    model_layout = dict(lay)
    model_layout.update({"per_group": bud["per_group"], "c_centroid": bud["centroid"],
                         "c_occupancy": bud["occupancy"], "c_shape": bud["shape"]})

    ds = ReplayGaussianDataset(
        root=args.root, max_points=cfg.max_points, group_size=cfg.group_size,
        drop_outside=True, sample_mode="stratified", crop_prob=0.0,
        stats_path=args.stats_path, seed=0, slot_redistribute=False,
        partition_mode=args.partition_mode, partition_block=args.partition_block,
        slot_sort=args.slot_sort, file_indices=[a.scene],
    )
    item = ds[0]
    x = item["target"][None, :, :3].cuda()
    m = item["mask"][None].cuda()

    flags = schedule_flags(a.step, args)
    model.cfg.shortcut_alpha = float(flags["shortcut_alpha"])
    model.cfg.folding_res_gain = float(flags["folding_res_gain"])
    model.cfg.decoder_refine_alpha = float(flags["decoder_refine_alpha"])
    base = effective_weights(a.step, args)

    if a.params == "local":
        keep = lambda n: "decompressor" in n and ("shape_xyz" in n or "fold_head" in n or "token_out" in n)
    elif a.params == "gen":
        keep = lambda n: n.startswith("gen_decoder.")
    elif a.params == "decoder":
        keep = lambda n: n.startswith("decoder.")
    elif a.params == "compressor":
        keep = lambda n: n.startswith("compressor.") or n.startswith("encoder.")
    else:
        keep = lambda n: True
    params = [p for n, p in model.named_parameters() if keep(n)]
    print(f"auditing {len(params)} tensors on the within-group geometry path, step {a.step}\n")

    grads, mags = {}, {}
    for name, _ in TERMS:
        w = float(base.get(name, 0.0))
        if w == 0.0:
            continue
        wt = dict(base)
        for other, _ in TERMS:
            if other != name:
                wt[other] = 0.0
        model.zero_grad(set_to_none=True)
        out = model(x, m, run_decode=True, run_gen=True)  # fresh graph each term
        res = total_loss(out, x, m, wt, {"xyz": 3}, model_layout, 0,
                         with_attrs=False, run_decode=True, run_gen=True)
        loss = res[0] if isinstance(res, tuple) else res
        if not torch.is_tensor(loss) or not loss.requires_grad:
            continue
        loss.backward(retain_graph=False)
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                       for p in params]).float()
        n = float(g.norm())
        if n < 1e-12:
            print(f"  {name:24s} w={w:7.2f}  |w*g|=0            <-- NO GRADIENT on this path")
            continue
        grads[name] = (g / n).cpu().numpy()
        mags[name] = n

    # The render term, measured the same way: everything else zeroed, fresh graph.
    w_render = float(base.get("w_render", 0.0))
    if w_render != 0.0 and item["camera"][0] > 0:
        from can3tok.losses import render_loss

        model.zero_grad(set_to_none=True)
        out = model(x, m, run_decode=True, run_gen=True)
        full = item["target"][None].cuda()
        r = render_loss(out["pred"].float(), full, m,
                        item["camera"][None], item["center"][None], item["scale"][None],
                        {"xyz": 0, "scale": 3, "rot": 6, "opacity": 10, "color": 11},
                        lam_dssim=float(getattr(args, "render_lam_dssim", 0.2)),
                        downscale=int(getattr(args, "render_downscale", 2)),
                        min_coverage=float(getattr(args, "render_min_coverage", 0.25)),
                        views=int(getattr(args, "render_views", 1)))
        loss = w_render * r["render"]
        if torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                           for p in params]).float()
            n = float(g.norm())
            if n < 1e-12:
                print(f"  {'w_render':24s} w={w_render:7.2f}  |w*g|=0            <-- NO GRADIENT")
            else:
                grads["w_render"], mags["w_render"] = (g / n).cpu().numpy(), n
                base = dict(base)
                base["w_render"] = w_render

    order = sorted(mags, key=lambda k: -mags[k])
    tot = sum(mags.values())
    print(f"{'term':24s} {'group':11s} {'weight':>8s} {'|w*g|':>10s} {'share':>7s}")
    grp = dict(TERMS + [RENDER_TERM])
    for k in order:
        print(f"{k:24s} {grp[k]:11s} {float(base[k]):8.2f} {mags[k]:10.3e} {mags[k]/tot:6.1%}")

    print(f"\npairwise cosine (>= +0.7 duplicate, <= -0.3 fighting)\n")
    hdr = "".join(f"{k[2:][:7]:>8s}" for k in order)
    print(f"{'':24s}{hdr}")
    for i in order:
        row = "".join(f"{float(np.dot(grads[i], grads[j])):8.2f}" for j in order)
        print(f"{i:24s}{row}")

    print("\nflags:", {k: round(float(v), 3) for k, v in flags.items() if isinstance(v, (int, float))})


def _args_from_launcher(path: Path) -> list:
    """Pull the train.py argv out of the launcher so the audit matches the real run."""
    import re
    import subprocess

    txt = path.read_text()
    # expand the launcher's own shell vars by sourcing it with train.py stubbed out
    stub = txt.replace('"$TORCHRUN"', "echo").replace("$TORCHRUN", "echo")
    stub = re.sub(r"^\s*nohup.*$", "", stub, flags=re.M)
    stub = re.sub(r"^\s*exec\s.*$", "", stub, flags=re.M)
    out = subprocess.run(["bash", "-c", stub], capture_output=True, text=True,
                         cwd=str(ROOT), env={"PATH": "/usr/bin:/bin", "AUDIT": "1"})
    line = ""
    for cand in (out.stdout or "").splitlines():
        if "--root" in cand and "--out_dir" in cand:
            line = cand
            break
    if not line:
        raise SystemExit("could not recover argv from the launcher; pass --extra manually\n"
                         + (out.stdout or "")[-2000:] + (out.stderr or "")[-2000:])
    toks = line.split()
    return [t for t in toks if not t.endswith("train.py") and t not in ("--standalone",)
            and not t.startswith("--nproc_per_node")]


if __name__ == "__main__":
    main()
