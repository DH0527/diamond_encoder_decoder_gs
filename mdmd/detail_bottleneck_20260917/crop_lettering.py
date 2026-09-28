"""Lettering + side-walkway ROI: photo / GTG / R1 70500 / C1 22000.

CPU encode, GPU raster with the leftover-VRAM cap. Does not touch C1/R1 training.
"""
import os, sys, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_d1_70500 as R

REPO = os.environ.get("REPO", os.path.abspath(os.path.join(HERE, "../..")))
os.chdir(REPO)

from can3tok.render import Camera, render_gaussians, psnr
from can3tok.io_utils import camera_from_vector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("OUT", os.path.join(HERE, "lettering_roi"))
BOX = (615, 855, 205, 322)  # x0,x1,y0,y1
CKPTS = [
    ("R1", os.environ.get(
        "R1_CKPT",
        "runs/R1_tfresume_20260917_054800/ckpt_step00070500.pt")),
    ("C1", os.environ.get(
        "C1_CKPT",
        "runs/C1_detail_20260916_021829/ckpt_step00022000.pt")),
]


def log(msg):
    print(msg, flush=True)


def rend(tt, gtm, cen, sc, cam):
    g = [z.cuda(non_blocking=True) for z in R.gauss(tt, gtm, cen, sc)]
    with torch.no_grad():
        img = render_gaussians(*g, cam).float().clamp(0, 1)
    for z in g:
        del z
    return img


def decode_pred(ckpt, val_index=166):
    args, cfg, model, val, step = R.load_cpu(ckpt)
    it = val[val_index]
    x = it["input"][None].float()
    mk = it["mask"][None].float()
    tg = it["target"][None].float()[..., :cfg.target_dim]
    kw = {}
    if "enc_input" in it:
        kw = dict(enc_x=it["enc_input"][None].float(),
                  enc_mask=it["enc_mask"][None].float())
    if "group_anchor" in it:
        kw["group_anchor"] = it["group_anchor"][None].float()
    t1 = time.time()
    with torch.no_grad():
        o = model(x, mk, run_decode=True, run_gen=False, **kw)
    apred = o.get("attr_pred", o["pred"]).float()[0]
    log(f"[fwd ] {ckpt} step={step} {time.time()-t1:.1f}s  "
        f"xyz_norm={args.normalize_pooler_xyz} shared={args.shared_cell_owner}")
    payload = {
        "step": int(step),
        "name": it["name"],
        "apred": apred,
        "tg": tg[0],
        "gtm": it["mask"] > 0.5,
        "cen": it["center"].float().view(1, 3),
        "sc": float(it["scale"]),
        "camera": np.asarray(it["camera"].numpy(), np.float32),
        "photo": it["photo"].float(),
        "flags": {
            "normalize_pooler_xyz": int(args.normalize_pooler_xyz),
            "shared_cell_owner": int(args.shared_cell_owner),
            "holdout_own_photo": int(args.holdout_own_photo),
        },
    }
    del o, model, x, mk, kw, val, args, cfg
    return payload


def crop(img, box):
    x0, x1, y0, y1 = box
    return img[:, y0:y1, x0:x1]


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    torch.cuda.set_per_process_memory_fraction(R.MEM_FRAC, 0)
    log(f"[mem] cap {R.MEM_FRAC:.2f}  device={torch.cuda.get_device_name(0)}")

    t0 = time.time()
    decoded = []
    for tag, ckpt in CKPTS:
        log(f"[load] {tag} {ckpt}")
        decoded.append((tag, decode_pred(ckpt)))

    first = decoded[0][1]
    cam = Camera(camera_from_vector(first["camera"]), device="cuda", downscale=1)
    ref = first["photo"]
    if ref.shape[-2:] != (cam.image_height, cam.image_width):
        ref = torch.nn.functional.interpolate(
            ref[None], size=(cam.image_height, cam.image_width),
            mode="bilinear", align_corners=False)[0]
    ref_c = ref.cuda()
    gtm, cen, sc = first["gtm"], first["cen"], first["sc"]

    i_gt = rend(first["tg"], gtm, cen, sc, cam)
    preds = []
    for tag, d in decoded:
        # C1/R1 pack independently; GTG uses that run's target (same NPZ).
        preds.append((tag, d["step"], rend(d["apred"], d["gtm"], d["cen"], d["sc"], cam)))

    x0, x1, y0, y1 = BOX
    metrics = {
        "box_x0x1y0y1": list(BOX),
        "name": first["name"],
        "full": {"gtg": float(psnr(i_gt, ref_c))},
        "C_lettering": {
            "gtg": float(psnr(crop(i_gt, BOX), crop(ref_c, BOX))),
        },
    }
    flag_by_tag = {tag: d["flags"] for tag, d in decoded}
    for tag, step, img in preds:
        metrics["full"][tag] = float(psnr(img, ref_c))
        metrics["C_lettering"][tag] = float(psnr(crop(img, BOX), crop(ref_c, BOX)))
        metrics[f"{tag}_step"] = step
        metrics[f"{tag}_flags"] = flag_by_tag[tag]

    log(f"[psnr] {json.dumps(metrics)}")
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    panels = [("photo", R.to_img(crop(ref, BOX)))]
    panels.append((f"GTG {metrics['C_lettering']['gtg']:.2f}", R.to_img(crop(i_gt, BOX))))
    for tag, step, img in preds:
        panels.append(
            (f"{tag} {step}  {metrics['C_lettering'][tag]:.2f}",
             R.to_img(crop(img, BOX))))

    h, w = panels[0][1].shape[:2]
    fig, ax = plt.subplots(1, len(panels),
                           figsize=(len(panels) * 4.4, 4.4 * h / max(w, 1) + 0.7),
                           dpi=150)
    if len(panels) == 1:
        ax = [ax]
    for i, (ttl, im) in enumerate(panels):
        ax[i].imshow(im)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(ttl, fontsize=11)
    fig.suptitle("C_lettering  WESTERN PACIFIC + walkway  vs photo dB", fontsize=12)
    fig.tight_layout()
    out_p = os.path.join(OUT, "C_lettering.png")
    fig.savefig(out_p)
    plt.close(fig)
    log(f"[out ] {out_p}")

    # also drop a copy next to the per-run boards
    copies = {
        "R1": os.path.join(HERE, "r1_70500", "C_lettering.png"),
        "C1": os.path.join(HERE, "c1_22000", "C_lettering.png"),
    }
    for tag, step, img in preds:
        dest = copies.get(tag)
        if not dest:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(3 * 4.2, 4.2 * h / max(w, 1) + 0.6), dpi=140)
        for i, (im, ttl) in enumerate([
            (R.to_img(crop(ref, BOX)), "photo"),
            (R.to_img(crop(i_gt, BOX)), f"GTG {metrics['C_lettering']['gtg']:.1f}"),
            (R.to_img(crop(img, BOX)), f"{tag} {metrics['C_lettering'][tag]:.1f}"),
        ]):
            ax[i].imshow(im)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(ttl, fontsize=11)
        fig.suptitle("C_lettering", fontsize=12)
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
        log(f"[out ] {dest}")

    log(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
