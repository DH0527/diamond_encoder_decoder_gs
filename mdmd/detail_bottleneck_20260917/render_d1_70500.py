"""D1 ckpt 렌더: 인코더/디코더는 CPU, 래스터만 GPU (C1 여유 VRAM 상한)."""
import os, sys, json, time
import numpy as np
import torch

REPO = os.environ.get("REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, REPO)
os.chdir(REPO)

from can3tok.train import build_parser, build_config, make_datasets, load_eval_views
from can3tok.model import build_model
from can3tok.render import Camera, render_gaussians, sh_dc_to_rgb, psnr
from can3tok.io_utils import camera_from_vector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = os.environ.get("CKPT", "runs/D1_moment_20260917_025742/ckpt_step00070500.pt")
OUT = os.environ.get("OUT", "mdmd/detail_bottleneck_20260917/d1_70500")
TAG = os.environ.get("TAG", "D1")
INDEX = 166
CROPS = {
    "A_hood_rail": (440, 880, 120, 230),
    "B_front_rail": (250, 540, 170, 500),
    # view3 WESTERN PACIFIC + walkway rail (x0,x1,y0,y1); same box as
    # mdmd/c1_ablation_20260917 lettering [615,205,855,322]
    "C_lettering": (615, 855, 205, 322),
}
MEM_FRAC = float(os.environ.get("MEM_FRAC", "0.10"))
COMPAT = ("normalize_pooler_xyz", "count_aware_template", "attr_slot_mask",
          "shared_cell_owner", "holdout_own_photo", "keep_extra_fullres",
          "reuse_prev_vis", "eval_pred_mask")


def log(msg):
    print(msg, flush=True)


def gpu_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 2), torch.cuda.max_memory_allocated() / (1024 ** 2)


def load_cpu(ckpt):
    cfgj = json.load(open(os.path.join(os.path.dirname(ckpt), "args.json")))
    argv = []
    for k, v in cfgj.items():
        if k in ("resume", "init_from", "eval_only", "out_dir", "init_skip"):
            continue
        if isinstance(v, bool):
            if v:
                argv.append("--" + k)
        elif isinstance(v, list):
            if v:
                argv += ["--" + k] + [str(x) for x in v]
        elif v is None:
            continue
        else:
            argv += ["--" + k, str(v)]
    argv += ["--out_dir", "/tmp/d1_70500_render"]
    args = build_parser().parse_args(argv)
    for k in COMPAT:
        if k not in cfgj:
            setattr(args, k, 0)
            log(f"  [compat] {k} -> 0")
    args.workers = 0
    tr, val = make_datasets(args)
    tr.anchor_spill = val.anchor_spill = int(getattr(args, "anchor_spill", 8))
    ev, held = load_eval_views(args)
    for d in (tr, val):
        d.view_exclude = set(held)
        # Ladder / view3 비교용: holdout 치환을 끄고 NPZ 자기 카메라를 쓴다.
        d.holdout_own_photo = False
    cfg = build_config(args, tr.target_dim, tr.sh_dim)
    model = build_model(cfg, allow_padded_tokens=getattr(args, "allow_padded_tokens", False)).cpu().eval()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    return args, cfg, model, val, ck.get("step")


def gauss(tt, msk, cen, sc):
    z = tt[msk]
    return (z[:, 0:3] * sc + cen,
            torch.exp(z[:, 3:6].clamp(-15, 5)) * sc,
            torch.nn.functional.normalize(z[:, 6:10], dim=-1),
            torch.sigmoid(z[:, 10:11]),
            sh_dc_to_rgb(z[:, 11:14]))


def to_img(t):
    return t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    torch.cuda.set_per_process_memory_fraction(MEM_FRAC, 0)
    log(f"[mem] cap {MEM_FRAC:.2f} of GPU (~{MEM_FRAC * 32760:.0f} MiB)  device={torch.cuda.get_device_name(0)}")

    t0 = time.time()
    log(f"[load] {TAG} {CKPT} on CPU")
    args, cfg, model, val, step = load_cpu(CKPT)
    log(f"[load] step={step}  flags xyz_norm={args.normalize_pooler_xyz} "
        f"count_tmpl={args.count_aware_template} slot_mask={args.attr_slot_mask} "
        f"shared={args.shared_cell_owner} holdout_train={args.holdout_own_photo} "
        f"render_cam=own  {time.time()-t0:.1f}s")

    it = val[INDEX]
    log(f"[data] {it['name']}  enc={'enc_input' in it}")
    x = it["input"][None].float()
    mk = it["mask"][None].float()
    tg = it["target"][None].float()[..., :cfg.target_dim]
    kw = {}
    if "enc_input" in it:
        kw = dict(enc_x=it["enc_input"][None].float(), enc_mask=it["enc_mask"][None].float())
    if "group_anchor" in it:
        kw["group_anchor"] = it["group_anchor"][None].float()

    t1 = time.time()
    with torch.no_grad():
        o = model(x, mk, run_decode=True, run_gen=False, **kw)
    apred = o.get("attr_pred", o["pred"]).float()[0]
    log(f"[fwd ] CPU decode {time.time()-t1:.1f}s  pred {tuple(apred.shape)}")
    del o, model, x, mk, kw
    torch.cuda.empty_cache()

    gtm = it["mask"] > 0.5
    cen = it["center"].float().view(1, 3)
    sc = float(it["scale"])
    cam = Camera(camera_from_vector(np.asarray(it["camera"].numpy(), np.float32)),
                 device="cuda", downscale=1)
    ref = it["photo"].float()
    if ref.shape[-2:] != (cam.image_height, cam.image_width):
        ref = torch.nn.functional.interpolate(
            ref[None], size=(cam.image_height, cam.image_width),
            mode="bilinear", align_corners=False)[0]
    log(f"[cam ] {cam.image_width}x{cam.image_height}  live={int(gtm.sum())}")

    def rend(tt):
        g = [z.cuda(non_blocking=True) for z in gauss(tt, gtm, cen, sc)]
        with torch.no_grad():
            img = render_gaussians(*g, cam).float().clamp(0, 1)
        for z in g:
            del z
        return img

    alloc, peak = gpu_mb()
    log(f"[gpu ] before raster alloc={alloc:.0f} peak={peak:.0f} MiB")
    t2 = time.time()
    i_gt = rend(tg[0])
    i_pr = rend(apred)
    alloc, peak = gpu_mb()
    log(f"[gpu ] after raster alloc={alloc:.0f} peak={peak:.0f} MiB  {time.time()-t2:.2f}s")

    ref_c = ref.cuda()
    p_gt = float(psnr(i_gt, ref_c))
    p_pr = float(psnr(i_pr, ref_c))
    crop_psnr = {}
    for k, (x0, x1, y0, y1) in CROPS.items():
        crop_psnr[k] = {
            "gtg": float(psnr(i_gt[:, y0:y1, x0:x1], ref_c[:, y0:y1, x0:x1])),
            TAG.lower(): float(psnr(i_pr[:, y0:y1, x0:x1], ref_c[:, y0:y1, x0:x1])),
        }
    log(f"[psnr] full gtg={p_gt:.2f}  {TAG}={p_pr:.2f}  crops={json.dumps(crop_psnr)}")

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({"step": int(step), "name": it["name"], "tag": TAG,
                   "full": {"gtg": p_gt, TAG.lower(): p_pr}, "crops": crop_psnr,
                   "gpu_peak_miB": peak, "fwd_s": time.time() - t0}, f, indent=2)

    fig, ax = plt.subplots(3, 1, figsize=(11, 3 * 11 * cam.image_height / cam.image_width + 0.9), dpi=130)
    for i, (im, ttl) in enumerate([
        (ref, "photograph"),
        (i_gt, f"GT Gaussians  {p_gt:.2f} dB"),
        (i_pr, f"{TAG} step {step}  {p_pr:.2f} dB"),
    ]):
        ax[i].imshow(to_img(im))
        ax[i].set_xticks([]); ax[i].set_yticks([]); ax[i].set_title(ttl, fontsize=10)
    fig.suptitle(f"{it['name']}  {TAG} {step}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    full_p = os.path.join(OUT, "full.png")
    fig.savefig(full_p)
    plt.close(fig)
    log(f"[out ] {full_p}")

    for name, (x0, x1, y0, y1) in CROPS.items():
        panels = [
            (to_img(ref[:, y0:y1, x0:x1]), "photo"),
            (to_img(i_gt[:, y0:y1, x0:x1]), f"GTG {crop_psnr[name]['gtg']:.1f}"),
            (to_img(i_pr[:, y0:y1, x0:x1]), f"{TAG} {crop_psnr[name][TAG.lower()]:.1f}"),
        ]
        h, w = panels[0][0].shape[:2]
        fig, ax = plt.subplots(1, 3, figsize=(3 * 4.2, 4.2 * h / max(w, 1) + 0.6), dpi=140)
        for i, (im, ttl) in enumerate(panels):
            ax[i].imshow(im)
            ax[i].set_xticks([]); ax[i].set_yticks([]); ax[i].set_title(ttl, fontsize=11)
        fig.suptitle(name, fontsize=12)
        fig.tight_layout()
        p = os.path.join(OUT, f"{name}.png")
        fig.savefig(p)
        plt.close(fig)
        log(f"[out ] {p}")

    log(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
