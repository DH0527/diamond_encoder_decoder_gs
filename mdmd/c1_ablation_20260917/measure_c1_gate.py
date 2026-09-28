"""C1 체크포인트에서 구조 vs 손실 vs 루프를 가르는 게이트를 한 파일에 남긴다.

채점 시점(index 166, step_028630)은 학습 extra-view 와 겹치지 않는 MAIN 이다.
프로브처럼 디코더를 맞추지 않는다. 저장된 가중치를 한 번 렌더할 뿐이다.
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "detail_bottleneck_20260917"))
from common import *
from can3tok.losses import _quat_to_R

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--index", type=int, default=166)
ap.add_argument("--out", required=True)
A = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)

CROPS = {"A_hood_rail": (440, 880, 120, 230),
         "B_front_rail": (250, 540, 170, 500),
         "D_letters": (695, 855, 222, 295)}

D = load(A.ckpt, "val", A.index, 1)
# C1 args keep holdout_own_photo=1, so the snapshot's own camera is replaced
# with a random held-out photo. Two checkpoints then get two different MAIN
# views and crop PSNR cannot be compared. Turn it off and re-fetch.
val = D["val"]
val.holdout_own_photo = False
it = val[A.index]
D["it"] = it
print(f"[setup] {it.get('name')} holdout={int(val.holdout_own_photo)} "
      f"cam={it['camera'][:3].tolist()}", flush=True)
model, cfg = D["model"], D["cfg"]
x = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1, 3); sc = float(it["scale"])
kw = {}
if "enc_input" in it:
    kw = dict(enc_x=it["enc_input"][None].cuda().float(),
              enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it:
    kw["group_anchor"] = it["group_anchor"][None].cuda().float()
cam = Camera(camera_from_vector(it["camera"].numpy().astype(np.float32)),
             device="cuda", downscale=1)

def rend(t):
    return render_gaussians(*gauss(t, gtm, cen, sc), cam)

def nn_idx(q, src):
    i = torch.empty(q.shape[0], dtype=torch.long, device=q.device)
    for s in range(0, q.shape[0], 4096):
        e = min(s + 4096, q.shape[0])
        i[s:e] = torch.cdist(q[s:e, :3], src[:, :3]).argmin(1)
    return i

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
z = o.get("z_compact")
pr = o.get("attr_pred", o["pred"]).float()[0]
geo = o["pred"].float()[0]
gt = tg[0]

with torch.no_grad():
    photo = it["photo"].cuda().float()
    if photo.shape[-2:] != (cam.image_height, cam.image_width):
        photo = F.interpolate(photo[None], size=(cam.image_height, cam.image_width),
                              mode="bilinear", align_corners=False)[0]
    im = rend(pr).clamp(0, 1)
    gtg = rend(gt).clamp(0, 1)
    # 위치만 예측, 속성은 최근접 GT. 위치가 난간을 덮는지의 천장.
    pl, gl = pr[gtm], gt[gtm]
    idx = nn_idx(pl, gl)
    swapped = pr.clone()
    swapped[gtm] = torch.cat([pl[:, :3], gl[idx, 3:]], dim=-1)
    im_swap = rend(swapped).clamp(0, 1)

def crop_psnr(a, b, box):
    x0, x1, y0, y1 = box
    return float(psnr(a[:, y0:y1, x0:x1], b[:, y0:y1, x0:x1]))

def aniso_of(t):
    ls = t[gtm][:, 3:6].clamp(-15, 5)
    return (ls.max(-1).values - ls.min(-1).values).exp()

ap_an, gt_an = aniso_of(pr), aniso_of(gt)
qp = F.normalize(pl[:, 6:10], dim=-1)
qt = F.normalize(gl[idx, 6:10], dim=-1)
dot = (qp * qt).sum(-1).abs().clamp(max=1.0)
ang = torch.rad2deg(2.0 * torch.acos(dot.clamp(max=1.0)))
lsq = gl[idx, 3:6].clamp(-15, 5)
an_nn = (lsq.max(-1).values - lsq.min(-1).values).exp()

def qnt(t, qs=(0.1, 0.5, 0.9)):
    return {f"p{int(q*100)}": float(torch.quantile(t.float(), q)) for q in qs}

rot_by = {}
for lo, hi, name in ((1, 2, "1-2"), (2, 4, "2-4"), (4, 8, "4-8"), (8, 1e9, ">8")):
    m = (an_nn >= lo) & (an_nn < hi)
    if int(m.sum()) < 50:
        continue
    rot_by[name] = {"n": int(m.sum()), "angle_p50": float(torch.quantile(ang[m].float(), 0.5))}

xyz_err = (pl[:, :3] - gl[idx, :3]).norm(dim=-1)
z_stats = {}
if z is not None:
    zf = z.detach().float()
    z_stats = {
        "shape": list(zf.shape),
        "std": float(zf.std()),
        "abs_mean": float(zf.abs().mean()),
    }

gate = {
    "ckpt": A.ckpt,
    "index": A.index,
    "name": it.get("name"),
    "n_live": int(gtm.sum()),
    "render": {
        "psnr_vs_gtg": float(psnr(im, gtg)),
        "psnr_vs_photo": float(psnr(im, photo)),
        "psnr_gtg_vs_photo": float(psnr(gtg, photo)),
        "swap_gt_attr_vs_gtg": float(psnr(im_swap, gtg)),
        "crops_vs_gtg": {k: crop_psnr(im, gtg, b) for k, b in CROPS.items()},
        "crops_vs_photo": {k: crop_psnr(im, photo, b) for k, b in CROPS.items()},
        "crops_swap_vs_gtg": {k: crop_psnr(im_swap, gtg, b) for k, b in CROPS.items()},
        "crops_gtg_vs_photo": {k: crop_psnr(gtg, photo, b) for k, b in CROPS.items()},
    },
    "geometry": {
        "xyz_nn_p50": float(torch.quantile(xyz_err, 0.5)),
        "xyz_nn_p90": float(torch.quantile(xyz_err, 0.9)),
    },
    "aniso": {
        "pred": {**qnt(ap_an), "lt4_pct": 100 * float((ap_an < 4).float().mean())},
        "gt": {**qnt(gt_an), "lt4_pct": 100 * float((gt_an < 4).float().mean())},
    },
    "rotation_deg": {
        "p50": float(torch.quantile(ang.float(), 0.5)),
        "p90": float(torch.quantile(ang.float(), 0.9)),
        "mean": float(ang.mean()),
        "by_gt_aniso": rot_by,
    },
    "latent": z_stats,
    "read": {
        "swap_minus_pred_gtg": float(psnr(im_swap, gtg) - psnr(im, gtg)),
        "note": ("swap>>pred 이면 위치는 충분하고 속성이 병목. "
                 "pred가 swap에 가까우면 속성은 이미 따라왔고 인코더/예산을 의심."),
    },
}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [("photo", photo.cpu()), ("gt gaussians", gtg.cpu()),
        ("pred", im.cpu()), ("pred xyz + GT attr", im_swap.cpu())]
for cn, (x0, x1, y0, y1) in CROPS.items():
    fig, ax = plt.subplots(len(rows), 1, figsize=(9.5, 2.2 * len(rows)))
    for a, (nm, ims) in zip(ax, rows):
        a.imshow(ims[:, y0:y1, x0:x1].clamp(0, 1).permute(1, 2, 0).numpy())
        extra = ""
        if nm == "pred":
            extra = f"  vs gtg {gate['render']['crops_vs_gtg'][cn]:.2f}dB"
        elif nm == "pred xyz + GT attr":
            extra = f"  vs gtg {gate['render']['crops_swap_vs_gtg'][cn]:.2f}dB"
        a.set_title(nm + extra, fontsize=8)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{A.out}_{cn}.png", dpi=120)
    print("wrote", f"{A.out}_{cn}.png")

json.dump(gate, open(A.out + ".json", "w"), indent=2)
print(json.dumps({k: gate[k] for k in ("render", "aniso", "rotation_deg", "geometry", "read")}, indent=2))
