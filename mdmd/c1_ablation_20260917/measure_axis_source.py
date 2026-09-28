"""Where can a needle axis come from? C1 40000, same camera as the gate.

Compares GT Gaussian principal axis against:
  - predicted Gaussian axis (C1)
  - decoded folding cell frame (what F2 used)
  - PCA of kNN xyz of GT centers
  - PCA of kNN xyz of pred centers
  - PCA of the whole cell's xyz

Then renders gated scale/rot swaps (keep pred color/opacity) to see whether
3D correspondence is usable if we only touch well-paired high-aniso points.
"""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "detail_bottleneck_20260917"))
from common import *
from can3tok.losses import _quat_to_R

CKPT = "runs/C1_detail_20260916_021829/ckpt_step00040000.pt"
OUT = "mdmd/c1_ablation_20260917/axis_source_c1_40000"
INDEX = 166
KNN = 16
TAU = 1e-3
CROPS = {"A_hood_rail": (440, 880, 120, 230),
         "B_front_rail": (250, 540, 170, 500),
         "D_letters": (695, 855, 222, 295)}


def princ_axis(log_s, q):
    R = _quat_to_R(F.normalize(q, dim=-1))
    k = log_s.argmax(-1)
    ax = R[torch.arange(R.shape[0], device=R.device), :, k]
    return ax / ax.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def ang_deg(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(-1).abs().clamp(0, 1)))


def knn_pca_axis(query, cloud, k=16):
    """First PC of k nearest cloud points, unsigned, (N,3)."""
    n = query.shape[0]
    axes = torch.empty(n, 3, device=query.device)
    bs = 2048
    k = min(k, cloud.shape[0])
    for s in range(0, n, bs):
        e = min(s + bs, n)
        d = torch.cdist(query[s:e], cloud)
        _, ix = d.topk(k, largest=False, dim=1)
        nb = cloud[ix]  # (b,k,3)
        nb = nb - nb.mean(1, keepdim=True)
        # batched SVD on (b,k,3)
        try:
            _, _, vh = torch.linalg.svd(nb, full_matrices=False)
            axes[s:e] = vh[:, 0, :]
        except RuntimeError:
            axes[s:e] = torch.tensor([1.0, 0.0, 0.0], device=query.device)
    nrm = axes.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return axes / nrm


def qnt(t):
    t = t.float()
    if t.numel() == 0:
        return {"n": 0}
    return {k: float(v) for k, v in {
        "n": int(t.numel()),
        "p10": torch.quantile(t, 0.1),
        "p50": torch.quantile(t, 0.5),
        "p90": torch.quantile(t, 0.9),
        "mean": t.mean(),
    }.items()}


def crop_psnr(a, b, box):
    x0, x1, y0, y1 = box
    return float(psnr(a[:, y0:y1, x0:x1], b[:, y0:y1, x0:x1]))


D = load(CKPT, "val", INDEX, 1)
val = D["val"]
val.holdout_own_photo = False
it = val[INDEX]
print(f"[setup] {it.get('name')} cam={it['camera'][:3].tolist()}", flush=True)
model, cfg = D["model"], D["cfg"]
x = it["input"][None].cuda().float()
mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1, 3)
sc = float(it["scale"])
kw = {}
if "enc_input" in it:
    kw = dict(enc_x=it["enc_input"][None].cuda().float(),
              enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it:
    kw["group_anchor"] = it["group_anchor"][None].cuda().float()
cam = Camera(camera_from_vector(it["camera"].numpy().astype(np.float32)),
             device="cuda", downscale=1)
G = int(model.layout["group_size"])

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
pr = o.get("attr_pred", o["pred"]).float()[0]
gt = tg[0]
ctx = o.get("_ctx") or {}
fold_rot = ctx.get("fold_rot")
if fold_rot is None and ctx.get("fold_frame") is not None:
    from can3tok.compressor import axis_angle_to_matrix
    fold_rot = axis_angle_to_matrix(ctx["fold_frame"][..., 3:6])
fold_rot = fold_rot[0].float() if fold_rot is not None else None  # (ng,3,3)

idx_all = torch.arange(pr.shape[0], device=pr.device)
live_i = idx_all[gtm]
pl, gl = pr[gtm], gt[gtm]
gid = (live_i // G).clamp(max=fold_rot.shape[0] - 1) if fold_rot is not None else None

# NN pred -> GT
nn = torch.empty(pl.shape[0], dtype=torch.long, device=pl.device)
for s in range(0, pl.shape[0], 4096):
    e = min(s + 4096, pl.shape[0])
    nn[s:e] = torch.cdist(pl[s:e, :3], gl[:, :3]).argmin(1)
xyz_err = (pl[:, :3] - gl[nn, :3]).norm(dim=-1)

ls_p = pl[:, 3:6].clamp(-15, 5)
ls_t = gl[nn, 3:6].clamp(-15, 5)
an_t = (ls_t.max(-1).values - ls_t.min(-1).values).exp()
an_p = (ls_p.max(-1).values - ls_p.min(-1).values).exp()
ax_p = princ_axis(ls_p, pl[:, 6:10])
ax_t = princ_axis(ls_t, gl[nn, 6:10])

hi = an_t > 8
gate = hi & (xyz_err < TAU)
print(f"live={int(gtm.sum())} hi={int(hi.sum())} gated={int(gate.sum())} "
      f"xyz_p50={float(xyz_err.median()):.2e}", flush=True)

# GT-side local PCA on the GT points themselves (orientation in the point cloud?)
gl_xyz = gl[:, :3]
gt_hi_mask = ((gl[:, 3:6].clamp(-15, 5).max(-1).values
               - gl[:, 3:6].clamp(-15, 5).min(-1).values).exp() > 8)
gt_hi_xyz = gl_xyz[gt_hi_mask]
gt_hi_ax = princ_axis(gl[gt_hi_mask, 3:6].clamp(-15, 5), gl[gt_hi_mask, 6:10])
pca_gt = knn_pca_axis(gt_hi_xyz, gl_xyz, k=KNN)
pca_gt_predcloud = knn_pca_axis(pl[hi, :3], pl[:, :3], k=KNN)  # pred-cloud PCA at pred hi points

cell_ax = None
ang_cell = None
if fold_rot is not None:
    cell_ax = fold_rot[gid][:, :, 0]
    cell_ax = cell_ax / cell_ax.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # folding aniso stretch is applied before rot; axis 0 is the first frame axis.
    # also try the longest cell axis if fold_aniso is present
    if ctx.get("fold_aniso") is not None:
        aniso = ctx["fold_aniso"][0].float()  # (ng,3)
        klong = aniso.abs().argmax(-1)
        cell_ax = fold_rot[gid][torch.arange(gid.shape[0], device=gid.device), :, klong[gid]]
        cell_ax = cell_ax / cell_ax.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ang_cell = ang_deg(cell_ax, ax_t)

# Whole-cell PCA of GT xyz (the geometric object F2 hoped folding captured)
n_g = int(pr.shape[0] // G)
cell_pca = torch.zeros(n_g, 3, device=pr.device)
cell_ratio = torch.full((n_g,), float("nan"), device=pr.device)
gt_full = gt
for g in range(n_g):
    sl = slice(g * G, (g + 1) * G)
    m = gtm[sl]
    if int(m.sum()) < 8:
        continue
    pts = gt_full[sl][m, :3]
    pts = pts - pts.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(pts, full_matrices=False)
    cell_pca[g] = Vh[0]
    if S[0] > 1e-12:
        cell_ratio[g] = (S[1] / S[0]).clamp(min=0)

ang_cell_pca = ang_deg(cell_pca[gid], ax_t) if gid is not None else None

angles = {
    "pred_vs_gt": qnt(ang_deg(ax_p, ax_t)),
    "pred_vs_gt_hi": qnt(ang_deg(ax_p[hi], ax_t[hi])),
    "pred_vs_gt_gated": qnt(ang_deg(ax_p[gate], ax_t[gate])),
    "fold_vs_gt_hi": qnt(ang_cell[hi]) if ang_cell is not None else None,
    "fold_vs_gt_gated": qnt(ang_cell[gate]) if ang_cell is not None else None,
    "cell_xyz_pca_vs_gt_hi": qnt(ang_cell_pca[hi]) if ang_cell_pca is not None else None,
    "knn_pca_GTcloud_vs_gt_hi": qnt(ang_deg(pca_gt, gt_hi_ax)),
    "knn_pca_PREDcloud_vs_gt_hi": qnt(ang_deg(pca_gt_predcloud, ax_t[hi])),
    "cell_l2_over_l1_p50": float(torch.nanmedian(cell_ratio)),
    "cell_line_pct_l2l1_lt0.1": float(100.0 * torch.nanmean((cell_ratio < 0.1).float())),
}

print(json.dumps(angles, indent=2), flush=True)


def rend(t):
    return render_gaussians(*gauss(t, gtm, cen, sc), cam).clamp(0, 1)


with torch.no_grad():
    gtg = rend(gt)
    im = rend(pr)

    def apply_sr(mask_live):
        out = pr.clone()
        live = out[gtm]
        live2 = live.clone()
        live2[mask_live, 3:10] = gl[nn[mask_live], 3:10]
        out[gtm] = live2
        return out

    variants = {
        "pred": pr,
        "swap_all_attr": None,
        "sr_all": apply_sr(torch.ones(pl.shape[0], dtype=torch.bool, device=pl.device)),
        "sr_gated": apply_sr(gate),
        "sr_hi": apply_sr(hi),
        "rot_gated": None,
        "scale_gated": None,
    }
    swapped = pr.clone()
    swapped[gtm] = torch.cat([pl[:, :3], gl[nn, 3:]], dim=-1)
    variants["swap_all_attr"] = swapped

    rg = pr.clone()
    lv = rg[gtm]
    lv2 = lv.clone()
    lv2[gate, 6:10] = gl[nn[gate], 6:10]
    rg[gtm] = lv2
    variants["rot_gated"] = rg

    sg = pr.clone()
    lv = sg[gtm]
    lv2 = lv.clone()
    lv2[gate, 3:6] = gl[nn[gate], 3:6]
    sg[gtm] = lv2
    variants["scale_gated"] = sg

    renders = {k: rend(v) for k, v in variants.items()}

report = {"angles": angles, "n": {
    "live": int(gtm.sum()), "hi_aniso8": int(hi.sum()),
    "gated_hi_and_xyz_lt_1e3": int(gate.sum()),
    "xyz_p50": float(xyz_err.median()), "tau": TAU, "knn": KNN,
}, "render_vs_gtg": {}}
for k, imk in renders.items():
    report["render_vs_gtg"][k] = {
        "full": float(psnr(imk, gtg)),
        "crops": {cn: crop_psnr(imk, gtg, b) for cn, b in CROPS.items()},
    }
    print(k, json.dumps(report["render_vs_gtg"][k]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
order = ["pred", "sr_gated", "sr_hi", "sr_all", "swap_all_attr"]
x0, x1, y0, y1 = CROPS["A_hood_rail"]
fig, ax = plt.subplots(len(order), 1, figsize=(9.5, 2.0 * len(order)))
for a, nm in zip(ax, order):
    db = report["render_vs_gtg"][nm]["crops"]["A_hood_rail"]
    a.imshow(renders[nm][:, y0:y1, x0:x1].cpu().permute(1, 2, 0).numpy())
    a.set_title(f"{nm}  vs gtg {db:.2f}dB", fontsize=8)
    a.axis("off")
fig.tight_layout()
fig.savefig(OUT + "_A_hood.png", dpi=120)
print("wrote", OUT + "_A_hood.png")
json.dump(report, open(OUT + ".json", "w"), indent=2)
print("wrote", OUT + ".json")
