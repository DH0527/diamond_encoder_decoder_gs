"""a_blocks 의 이득이 3D 인지 한 장의 그림인지 가른다.

사다리가 말한 것: 렌더 이미지에 기울기가 닿는 파라미터는 attr_decoder 3.66M 뿐이고,
그 안에서 난간과 글자를 만드는 것은 self-attention 블록 3.16M 이다 (한 시점 400스텝에
full 20.42 -> 31.44). 남은 질문은 그것이 가우시안을 실제로 얇고 방향 있게 만든 것인지,
아니면 그 한 시점의 픽셀만 맞춘 것인지다.

그래서 최적화는 시점 하나만 보고, 채점은 그 시점을 쓰지 않은 다른 카메라들에서 한다.
베이스 모델은 그 카메라들로 학습된 적이 있으므로 (loader 의 extra_real_views 풀) 이
비교는 베이스에 유리한 쪽으로 기울어 있다. 그래도 블록이 이기면 결론은 안전하다.
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--index", type=int, default=166)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--lr_d", type=float, default=1e-4)
ap.add_argument("--views", type=int, default=6)
ap.add_argument("--edge", type=float, default=1.0)
ap.add_argument("--out", default="")
A = ap.parse_args()

D = load(A.ckpt, "val", A.index, 1)
model, cfg, val = D["model"], D["cfg"], D["val"]
val.extra_real_views = int(A.views)
val.view_downscale = 2
val.keep_extra_fullres = True
it = val[A.index]                                  # 카메라와 점집합을 한 번에, 자기모순 없이

x  = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1,3); sc = float(it["scale"])
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()

def mkcam(vec):
    return Camera(camera_from_vector(np.asarray(vec, np.float32)), device="cuda", downscale=1)
def fit(img, c):
    if img.shape[-2:] == (c.image_height, c.image_width): return img
    return F.interpolate(img[None], size=(c.image_height, c.image_width),
                         mode="bilinear", align_corners=False)[0]

cam = mkcam(it["camera"].numpy())
ref = fit(it["photo"].cuda().float(), cam)
ecams = [mkcam(v) for v in it["extra_cams"].numpy()]
eimgs = [fit(im.cuda().float(), c) for im, c in zip(it["extra_imgs"], ecams)]
if not ecams:
    raise SystemExit("extra_cams 가 비었다. view_pool 이 없거나 풀에 남은 시점이 없다.")
print(f"[setup] {it['name']}  main {cam.image_width}x{cam.image_height}  "
      f"held views {len(ecams)}  pts {int(gtm.sum())}")

def rend(tt, c): return render_gaussians(*gauss(tt, gtm, cen, sc), c)

with torch.no_grad():
    i_gt = rend(tg[0], cam).clamp(0,1)                      # 최적화 타깃 (main view)
    e_gt = [rend(tg[0], c).clamp(0,1) for c in ecams]       # 채점 기준 (held views)
TGT = i_gt

def score(pred):
    """main 은 최적화한 시점이므로 참고용, held 는 판정용."""
    with torch.no_grad():
        m = rend(pred, cam).clamp(0,1)
        o = {"main_gtg": float(psnr(m, i_gt)), "main_photo": float(psnr(m, ref))}
        hg, hp, ims = [], [], []
        for c, g, p in zip(ecams, e_gt, eimgs):
            r = rend(pred, c).clamp(0,1)
            hg.append(float(psnr(r, g))); hp.append(float(psnr(r, p))); ims.append(r)
        o["held_gtg"] = float(np.mean(hg)); o["held_photo"] = float(np.mean(hp))
        o["held_gtg_each"] = [round(v,2) for v in hg]
    return o, ims

res = {}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
z0 = o["z_compact"].detach().float().clone()
with torch.no_grad():
    base = o.get("attr_pred", o["pred"]).float()[0]
res["base"], base_ims = score(base)
print("[base ] " + json.dumps(res["base"]))
with torch.no_grad():
    gt_s, gt_ims = score(tg[0])
res["gt_gaussians"] = gt_s
print("[gtg  ] " + json.dumps(gt_s))

model.train()
for m in model.modules():
    if isinstance(m, torch.nn.Dropout): m.eval()
model.cfg.checkpoint_decode = True
for mod in model.modules():
    if hasattr(mod, "patch_chunk"): mod.patch_chunk = 32
for p in model.parameters(): p.requires_grad_(False)
bp = list(model.attr_decoder.blocks.parameters())
for p in bp: p.requires_grad_(True)
print(f"[blocks] {sum(p.numel() for p in bp)/1e6:.3f}M params, main view 만 보고 최적화")

def sobel(t):
    k = t.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3).repeat(3,1,1,1)
    gx = F.conv2d(t[None], k, padding=1, groups=3)
    gy = F.conv2d(t[None], k.transpose(2,3), padding=1, groups=3)
    return torch.cat([gx,gy],1)[0]

zf = z0.clone()
def pred_now():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d = model.decode_compact(zf)
    return d.get("attr_pred", d["pred"]).float()[0]

TGT_E = sobel(TGT).detach()
opt = torch.optim.Adam(bp, lr=A.lr_d)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.iters, eta_min=0.0)
t0 = time.time()
for i in range(A.iters):
    opt.zero_grad(set_to_none=True)
    img = rend(pred_now(), cam)
    loss = (img - TGT).abs().mean() + A.edge * (sobel(img) - TGT_E).abs().mean()
    loss.backward(); opt.step(); sch.step()
    if (i+1) % max(A.iters//4,1) == 0:
        s, _ = score(pred_now())
        print(f"[blocks] {i+1:4d}/{A.iters} L1 {loss.item():.5f}  " + json.dumps(s), flush=True)
res["blocks"], blk_ims = score(pred_now())
print(f"[blocks] done in {time.time()-t0:.0f}s")

print("\n===== main 은 학습한 시점, held 는 안 본 시점 =====")
for k in ["base", "blocks", "gt_gaussians"]:
    v = res[k]
    print(f"{k:13s} main_gtg {v['main_gtg']:6.2f}  held_gtg {v['held_gtg']:6.2f}  "
          f"held_photo {v['held_photo']:6.2f}   each {v['held_gtg_each']}")

if A.out:
    json.dump(res, open(A.out + ".json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(3, len(ecams))
    fig, ax = plt.subplots(n, 3, figsize=(16, 3.4*n), squeeze=False)
    for r in range(n):
        for c, (nm, im) in enumerate([("photo", eimgs[r]), ("base", base_ims[r]),
                                      ("blocks", blk_ims[r])]):
            ax[r][c].imshow(im.clamp(0,1).permute(1,2,0).cpu().numpy()); ax[r][c].axis("off")
            ax[r][c].set_title(f"held view {r}  {nm}", fontsize=8)
    fig.tight_layout(); fig.savefig(A.out + "_held.png", dpi=110)
    print("wrote", A.out + "_held.png")
