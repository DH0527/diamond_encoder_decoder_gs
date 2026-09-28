"""이 파라미터화가 난간 속성을 표현할 수 있는가. 이미지를 거치지 않고 판정한다.

swap 이 증명한 것: 예측 위치 79,821개에 GT 속성을 최근접으로 옮기면 난간이 선으로 나온다.
즉 목표 속성값은 이 점집합 위에 존재한다.
viewgen / viewgen_mv 가 보인 것: 그 목표를 이미지로 찾게 하면 감독한 시점만 좋아지고
미관측 시점은 나빠진다. 속성은 시점에서 과소결정이니 당연하다.

그래서 여기서는 그 목표 속성을 per-point 회귀 목표로 직접 준다. 손실은 이미지를 전혀
보지 않으므로, 렌더 지표가 오르면 그것은 시점과 무관한 3D 이득이다.

  표현 가능하다  -> 벽은 학습 신호(집합 매칭 평균화 / 시점 수 / 가중치)다.
  표현 불가능하다 -> 벽은 파라미터화다: 셀당 8채널 코드 + PE + 공유 블록으로는
                     그룹 내 256점의 난간/글자 패턴을 담을 수 없다.
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--index", type=int, default=166)
ap.add_argument("--iters", type=int, default=800)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--views", type=int, default=6)
ap.add_argument("--scope", default="attrd", choices=["blocks", "attrd"])
ap.add_argument("--out", default="")
A = ap.parse_args()

CROPS = {"A_hood_rail": (440, 880, 120, 230), "B_front_rail": (250, 540, 170, 500),
         "D_letters": (695, 855, 222, 295)}

D = load(A.ckpt, "val", A.index, 1)
model, cfg, val = D["model"], D["cfg"], D["val"]
val.extra_real_views = int(A.views); val.view_downscale = 2; val.keep_extra_fullres = True
it = val[A.index]
x  = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1,3); sc = float(it["scale"])
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()

def mkcam(v): return Camera(camera_from_vector(np.asarray(v, np.float32)), device="cuda", downscale=1)
def fit_im(im, c):
    if im.shape[-2:] == (c.image_height, c.image_width): return im
    return F.interpolate(im[None], size=(c.image_height,c.image_width), mode="bilinear",
                         align_corners=False)[0]
cam = mkcam(it["camera"].numpy()); ref = fit_im(it["photo"].cuda().float(), cam)
ecams = [mkcam(v) for v in it["extra_cams"].numpy()]
def rend(t, c): return render_gaussians(*gauss(t, gtm, cen, sc), c)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
z0 = o["z_compact"].detach().float().clone()
with torch.no_grad():
    base = o.get("attr_pred", o["pred"]).float()[0]
gt = tg[0]

def nn_take(q, src):
    """살아 있는 슬롯만 후보로 삼는다.

    텐서는 1024그룹 x 256슬롯 = 262,144 행인데 실제 가우시안은 79,821개다. 죽은 행까지
    후보에 넣으면 살아 있는 점이 죽은 행의 좌표에서 속성을 가져와 목표가 오염된다.
    """
    idx = torch.empty(q.shape[0], dtype=torch.long, device=q.device)
    for s in range(0, q.shape[0], 4096):
        e = min(s+4096, q.shape[0])
        idx[s:e] = torch.cdist(q[s:e, :3], src[:, :3]).argmin(dim=1)
    return src[idx]
with torch.no_grad():
    TA = gt[:, 3:].clone()
    TA[gtm] = nn_take(base[gtm], gt[gtm])[:, 3:]        # 목표 속성 (예측 위치 위에서)
    TA = TA.contiguous()
print(f"[setup] {it['name']}  pts {int(gtm.sum())}  target attrs {tuple(TA.shape)}  "
      f"held views {1+len(ecams)}")

with torch.no_grad():
    g_main = rend(gt, cam).clamp(0,1)
    g_hld = [rend(gt, c).clamp(0,1) for c in ecams]

def attr_l1(p):
    """log_scale / quat(부호정렬) / opacity / sh 의 L1. 렌더되는 점만 센다."""
    p = p[gtm]; T = TA[gtm]
    ps, pq, po, pc = p[:, 3:6], p[:, 6:10], p[:, 10:11], p[:, 11:]
    ts, tq, to, tc = T[:, 0:3], T[:, 3:7], T[:, 7:8], T[:, 8:]
    pq = F.normalize(pq, dim=-1); tq = F.normalize(tq, dim=-1)
    sgn = torch.sign((pq*tq).sum(-1, keepdim=True)); sgn = torch.where(sgn == 0, 1.0, sgn)
    d = {"sc": (ps-ts).abs().mean(), "rot": (pq - sgn*tq).abs().mean(),
         "op": (po-to).abs().mean(), "sh": (pc-tc).abs().mean()}
    d["all"] = d["sc"] + d["rot"] + d["op"] + d["sh"]
    return d

def score(p):
    with torch.no_grad():
        m = rend(p, cam).clamp(0,1)
        o = {"MAIN_gtg": float(psnr(m, g_main)), "MAIN_photo": float(psnr(m, ref))}
        for cn,(x0,x1,y0,y1) in CROPS.items():
            o[cn] = float(psnr(m[:, y0:y1, x0:x1], g_main[:, y0:y1, x0:x1]))
        h = [float(psnr(rend(p, c).clamp(0,1), g)) for c, g in zip(ecams, g_hld)]
        o["held_gtg"] = float(np.mean(h)) if h else 0.0
        a = attr_l1(p)
        o.update({f"L1_{k}": float(v) for k, v in a.items()})
    return o, m

res = {}
res["base"], img_base = score(base)
print("[base   ] " + json.dumps({k: round(v,3) for k,v in res["base"].items()}))
# 상한: 목표 속성을 그대로 쓴 렌더. 이 아래로만 갈 수 있다.
with torch.no_grad():
    oracle = torch.cat([base[:, :3], TA], dim=-1)
res["oracle"], img_or = score(oracle)
print("[oracle ] " + json.dumps({k: round(v,3) for k,v in res["oracle"].items()}))

model.train()
for m in model.modules():
    if isinstance(m, torch.nn.Dropout): m.eval()
model.cfg.checkpoint_decode = True
for mod in model.modules():
    if hasattr(mod, "patch_chunk"): mod.patch_chunk = 32
for p in model.parameters(): p.requires_grad_(False)
tgt_mod = model.attr_decoder.blocks if A.scope == "blocks" else model.attr_decoder
bp = list(tgt_mod.parameters())
for p in bp: p.requires_grad_(True)
print(f"[{A.scope}] {sum(p.numel() for p in bp)/1e6:.3f}M params, 손실은 이미지를 보지 않는다")

zf = z0.clone()
def pred_now():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d = model.decode_compact(zf)
    return d.get("attr_pred", d["pred"]).float()[0]

opt = torch.optim.Adam(bp, lr=A.lr)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.iters, eta_min=0.0)
t0 = time.time()
for i in range(A.iters):
    opt.zero_grad(set_to_none=True)
    d = attr_l1(pred_now())
    d["all"].backward(); opt.step(); sch.step()
    if (i+1) % max(A.iters//8,1) == 0:
        s, _ = score(pred_now())
        print(f"[{A.scope}] {i+1:4d}/{A.iters}  "
              + json.dumps({k: round(v,3) for k,v in s.items()}), flush=True)
res[A.scope], img_fin = score(pred_now())
print(f"[{A.scope}] done in {time.time()-t0:.0f}s")

print("\n===== 속성 공간 회귀. 렌더 지표는 어느 시점도 손실에 쓰이지 않았다 =====")
for k in ["base", A.scope, "oracle"]:
    v = res[k]
    print(f"{k:8s} L1 {v['L1_all']:6.3f} (sc {v['L1_sc']:.3f} rot {v['L1_rot']:.3f}) | "
          f"MAIN {v['MAIN_gtg']:6.2f}  held {v['held_gtg']:6.2f}  A_hood {v['A_hood_rail']:6.2f}  "
          f"D_letters {v['D_letters']:6.2f}")

if A.out:
    json.dump(res, open(A.out + ".json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for cn,(x0,x1,y0,y1) in CROPS.items():
        panes = [("photo", ref), ("gtg", g_main), ("base", img_base),
                 (A.scope, img_fin), ("oracle attrs", img_or)]
        fig, ax = plt.subplots(len(panes), 1, figsize=(9, 2.2*len(panes)))
        for a,(nm,im) in zip(ax, panes):
            a.imshow(im[:, y0:y1, x0:x1].clamp(0,1).permute(1,2,0).cpu().numpy())
            a.set_title(nm, fontsize=8); a.axis("off")
        fig.tight_layout(); fig.savefig(f"{A.out}_{cn}.png", dpi=115)
        print("wrote", f"{A.out}_{cn}.png")
