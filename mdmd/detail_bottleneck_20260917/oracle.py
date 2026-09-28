"""난간 디테일이 어디서 죽는지 가르는 오라클 사다리.

M0 base      : 현재 모델
M2 zopt      : 디코더 동결, z_compact(16,384) 만 최적화  -> 인코더/손실 탓인가?
M3 zdec      : z + 디코더 전체 최적화 (이 씬 하나에 과적합) -> 구조 용량 한계인가?
M4 direct    : 가우시안 79,821개를 직접 최적화 -> 렌더러/지표 sanity ceiling
M5 head      : z 고정, attr_decoder 의 head_scale/head_rot 만 최적화
M6 headz     : M5 + z 도 함께

M5/M6 은 M2(난간 없음) 와 M3(난간 선명) 사이를 가른다. 난간 밴드에서 위치와
alpha 는 이미 맞고(nearest3d p50 2.71px, alpha 0.91 vs 0.95) 틀리는 것은 모양뿐인데
-- sigma_min p90 GT 12.29px vs 예측 3.85px, sigma_max p90 26.18 vs 8.85 --
scale 캡은 로그 +-3(약 20배)이라 열려 있다. 즉 캡이 막은 것이 아니라 학습된
매핑이 평균으로 눌린 것이라는 가설이고, 그 매핑이 정확히 이 두 헤드에 있다면
헤드만 풀어도 난간이 나와야 한다. 나오면 처방은 짝맞춘 scale+회전 손실이고,
안 나오면 벽은 디코더 본체(폴딩 프레임/shape 3채널)다.
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--index", type=int, default=166)
ap.add_argument("--modes", default="base,zopt,zdec,direct")
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--lr_z", type=float, default=3e-2)
ap.add_argument("--lr_d", type=float, default=1e-4)
ap.add_argument("--lr_h", type=float, default=1e-3)
ap.add_argument("--target", default="gtg", choices=["gtg","photo"])
ap.add_argument("--out", default="")
ap.add_argument("--edge", type=float, default=1.0)
A = ap.parse_args()

CROPS = {"A_hood_rail": (440, 880, 120, 230), "B_front_rail": (250, 540, 170, 500),
         "C_lettering": (615, 855, 205, 322),
         # C_lettering 은 판넬이 면적을 지배해서 글자가 깨져도 dB 가 오른다.
         # WESTERN PACIFIC 획만 남긴 상자.
         "D_letters": (695, 855, 222, 295)}

D = load(A.ckpt, "val", A.index, 1)
model, it, cam, ref, cfg = D["model"], D["it"], D["cam"], D["ref"], D["cfg"]
x  = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1,3); sc = float(it["scale"])
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()

def rend(tt):
    return render_gaussians(*gauss(tt, gtm, cen, sc), cam)
def crops(img, tgt):
    out = {}
    for k,(x0,x1,y0,y1) in CROPS.items():
        out[k] = float(psnr(img[:, y0:y1, x0:x1], tgt[:, y0:y1, x0:x1]))
    out["full"] = float(psnr(img, tgt))
    return out

with torch.no_grad():
    i_gt = rend(tg[0]).clamp(0,1)
TGT = i_gt if A.target == "gtg" else ref
print(f"[setup] {it['name']}  {cam.image_width}x{cam.image_height}  pts {int(gtm.sum())}  target={A.target}")
print(f"[ref  ] GT-gaussians vs photo: " + json.dumps(crops(i_gt, ref)))

res = {}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
z0 = o["z_compact"].detach().float().clone()
with torch.no_grad():
    base = o.get("attr_pred", o["pred"]).float()[0]
    i_b = rend(base).clamp(0,1)
res["M0_base"] = {"vs_target": crops(i_b, TGT), "vs_photo": crops(i_b, ref)}
print("[M0   ] base          " + json.dumps(res["M0_base"]["vs_target"]))

def enable_ckpt():
    """decoder 의 chunk gradient-checkpoint 는 self.training 에만 걸려 있다."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout): m.eval()
    model.cfg.checkpoint_decode = True
    for mod in model.modules():
        if hasattr(mod, "patch_chunk"): mod.patch_chunk = 32
    print("[mem  ] checkpointing on, patch_chunk=32")

def sobel(t):
    k = t.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3).repeat(3,1,1,1)
    gx = F.conv2d(t[None], k, padding=1, groups=3)
    gy = F.conv2d(t[None], k.transpose(2,3), padding=1, groups=3)
    return torch.cat([gx,gy],1)[0]

TGT_E = None
def run_opt(name, params, lrs, iters, get_pred):
    global TGT_E
    if TGT_E is None: TGT_E = sobel(TGT).detach()
    opt = torch.optim.Adam([{"params": p, "lr": l} for p, l in zip(params, lrs)])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.0)
    t0 = time.time(); best = None
    for i in range(iters):
        opt.zero_grad(set_to_none=True)
        pr = get_pred()
        img = rend(pr)
        loss = (img - TGT).abs().mean()
        if A.edge > 0: loss = loss + A.edge * (sobel(img) - TGT_E).abs().mean()
        loss.backward(); opt.step(); sch.step()
        if (i+1) % max(iters//8,1) == 0:
            with torch.no_grad():
                im = rend(get_pred()).clamp(0,1)
                c = crops(im, TGT)
            print(f"[{name:5s}] {i+1:4d}/{iters} L1 {loss.item():.5f}  " + json.dumps(c), flush=True)
            best = (im.detach(), c)
    with torch.no_grad():
        im = rend(get_pred()).clamp(0,1)
    print(f"[{name:5s}] done in {time.time()-t0:.0f}s")
    return im, crops(im, TGT), crops(im, ref)

modes = A.modes.split(",")
imgs = {"photo": ref, "gtg": i_gt, "base": i_b}

if "zopt" in modes:
    enable_ckpt()
    for p in model.parameters(): p.requires_grad_(False)
    z = torch.nn.Parameter(z0.clone())
    def gp():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = model.decode_compact(z)
        return d.get("attr_pred", d["pred"]).float()[0]
    im, ct, cp = run_opt("zopt", [z], [A.lr_z], A.iters, gp)
    res["M2_zopt"] = {"vs_target": ct, "vs_photo": cp}; imgs["zopt"] = im

if "zdec" in modes:
    enable_ckpt()
    for p in model.parameters(): p.requires_grad_(False)
    dec = [p for n,p in model.named_parameters() if n.startswith(("decompressor","decoder","attr_decoder","joint"))]
    for p in dec: p.requires_grad_(True)
    z = torch.nn.Parameter(z0.clone())
    def gp2():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = model.decode_compact(z)
        return d.get("attr_pred", d["pred"]).float()[0]
    im, ct, cp = run_opt("zdec", [z, dec], [A.lr_z, A.lr_d], A.iters, gp2)
    res["M3_zdec"] = {"vs_target": ct, "vs_photo": cp, "n_dec_params": sum(p.numel() for p in dec)}
    imgs["zdec"] = im
    for p in model.parameters(): p.requires_grad_(False)

SHAPE_H  = ["head_scale", "head_rot"]
APPEAR_H = ["head_color", "head_sh", "head_opacity"]

def shape_heads(which="all", group=SHAPE_H):
    """which: all | weight | bias  (bias 3+4 개는 모든 점의 log-scale/rot 을 통째로 옮긴다)"""
    ad = getattr(model, "attr_decoder", None)
    if ad is None:
        raise SystemExit("attr_decoder 가 없는 체크포인트다 (attr_decoder_layers=0?)")
    named = []
    for h in group:
        m = getattr(ad, h, None)
        if m is not None: named += list(m.named_parameters())
    if which == "weight":
        named = [(n, p) for n, p in named if "bias" not in n]
    elif which == "bias":
        named = [(n, p) for n, p in named if "bias" in n]
    return [p for _, p in named]

HEAD_MODES = ["head", "headw", "headb", "headz", "headc", "heada"]
if any(m in modes for m in HEAD_MODES):
    enable_ckpt()
    ALL = shape_heads("all", SHAPE_H + APPEAR_H)
    ORIG = [p.detach().clone() for p in ALL]

    def run_head(tag, key, which, with_z, group=SHAPE_H):
        for p in model.parameters(): p.requires_grad_(False)
        with torch.no_grad():                            # 매 모드 독립 출발점
            for p, s in zip(ALL, ORIG): p.copy_(s)
        hp = shape_heads(which, group)
        for p in hp: p.requires_grad_(True)
        n_hp = sum(p.numel() for p in hp)
        print(f"[{tag:5s}] {'+'.join(group)} {which}: {n_hp} params  z={'free' if with_z else 'fixed'}")
        zt = torch.nn.Parameter(z0.clone()) if with_z else z0.clone()
        def gp_h():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d = model.decode_compact(zt)
            return d.get("attr_pred", d["pred"]).float()[0]
        ps = [hp, [zt]] if with_z else [hp]
        lr = [A.lr_h, A.lr_z] if with_z else [A.lr_h]
        im, ct, cp = run_opt(tag, ps, lr, A.iters, gp_h)
        res[key] = {"vs_target": ct, "vs_photo": cp, "n_params": n_hp}
        imgs[tag] = im

    if "head" in modes:  run_head("head",  "M5_head",  "all",    False)
    if "headw" in modes: run_head("headw", "M5w_head_weight", "weight", False)
    if "headb" in modes: run_head("headb", "M5b_head_bias",   "bias",   False)
    if "headz" in modes: run_head("headz", "M6_headz", "all",    True)
    # 글자는 색/SH 로 그려진다. shape 헤드만 풀면 획이 갈라질 수 있다.
    if "headc" in modes: run_head("headc", "M7_appear", "all", False, APPEAR_H)
    if "heada" in modes: run_head("heada", "M8_shape_appear", "all", False, SHAPE_H + APPEAR_H)

    with torch.no_grad():
        for p, s in zip(ALL, ORIG): p.copy_(s)
    for p in model.parameters(): p.requires_grad_(False)

# ---- 부분 해동 사다리: z 고정, 디코딩 경로의 모듈을 단위별로 푼다 -------------
# 헤드(2.8k)는 24.8 에서 막히고 전체(z+디코더)는 30.8 이다. 30 을 내는 최소 단위를 찾는다.
A_XATTN = ["attr_decoder." + n for n in
           ("xattn", "xnorm", "to_code_tok", "to_shape_tok", "nbr_emb", "xattn_gate")]
A_HEADS = ["attr_decoder." + n for n in
           ("head_scale", "head_rot", "head_opacity", "head_color", "head_sh",
            "head_nudge", "scale_base_a")]
A_COND  = ["attr_decoder.in_proj", "attr_decoder.slot_emb"]
BODY_SETS = {
    "b_attrd": ["attr_decoder"],                                        #  3.66M
    "b_fold":  ["decompressor.fold_head", "decompressor.shape_xyz"],    #  2.36M
    "b_dcmp":  ["decompressor"],                                        # 15.98M
    "b_cdec":  ["decoder"],                                             # 50.75M
    "b_dec":   ["decompressor", "decoder", "attr_decoder"],             # 70.38M, z 고정 천장
    # attr_decoder 내부 사다리. 헤드만 24.8, 모듈 전체 31.7 인 그 사이를 가른다.
    # 모두 lr_d 로 돌려야 b_attrd(31.7) 와 같은 조건에서 비교된다.
    "a_blocks": ["attr_decoder.blocks"],                                #  3.16M
    "a_cond":   A_COND,                                                 #  0.22M
    "a_xattn":  A_XATTN,                                                #  0.28M
    "a_heads":  A_HEADS,                                                #  2.8k, 같은 lr 대조군
    "a_noblk":  A_COND + A_XATTN + A_HEADS,                             # 블록 빼고 전부
}
if any(m in modes for m in BODY_SETS):
    enable_ckpt()
    SNAP = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    zf = z0.clone()
    def gp_b():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = model.decode_compact(zf)
        return d.get("attr_pred", d["pred"]).float()[0]

    for tag in [t for t in BODY_SETS if t in modes]:
        model.load_state_dict(SNAP, strict=False)       # 모드마다 독립 출발점
        for p in model.parameters(): p.requires_grad_(False)
        bp = []
        for path in BODY_SETS[tag]:
            mod = model
            for a in path.split("."): mod = getattr(mod, a, None)
            if mod is None: continue                    # 이 설정에서 없는 갈래 (head_sh 등)
            ps = [mod] if isinstance(mod, torch.nn.Parameter) else list(mod.parameters())
            for p in ps: p.requires_grad_(True)
            bp += ps
        n_bp = sum(p.numel() for p in bp)
        print(f"[{tag:7s}] {'+'.join(BODY_SETS[tag])}: {n_bp/1e6:.3f}M params  z=fixed")
        im, ct, cp = run_opt(tag, [bp], [A.lr_d], A.iters, gp_b)
        res[tag] = {"vs_target": ct, "vs_photo": cp, "n_params": n_bp}
        imgs[tag] = im

    model.load_state_dict(SNAP, strict=False)
    for p in model.parameters(): p.requires_grad_(False)

if "direct" in modes:
    gp3_t = torch.nn.Parameter(base.clone())
    def gp3(): return gp3_t
    im, ct, cp = run_opt("dirct", [gp3_t], [3e-3], A.iters, gp3)
    res["M4_direct"] = {"vs_target": ct, "vs_photo": cp}; imgs["direct"] = im

print("\n===== SUMMARY (PSNR vs %s) =====" % A.target)
for k in ["M0_base","M2_zopt","M3_zdec","M4_direct",
          "M5_head","M5w_head_weight","M5b_head_bias","M6_headz",
          "M7_appear","M8_shape_appear",
          "b_attrd","b_fold","b_dcmp","b_cdec","b_dec",
          "a_blocks","a_cond","a_xattn","a_heads","a_noblk"]:
    if k in res:
        v = res[k]["vs_target"]
        print(f"{k:16s} full {v['full']:6.2f}   A_hood {v['A_hood_rail']:6.2f}   "
              f"B_front {v['B_front_rail']:6.2f}   C_letter {v['C_lettering']:6.2f}   "
              f"D_letters {v['D_letters']:6.2f}")

if A.out:
    json.dump(res, open(A.out + ".json","w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order = [k for k in ["photo","gtg","base","zopt","head","headw","headb","headc",
                         "heada","headz","a_heads","a_cond","a_xattn","a_noblk",
                         "a_blocks","b_attrd","b_fold","b_dcmp","b_cdec","b_dec",
                         "zdec","direct"] if k in imgs]
    # 다음 질문마다 400스텝을 다시 돌리지 않도록 원본 렌더를 남긴다.
    np.savez_compressed(A.out + "_imgs.npz",
                        **{k: v.detach().clamp(0, 1).cpu().numpy().astype(np.float16)
                           for k, v in imgs.items()})
    print("wrote", A.out + "_imgs.npz")
    for cname,(x0,x1,y0,y1) in CROPS.items():
        n=len(order); fig,ax=plt.subplots(n,1,figsize=(9, n*9*(y1-y0)/(x1-x0)+0.7), dpi=130)
        for k,nm in enumerate(order):
            ax[k].imshow(imgs[nm].clamp(0,1)[:,y0:y1,x0:x1].permute(1,2,0).cpu().numpy())
            ax[k].set_xticks([]); ax[k].set_yticks([]); ax[k].set_title(nm, fontsize=9)
        fig.tight_layout(); fig.savefig(f"{A.out}_{cname}.png"); print("wrote", f"{A.out}_{cname}.png")
