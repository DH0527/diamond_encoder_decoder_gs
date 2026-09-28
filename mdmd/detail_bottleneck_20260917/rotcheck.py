"""A[rot 0.016] 이 낮은 오차인지, 손실이 각도를 납작하게 눌러 낮아 보이는 것인지.

quat_loss = 1 - |dot(q_pred, q_tgt)| 이고 단위 사원수에서 |dot| = cos(theta/2) 이므로
이 값은 각도 오차의 2차 함수다: 작은 theta 에서 1 - cos(theta/2) ~ theta^2 / 8.
따라서 20도짜리 오차도 0.015 로 보고된다. 여기서는 실제 각도 분포를 직접 잰다.

대조군 두 개를 같이 둔다.
  group_mean : 그룹의 평균 사원수를 그 그룹 256점 전부에 쓴 예측. 평균화의 하한.
  gt_shuffle : 그룹 안에서 GT 방향을 무작위로 섞은 것. 그룹 내 방향장이 실제로 얼마나
               다양한지, 즉 평균화로 잃는 것이 얼마인지의 눈금.
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--index", type=int, default=166)
ap.add_argument("--out", default="")
A = ap.parse_args()

D = load(A.ckpt, "val", A.index, 1)
model, cfg = D["model"], D["cfg"]
it = D["it"]
x  = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
pred = o.get("attr_pred", o["pred"]).float()[0]
gt = tg[0]

G = 256
def nn_idx(q, src):
    idx = torch.empty(q.shape[0], dtype=torch.long, device=q.device)
    for s in range(0, q.shape[0], 4096):
        e = min(s+4096, q.shape[0])
        idx[s:e] = torch.cdist(q[s:e, :3], src[:, :3]).argmin(dim=1)
    return idx

pl, gl = pred[gtm], gt[gtm]
idx = nn_idx(pl, gl)
qp = F.normalize(pl[:, 6:10], dim=-1)
qt = F.normalize(gl[idx, 6:10], dim=-1)

def report(name, a, b):
    dot = (a*b).sum(-1).abs().clamp(max=1.0)
    ang = torch.rad2deg(2.0 * torch.acos(dot.clamp(max=1.0)))
    ql = (1.0 - dot)
    p = lambda t, q: float(torch.quantile(t.float(), q))
    r = {"quat_loss": float(ql.mean()),
         "angle_mean": float(ang.mean()),
         "angle_p50": p(ang, 0.5), "angle_p90": p(ang, 0.9),
         "L1_per_comp": float((a - torch.sign((a*b).sum(-1, keepdim=True))*b).abs().mean())}
    print(f"[{name:11s}] quat_loss {r['quat_loss']:.4f}  각도 mean {r['angle_mean']:5.1f}deg  "
          f"p50 {r['angle_p50']:5.1f}  p90 {r['angle_p90']:5.1f}   per-comp L1 {r['L1_per_comp']:.3f}")
    return r

print(f"[setup] {it['name']}  살아있는 점 {int(gtm.sum())}")
print(f"[변환  ] quat_loss 0.016 -> 각도 {np.rad2deg(2*np.arccos(1-0.016)):.1f}deg "
      f"| 0.100 -> {np.rad2deg(2*np.arccos(1-0.100)):.1f}deg "
      f"| 0.300 -> {np.rad2deg(2*np.arccos(1-0.300)):.1f}deg")

res = {"pred_vs_nnGT": report("pred", qp, qt)}

# 그룹 평균 사원수: 그룹 내 방향을 하나로 눌렀을 때. 부호를 첫 원소에 맞춰 정렬해야 평균이 의미를 갖는다.
ng = gt.shape[0] // G
qg = F.normalize(gt[:, 6:10], dim=-1).reshape(ng, G, 4)
mg = gt[:, 6:10].new_zeros(ng, G, 4)
ref = qg[:, :1]
sgn = torch.sign((qg*ref).sum(-1, keepdim=True)); sgn = torch.where(sgn == 0, 1.0, sgn)
mean_q = F.normalize((qg*sgn).mean(dim=1, keepdim=True), dim=-1).expand(-1, G, -1)
mg = mean_q.reshape(-1, 4)
res["groupmean_vs_GT"] = report("group_mean", mg[gtm], F.normalize(gt[gtm][:, 6:10], dim=-1))

# 그룹 내 무작위 섞기: 그룹 내 방향장의 실제 다양성 눈금
perm = torch.stack([torch.randperm(G, device=gt.device) for _ in range(ng)])
sh = qg.gather(1, perm.unsqueeze(-1).expand(-1, -1, 4)).reshape(-1, 4)
res["shuffled_vs_GT"] = report("gt_shuffle", sh[gtm], F.normalize(gt[gtm][:, 6:10], dim=-1))

print("\n===== 읽는 법 =====")
print("pred 가 group_mean 과 비슷하면 모델은 그룹당 방향 하나를 내고 있는 것이고,")
print("gt_shuffle 이 크면 그룹 내 방향장이 실제로 다양해서 평균화로 잃는 것이 크다는 뜻이다.")

# 등방성 가우시안에서는 회전이 관측되지 않으므로 위 각도는 부풀려져 있다.
# GT 이방성으로 갈라야 정직하다. 난간은 이방성이 큰 쪽에 있다.
print("\n===== GT 이방성별 (등방성에서는 회전이 무의미하다) =====")
ls = gl[:, 3:6].clamp(-15, 5)
smin, smax = ls.min(-1).values.exp(), ls.max(-1).values.exp()
aniso = (smax / smin.clamp(min=1e-12))
qm = F.normalize(mg[gtm], dim=-1)
qg_live = F.normalize(gl[:, 6:10], dim=-1)
def ang(a, b):
    return torch.rad2deg(2*torch.acos((a*b).sum(-1).abs().clamp(max=1.0)))
a_pred, a_mean = ang(qp, qt), ang(qm, qg_live)
buckets = [(1.0,2.0),(2.0,4.0),(4.0,8.0),(8.0,1e9)]
res["by_aniso"] = {}
for lo, hi in buckets:
    m = (aniso >= lo) & (aniso < hi)
    if int(m.sum()) < 100: continue
    pm = float(torch.quantile(a_pred[m].float(), 0.5))
    mm = float(torch.quantile(a_mean[m].float(), 0.5))
    tag = f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f">{lo:.0f}"
    res["by_aniso"][tag] = {"n": int(m.sum()), "pred_p50": pm, "groupmean_p50": mm}
    print(f"  aniso {tag:>5s}  n {int(m.sum()):6d}   pred p50 {pm:5.1f}deg   "
          f"group_mean p50 {mm:5.1f}deg   {'모델이 평균보다 나쁨' if pm > mm else '모델이 평균보다 나음'}")
if A.out: json.dump(res, open(A.out + ".json", "w"), indent=1)
