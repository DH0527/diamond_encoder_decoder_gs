"""고친 intra_group_hungarian_attr_loss 가 이전과 같은 값을 내는지, 얼마나 빨라졌는지.

pinned 전송과 스레드 풀은 배정 결과를 바꾸지 않아야 한다. 바꾸면 학습이 달라진다.
비교 기준은 같은 입력에 대한 예전 구현(단일 스레드, pageable .cpu())이다.
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import linear_sum_assignment
from can3tok.losses import intra_group_hungarian_attr_loss, ATTR_STD

CK = sys.argv[1] if len(sys.argv) > 1 else "runs/T16k_20260913_093332/ckpt_step00070000.pt"
D = load(CK, "val", 166, 1)
model, cfg, it = D["model"], D["cfg"], D["it"]
x = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
pred = o.get("attr_pred", o["pred"]).float()
LAYOUT = {"scale": 3, "rot": 6, "opacity": 10, "color": 11, "sh": 14}
gsz = int(cfg.group_size)


def old_impl(pred, target, mask, group_size, layout, min_live=8):
    """예전 구현 그대로: 단일 스레드 + pageable 전송."""
    b, n, c = pred.shape
    ng = n // int(group_size); bg = b * ng
    a_s, a_r, a_o = layout["scale"], layout["rot"], layout["opacity"]
    p = pred[:, :ng*group_size].reshape(bg, group_size, c).float()
    t = target[:, :ng*group_size].reshape(bg, group_size, c).float()
    m = mask[:, :ng*group_size].reshape(bg, group_size) > 0.5
    with torch.no_grad():
        cost = torch.cdist(p[..., 0:3], t[..., 0:3])
        inv = ~m
        cost = cost.masked_fill(inv.unsqueeze(2), 1.0e6).masked_fill(inv.unsqueeze(1), 1.0e6)
        cost_np = cost.cpu().numpy(); m_np = m.cpu().numpy()
    groups, rows, cols = [], [], []
    for i in range(bg):
        li = np.flatnonzero(m_np[i])
        if li.size < min_live: continue
        r, col = linear_sum_assignment(cost_np[i][np.ix_(li, li)])
        groups.append(np.full(r.shape[0], i, dtype=np.int64)); rows.append(li[r]); cols.append(li[col])
    gi = torch.from_numpy(np.concatenate(groups)).cuda()
    ri = torch.from_numpy(np.concatenate(rows)).cuda()
    ci = torch.from_numpy(np.concatenate(cols)).cuda()
    ps, ts = p[gi, ri], t[gi, ci]
    pq = ps[..., a_r:a_r+4] * torch.sign(ps[..., a_r:a_r+1].detach() + 1e-12)
    tq = ts[..., a_r:a_r+4] * torch.sign(ts[..., a_r:a_r+1].detach() + 1e-12)
    sd = torch.tensor(ATTR_STD[:11], device=p.device, dtype=p.dtype).clamp(min=1e-3)
    return (((ps[..., a_s:a_s+3] - ts[..., a_s:a_s+3]).abs()/sd[0:3]).mean(),
            ((ps[..., a_o:a_o+1] - ts[..., a_o:a_o+1]).abs()/sd[7:8]).mean(),
            ((pq - tq).abs()/sd[3:7]).mean())


def timed(fn, reps=5):
    fn()                                  # 워밍업 (pinned 버퍼 할당, 풀 생성)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): r = fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/reps, r

dt_old, r_old = timed(lambda: old_impl(pred, tg, mk, gsz, LAYOUT))
dt_new, r_new = timed(lambda: intra_group_hungarian_attr_loss(pred, tg, mk, gsz, LAYOUT))
print(f"[old] {dt_old*1e3:7.1f} ms   scale {float(r_old[0]):.6f}  opacity {float(r_old[1]):.6f}  rot {float(r_old[2]):.6f}")
print(f"[new] {dt_new*1e3:7.1f} ms   scale {float(r_new[0]):.6f}  opacity {float(r_new[1]):.6f}  rot {float(r_new[2]):.6f}")
same = all(abs(float(a)-float(b)) < 1e-6 for a, b in zip(r_old, r_new))
print(f"[결과] 동일: {same}    속도 {dt_old/dt_new:.2f}x    "
      f"워커 {os.environ.get('CAN3TOK_HUNG_WORKERS','16')}")

# 기울기도 같은지. 값이 같아도 그래프가 다르면 학습이 달라진다.
def grads(fn):
    p2 = pred.detach().clone().requires_grad_(True)
    s, o_, r = fn(p2)
    (s + o_ + r).backward()
    return p2.grad
g_old = grads(lambda q: old_impl(q, tg, mk, gsz, LAYOUT))
g_new = grads(lambda q: intra_group_hungarian_attr_loss(q, tg, mk, gsz, LAYOUT))
print(f"[기울기] 최대 절대차 {float((g_old-g_new).abs().max()):.3e}  "
      f"(old 최대 {float(g_old.abs().max()):.3e})")
