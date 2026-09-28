"""난간 가우시안이 어느 셀에 들어 있고, 그 셀이 어떤 모양인지."""
import os,sys,json,argparse
import numpy as np, torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
ap=argparse.ArgumentParser()
ap.add_argument("--ckpt",required=True); ap.add_argument("--index",type=int,default=166)
A=ap.parse_args()
CROPS={"A_hood_rail":(440,880,120,230),"B_front_rail":(250,540,170,500)}
D=load(A.ckpt,"val",A.index,1); it=D["it"]; cam=D["cam"]; cfg=D["cfg"]
G=int(cfg.group_size); NC=int(cfg.max_points)//G
t=it["target"].cuda().float()[...,:cfg.target_dim]; m=(it["mask"].cuda()>0.5)
cen=it["center"].cuda().float().view(1,3); sc=float(it["scale"])
xyz_w = t[:,0:3]*sc+cen
# project
P = cam.full_proj_transform
h = torch.cat([xyz_w, torch.ones_like(xyz_w[:,:1])],1) @ P
ndc = h[:,:3]/h[:,3:4].clamp(min=1e-6)
u = (ndc[:,0]*0.5+0.5)*cam.image_width; v=(ndc[:,1]*0.5+0.5)*cam.image_height
depth = (torch.cat([xyz_w,torch.ones_like(xyz_w[:,:1])],1) @ cam.world_view_transform)[:,2]
opac = torch.sigmoid(t[:,10])
slot = torch.arange(t.shape[0], device=t.device)
cell = slot // G
print(f"cells {NC}  group {G}  occupied {int(m.sum())}  occ/cell mean {float(m.sum())/NC:.1f}")
occ_per_cell = torch.zeros(NC, device=t.device).index_add_(0, cell, m.float())
print(f"occ/cell  p10 {occ_per_cell.quantile(.1):.0f}  median {occ_per_cell.median():.0f}  p90 {occ_per_cell.quantile(.9):.0f}  max {occ_per_cell.max():.0f}  empty cells {(occ_per_cell==0).sum().item()}")
for nm,(x0,x1,y0,y1) in CROPS.items():
    sel = m & (u>=x0)&(u<x1)&(v>=y0)&(v<y1)&(depth>0.1)&(opac>0.3)
    n=int(sel.sum()); c_sel=cell[sel]
    uc,cnt = torch.unique(c_sel, return_counts=True)
    print(f"\n### {nm}: {n} gaussians  ({100*n/int(m.sum()):.2f}% of scene)  spread over {uc.numel()} cells")
    frac = cnt.float()/occ_per_cell[uc].clamp(min=1)
    print(f"  cell 안에서 이 영역이 차지하는 비율: median {frac.median():.3f}  p90 {frac.quantile(.9):.3f}  =1.0 인 셀 {(frac>0.95).sum().item()}개")
    # 상위 셀들의 기하 모양 (PCA)
    top = uc[cnt.argsort(descending=True)[:12]]
    print("  상위 셀 (이 영역 점수 / 셀 총점 / 셀 PCA 축비 l1:l2:l3 / 셀 반경)")
    for c in top.tolist():
        pts = xyz_w[m & (cell==c)]
        q = pts - pts.mean(0, keepdim=True)
        ev = torch.linalg.svdvals(q)**2/max(pts.shape[0]-1,1)
        ev = ev/ev.max()
        rad = q.norm(dim=1).median()
        nin = int(((cell==c)&sel).sum())
        print(f"    cell {c:4d}  {nin:4d}/{int(occ_per_cell[c]):4d}  {ev[0]:.2f}:{ev[1]:.2f}:{ev[2]:.2f}  r={rad:.3f}")
# 전체 셀의 모양 통계
evs=[]
for c in range(NC):
    pts = xyz_w[m & (cell==c)]
    if pts.shape[0] < 8: continue
    q = pts - pts.mean(0,keepdim=True)
    ev = torch.linalg.svdvals(q)**2
    evs.append((ev/ev.max()).cpu().numpy())
evs=np.array(evs)
print(f"\n### 전체 셀 {len(evs)}개 PCA 축비")
print(f"  l2/l1 median {np.median(evs[:,1]):.3f}   l3/l1 median {np.median(evs[:,2]):.3f}")
print(f"  l3/l1 < 0.05 (판/선 모양) 셀 비율 {100*np.mean(evs[:,2]<0.05):.1f}%")
print(f"  l2/l1 < 0.10 (선 모양) 셀 비율 {100*np.mean(evs[:,1]<0.10):.1f}%")
