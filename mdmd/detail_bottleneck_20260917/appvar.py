"""셀 안에서 속성(scale/rot/opacity/color)이 위치에 따라 얼마나 변하는가: GT vs 예측."""
import os,sys,argparse
import numpy as np, torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--index",type=int,default=166); A=ap.parse_args()
D=load(A.ckpt,"val",A.index,1); it=D["it"]; cfg=D["cfg"]; model=D["model"]
G=int(cfg.group_size); NC=int(cfg.max_points)//G
x=it["input"][None].cuda().float(); mk=it["mask"][None].cuda().float()
t=it["target"].cuda().float()[...,:cfg.target_dim]; m=(it["mask"].cuda()>0.5)
kw={}
if "enc_input" in it: kw=dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"]=it["group_anchor"][None].cuda().float()
with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
    o=model(x,mk,run_decode=True,run_gen=False,**kw)
p=o.get("attr_pred",o["pred"]).float()[0]
cell=torch.arange(t.shape[0],device=t.device)//G
CH={"xyz":slice(0,3),"logscale":slice(3,6),"rot":slice(6,10),"opacity":slice(10,11),"sh_dc":slice(11,14)}
print(f"{'channel':10s} {'GT 셀내 std':>12s} {'pred 셀내 std':>14s} {'비율':>7s} {'GT 셀간 std':>12s} {'pred 셀간 std':>14s}")
for nm,sl in CH.items():
    gt=t[:,sl]; pr=p[:,sl]
    rows=[]
    for c in range(NC):
        s=m&(cell==c)
        if int(s.sum())<16: continue
        a=gt[s]; b=pr[s]
        rows.append((a.std(0).mean().item(), b.std(0).mean().item(), a.mean(0), b.mean(0)))
    gi=np.mean([r[0] for r in rows]); pi=np.mean([r[1] for r in rows])
    gb=torch.stack([r[2] for r in rows]).std(0).mean().item()
    pb=torch.stack([r[3] for r in rows]).std(0).mean().item()
    print(f"{nm:10s} {gi:12.4f} {pi:14.4f} {pi/max(gi,1e-9):7.2f} {gb:12.4f} {pb:14.4f}")
# 위치와 속성의 셀내 상관: 속성이 셀 안에서 '어디냐'에 따라 변하는가
print("\n셀 내부에서 위치->속성 설명력 (R^2, 선형):")
for nm,sl in CH.items():
    if nm=="xyz": continue
    r2g=[];r2p=[]
    for c in range(NC):
        s=m&(cell==c); n=int(s.sum())
        if n<32: continue
        X=t[s,0:3]; X=X-X.mean(0,keepdim=True)
        X=torch.cat([X,torch.ones(n,1,device=X.device)],1)
        for src,acc in ((t[s,sl],r2g),(p[s,sl],r2p)):
            Y=src-src.mean(0,keepdim=True)
            sol=torch.linalg.lstsq(X,Y).solution
            res=(Y-X@sol).pow(2).sum(); tot=Y.pow(2).sum()
            if tot>1e-12: acc.append(1-(res/tot).item())
    print(f"  {nm:10s} GT {np.mean(r2g):.3f}   pred {np.mean(r2p):.3f}")
