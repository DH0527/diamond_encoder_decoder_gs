import os,sys,torch,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from can3tok.losses import intra_group_attr_moment_loss, intra_group_attr_set_loss
ck=sys.argv[1]
D=load(ck,"val",166,1); it=D["it"]; cfg=D["cfg"]; model=D["model"]
x=it["input"][None].cuda().float(); mk=it["mask"][None].cuda().float()
tg=it["target"][None].cuda().float()[...,:cfg.target_dim]
kw={}
if "enc_input" in it: kw=dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"]=it["group_anchor"][None].cuda().float()
with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
    o=model(x,mk,run_decode=True,run_gen=False,**kw)
ap=o.get("attr_pred",o["pred"]).float()
lay=dict(cfg.attr_layout) if hasattr(cfg,"attr_layout") else {"scale":3,"rot":6,"opacity":10,"color":11,"sh":11}
G=int(cfg.group_size)
sp,sl=intra_group_attr_moment_loss(ap,tg,mk,G,lay)
aset=intra_group_attr_set_loss(ap,tg,mk,G,lay,chunk_groups=96)
print(f"실측 항 값:  attr_spread {sp.item():.4f}   attr_slope {sl.item():.4f}   attr_set {aset.item():.4f}")
print(f"학습로그의 aset 는 1.63 부근 -> 동일 스케일 확인")
print(f"\n같은 기여도(=aset*2.0 수준)를 주려면:")
print(f"  w_attr_spread ~ {2.0*aset.item()/max(sp.item(),1e-6):.1f}")
print(f"  w_attr_slope  ~ {2.0*aset.item()/max(sl.item(),1e-6):.1f}")
