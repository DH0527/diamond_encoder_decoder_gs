import os,sys,torch
R=os.environ["REPO"]; sys.path.insert(0,R); os.chdir(R)
from can3tok.losses import intra_group_attr_moment_loss
torch.manual_seed(0)
B,G,NG,C=1,256,8,14
lay={"scale":3,"rot":6,"opacity":10,"color":11,"sh":11}
tgt=torch.randn(B,G*NG,C).cuda(); msk=torch.ones(B,G*NG).cuda()
mu=tgt.reshape(B*NG,G,C).mean(1,keepdim=True)
def shrink(a):
    v=tgt.reshape(B*NG,G,C)
    out=(mu+a*(v-mu)).reshape(B,G*NG,C).contiguous()
    out[...,0:3]=tgt[...,0:3]          # 위치는 GT 유지 (실측: xyz 비율 0.98)
    return out
print(" 분산비   spread    slope    grad_norm  (0.39 = 실측 logscale)")
for a in (1.0,0.8,0.5,0.39,0.2,0.05):
    pr=torch.nn.Parameter(shrink(a))
    sp,sl=intra_group_attr_moment_loss(pr,tgt,msk,G,lay)
    (sp+sl).backward()
    print(f"  {a:4.2f}   {sp.item():7.4f}  {sl.item():7.4f}   {pr.grad.norm().item():9.4f}")
# 기울기 방향 확인: 한 스텝 내려가면 분산비가 올라가야 한다
pr=torch.nn.Parameter(shrink(0.39))
def ratio(x):
    v=x.reshape(B*NG,G,C)[...,3:]; t=tgt.reshape(B*NG,G,C)[...,3:]
    return (v.std(1).mean()/t.std(1).mean()).item()
r0=ratio(pr.data)
opt=torch.optim.SGD([pr],lr=5.0)
for _ in range(30):
    opt.zero_grad(); sp,sl=intra_group_attr_moment_loss(pr,tgt,msk,G,lay); (sp+sl).backward(); opt.step()
print(f"\nSGD 30스텝: 속성 분산비 {r0:.3f} -> {ratio(pr.data):.3f} (1.0 로 가야 정상)")
