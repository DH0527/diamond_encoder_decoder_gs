"""한 스냅샷을 자기 카메라에서 크게 렌더."""
import os, sys, json, argparse
import numpy as np, torch
R = os.environ.get("REPO"); sys.path.insert(0, R); os.chdir(R)
from can3tok.train import build_parser, build_config, make_datasets, load_eval_views
from can3tok.model import build_model
from can3tok.render import Camera, render_gaussians, sh_dc_to_rgb, psnr
from can3tok.io_utils import camera_from_vector
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--split", default="val"); ap.add_argument("--index", type=int, required=True)
ap.add_argument("--downscale", type=int, default=1)
ap.add_argument("--title", default="")
a = ap.parse_args()

cfgj = json.load(open(os.path.join(os.path.dirname(a.ckpt), "args.json"))); argv=[]
for k,v in cfgj.items():
    if k in ("resume","init_from","eval_only","out_dir","init_skip"): continue
    if isinstance(v,bool):
        if v: argv.append("--"+k)
    elif isinstance(v,list):
        if v: argv += ["--"+k]+[str(x) for x in v]
    elif v is None: continue
    else: argv += ["--"+k,str(v)]
argv += ["--out_dir","/tmp/x_one"]
args = build_parser().parse_args(argv)
for _k in ("normalize_pooler_xyz","count_aware_template","attr_slot_mask",
           "shared_cell_owner","holdout_own_photo","keep_extra_fullres",
           "reuse_prev_vis","eval_pred_mask"):
    if _k not in cfgj:
        setattr(args,_k,0); print(f"  [compat] {_k} -> 0")
tr,val = make_datasets(args)
tr.anchor_spill = val.anchor_spill = int(getattr(args,"anchor_spill",8))
ev,held = load_eval_views(args)
for d in (tr,val): d.view_exclude=set(held)
cfg = build_config(args, tr.target_dim, tr.sh_dim)
model = build_model(cfg, allow_padded_tokens=getattr(args,"allow_padded_tokens",False)).cuda().eval()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"], strict=False); step = ck.get("step")

DS = val if a.split=="val" else tr
it = DS[a.index]
x  = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
kw = {}
if "enc_input" in it: kw=dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"]=it["group_anchor"][None].cuda().float()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
apred=(o.get("attr_pred",o["pred"])).float()[0]; t=tg[0]
gtm = it["mask"].cuda()>0.5
cen = it["center"].cuda().float().view(1,3); sc=float(it["scale"])
def gauss(tt,msk):
    z=tt[msk]
    return (z[:,0:3]*sc+cen, torch.exp(z[:,3:6].clamp(-15,5))*sc,
            torch.nn.functional.normalize(z[:,6:10],dim=-1),
            torch.sigmoid(z[:,10:11]), sh_dc_to_rgb(z[:,11:14]))
cam = Camera(camera_from_vector(np.asarray(it["camera"].numpy(), np.float32)), device="cuda", downscale=a.downscale)
ref = it["photo"].cuda().float()
if ref.shape[-2:] != (cam.image_height, cam.image_width):
    ref = torch.nn.functional.interpolate(ref[None], size=(cam.image_height,cam.image_width),
                                          mode="bilinear", align_corners=False)[0]
with torch.no_grad():
    i_gt = render_gaussians(*gauss(t,gtm), cam)
    i_pr = render_gaussians(*gauss(apred,gtm), cam)
img=lambda z: z.detach().clamp(0,1).permute(1,2,0).cpu().numpy()
print(f"{it['name']}  {cam.image_width}x{cam.image_height}  GT {psnr(i_gt,ref):.2f}  model {psnr(i_pr,ref):.2f}")
fig,ax=plt.subplots(3,1,figsize=(11, 3*11*cam.image_height/cam.image_width+0.9), dpi=130)
for k,(im,ttl) in enumerate([(ref,"real photograph (USED IN TRAINING - fit, not generalisation)"),
                             (i_gt,"GT Gaussians truncated to 262,144   %.2f dB"%psnr(i_gt,ref)),
                             (i_pr,"model from 16,384 latent   %.2f dB"%psnr(i_pr,ref))]):
    ax[k].imshow(img(im)); ax[k].set_xticks([]); ax[k].set_yticks([]); ax[k].set_title(ttl,fontsize=10)
fig.suptitle(a.title or f"step {step}", fontsize=11)
fig.tight_layout(rect=(0,0,1,0.98)); fig.savefig(a.out); print("wrote",a.out)
