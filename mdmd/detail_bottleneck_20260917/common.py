"""체크포인트 + 스냅샷 로딩 공통부 (render_one.py 와 동일한 compat 가드)."""
import os, sys, json
import numpy as np, torch
R = os.environ.get("REPO"); sys.path.insert(0, R); os.chdir(R)
from can3tok.train import build_parser, build_config, make_datasets, load_eval_views
from can3tok.model import build_model
from can3tok.render import Camera, render_gaussians, sh_dc_to_rgb, psnr
from can3tok.io_utils import camera_from_vector

COMPAT = ("normalize_pooler_xyz","count_aware_template","attr_slot_mask",
          "shared_cell_owner","holdout_own_photo","keep_extra_fullres",
          "reuse_prev_vis","eval_pred_mask","attr_frame_needles",
          "attr_detach_scale_rot","w_attr_local_set","attr_local_set_k")

def load(ckpt, split="val", index=0, downscale=1):
    cfgj = json.load(open(os.path.join(os.path.dirname(ckpt), "args.json"))); argv=[]
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
    for k in COMPAT:
        if k not in cfgj: setattr(args,k,0)
    tr,val = make_datasets(args)
    tr.anchor_spill = val.anchor_spill = int(getattr(args,"anchor_spill",8))
    ev,held = load_eval_views(args)
    for d in (tr,val): d.view_exclude=set(held)
    cfg = build_config(args, tr.target_dim, tr.sh_dim)
    model = build_model(cfg, allow_padded_tokens=getattr(args,"allow_padded_tokens",False)).cuda().eval()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    DS = val if split=="val" else tr
    it = DS[index]
    cam = Camera(camera_from_vector(np.asarray(it["camera"].numpy(), np.float32)),
                 device="cuda", downscale=downscale)
    ref = it["photo"].cuda().float()
    if ref.shape[-2:] != (cam.image_height, cam.image_width):
        ref = torch.nn.functional.interpolate(ref[None], size=(cam.image_height,cam.image_width),
                                              mode="bilinear", align_corners=False)[0]
    return dict(args=args, cfg=cfg, model=model, it=it, cam=cam, ref=ref,
                step=ck.get("step"), tr=tr, val=val)

def gauss(tt, msk, cen, sc):
    z = tt[msk]
    return (z[:,0:3]*sc+cen, torch.exp(z[:,3:6].clamp(-15,5))*sc,
            torch.nn.functional.normalize(z[:,6:10],dim=-1),
            torch.sigmoid(z[:,10:11]), sh_dc_to_rgb(z[:,11:14]))
