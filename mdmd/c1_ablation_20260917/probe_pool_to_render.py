"""Intervene on pool input features while preserving analytic anchors.

All arms use the same checkpoint, chunk sizes, cameras and BF16 precision.
Only read-only inference; memory is capped to share GPU 3 with other work.
"""
from pathlib import Path
import sys,json
from argparse import Namespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from PIL import Image
from can3tok.train import make_datasets,build_config
from can3tok.model import build_model
from can3tok.schedule import schedule_flags
from can3tok.io_utils import camera_from_vector
from can3tok.render import Camera,render_gaussians,sh_dc_to_rgb,psnr
from probe_rail_detail import OUT,ROIS,metrics,png,qnt

def main():
    torch.set_num_threads(4);torch.cuda.set_device(3)
    torch.cuda.set_per_process_memory_fraction(.15,3)
    dev='cuda:3';setup=json.load(open(OUT/'setup.json'))
    ck=torch.load(setup['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    args=Namespace(**ck['args']);_,ds=make_datasets(args);ds.extra_real_views=0;it=ds[166]
    cfg=build_config(args,ds.target_dim,ds.sh_dim);cfg.patch_chunk=24;cfg.pool_chunk=8
    model=build_model(cfg,allow_padded_tokens=args.allow_padded_tokens)
    model.load_state_dict(ck['model'],strict=True)
    flags=schedule_flags(int(ck['step']),args)
    for k in ('decoder_refine_alpha','folding_res_gain','attr_detach_geometry'):setattr(cfg,k,flags[k])
    model.set_encoder_residual(flags['encoder_residual']);model.to(dev).eval();del ck
    x=it['input'][None].to(dev);m=it['mask'][None].to(dev)
    ex=it['enc_input'][None].to(dev);em=it['enc_mask'][None].to(dev);live=m[0]>.5
    sc=float(it['scale']);cen=it['center'].to(dev);target=it['target'][:,:14].to(dev)
    def unpack(p):
        p=p[live].float()
        return(p[:,:3]*sc+cen,p[:,3:6].clamp(-15,5).exp()*sc,p[:,6:10],p[:,10:11].sigmoid(),sh_dc_to_rgb(p[:,11:14]))
    vp=json.load(open(args.view_pool));pool=next(v for k,v in vp.items() if 'train_colmap' in k)
    cams={vi:Camera(camera_from_vector(pool[vi]['cam']),device=dev) for vi in (3,93)}
    photos={vi:torch.as_tensor(np.array(Image.open(pool[vi]['image']).convert('RGB')),device=dev).permute(2,0,1).float()/255 for vi in cams}
    refs={vi:render_gaussians(*unpack(target),cam).detach() for vi,cam in cams.items()}
    results={};base=None;base_images={}
    for arm in ('baseline','no_local_xyz','mean_local_attributes'):
        handle=None
        if arm!='baseline':
            def intervene(module,inputs):
                f,mask=inputs;f=f.clone()
                if arm=='no_local_xyz':f[...,:3]=0
                else:
                    mean=(f[...,3:]*mask[...,None]).sum(-2,keepdim=True)/mask.sum(-1,keepdim=True)[...,None].clamp(min=1)
                    f[...,3:]=torch.where(mask[...,None]>.5,mean,f[...,3:])
                return f,mask
            handle=model.encoder.pooler.register_forward_pre_hook(intervene)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(x,m,run_gen=False,enc_x=ex,enc_mask=em)
        if handle:handle.remove()
        p=out['attr_pred'][0].float();z=out['z_compact'].float()
        if base is None:base={'pred':p.clone(),'z':z.clone()}
        row={'xyz_change_world':qnt(((p[live,:3]-base['pred'][live,:3]).norm(dim=-1)*sc).cpu()),
             'compact_shape_relative_l2':float((z[:,5:8]-base['z'][:,5:8]).norm()/base['z'][:,5:8].norm().clamp(min=1e-12)),
             'compact_app_relative_l2':float((z[:,8:]-base['z'][:,8:]).norm()/base['z'][:,8:].norm().clamp(min=1e-12)),
             'view_scores':{}}
        with torch.no_grad():
            for vi,cam in cams.items():
                im=render_gaussians(*unpack(p),cam)
                if arm=='baseline':base_images[vi]=im.clone()
                png(f'view{vi}_pool_{arm}.png',im)
                score=metrics(im,photos[vi],refs[vi],ROIS[vi])
                score['baseline_render_psnr']=psnr(im,base_images[vi])
                score['baseline_image_mae']=float((im-base_images[vi]).abs().mean())
                row['view_scores'][str(vi)]=score
        results[arm]=row
        del out,p,z;torch.cuda.empty_cache()
        print(arm,'complete',flush=True)
    result={'checkpoint':setup['checkpoint'],'snapshot':ds.files[166],
            'precision':'CUDA BF16 autocast, FP32 render','patch_chunk':24,'pool_chunk':8,
            'all_arms_mask':'same target mask','analytic_anchors':'computed from original unchanged input in all arms',
            'max_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'arms':results}
    (OUT/'pool_to_render.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
