"""Complete the original/selected SH comparison without another model forward."""
from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from can3tok.render import Camera,render_gaussians,sh_dc_to_rgb
from can3tok.io_utils import load_npz_state,camera_from_vector
from probe_rail_detail import OUT,ROIS,metrics,png

def main():
    torch.set_num_threads(4);torch.cuda.set_device(3);dev='cuda:3'
    bundle=np.load(OUT/'gaussian_bundle.npz');setup=json.load(open(OUT/'setup.json'))
    gs=load_npz_state(setup['snapshot']);mask=bundle['mask'];src=bundle['source_index'][mask]
    tar=torch.as_tensor(bundle['target'][mask],device=dev);s=float(bundle['scale']);c=torch.as_tensor(bundle['center'],device=dev)
    values=(tar[:,:3]*s+c,tar[:,3:6].clamp(-15,5).exp()*s,tar[:,6:10],tar[:,10:11].sigmoid())
    coeff=torch.cat([torch.as_tensor(gs['color'][src],device=dev).reshape(-1,1,3),torch.as_tensor(gs['sh'][src],device=dev).reshape(-1,15,3)],1).transpose(1,2)
    spec=importlib.util.spec_from_file_location('sh_local','/data/daeho/aabb/gaussian-splatting/utils/sh_utils.py');sh=importlib.util.module_from_spec(spec);spec.loader.exec_module(sh)
    pools=json.load(open(ROOT/'assets/view_pool_speedy_both.json'));pool=next(v for k,v in pools.items() if 'train_colmap' in k)
    rows=json.load(open(OUT/'render_metrics.json'))
    with torch.no_grad():
        for vi in (3,93):
            cam=Camera(camera_from_vector(pool[vi]['cam']),device=dev)
            color=(sh.eval_sh(3,coeff,F.normalize(values[0]-cam.camera_center,dim=-1))+.5).clamp(min=0)
            image=render_gaussians(*values,color,cam)
            ref=render_gaussians(*values,sh_dc_to_rgb(tar[:,11:14]),cam)
            photo=torch.as_tensor(np.array(Image.open(pool[vi]['image']).convert('RGB')),device=dev).permute(2,0,1).float()/255
            png(f'view{vi}_target_sh3.png',image)
            rows[str(vi)]['target_sh3']=metrics(image,photo,ref,ROIS[vi])
    (OUT/'render_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('selected SH renders complete')

if __name__=='__main__':main()
