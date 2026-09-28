"""Compare old/shared-owner targets before training; renders only, no model."""
from pathlib import Path
from argparse import Namespace
import sys, json, gc
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from can3tok.train import make_datasets
from can3tok.io_utils import camera_from_vector
from can3tok.render import Camera,render_gaussians,sh_dc_to_rgb,psnr
OUT=Path(__file__).parent

def main():
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.10,0)
    raw=json.loads((ROOT/'runs/S16k_20260910_012456/args.json').read_text())
    _,old=make_datasets(Namespace(**raw))
    _,new=make_datasets(Namespace(**dict(raw,shared_cell_owner=1)))
    # Template reordering is only a permutation. Avoid solving Hungarian
    # assignments when rendering a set; source sets are checked against preflight.
    for ds in (old,new):
        ds.extra_real_views=0;ds.photo_map={};ds.view_pool=[];ds.slot_sort='morton'
    vp=json.loads((ROOT/raw['view_pool']).read_text());rows=[]
    for vi in (25,166,229,382):
        items=[old[vi],new[vi]];scene=str(items[0]['scene_key'])
        pool=next(v for k,v in vp.items() if Path(k).name==Path(scene).name)
        row=dict(val_index=vi,path=old.files[vi],used=[int(i['mask'].sum()) for i in items],views={})
        for camera_idx in (3,93):
            e=pool[camera_idx];cam=Camera(camera_from_vector(e['cam']),device='cuda:0')
            photo=torch.from_numpy(np.array(Image.open(e['image']).convert('RGB'))).permute(2,0,1).float().to('cuda:0')/255
            images=[]
            with torch.no_grad():
                for item in items:
                    live=item['mask']>.5;t=item['target'][live,:14].to('cuda:0')
                    s=float(item['scale']);c=item['center'].to('cuda:0')
                    images.append(render_gaussians(t[:,:3]*s+c,t[:,3:6].clamp(-15,5).exp()*s,
                                                   t[:,6:10],t[:,10:11].sigmoid(),sh_dc_to_rgb(t[:,11:14]),cam))
            def rgb(im):return (im.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            Image.fromarray(np.concatenate([rgb(photo)]+[rgb(im) for im in images],axis=1)).save(OUT/f'targets_val{vi}_view{camera_idx}.png')
            score=dict(old_photo_psnr=psnr(images[0],photo),shared_photo_psnr=psnr(images[1],photo),
                       old_shared_psnr=psnr(images[0],images[1]),full_resolution=list(photo.shape[-2:]))
            if vi in (25,166) and camera_idx==3:
                sl=(slice(None),slice(118,345),slice(395,620))
                score['upper_rail_old_photo_psnr']=psnr(images[0][sl],photo[sl])
                score['upper_rail_shared_photo_psnr']=psnr(images[1][sl],photo[sl])
                Image.fromarray(np.concatenate([rgb(photo[sl])]+[rgb(im[sl]) for im in images],axis=1)).save(OUT/f'upper_rail_val{vi}.png')
            row['views'][str(camera_idx)]=score
        rows.append(row);print(json.dumps(row),flush=True)
        del items,images;gc.collect();torch.cuda.empty_cache()
    (OUT/'target_render.json').write_text(json.dumps(dict(
        description='Native resolution, selected DC-only target renders, no model or training; columns photo / old target / shared-owner target',
        rows=rows),indent=2)+'\n')

if __name__=='__main__':main()
