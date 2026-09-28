"""Read-only checks of the proposed B-only data contract; no model or training."""
from pathlib import Path
from argparse import Namespace
import sys, json, gc
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from can3tok.train import make_datasets
from can3tok.schedule import global_lr, schedule_flags

OUT = Path(__file__).parent

def main():
    torch.set_num_threads(4)
    raw = json.loads((ROOT / 'runs/S16k_20260910_012456/args.json').read_text())
    shared = dict(raw, normalize_pooler_xyz=0, count_aware_template=0,
                  attr_slot_mask=0, shared_cell_owner=1, holdout_own_photo=1,
                  keep_extra_fullres=1, reuse_prev_vis=0, eval_pred_mask=1)
    _, old = make_datasets(Namespace(**raw))
    _, new = make_datasets(Namespace(**shared))
    # Geometry-only CPU check: neither photo augmentation nor model execution.
    for ds in (old, new):
        ds.extra_real_views = 0
        ds.photo_map = {}
        ds.view_pool = []
    capture = {}
    pack = new._pack_shared
    def traced(*args, **kwargs):
        t, e = pack(*args, **kwargs)
        capture['t'], capture['e'] = t, e
        return t, e
    new._pack_shared = traced
    rows = []
    for vi in (25, 76, 127, 166, 229, 280, 331, 382):
        a, b = old[vi], new[vi]
        sa = a['source_index'].numpy(); sa = sa[sa >= 0]
        sb = b['source_index'].numpy(); sb = sb[sb >= 0]
        t, e = capture['t'], capture['e']
        ec = (e.reshape(new.num_groups, new.group_in) >= 0).sum(1)
        tc = (t.reshape(new.num_groups, new.group_size) >= 0).sum(1)
        owner = np.full(max(int(e.max()), int(t.max())) + 1, -1)
        ei = np.flatnonzero(e >= 0); owner[e[ei]] = ei // new.group_in
        ti = np.flatnonzero(t >= 0)
        mismatches = int((owner[t[ti]] != ti // new.group_size).sum())
        xyz = b['enc_input'][:, :3].numpy().reshape(new.num_groups, new.group_in, 3)
        em = b['enc_mask'].numpy().reshape(new.num_groups, new.group_in)
        tx = b['target'][:, :3].numpy().reshape(new.num_groups, new.group_size, 3)
        tm = b['mask'].numpy().reshape(new.num_groups, new.group_size)
        ce = (xyz * em[..., None]).sum(1) / ec.clip(1)[:, None]
        ct = (tx * tm[..., None]).sum(1) / tc.clip(1)[:, None]
        extent = np.sqrt((((xyz-ce[:,None])**2)*em[...,None]).sum(1)/ec.clip(1)[:,None]).mean(1)
        sat = ec > new.group_size
        discrepancy = np.linalg.norm(ce-ct,axis=1) / extent.clip(1e-5)
        row = dict(val_index=vi, snapshot=old.files[vi],
                   valid=int(a['num_points_valid']), old_used=len(sa), shared_used=len(sb),
                   old_sources_removed=len(np.setdiff1d(sa, sb)),
                   shared_sources_added=len(np.setdiff1d(sb, sa)),
                   encoder_live=int(ec.sum()), ownership_mismatches=mismatches,
                   overfull_cells=int(sat.sum()),
                   within_owner_truncated_points=int(np.maximum(ec-new.group_size,0).sum()),
                   saturated_centroid_gap_over_mean_axis_std_p50=float(np.median(discrepancy[sat])) if sat.any() else None,
                   saturated_count_encoder_p50=float(np.median(ec[sat])) if sat.any() else None)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del a,b;gc.collect()
    lr = []
    for step in (8000,12000,16000,24000,40000,43000,48000):
        args = Namespace(**raw); short=Namespace(**dict(raw,max_steps=16000))
        lr.append(dict(step=step,lr_horizon_64000=global_lr(step,args),
                       lr_horizon_16000=global_lr(step,short),flags=schedule_flags(step,args)))
    report=dict(source_args=str(ROOT/'runs/S16k_20260910_012456/args.json'),
                description='CPU dataset geometry only; no training/model; no image-quality claim from point counts',
                flags={k:shared[k] for k in shared if k not in raw}, rows=rows, schedule=lr)
    (OUT/'preflight_data.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':main()
