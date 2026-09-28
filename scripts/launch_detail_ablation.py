"""Prepare or launch B0/C1 with the S16k learning-rate horizon preserved.

Default is a dry run. Both variants keep shared_cell_owner=0 because the
2026-09-16 preflight found large target-render regressions in its prefix policy.
An executed run trains toward 64000 steps; it does not auto-stop at review gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.argv_from_argsjson import ns_to_argv
from can3tok.train import build_parser


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=('B0', 'C1'), required=True)
    ap.add_argument('--gpus', default='0,1,2')
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--plan-out', default='', help='write resolved plan JSON, including on dry run')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--detached', action='store_true')
    a = ap.parse_args()
    if a.detached and not a.execute:
        ap.error('--detached requires --execute')
    gpu_ids = a.gpus.split(',')
    if len(gpu_ids) != 3 or len(set(gpu_ids)) != 3 or not all(x.isdigit() for x in gpu_ids):
        ap.error('this matched experiment requires three distinct GPU indices')
    os.chdir(ROOT)
    src = ROOT/'runs/S16k_20260910_012456/args.json'
    raw = json.loads(src.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out = Path(a.out_dir or f'runs/{a.variant}_detail_{stamp}').resolve()
    overrides = dict(out_dir=str(out), resume='', init_from='', max_steps=64000,
                     normalize_pooler_xyz=int(a.variant == 'C1'),
                     count_aware_template=0, attr_slot_mask=0,
                     shared_cell_owner=0, holdout_own_photo=1,
                     keep_extra_fullres=1, reuse_prev_vis=0, eval_pred_mask=1,
                     use_fixed_anchor_center=0)
    argv = ns_to_argv(argparse.Namespace(**dict(raw, **overrides)))
    resolved = vars(build_parser().parse_args(argv))
    envbin = Path('/home/super/anaconda3/envs/can3tok/bin')
    command = [str(envbin/'torchrun'), '--standalone', '--nproc_per_node=3',
               'train.py', *argv]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT/'can3tok').glob('*.py'))}
    manifest = dict(variant=a.variant, source_args=str(src),
                    source_args_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                    core_source_sha256=hashes, gpus=a.gpus, world_size=3,
                    status='planned', args=resolved, command=command,
                    review_steps=[2000,8000,16000,24000,40000,43000,48000,64000],
                    note='No automatic pause at review steps. B0 and C1 differ only in pooler xyz normalization; source view RNG is not yet fully seeded.')
    if a.plan_out:
        plan=Path(a.plan_out);plan.parent.mkdir(parents=True,exist_ok=True)
        plan.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(mode='execute' if a.execute else 'dry-run', variant=a.variant,
                         GPUs=a.gpus, overrides=overrides),indent=2),flush=True)
    if not a.execute:
        return
    out.mkdir(parents=True,exist_ok=False)
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpus
    for key,value in dict(CHAMFER_PAIR_CHUNK='4096',OMP_NUM_THREADS='4',
                          PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True').items():
        env.setdefault(key,value)
    if a.detached:
        with (out/'train.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'run.pid').write_text(str(proc.pid)+'\n')
        print(f'PID={proc.pid}; log={out / "train.log"}',flush=True)
    else:
        os.execvpe(command[0],command,env)


if __name__ == '__main__':
    main()
