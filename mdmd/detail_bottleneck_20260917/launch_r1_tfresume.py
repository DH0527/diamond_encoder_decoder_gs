"""T16k와 같은 계약으로 70k 재개. argv는 T args.json + 아래 오버라이드만."""
import json, os, sys
from datetime import datetime

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TARGS = os.path.join(REPO, "runs/T16k_20260913_093332/args.json")
SKIP = {"resume", "init_from", "eval_only", "out_dir", "init_skip"}
cfg = json.load(open(TARGS))
argv = []
for k, v in cfg.items():
    if k in SKIP:
        continue
    if isinstance(v, bool):
        if v:
            argv.append("--" + k)
    elif isinstance(v, list):
        if v:
            argv += ["--" + k] + [str(x) for x in v]
    elif v is None:
        continue
    else:
        argv += ["--" + k, str(v)]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = os.environ.get("R1_OUT", f"runs/R1_tfresume_{stamp}")
argv += [
    "--out_dir", out,
    "--resume", "runs/T16k_20260913_093332/ckpt_step00070000.pt",
    "--max_steps", "72000",
    "--lr", "0.0002",
    "--save_every", "500",
    "--eval_every", "10000",
    "--log_every", "50",
    "--shared_cell_owner", "0",
    "--normalize_pooler_xyz", "0",
    "--count_aware_template", "0",
    "--attr_slot_mask", "0",
    "--holdout_own_photo", "0",
    "--keep_extra_fullres", "0",
    "--reuse_prev_vis", "0",
    "--eval_pred_mask", "0",
    "--w_attr_spread", "0.0",
    "--w_attr_slope", "0.0",
]
print("OUT_DIR=" + out, flush=True)
os.chdir(REPO)
os.execv(sys.executable, [sys.executable, "-u", os.path.join(REPO, "train.py")] + argv)
