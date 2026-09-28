"""GPU 없이 w_aniso 배선만 확인한다. 스케줄과 LR 이 의도한 값으로 나오는지."""
import inspect, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import can3tok.schedule as S
from can3tok.train import build_parser

d = json.load(open("runs/T16k_20260913_093332/args.json"))
argv = []
for k, v in d.items():
    if k in ("resume", "init_from", "eval_only", "out_dir", "init_skip"): continue
    if isinstance(v, bool):
        if v: argv.append("--" + k)
    elif isinstance(v, list):
        if v: argv += ["--" + k] + [str(x) for x in v]
    elif v is None: continue
    else: argv += ["--" + k, str(v)]
argv += ["--out_dir", "/tmp/x", "--w_aniso", "5.0", "--w_attr_hung_scale", "5.0",
         "--max_steps", "140000"]
a = build_parser().parse_args(argv)

wfn = next(n for n, o in vars(S).items()
           if inspect.isfunction(o) and "weight" in n.lower() and "attr_param" not in n)
w = getattr(S, wfn)(70000, a)
print(f"[schedule] {wfn}(step=70000)")
for k in ("w_aniso", "w_scale", "w_cov3d", "w_rot", "w_attr_hung_scale", "w_attr_hung_rot"):
    print(f"  {k:20s} = {w.get(k)}")
a72 = build_parser().parse_args(argv[:-2] + ["--max_steps", "72000"])
print(f"[lr] step 70000: max_steps=140000 -> {S.global_lr(70000, a):.3e}   "
      f"H1 의 max_steps=72000 -> {S.global_lr(70000, a72):.3e}")
