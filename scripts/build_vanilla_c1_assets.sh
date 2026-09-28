#!/usr/bin/env bash
# vanilla 3DGS replay → C1 계약(1024 anchors, train+truck, blockB split) 자산.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}
DEVICE=${DEVICE:-cuda:0}

echo "[vanilla assets] train/truck n1024 + truck photo/view + merge"
CUDA_VISIBLE_DEVICES="${DEVICE##*:}" "$PYTHON" -u - <<'PY'
import glob, importlib.util, json, os
import numpy as np

spec = importlib.util.spec_from_file_location("build_assets", "tools/build_assets.py")
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)

TRAIN_REPLAY = "/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/train/replay"
TRUCK_REPLAY = "/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/truck/replay"
TRUCK_COLMAP = "/data/daeho/truck_colmap"
SCENE_TRUCK = "/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/truck"
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"

def cam_only(path):
    raw = np.load(path, allow_pickle=True)
    return np.asarray(raw["cam"], np.float32).reshape(-1)

def ensure_anchors(tag, replay, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        a = np.load(out)
        print(f"reuse {out} shape={a.shape}", flush=True)
        return a
    files = sorted(glob.glob(os.path.join(replay, "step_*.npz")))
    print(f"k-means 1024 {tag} n={len(files)} device={device}", flush=True)
    anch, cnt = ba.build_anchors(files, 1024, 100, 30, 42, device)
    np.save(out, anch)
    print(f"wrote {out} empty={int((cnt==0).sum())} p50={int(np.median(cnt))} max={int(cnt.max())}", flush=True)
    return anch

def ensure_stats(tag, replay, out):
    if os.path.exists(out):
        print(f"reuse {out}", flush=True)
        return json.load(open(out))
    files = sorted(glob.glob(os.path.join(replay, "step_*.npz")))
    print(f"stats {tag} ...", flush=True)
    stats = ba.build_stats(files, 50, 0.02)
    stats["root"] = os.path.abspath(replay)
    json.dump(stats, open(out, "w"), indent=2)
    print(f"wrote {out} scale={stats['scale']:.4f}", flush=True)
    return stats

def ensure_truck_photos():
    pm_p = "assets/npz_to_image_vanilla_truck.json"
    vp_p = "assets/view_pool_vanilla_truck.json"
    sp_p = "assets/split_vanilla_truck.json"
    files = sorted(glob.glob(os.path.join(TRUCK_REPLAY, "step_*.npz")))
    if not os.path.exists(sp_p):
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(files))
        val = sorted(perm[:300].tolist()); tr = sorted(perm[300:].tolist())
        json.dump({"root": os.path.abspath(TRUCK_REPLAY), "n_total": len(files),
                   "n_train": len(tr), "n_val": len(val), "seed": 42,
                   "train": tr, "val": val}, open(sp_p, "w"))
        print(f"wrote {sp_p}", flush=True)
    if os.path.exists(pm_p) and os.path.exists(vp_p):
        print(f"reuse {pm_p} {vp_p}", flush=True)
        return
    print(f"COLMAP cameras {TRUCK_COLMAP} ...", flush=True)
    views = ba.read_colmap_cameras(TRUCK_COLMAP)
    print(f"  {len(views)} photographs", flush=True)
    V = np.stack([v["cam"] for v in views])[:, 4:]
    mapping, worst, unmatched = {}, 0.0, []
    for i, p in enumerate(files):
        c = cam_only(p)
        d = np.abs(V - c[4:]).max(axis=1)
        j = int(d.argmin())
        worst = max(worst, float(d[j]))
        if d[j] > 1e-3:
            unmatched.append(os.path.basename(p))
            continue
        mapping[os.path.basename(p)] = j
        if (i + 1) % 500 == 0:
            print(f"  matched {i+1}/{len(files)} worst={worst:.3e}", flush=True)
    print(f"  matched {len(mapping)}/{len(files)} worst={worst:.3e}", flush=True)
    if unmatched:
        raise SystemExit(f"UNMATCHED {len(unmatched)}: {unmatched[:5]}")
    pm = {k: {"image": views[j]["image"], "scene": SCENE_TRUCK} for k, j in mapping.items()}
    json.dump(pm, open(pm_p, "w"), indent=1)
    json.dump({SCENE_TRUCK: [{"image": v["image"], "cam": v["cam"].tolist()} for v in views]},
              open(vp_p, "w"))
    print(f"wrote {pm_p} {vp_p}", flush=True)

os.makedirs("assets", exist_ok=True)
ensure_stats("vanilla_train", TRAIN_REPLAY, "assets/stats_vanilla_train.json")
ensure_stats("vanilla_truck", TRUCK_REPLAY, "assets/stats_vanilla_truck.json")
ensure_anchors("vanilla_train", TRAIN_REPLAY, "assets/anchors_vanilla_train_n1024.npy")
ensure_anchors("vanilla_truck", TRUCK_REPLAY, "assets/anchors_vanilla_truck_n1024.npy")
ensure_truck_photos()
print("per-scene assets done", flush=True)
PY

echo "[vanilla assets] merge photo_map + view_pool"
"$PYTHON" tools/merge_assets.py \
  --roots /data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/train/replay,/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/truck/replay \
  --tags vanilla_train,vanilla_truck \
  --out_tag vanilla_both \
  --assets assets

echo "[vanilla assets] blockB split (same indices as Speedy; file lists match)"
"$PYTHON" - <<'PY'
import json
src = json.load(open("assets/split_speedy_both_blockB.json"))
src["roots"] = [
    "/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/train/replay",
    "/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/truck/replay",
]
src["source_split"] = "assets/split_speedy_both_blockB.json"
json.dump(src, open("assets/split_vanilla_both_blockB.json", "w"))
print("wrote assets/split_vanilla_both_blockB.json",
      "n_train", src["n_train"], "n_val", src["n_val"])
PY

"$PYTHON" - <<'PY'
import json, os, numpy as np
need = [
    "assets/stats_vanilla_train.json",
    "assets/stats_vanilla_truck.json",
    "assets/anchors_vanilla_train_n1024.npy",
    "assets/anchors_vanilla_truck_n1024.npy",
    "assets/npz_to_image_vanilla_both.json",
    "assets/view_pool_vanilla_both.json",
    "assets/split_vanilla_both_blockB.json",
]
for p in need:
    assert os.path.exists(p), p
    print("ok", p)
a = np.load("assets/anchors_vanilla_train_n1024.npy")
b = np.load("assets/anchors_vanilla_truck_n1024.npy")
assert a.shape == (1024, 3), a.shape
assert b.shape == (1024, 3), b.shape
pm = json.load(open("assets/npz_to_image_vanilla_both.json"))
print("photo_map", len(pm))
assert len(pm) == 6000, len(pm)
vp = json.load(open("assets/view_pool_vanilla_both.json"))
print("view_pool scenes", list(vp))
print("vanilla C1 assets ready")
PY
