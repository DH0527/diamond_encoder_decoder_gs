#!/usr/bin/env python3
"""
Match replay .npz camera poses to COLMAP images and optionally inject the matched
GT image path into each .npz.

Usage example:
  python tools/match_replay_npz_to_colmap.py \
    --colmap_path /data/daeho/train_colmap \
    --replay_dir /data/daeho/gaussian-splatting/output/replay_out_/replay \
    --out_json /tmp/matched_index.json \
    --inject

This script uses the project's COLMAP reader (scene.colmap_loader) to load
intrinsics/extrinsics and matches each replay npz by a simple score:
  score = w_R * ||R_npz - R_col||_F + w_T * ||T_npz - T_col||_2 + w_K * sum(|K_diff|)

By default it will not modify the npz files. Use --inject to add two fields to
each npz file: 'gt_image_path' (bytes / str) and 'match_score' (float).
A backup of the original npz is created with suffix '.orig'.

"""

import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Allow running this script from any cwd by adding the package root to sys.path.
import sys
# project layout: <project_root>/tools/this_script -> we want <project_root> on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# import colmap reader from this project
try:
    from scene.colmap_loader import (
        read_extrinsics_binary, read_intrinsics_binary,
        read_extrinsics_text, read_intrinsics_text, qvec2rotmat,
    )
except Exception:
    # If importing the scene package fails (missing optional deps), load the
    # colmap_loader.py file directly to avoid importing other modules.
    import importlib.util
    _colmap_path = os.path.join(str(_PROJECT_ROOT), "scene", "colmap_loader.py")
    spec = importlib.util.spec_from_file_location("colmap_loader_local", _colmap_path)
    colmap_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(colmap_loader)

    read_extrinsics_binary = colmap_loader.read_extrinsics_binary
    read_intrinsics_binary = colmap_loader.read_intrinsics_binary
    read_extrinsics_text = colmap_loader.read_extrinsics_text
    read_intrinsics_text = colmap_loader.read_intrinsics_text
    qvec2rotmat = colmap_loader.qvec2rotmat


def build_dataset_views(colmap_path: str, images_subdir: str = "images"):
    """Read COLMAP intrinsics/extrinsics and build a list of views with R,T,K and image_path."""
    extr_bin = os.path.join(colmap_path, "sparse/0", "images.bin")
    cam_bin = os.path.join(colmap_path, "sparse/0", "cameras.bin")
    extr_txt = os.path.join(colmap_path, "sparse/0", "images.txt")
    cam_txt = os.path.join(colmap_path, "sparse/0", "cameras.txt")

    use_bin = False
    if os.path.exists(extr_bin) and os.path.exists(cam_bin):
        use_bin = True

    if use_bin:
        cam_extrinsics = read_extrinsics_binary(extr_bin)
        cam_intrinsics = read_intrinsics_binary(cam_bin)
    else:
        cam_extrinsics = read_extrinsics_text(extr_txt)
        cam_intrinsics = read_intrinsics_text(cam_txt)

    views = []
    for img_id, extr in cam_extrinsics.items():
        intr = cam_intrinsics[extr.camera_id]
        # COLMAP qvec -> rotation
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        params = intr.params
        # map camera params to fx,fy,cx,cy depending on model param length
        if len(params) >= 4:
            fx, fy, cx, cy = float(params[0]), float(params[1]), float(params[2]), float(params[3])
        elif len(params) == 3:
            fx = float(params[0]); fy = fx; cx = float(params[1]); cy = float(params[2])
        elif len(params) == 1:
            fx = fy = float(params[0]); cx = None; cy = None
        else:
            fx = fy = cx = cy = None

        image_path = os.path.join(colmap_path, images_subdir, extr.name)

        views.append({
            "image_path": image_path,
            "image_name": extr.name,
            "R": R,
            "T": T,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        })

    return views


def load_camera_from_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # state may be stored as an object array; handle both
    if "state_t" in data:
        st = data["state_t"]
        try:
            state = st.item()
        except Exception:
            state = st
    else:
        # try common keys
        keys = list(data.files)
        if len(keys) == 1:
            state = data[keys[0]].item()
        else:
            # cannot find state_t
            raise RuntimeError(f"npz {npz_path} does not contain 'state_t' key; found keys: {keys}")

    cam = state["camera"]
    def to_np(x):
        # cam entries could be numpy scalars, arrays, or Python scalars
        return np.array(x) if not isinstance(x, np.ndarray) else x

    fx = float(np.asarray(cam.get("fx", 0.0)))
    fy = float(np.asarray(cam.get("fy", fx)))
    cx = None
    cy = None
    try:
        cx = float(np.asarray(cam.get("cx", 0.0)))
        cy = float(np.asarray(cam.get("cy", 0.0)))
    except Exception:
        cx = cy = None

    R = np.asarray(cam.get("R"))
    T = np.asarray(cam.get("T"))
    # ensure shapes
    R = R.reshape((3, 3)) if R.size == 9 else R
    T = T.reshape((3,)) if T.size == 3 else T

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "R": R, "T": T}


def pose_score(cam_a, cam_b, w_R=1.0, w_T=1.0, w_K=0.01):
    R1, T1 = cam_a["R"], cam_a["T"]
    R2, T2 = cam_b["R"], cam_b["T"]

    # if shapes are not consistent, make safe
    sR = np.linalg.norm(R1 - R2)
    sT = np.linalg.norm(T1 - T2)

    sK = 0.0
    if cam_a.get("fx") is not None and cam_b.get("fx") is not None:
        sK += abs((cam_a.get("fx") or 0.0) - (cam_b.get("fx") or 0.0))
        sK += abs((cam_a.get("fy") or 0.0) - (cam_b.get("fy") or 0.0))
    if cam_a.get("cx") is not None and cam_b.get("cx") is not None:
        sK += abs((cam_a.get("cx") or 0.0) - (cam_b.get("cx") or 0.0))
        sK += abs((cam_a.get("cy") or 0.0) - (cam_b.get("cy") or 0.0))

    return float(w_R * sR + w_T * sT + w_K * sK)


def match_npz_to_views(npz_files, views, weights, inject=False, backup=True, out_json=None, verbose=False):
    results = {}
    for npz_path in tqdm(npz_files, desc="Matching npz files"):
        cam_npz = load_camera_from_npz(npz_path)

        best_score = float("inf")
        best_view = None
        for v in views:
            s = pose_score(cam_npz, v, w_R=weights[0], w_T=weights[1], w_K=weights[2])
            if s < best_score:
                best_score = s
                best_view = v

        results[os.path.basename(npz_path)] = {"npz_path": npz_path, "image_path": best_view["image_path"], "image_name": best_view["image_name"], "score": float(best_score)}

        if inject:
            # backup original
            if backup:
                bak = npz_path + ".orig"
                if not os.path.exists(bak):
                    shutil.copy2(npz_path, bak)
            # load original payload and re-save with new fields
            with np.load(npz_path, allow_pickle=True) as z:
                payload = {k: z[k] for k in z.files}

            # add fields
            payload["gt_image_path"] = np.array(best_view["image_path"]).astype(object)
            payload["match_score"] = np.array(best_score)

            # write back
            tmp_out = npz_path + ".tmp"
            np.savez_compressed(tmp_out, **payload)
            shutil.move(tmp_out, npz_path)

    if out_json:
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--colmap_path", type=str, required=True, help="Path to original COLMAP folder (root) containing sparse/0/)")
    p.add_argument("--replay_dir", type=str, required=True, help="Folder with replay npz files (e.g. output/.../replay)")
    p.add_argument("--out_json", type=str, default=None, help="Path to write matched index JSON")
    p.add_argument("--inject", action="store_true", help="Inject match result into each npz (creates .orig backup)")
    p.add_argument("--no-backup", dest="backup", action="store_false", help="Do not backup original npz before injecting")
    p.add_argument("--wR", type=float, default=1.0, help="Weight for rotation term")
    p.add_argument("--wT", type=float, default=1.0, help="Weight for translation term")
    p.add_argument("--wK", type=float, default=0.01, help="Weight for intrinsics term")
    p.add_argument("--ext", type=str, default=".npz", help="Replay file extension to search for")
    p.add_argument("--images_subdir", type=str, default="images", help="COLMAP images subdir name (usually 'images')")
    p.add_argument("--limit", type=int, default=0, help="Limit number of npz files to process (0 = no limit)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    replay_dir = args.replay_dir
    npz_files = sorted([str(p) for p in Path(replay_dir).glob(f"*{args.ext}")])
    if len(npz_files) == 0:
        raise SystemExit(f"No {args.ext} files found in {replay_dir}")
    if args.limit and args.limit > 0:
        np.random.shuffle(npz_files)
        npz_files = npz_files[: args.limit]

    print(f"Found {len(npz_files)} replay files. Reading COLMAP data from {args.colmap_path} ...")
    views = build_dataset_views(args.colmap_path, images_subdir=args.images_subdir)
    print(f"Loaded {len(views)} colmap views (images)")

    weights = (args.wR, args.wT, args.wK)
    results = match_npz_to_views(npz_files, views, weights, inject=args.inject, backup=args.backup, out_json=args.out_json, verbose=args.debug)

    if args.out_json:
        print(f"Wrote matches to {args.out_json}")
    else:
        print("Matching complete (no json written).")

    # print a few examples
    for k, v in list(results.items())[:10]:
        print(k, "->", v["image_path"], "score=", v["score"])


if __name__ == "__main__":
    main()
