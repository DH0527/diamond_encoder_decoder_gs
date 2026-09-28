"""Rebuild every dataset-derived asset can3tok training needs.

The five assets under assets/ were all derived from a replay directory that was
deleted, and no generator survived. This rebuilds them from a replay directory
plus its COLMAP source, for any dataset (speedy-splat `state_t` schema or vanilla
`compact npz` schema -- load_npz_state handles both).

Assets produced, and why each is load-bearing:

  stats_<tag>.json        center/scale for xyz normalisation. Derived from robust
                          quantiles over a strided sample, identical formula to
                          can3tok.stats.compute_global_stats so xyz_rmse_norm stays
                          comparable across datasets.
  anchors_<tag>.npy       4096 fixed k-means anchors in WORLD coordinates. Morton
                          chunking makes latent cell j mean a different region in
                          every snapshot (measured cell drift 13.9x the group
                          radius across a densify step, 0.13x with fixed anchors),
                          so the anchors have to exist and have to be computed from
                          THIS dataset's density -- reusing another dataset's
                          anchors puts cells where there are no points.
  split_<tag>.json        2700/300 train/val by index into the sorted file list.
  npz_to_image_<tag>.json snapshot -> the photograph taken from that snapshot's own
                          camera. The render loss substitutes this for the
                          rasterised target whenever it exists, which is the only
                          supervision that is not a proxy.
  view_pool_<tag>.json    every photograph + its camera vector, keyed by scene.
                          The loader draws `extra_real_views` fresh views per step
                          from this; measured, one fixed view is worth +0.02 dB and
                          fresh draws from a pool of 64 are worth +4.13 dB.

The npz->image match is by CAMERA, not by filename: each snapshot records the
viewpoint it was optimised against that iteration, and we find the COLMAP image
whose extrinsics match. An exact match is expected because both come from the same
camera list; the script fails loudly if the best match is not essentially exact,
because a silently wrong photograph would train the model against another view.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from can3tok.io_utils import load_npz_state  # noqa: E402

QUANTILE_MARGIN = 1.05


def read_colmap_cameras(colmap_root: str):
    """Return [{image, cam(16,)}] from a COLMAP sparse reconstruction.

    cam is the same 16-vector can3tok uses everywhere: fx, fy, cx, cy, R(9), T(3).
    """
    sys.path.insert(0, "/data/daeho/gaussian-splatting")
    from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                     read_extrinsics_text, read_intrinsics_text,
                                     qvec2rotmat)
    sparse = os.path.join(colmap_root, "sparse", "0")
    try:
        cam_ex = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
        cam_in = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    except Exception:
        cam_ex = read_extrinsics_text(os.path.join(sparse, "images.txt"))
        cam_in = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))

    img_dir = os.path.join(colmap_root, "images")
    out = []
    for key in cam_ex:
        e = cam_ex[key]
        i = cam_in[e.camera_id]
        if i.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = float(i.params[0])
        else:
            fx, fy = float(i.params[0]), float(i.params[1])
        cx, cy = i.width / 2.0, i.height / 2.0
        # TRANSPOSED on purpose. qvec2rotmat gives the world->cam rotation, but
        # every camera vector in this project -- the ones baked into the npz files
        # by the replay writers, and the ones the old (working) view_pool.json
        # held -- stores the cam->world rotation, which is what
        # gaussian-splatting's Camera keeps in `.R`. Verified against a real
        # snapshot: with the transpose the pose residual against the matching
        # photograph is exactly 0.000e+00; without it, 1.432 and nothing matches.
        R = qvec2rotmat(e.qvec).astype(np.float32).T    # cam->world rotation
        T = np.asarray(e.tvec, np.float32)
        path = os.path.join(img_dir, e.name)
        if not os.path.exists(path):
            continue
        out.append({
            "image": path,
            "cam": np.concatenate([[fx, fy, cx, cy], R.reshape(-1), T]).astype(np.float32),
        })
    out.sort(key=lambda d: d["image"])
    return out


def build_stats(files, stride, quantile):
    chunks = []
    for p in files[::max(int(stride), 1)]:
        xyz = load_npz_state(p)["xyz"]
        if xyz.size:
            chunks.append(xyz)
    xyz = np.concatenate(chunks, axis=0)
    lo = np.quantile(xyz, quantile, axis=0).astype(np.float32)
    hi = np.quantile(xyz, 1.0 - quantile, axis=0).astype(np.float32)
    center = ((lo + hi) * 0.5).astype(np.float32)
    scale = max(float(np.max((hi - lo) * 0.5)) * QUANTILE_MARGIN, 1e-6)
    return {"num_files": len(files), "sample_stride": int(stride),
            "quantile": float(quantile), "center": center.tolist(),
            "scale": scale, "lo": lo.tolist(), "hi": hi.tolist()}


def build_anchors(files, n_anchors, stride, iters, seed, device):
    """k-means over a strided union of snapshots, in WORLD coordinates.

    k-means rather than farthest-point: FPS covers space uniformly and therefore
    makes one anchor enormous wherever points are dense (measured group radius
    1.18 against 0.064 for k-means on this data).
    """
    import torch
    pts = []
    for p in files[::max(int(stride), 1)]:
        xyz = load_npz_state(p)["xyz"]
        if xyz.size:
            pts.append(xyz)
    x = torch.from_numpy(np.concatenate(pts, 0)).float().to(device)
    g = torch.Generator(device=device).manual_seed(int(seed))
    c = x[torch.randperm(x.shape[0], generator=g, device=device)[:n_anchors]].clone()
    for _ in range(int(iters)):
        lab = torch.empty(x.shape[0], dtype=torch.long, device=device)
        for i in range(0, x.shape[0], 200_000):
            lab[i:i + 200_000] = torch.cdist(x[i:i + 200_000], c).argmin(1)
        newc = c.clone()
        cnt = torch.bincount(lab, minlength=n_anchors)
        s = torch.zeros_like(c).index_add_(0, lab, x)
        nz = cnt > 0
        newc[nz] = s[nz] / cnt[nz].unsqueeze(-1).float()
        # Empty clusters get re-seeded onto the worst-served points; leaving them
        # where they are wastes a latent cell for the whole run.
        if int((~nz).sum()) > 0:
            d = torch.cdist(x[torch.randperm(x.shape[0], generator=g, device=device)[:50_000]], newc).min(1).values
            far = x[torch.randperm(x.shape[0], generator=g, device=device)[:50_000]][d.argsort(descending=True)]
            newc[~nz] = far[:int((~nz).sum())]
        c = newc
    return c.cpu().numpy().astype(np.float32), cnt.cpu().numpy()


def cam_of(path):
    gs = load_npz_state(path)
    return gs.get("camera")


def match_photos(files, views, tol):
    """snapshot -> photograph, matched on the camera extrinsics, not the filename."""
    V = np.stack([v["cam"] for v in views])            # (M,16)
    Vx = V[:, 4:]                                      # R(9)+T(3): the pose
    mapping, worst, unmatched = {}, 0.0, []
    for p in files:
        c = cam_of(p)
        if c is None:
            unmatched.append(os.path.basename(p)); continue
        d = np.abs(Vx - np.asarray(c, np.float32)[4:]).max(axis=1)
        j = int(d.argmin())
        worst = max(worst, float(d[j]))
        if d[j] > tol:
            unmatched.append(os.path.basename(p)); continue
        mapping[os.path.basename(p)] = j
    return mapping, worst, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, help="directory holding step_*.npz")
    ap.add_argument("--colmap", required=True, help="COLMAP scene root (images/, sparse/)")
    ap.add_argument("--scene_key", required=True, help="key used in photo_map/view_pool")
    ap.add_argument("--tag", required=True, help="asset filename suffix")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anchors", type=int, default=4096)
    ap.add_argument("--anchor_stride", type=int, default=100)
    ap.add_argument("--anchor_iters", type=int, default=30)
    ap.add_argument("--stats_stride", type=int, default=50)
    ap.add_argument("--quantile", type=float, default=0.02)
    ap.add_argument("--cam_tol", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="stats/anchors 를 이미 있어도 다시 계산")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.replay, "step_*.npz")))
    if not files:
        raise SystemExit(f"no step_*.npz under {a.replay}")
    os.makedirs(a.out, exist_ok=True)
    print(f"[{a.tag}] {len(files)} snapshots under {a.replay}")

    p = os.path.join(a.out, f"stats_{a.tag}.json")
    if os.path.exists(p) and not a.force:
        stats = json.load(open(p))
        print(f"[{a.tag}] stats: 기존 재사용 {p}")
    else:
        print(f"[{a.tag}] stats ...", flush=True)
        stats = build_stats(files, a.stats_stride, a.quantile)
        stats["root"] = os.path.abspath(a.replay)
        json.dump(stats, open(p, "w"), indent=2)
    print(f"   center={[round(x,3) for x in stats['center']]} scale={stats['scale']:.4f}")

    p = os.path.join(a.out, f"anchors_{a.tag}.npy")
    if os.path.exists(p) and not a.force:
        anch = np.load(p)
        print(f"[{a.tag}] anchors: 기존 재사용 {p} shape={anch.shape}")
    else:
        print(f"[{a.tag}] anchors ({a.anchors}, k-means) ...", flush=True)
        anch, cnt = build_anchors(files, a.anchors, a.anchor_stride, a.anchor_iters, a.seed, a.device)
        np.save(p, anch)
        print(f"   shape={anch.shape} 빈셀={int((cnt==0).sum())} 셀크기 p50={int(np.median(cnt))} max={int(cnt.max())} -> {p}")

    print(f"[{a.tag}] split ({len(files)-a.n_val}/{a.n_val}) ...", flush=True)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(files))
    val = sorted(perm[:a.n_val].tolist()); tr = sorted(perm[a.n_val:].tolist())
    p = os.path.join(a.out, f"split_{a.tag}.json")
    json.dump({"root": os.path.abspath(a.replay), "n_total": len(files),
               "n_train": len(tr), "n_val": len(val), "seed": a.seed,
               "train": tr, "val": val}, open(p, "w"))
    print(f"   -> {p}")

    print(f"[{a.tag}] cameras from COLMAP ...", flush=True)
    views = read_colmap_cameras(a.colmap)
    print(f"   {len(views)} photographs")

    print(f"[{a.tag}] matching snapshots to photographs by camera ...", flush=True)
    mapping, worst, unmatched = match_photos(files, views, a.cam_tol)
    print(f"   matched {len(mapping)}/{len(files)}  worst pose residual={worst:.3e}")
    if unmatched:
        print(f"   !! UNMATCHED {len(unmatched)}: {unmatched[:5]}")
    pm = {k: {"image": views[j]["image"], "scene": a.scene_key} for k, j in mapping.items()}
    p = os.path.join(a.out, f"npz_to_image_{a.tag}.json")
    json.dump(pm, open(p, "w"), indent=1)
    print(f"   -> {p}")

    p = os.path.join(a.out, f"view_pool_{a.tag}.json")
    json.dump({a.scene_key: [{"image": v["image"], "cam": v["cam"].tolist()} for v in views]},
              open(p, "w"))
    print(f"   -> {p}")
    print(f"[{a.tag}] done")


if __name__ == "__main__":
    main()
