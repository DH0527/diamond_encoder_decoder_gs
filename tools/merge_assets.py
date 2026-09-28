"""Merge per-scene assets into the multi-scene forms train.py consumes.

Per-scene assets stay separate where the code now takes a list (stats, anchors --
one entry per --root, in the same order) and are merged where the code takes a
single mapping keyed by scene (photo_map, view_pool). The split is merged into
GLOBAL indices over the concatenated file list, which is the order
ReplayGaussianDataset builds: roots in the order given, each sorted.

The split has to be rebuilt rather than concatenated: each per-scene split holds
indices into that scene's own file list, and scene 1's index 0 is global index
len(scene0_files).
"""
from __future__ import annotations
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from can3tok.data import list_npz_files  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True, help="comma-separated, same order as --root")
    ap.add_argument("--tags", required=True, help="comma-separated asset tags, aligned")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--assets", default="assets")
    a = ap.parse_args()

    roots = [r.strip() for r in a.roots.split(",") if r.strip()]
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    if len(roots) != len(tags):
        raise SystemExit(f"{len(roots)} roots vs {len(tags)} tags")

    files, offs, off = [], [], 0
    for r in roots:
        fs = list_npz_files(r)
        offs.append(off); off += len(fs); files.extend(fs)
        print(f"   {r}: {len(fs)} files (global offset {offs[-1]})")

    # split -> global indices
    tr, va = [], []
    for si, t in enumerate(tags):
        sp = json.load(open(os.path.join(a.assets, f"split_{t}.json")))
        tr += [i + offs[si] for i in sp["train"]]
        va += [i + offs[si] for i in sp["val"]]
    tr, va = sorted(tr), sorted(va)
    p = os.path.join(a.assets, f"split_{a.out_tag}.json")
    json.dump({"roots": roots, "n_total": len(files), "n_train": len(tr),
               "n_val": len(va), "train": tr, "val": va}, open(p, "w"))
    print(f"split -> {p}  ({len(tr)}/{len(va)})")

    # photo_map: {npz_basename: {image, scene}}. Basenames COLLIDE across scenes
    # (both hold step_012000.npz), so the merged map is keyed by basename only and
    # would silently hand scene A's photograph to scene B. Key it by the path the
    # loader actually sees instead, and keep the basename form as a fallback for
    # the single-scene case.
    pm = {}
    for si, t in enumerate(tags):
        src = json.load(open(os.path.join(a.assets, f"npz_to_image_{t}.json")))
        for k, v in src.items():
            pm[os.path.join(roots[si], k)] = v
    p = os.path.join(a.assets, f"npz_to_image_{a.out_tag}.json")
    json.dump(pm, open(p, "w"), indent=1)
    print(f"photo_map -> {p}  ({len(pm)} entries, keyed by FULL PATH)")

    vp = {}
    for t in tags:
        vp.update(json.load(open(os.path.join(a.assets, f"view_pool_{t}.json"))))
    p = os.path.join(a.assets, f"view_pool_{a.out_tag}.json")
    json.dump(vp, open(p, "w"))
    print(f"view_pool -> {p}  ({len(vp)} scenes: " +
          ", ".join(f"{k.split('/')[-1]}={len(v)}" for k, v in vp.items()) + ")")

    print("\nstats / anchors stay per-scene, pass them as comma-separated lists:")
    print("   --stats_path "    + ",".join(f"{a.assets}/stats_{t}.json" for t in tags))
    print("   --scene_anchors " + ",".join(f"{a.assets}/anchors_{t}.npy" for t in tags))


if __name__ == "__main__":
    main()
