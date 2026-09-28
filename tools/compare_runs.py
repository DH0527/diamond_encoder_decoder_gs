"""Gate table for the fix_2 run against the fix baseline.

Reads eval/step*/metrics.json from one or more run dirs and prints the numbers the
plan's go/no-go gates are defined on. The point of the extra columns is that plain
rmse is dominated by the ~1% of Morton-jump groups and is only weakly related to
how detailed the reconstruction looks; ``relq`` is the scale-free version.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

COLS = [
    ("step", "step", "{:>6.0f}"),
    ("codec_rmse", "codec_xyz_rmse_norm", "{:>10.5f}"),
    ("offset", "codec_offset_rmse_norm", "{:>9.5f}"),
    ("centroid", "codec_centroid_rmse_norm", "{:>9.5f}"),
    ("relq_p50", "codec_rel_offset_p50", "{:>9.3f}"),
    ("relq_p90", "codec_rel_offset_p90", "{:>9.3f}"),
    ("chamfer", "codec_chamfer_norm", "{:>9.5f}"),
    ("gen_rmse", "gen_xyz_rmse_norm", "{:>9.5f}"),
    ("hf", "latent_hf_ratio", "{:>7.3f}"),
    ("shape_hf", "shape_hf_ratio", "{:>9.3f}"),
    ("sstd_min", "shape_chan_std_min", "{:>9.3f}"),
    ("sstd_max", "shape_chan_std_max", "{:>9.3f}"),
    ("dfrac", "direct_frac", "{:>7.2f}"),
    ("res", "res_ratio", "{:>7.3f}"),
    ("rad999", "codec_group_radius_p999", "{:>8.3f}"),
    ("lam2", "codec_aniso_lam2_pred", "{:>7.3f}"),
    ("lam2_gt", "codec_aniso_lam2_gt", "{:>8.3f}"),
    ("tmpl", "codec_template_erank_pred", "{:>7.1f}"),
    ("tmpl_gt", "codec_template_erank_gt", "{:>8.1f}"),
]


def load(run: str):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "eval", "step*", "metrics.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for run in a.runs:
        rows = load(run)
        print(f"\n===== {os.path.basename(run.rstrip('/'))}  ({len(rows)} evals) =====")
        if not rows:
            print("  (no eval yet)")
            continue
        have = [c for c in COLS if any(c[1] in r for r in rows)]
        print("".join(f"{c[0]:>{len(c[2].format(0))}}" for c in have))
        for r in rows:
            out = []
            for name, key, fmt in have:
                v = r.get(key)
                out.append(fmt.format(v) if isinstance(v, (int, float)) else " " * len(fmt.format(0)))
            print("".join(out))


if __name__ == "__main__":
    main()
