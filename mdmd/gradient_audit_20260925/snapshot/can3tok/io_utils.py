"""NPZ Gaussian-state IO, target-channel packing and small file helpers.

The channel layout intentionally matches the original CoordFirstGSAE pipeline so
that metrics (xyz_rmse_norm in particular) stay comparable with previous runs:

    idx  0: 3   xyz_norm      (xyz - center) / scale
    idx  3: 3   log_scale     log(clip(scaling / scale, 1e-8))
    idx  6: 4   rot           unit quaternion
    idx 10: 1   opacity       logit(opacity)
    idx 11: 3   color         SH DC term
    idx 14: 45  sh            features_rest flattened
    total 59
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np


def _as_float32(x, shape_last: int = -1) -> np.ndarray:
    a = np.asarray(x)
    if shape_last > 0 and (a.ndim == 1 or a.shape[-1] != shape_last):
        a = a.reshape(-1, shape_last)
    return a.astype(np.float32, copy=False)


def normalize_quat_np(q, eps: float = 1e-8) -> np.ndarray:
    q = _as_float32(q, 4)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(n, eps)
    fallback = np.zeros_like(q)
    fallback[..., 0] = 1.0
    return np.where(n > eps, q, fallback).astype(np.float32)


def logit_np(alpha, eps: float = 1e-6) -> np.ndarray:
    a = np.clip(_as_float32(alpha, 1), eps, 1.0 - eps)
    return np.log(a / (1.0 - a)).astype(np.float32)


def load_replay_dict(path: str) -> Dict:
    """Return ``{camera, gaussians, ids}`` from either pickled ``state_t`` or compact flat npz."""
    raw = np.load(path, allow_pickle=True)
    files = set(raw.files)
    if "state_t" in files or "state" in files:
        state = raw["state_t"].item() if "state_t" in files else raw["state"].item()
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported NPZ structure in {path}")
        return state
    if "xyz" in files:
        cam_vec = np.asarray(raw["cam"], np.float32).reshape(-1)
        if cam_vec.size < 16:
            raise ValueError(f"compact npz camera vector too short in {path}: {cam_vec.shape}")
        cam = {
            "fx": float(cam_vec[0]),
            "fy": float(cam_vec[1]),
            "cx": float(cam_vec[2]),
            "cy": float(cam_vec[3]),
            "R": cam_vec[4:13].reshape(3, 3).astype(np.float32),
            "T": cam_vec[13:16].astype(np.float32),
        }
        if "image_name" in files:
            cam["image_name"] = str(raw["image_name"])
        if "image_wh" in files:
            wh = np.asarray(raw["image_wh"]).reshape(-1)
            cam["image_width"] = int(wh[0])
            cam["image_height"] = int(wh[1])
        gs = {
            "xyz": raw["xyz"],
            "scaling": raw["scaling"],
            "rot": raw["rot"],
            "opacity": raw["opacity"],
            "color": raw["color"],
        }
        if "features_rest" in files:
            gs["features_rest"] = raw["features_rest"]
        ids = raw["ids"] if "ids" in files else np.arange(np.asarray(gs["xyz"]).shape[0], dtype=np.uint32)
        out = {"camera": cam, "gaussians": gs, "ids": ids}
        if "pruning_scores" in files:
            out["pruning_scores"] = raw["pruning_scores"]
        if "reward" in files:
            vec = np.asarray(raw["reward"], np.float32).reshape(-1)
            keys = ("scalar", "quality", "eff", "loss_t", "loss_prev", "n_t", "n_prev", "unsquashed")
            reward = {}
            for i, k in enumerate(keys):
                if i >= vec.size or np.isnan(vec[i]):
                    reward[k] = None
                elif k in ("n_t", "n_prev"):
                    reward[k] = int(vec[i])
                else:
                    reward[k] = float(vec[i])
            out["reward_t"] = reward
        return out
    raise ValueError(f"Unsupported NPZ structure in {path}: {raw.files}")


def load_npz_state(path: str) -> Dict[str, np.ndarray]:
    """Load one replay frame into a flat dict of float32 arrays."""
    state = load_replay_dict(path)
    gs = state.get("gaussians", state)
    if not isinstance(gs, dict):
        raise ValueError(f"Unsupported Gaussian payload in {path}")

    def require(name: str) -> np.ndarray:
        if name not in gs:
            raise ValueError(f"missing '{name}' in {path}")
        return gs[name]

    out: Dict[str, np.ndarray] = {
        "xyz": _as_float32(require("xyz"), 3),
        "scaling": _as_float32(require("scaling"), 3),
        "rot": normalize_quat_np(require("rot")),
        "opacity": _as_float32(require("opacity"), 1),
    }
    n = out["xyz"].shape[0]
    out["color"] = _as_float32(gs.get("color", np.zeros((n, 3), np.float32)), 3)
    if "features_rest" in gs:
        fr = np.asarray(gs["features_rest"])
        out["sh"] = fr.reshape(fr.shape[0], -1).astype(np.float32, copy=False)
    else:
        out["sh"] = np.zeros((n, 0), np.float32)

    # The frame's own camera, when present. Packed into one flat vector so the
    # default collate handles it -- a dict of ragged numpy arrays does not
    # survive a DataLoader without a custom collate_fn.
    cam = state.get("camera", None)
    if isinstance(cam, dict) and all(k in cam for k in ("fx", "fy", "cx", "cy", "R", "T")):
        out["camera"] = np.concatenate([
            np.array([cam["fx"], cam["fy"], cam["cx"], cam["cy"]], np.float32).reshape(-1),
            np.asarray(cam["R"], np.float32).reshape(-1),
            np.asarray(cam["T"], np.float32).reshape(-1),
        ]).astype(np.float32)                      # 4 + 9 + 3 = 16
    ids = state.get("ids", None)
    if ids is not None:
        out["ids"] = np.asarray(ids).reshape(-1)
    scores = _coerce_pruning_scores(state.get("pruning_scores", None), n, out.get("ids"))
    if scores is not None:
        out["pruning_scores"] = scores
    return out


def _coerce_pruning_scores(raw, n: int, ids=None):
    """Align a replay/flat pruning payload to the loaded Gaussian order.

    Upstream stores ``{scores, meta}`` rather than a bare vector. A length match
    against the current xyz is required; a same-length vector from another
    snapshot is rejected when ids are present and disagree.
    """
    if raw is None:
        return None
    src_ids = None
    scores = raw
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item()
    if isinstance(raw, dict):
        scores = raw.get("scores", raw.get("pruning_scores"))
        src_ids = raw.get("ids", raw.get("gaussian_ids"))
    if scores is None:
        return None
    scores = np.asarray(scores).reshape(-1)
    if scores.size != n or not np.isfinite(scores.astype(np.float64)).any():
        return None
    if src_ids is not None and ids is not None:
        src_ids = np.asarray(src_ids).reshape(-1)
        ids = np.asarray(ids).reshape(-1)
        if src_ids.size == n and ids.size == n and not np.array_equal(src_ids, ids):
            order = {int(i): k for k, i in enumerate(src_ids)}
            try:
                scores = scores[np.array([order[int(i)] for i in ids], dtype=np.int64)]
            except KeyError:
                return None
    return np.nan_to_num(scores.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def camera_from_vector(v) -> Dict:
    """Inverse of the packing above -> the dict ``render.Camera`` expects."""
    v = np.asarray(v, np.float64).reshape(-1)
    return {"fx": v[0], "fy": v[1], "cx": v[2], "cy": v[3],
            "R": v[4:13].reshape(3, 3), "T": v[13:16]}


def transform_camera_vector(cam_vec, R_w, s: float, t_w) -> np.ndarray:
    """Move the camera by the same world similarity the augmentation applied.

    Augmentation maps ``x -> s R x + t_w``. Writing world-to-camera as
    ``x_c = R_wc x + T`` (which is what ``getWorld2View2`` builds, with the
    stored ``R`` equal to ``R_wc^T``), the camera that sees the augmented cloud
    exactly as the original camera saw the original cloud is

        R_wc' = R_wc R^T          T' = s T - R_wc R^T t_w

    because then ``x_c' = s (R_wc x + T)``. Camera-space coordinates come out
    uniformly scaled by ``s``, and perspective projection is invariant to that:
    ``x/z`` is unchanged, and the Gaussians' own scales were multiplied by the
    same ``s``, so their projected footprints are unchanged too. The render is
    therefore identical, not merely similar -- which is what makes it safe to
    train a render loss on augmented samples.
    """
    v = np.asarray(cam_vec, np.float64).reshape(-1)
    if v.shape[0] != 16 or v[0] == 0.0:
        return np.asarray(cam_vec, np.float32)
    R_cam, T = v[4:13].reshape(3, 3), v[13:16]
    R_w = np.asarray(R_w, np.float64).reshape(3, 3)
    t_w = np.asarray(t_w, np.float64).reshape(3)
    R_cam_new = R_w @ R_cam                    # since R_wc' = R_wc R^T
    T_new = float(s) * T - R_cam_new.T @ t_w
    return np.concatenate([v[:4], R_cam_new.reshape(-1), T_new]).astype(np.float32)


def gaussian_target_channels(gs: Dict[str, np.ndarray], center, scale: float):
    """Pack a Gaussian dict into the 59-channel target tensor + layout dict."""
    center = np.asarray(center, np.float32).reshape(1, 3)
    scale = float(scale)
    xyz_norm = (gs["xyz"].reshape(-1, 3) - center) / scale
    log_scale = np.log(np.clip(gs["scaling"].reshape(-1, 3) / scale, 1e-8, None)).astype(np.float32)
    rot = normalize_quat_np(gs["rot"])
    opacity = logit_np(gs["opacity"])
    color = _as_float32(gs.get("color", np.zeros((xyz_norm.shape[0], 3), np.float32)), 3)
    sh = gs.get("sh", None)
    if sh is None or np.size(sh) == 0:
        sh = np.zeros((xyz_norm.shape[0], 0), np.float32)
    else:
        sh = np.asarray(sh).reshape(xyz_norm.shape[0], -1).astype(np.float32, copy=False)

    target = np.concatenate(
        [xyz_norm.astype(np.float32), log_scale, rot, opacity, color, sh], axis=1
    ).astype(np.float32)
    layout = {
        "xyz": 0,
        "scale": 3,
        "rot": 6,
        "opacity": 10,
        "color": 11,
        "sh": 14,
        "target_dim": int(target.shape[1]),
        "sh_dim": int(sh.shape[1]),
    }
    return target, layout


def denormalize_xyz(xyz_norm: np.ndarray, center, scale: float) -> np.ndarray:
    center = np.asarray(center, np.float32).reshape(1, 3)
    return xyz_norm.astype(np.float32) * float(scale) + center


def write_ply(path: str, xyz: np.ndarray, color: np.ndarray = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    if color is None:
        rgb = np.full((xyz.shape[0], 3), 190, np.uint8)
    else:
        c = np.asarray(color, np.float32).reshape(-1, 3)
        lo = np.nanpercentile(c, 1.0)
        hi = np.nanpercentile(c, 99.0)
        rgb = np.clip((c - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
