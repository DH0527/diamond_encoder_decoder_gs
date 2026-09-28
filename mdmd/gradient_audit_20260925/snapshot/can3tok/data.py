"""Replay-frame Gaussian dataset.

Improvements over the previous loader
------------------------------------
* ``stratified`` sampling: instead of ``linspace`` decimation of the Morton
  order, the Morton order is cut into exactly ``max_points`` equal bins and the
  most important Gaussian of each bin is kept. Spatial coverage becomes uniform
  by construction while high-importance points still win inside their bin.
* ``window`` multi-crop: with probability ``crop_prob`` a random contiguous
  Morton window is used instead. Dense frames have >2x ``max_points`` Gaussians,
  so a single fixed subsample means the model never sees half of the scene;
  different windows across epochs recover the full distribution.
* geometric augmentation (small rotation, scale/translation jitter, optional
  mirror) with correct quaternion/log-scale bookkeeping. Beyond the usual
  regularisation this is what makes the latent space approximately equivariant,
  which EQ-VAE (ICML'25) showed markedly speeds up downstream diffusion.
"""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io_utils import gaussian_target_channels, load_npz_state, transform_camera_vector
from .morton import kd_equal_groups, morton3d_np, morton_equal_groups
from .stats import load_or_create_stats
from .template import fibonacci_ball, fibonacci_prefix

SAMPLE_MODES = ("stratified", "importance_even", "morton", "first")
PARTITION_MODES = ("morton", "kd")
SLOT_SORTS = ("morton", "template")


def _norm_image_id(path: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))


def images_match(a, b) -> bool:
    if not a or not b:
        return False
    na, nb = _norm_image_id(a), _norm_image_id(b)
    return na == nb or os.path.basename(na) == os.path.basename(nb)


def list_npz_files(root: str):
    return sorted(glob.glob(os.path.join(root, "step_*.npz")))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions, broadcasting on the batch."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    ).astype(np.float32)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    t = float(R[0, 0] + R[1, 1] + R[2, 2])
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q.astype(np.float32)
    return q / max(float(np.linalg.norm(q)), 1e-8)


def _euler_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


class ReplayGaussianDataset(Dataset):
    def __init__(
        self,
        root: str,
        max_points: int = 262144,
        max_input_points: int = 0,
        stats_path: Optional[str] = None,
        stats_stride: int = 50,
        stats_quantile: float = 0.02,
        file_indices: Optional[Sequence[int]] = None,
        single_index: int = -1,
        start_index: int = 0,
        max_files: int = 0,
        drop_outside: bool = True,
        morton_bits: int = 10,
        sample_mode: str = "stratified",
        chunk_size: int = 64,
        group_size: int = 32,
        crop_prob: float = 0.0,
        augment: bool = False,
        aug_rot_deg: float = 10.0,
        aug_scale_jitter: float = 0.03,
        aug_shift: float = 0.01,
        aug_mirror: bool = False,
        density_bins: int = 32,
        # True 이면 importance 에 inverse local density 를 섞어 희소 구조를 보호
        density_aware_sample: bool = True,
        density_importance_weight: float = 0.35,
        # Spread real points across all groups instead of packing into a prefix
        # (the old layout left ~24-70% of compact cells encoding empty pads).
        slot_redistribute: bool = True,
        # How to cut the selected cloud into num_groups: Morton-contiguous or
        # balanced kd-tree (equal count, spatially compact).
        partition_mode: str = "kd",
        # Local kd inside a run of ``partition_block`` Morton-consecutive groups.
        # 0 disables (pure Morton-contiguous). Global kd (partition_mode="kd" with
        # redistribute) was measured to *hurt*; local kd keeps the Morton block
        # structure -- and therefore prefix packing, stationary slot counts and the
        # cell -> Z-order-grid mapping -- while removing the Morton-jump groups
        # that hold >55% of the MSE. Measured on val: PCA-28 ceiling 0.00606 ->
        # 0.00555 and group radius p99.9 0.38 -> 0.23.
        partition_block: int = 0,
        # Slot order inside a group. "morton" is the raw Morton order, which for
        # small groups degenerates into the npz file order (a tie-break, not a
        # function of geometry) -- those groups reconstruct at relq ~= 1.0, i.e. no
        # better than putting every point on the centroid. "pca" sorts along the
        # group's own major axis, which is a deterministic geometric function and
        # therefore learnable. Measured: PCA-12 ceiling 0.00919 -> 0.00716 (-22%).
        slot_sort: str = "morton",
        # Path to an (num_groups, 3) npy of fixed per-scene anchors in normalised
        # coordinates. When given, group membership comes from nearest anchor
        # instead of from the Morton order -- see _anchor_assign.
        scene_anchors: str = "",
        # Cache selected indices + slot packing per npz. World-model cells need
        # the same Gaussian in the same slot every epoch; stratified RNG and
        # independent encoder sampling were rewriting slot 17 every visit.
        cache_slots: bool = False,
        seed: int = 0,
    ) -> None:
        if sample_mode not in SAMPLE_MODES:
            raise ValueError(f"sample_mode must be one of {SAMPLE_MODES}")
        if partition_mode not in PARTITION_MODES:
            raise ValueError(f"partition_mode must be one of {PARTITION_MODES}")
        if slot_sort not in SLOT_SORTS:
            raise ValueError(f"slot_sort must be one of {SLOT_SORTS}")
        # MULTI-SCENE. `root`, `stats_path` and `scene_anchors` may each be a
        # comma-separated list, one entry per scene, and every file remembers which
        # scene it came from.
        #
        # Why per-scene normalisation AND per-scene anchors, rather than one global
        # frame with one shared anchor set: measured on the two speedy scenes,
        # a shared set leaves 43.7% of train's 4096 latent cells with no points in
        # them (truck 16.9%) because the two scenes occupy different regions and
        # have different shapes -- normalising each to its own cube does not align
        # them either (that was worse still, 62.8%). Per-scene anchors put every
        # cell where that scene actually has points: 0.0% empty for both.
        #
        # Cell j then means "near anchor j OF THIS SCENE". That is exactly what the
        # world model needs -- cell j must be stable across the SNAPSHOTS of one
        # scene, which it is -- and requiring it to mean the same absolute region
        # across different scenes is not achievable anyway, since the scenes are
        # different objects.
        roots = [r.strip() for r in str(root).split(",") if r.strip()]
        self.roots = roots
        self.files, self.file_scene = [], []
        for si, r in enumerate(roots):
            fs = list_npz_files(r)
            if not fs:
                raise FileNotFoundError(f"no step_*.npz under {r}")
            self.files.extend(fs)
            self.file_scene.extend([si] * len(fs))
        self.file_scene = np.asarray(self.file_scene, np.int64)
        if not self.files:
            raise FileNotFoundError(f"no step_*.npz under {root}")
        # Any subset of the file list has to carry its scene tags with it, or a
        # file ends up normalised by another scene's center/scale with no error.
        if single_index >= 0:
            sel = [single_index]
        elif file_indices is not None:
            sel = list(file_indices)
        else:
            sel = list(range(len(self.files)))[start_index:]
            if max_files > 0:
                sel = sel[:max_files]
        self.files = [self.files[i] for i in sel]
        self.file_scene = self.file_scene[np.asarray(sel, np.int64)]

        if max_points % group_size != 0:
            raise ValueError(f"max_points={max_points} must be a multiple of group_size={group_size}")

        sps = [x.strip() for x in str(stats_path or "").split(",")] if stats_path else []
        if sps and len(sps) != len(roots):
            raise ValueError(f"stats_path has {len(sps)} entries but there are "
                             f"{len(roots)} roots; give one per scene or none")
        self.centers, self.scales = [], []
        for si, r in enumerate(roots):
            st = load_or_create_stats(r, sps[si] if sps else None,
                                      stride=stats_stride, quantile=stats_quantile)
            self.centers.append(np.asarray(st["center"], np.float32))
            self.scales.append(float(st["scale"]))
        # Scene 0 stays reachable as `.center` / `.scale` so single-scene callers
        # and every existing diagnostic keep working unchanged.
        self.center = self.centers[0]
        self.scale = self.scales[0]

        self.max_points = int(max_points)
        # Encoder capacity. Selection targets this; the decoder still emits
        # max_points Gaussians, so the two are no longer forced to be equal.
        self.max_input_points = int(max_input_points) or int(max_points)
        # Points per group on the ENCODER side. num_groups is fixed by z_compact,
        # so a bigger input capacity means more points per group, not more groups.
        self.group_in = max(1, self.max_input_points // (int(max_points) // int(group_size)))
        self.group_size = int(group_size)
        self.drop_outside = bool(drop_outside)
        self.morton_bits = int(morton_bits)
        self.sample_mode = sample_mode
        self.chunk_size = max(int(chunk_size), 1)
        self.crop_prob = float(crop_prob)
        self.augment = bool(augment)
        self.aug_rot_deg = float(aug_rot_deg)
        self.aug_scale_jitter = float(aug_scale_jitter)
        self.aug_shift = float(aug_shift)
        self.aug_mirror = bool(aug_mirror)
        self.density_bins = int(density_bins)
        self.density_aware_sample = bool(density_aware_sample)
        self.density_importance_weight = float(density_importance_weight)
        self.slot_redistribute = bool(slot_redistribute)
        self.partition_mode = str(partition_mode)
        self.partition_block = max(int(partition_block), 0)
        self.slot_sort = str(slot_sort)
        self.cache_slots = bool(cache_slots)
        self.shared_cell_owner = False
        self.holdout_own_photo = False
        self.keep_extra_fullres = False
        self.count_aware_template = False
        self.scene_names: list = []
        # LRU, not unbounded: with shuffle + N workers a full cache is
        # ~2 MB * n_files * n_workers and will OOM. 128 files is ~0.5 GB.
        self._slot_cache: OrderedDict[str, Tuple] = OrderedDict()
        self._slot_cache_max = 128
        self.epoch = 0
        self.seed = int(seed)
        self.num_groups = self.max_points // self.group_size
        # One anchor set PER SCENE, each already checked against that scene's own
        # center/scale. `scene_anchors` is a comma-separated list aligned with
        # `root`; a single path is broadcast to every scene, which keeps the
        # single-scene call identical to before.
        self.scene_anchors_list = [None] * len(roots)
        aps = [x.strip() for x in str(scene_anchors or "").split(",") if x.strip()]
        if aps and len(aps) not in (1, len(roots)):
            raise ValueError(f"scene_anchors has {len(aps)} entries but there are "
                             f"{len(roots)} roots; give one per scene, or one for all")
        if len(aps) == 1 and len(roots) > 1:
            aps = aps * len(roots)
        for si, ap in enumerate(aps):
            if not os.path.exists(ap):
                raise FileNotFoundError(f"scene_anchors not found: {ap}")
            a = np.load(ap).astype(np.float32).reshape(-1, 3)
            if a.shape[0] != self.num_groups:
                raise ValueError(
                    f"{ap} has {a.shape[0]} anchors but the layout needs "
                    f"{self.num_groups} (max_points/group_size). One anchor per latent "
                    "cell is the whole point -- a mismatch would silently fall back to "
                    "Morton chunking.")
            c_i, s_i = self.centers[si], self.scales[si]

            # The anchors on disk are in WORLD units while every point this class
            # handles is normalised. Assigning normalised points to world anchors
            # would not raise -- every point would take the same nearest anchor and
            # 4095 of 4096 cells would be empty, with no error anywhere. So decide
            # the space explicitly, verify, and say which branch was taken.
            def _inside(p):
                return float(np.all(np.abs(p) <= 1.0, axis=1).mean())
            raw_in, norm_in = _inside(a), _inside((a - c_i) / s_i)
            if norm_in > raw_in:
                a = (a - c_i) / s_i
                space = f"world -> normalised ({norm_in:.1%} inside the unit cube)"
            else:
                space = f"already normalised ({raw_in:.1%} inside the unit cube)"
            if max(raw_in, norm_in) < 0.5:
                raise ValueError(
                    f"{ap} fits the unit cube in neither space (raw {raw_in:.1%}, "
                    f"normalised {norm_in:.1%}); probably built against different "
                    f"stats than {c_i}/{s_i}")
            print(f"[data] scene {si} anchors: {a.shape[0]}, {space}", flush=True)
            self.scene_anchors_list[si] = np.ascontiguousarray(a.astype(np.float32))
        self.scene_anchors = self.scene_anchors_list[0]

        probe = load_npz_state(self.files[0])
        _, layout = gaussian_target_channels(probe, self.center, self.scale)
        self.layout = layout
        self.target_dim = int(layout["target_dim"])
        self.sh_dim = int(layout["sh_dim"])
        self.input_dim = self.target_dim + 4

    def __len__(self) -> int:
        return len(self.files)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _importance(
        self, gs: Dict[str, np.ndarray], target: np.ndarray, xyz_norm: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Selection score for which Gaussians occupy the K slots.

        ``pruning_scores`` is an upstream densify/prune signal. Held-out fitting
        showed importance pruning beats random by >5 dB at the same K, so it is
        used here for *sampling only*. It is not written into the encoder
        features: ``decode(z_compact)`` never sees it.
        """
        score = target[:, self.layout["opacity"]].astype(np.float32).copy()
        ps = gs.get("pruning_scores", None)
        if ps is not None and ps.shape[0] == score.shape[0]:
            ps = ps.astype(np.float32)
            ps = (ps - np.nanmean(ps)) / (np.nanstd(ps) + 1e-6)
            score = score + 0.25 * np.nan_to_num(ps)
        if self.density_aware_sample and xyz_norm is not None and self.density_importance_weight > 0:
            dens = self._local_density(xyz_norm)  # 0~1, 밀집일수록 큼
            score = score + float(self.density_importance_weight) * (1.0 - dens)
        return score

    def _local_density(self, xyz_norm: np.ndarray) -> np.ndarray:
        """점별 로컬 밀도 (0~1). 샘플링·입력 피처 공용."""
        b = self.density_bins
        q = np.clip(((xyz_norm + 1.0) * 0.5 * b).astype(np.int64), 0, b - 1)
        flat = (q[:, 0] * b + q[:, 1]) * b + q[:, 2]
        counts = np.bincount(flat, minlength=b ** 3).astype(np.float32)
        d = counts[flat]
        return (np.log1p(d) / float(np.log1p(counts.max() + 1e-6))).astype(np.float32)

    def _augment(self, xyz_norm, target, rng, center=None, scale=None):
        deg = self.aug_rot_deg
        if deg > 0:
            ang = np.deg2rad(rng.uniform(-deg, deg, size=3)).astype(np.float32)
            R = _euler_rotation(float(ang[0]), float(ang[1]), float(ang[2]))
        else:
            R = np.eye(3, dtype=np.float32)
        if self.aug_mirror and rng.random() < 0.5:
            # Reflection is not a rotation: only safe when quaternions are unused.
            R = R @ np.diag(np.array([-1.0, 1.0, 1.0], np.float32))
            mirrored = True
        else:
            mirrored = False
        s = 1.0 + float(rng.uniform(-self.aug_scale_jitter, self.aug_scale_jitter))
        shift = rng.uniform(-self.aug_shift, self.aug_shift, size=3).astype(np.float32)

        xyz_new = (xyz_norm @ R.T) * s + shift[None, :]
        target = target.copy()
        target[:, 0:3] = xyz_new
        sl = self.layout["scale"]
        target[:, sl : sl + 3] = target[:, sl : sl + 3] + np.float32(np.log(s))
        if not mirrored and deg > 0:
            rq = self.layout["rot"]
            q = _rot_to_quat(R)[None, :]
            target[:, rq : rq + 4] = _quat_mul(q, target[:, rq : rq + 4])
        # The same map in world units, for the camera: normalised
        # x_n -> s R x_n + shift becomes x -> s R x + (c - s R c + shift*scale).
        c_i = self.center if center is None else center
        s_i = self.scale if scale is None else float(scale)
        t_w = c_i - s * (R @ c_i) + shift * s_i
        # The map in NORMALISED units, for the fixed scene anchors. They live in
        # the same space as xyz_norm, so anything that moves the cloud has to move
        # them identically -- otherwise the cloud rotates away from a stationary
        # anchor grid and nearest-anchor assignment scrambles: measured at
        # aug_rot_deg=8, an 8 degree rotation displaces a point at radius 0.5 by
        # ~0.07 while a group radius is 0.0014-0.04, i.e. 2-50 group radii, so the
        # local density stops matching the anchor density, anchors overflow and
        # spill their surplus, and 81% of the output slots came back empty.
        return xyz_new.astype(np.float32), target, (R, s, t_w), (R, s, shift)

    def _select(self, order, importance, xyz_norm, rng, cap=None):
        """Choose exactly ``cap`` indices (default ``max_points``), or all when fewer exist."""
        n = order.shape[0]
        s = int(cap) if cap else self.max_points
        if n <= s:
            return order

        if self.sample_mode == "first":
            return order[:s]
        if self.sample_mode == "morton":
            pos = np.linspace(0, n - 1, s, dtype=np.int64)
            return order[pos]

        if self.sample_mode == "importance_even":
            n_imp = max(1, s // 4)
            even_pos = np.linspace(0, n - 1, s - n_imp, dtype=np.int64)
            even_idx = order[even_pos]
            imp_idx = np.argsort(-importance, kind="stable")[:n_imp]
            merged = np.unique(np.concatenate([even_idx, imp_idx]))
            if merged.shape[0] < s:
                used = np.zeros(n, dtype=bool)
                used[merged] = True
                extra = order[~used[order]]
                merged = np.concatenate([merged, extra[: s - merged.shape[0]]])
            codes = morton3d_np(xyz_norm[merged], bits=self.morton_bits)
            return merged[np.argsort(codes, kind="stable")[:s]]

        # ---- stratified (default) -------------------------------------
        # crop: Morton 연속 구간 262k — 밀집 프레임에서 매 epoch 다른 부분 집합
        if self.crop_prob > 0.0 and rng.random() < self.crop_prob:
            start = int(rng.integers(0, n - s + 1))
            return order[start : start + s]

        # Morton 순서를 정확히 s 개 bin 으로 나누고, bin 마다 importance 최대 1점
        bin_id = (np.arange(n, dtype=np.int64) * s) // n
        imp_sorted = importance[order]
        pick = np.lexsort((-imp_sorted, bin_id))
        first_of_bin = np.ones(n, dtype=bool)
        sorted_bins = bin_id[pick]
        first_of_bin[1:] = sorted_bins[1:] != sorted_bins[:-1]
        chosen_pos = np.sort(pick[first_of_bin])
        if chosen_pos.shape[0] < s:
            missing = s - chosen_pos.shape[0]
            rest = np.setdiff1d(np.arange(n), chosen_pos, assume_unique=False)
            # density-aware 보충: 아직 안 뽑힌 점 중 희소한 쪽을 우선
            if self.density_aware_sample and rest.size > 0:
                dens = self._local_density(xyz_norm)
                rest_score = -dens[order[rest]]
                rest = rest[np.argsort(rest_score, kind="stable")]
            chosen_pos = np.sort(np.concatenate([chosen_pos, rest[:missing]]))
        return order[chosen_pos[:s]]

    def _anchor_assign(self, xyz_sel: np.ndarray, gsz: int, ng: int,
                       anchors: Optional[np.ndarray] = None,
                       spill: Optional[int] = None) -> np.ndarray:
        """Assign selected points to FIXED per-scene anchors, one group per anchor.

        Morton chunking makes latent cell *j* mean a different region in every
        snapshot: the chunk boundaries are defined by the point ORDER, so a
        densification step that inserts points anywhere before cell j shifts cell
        j's contents. Measured on adjacent snapshot pairs whose point count
        changes, the cell centre moves 13.9x the group radius; on pairs with no
        densification it moves 0.1x, which is why the defect is invisible unless
        the right pairs are compared. A world model trained to map z_t -> z_{t+1}
        cannot work through that.

        With fixed anchors, cell j is "whatever is near anchor j" in every
        snapshot, and the measured drift falls to 0.13x. k-means anchors rather
        than farthest-point ones: FPS covers space uniformly and therefore makes
        one anchor enormous wherever points are dense (group radius 1.18 against
        0.064 for k-means).

        Each anchor keeps its ``gsz`` nearest assigned points. Anchors that draw
        more than that lose the surplus; anchors that draw fewer leave pad slots,
        which the mask already handles.
        """
        anch = self.scene_anchors if anchors is None else anchors
        n = xyz_sel.shape[0]
        slot_src = -np.ones(ng * gsz, dtype=np.int64)
        if n == 0:
            return slot_src
        # Nearest anchors per point, chunked: the full (n, ng) matrix at n=262144,
        # ng=4096 is 4 GB in float32 and this runs in a dataloader worker.
        #
        # SPILLOVER. A fixed anchor set cannot match a density that changes across
        # the dataset -- these snapshots run 81k to 565k points -- so with a hard
        # cap of `gsz` per anchor, dense anchors overflow and drop their surplus
        # while sparse ones leave slots idle. Measured with nearest-anchor only:
        # 26.1% of the selected points dropped on step_005070, 15.9% on
        # step_018020, 0.4-5.9% on sparse snapshots. Letting a point fall through
        # to its 2nd..8th nearest anchor when the nearest is full recovers most of
        # it: 26.1 -> 15.6% and 15.9 -> 6.1%. The remaining loss is the genuinely
        # over-budget case, where the selection already fills every slot.
        k = int(getattr(self, "anchor_spill", 8) if spill is None else spill)
        k = int(min(max(k, 1), ng))
        idx = np.empty((n, k), np.int64)
        dst = np.empty((n, k), np.float32)
        a2 = (anch ** 2).sum(1)[None, :]
        for i0 in range(0, n, 16384):
            p = xyz_sel[i0: i0 + 16384]
            d = (p ** 2).sum(1)[:, None] + a2 - 2.0 * (p @ anch.T)
            j = np.argpartition(d, min(k, d.shape[1] - 1), axis=1)[:, :k]
            dd = np.take_along_axis(d, j, 1)
            o = np.argsort(dd, axis=1)
            idx[i0: i0 + 16384] = np.take_along_axis(j, o, 1)
            dst[i0: i0 + 16384] = np.take_along_axis(dd, o, 1)

        # Closest-first greedy, one vectorised round per fallback rank. Points with
        # the best available match claim their anchor first, so a spilled point is
        # one that genuinely lost the competition rather than one that arrived late.
        cap = np.zeros(ng, np.int64)
        owner = np.full(n, -1, np.int64)
        pend = np.argsort(dst[:, 0], kind="stable")
        for h in range(k):
            if pend.size == 0:
                break
            cand = idx[pend, h]
            o = np.lexsort((dst[pend, h], cand))
            pend, cand = pend[o], cand[o]
            starts = np.flatnonzero(np.r_[True, cand[1:] != cand[:-1]])
            rank = np.arange(cand.size, dtype=np.int64) - np.repeat(
                starts, np.diff(np.r_[starts, cand.size]))
            take = rank < (gsz - cap[cand])
            owner[pend[take]] = cand[take]
            np.add.at(cap, cand[take], 1)
            pend = pend[~take]

        # Sparse global capacity repair.  Local k-nearest spill preserves
        # locality, but can drop points while distant anchors still have empty
        # slots.  Repeatedly offer the remaining points to all non-full anchors.
        # This fills the available global budget without constructing the full
        # n_points x n_anchors matrix in memory.
        if (getattr(self, "anchor_assignment", "spill") == "capacitated"
                and pend.size > 0 and int(cap.sum()) < min(n, ng * gsz)):
            while pend.size > 0:
                free = np.flatnonzero(cap < gsz)
                if free.size == 0:
                    break
                best_a = np.empty(pend.size, np.int64)
                best_d = np.empty(pend.size, np.float32)
                af = anch[free]
                af2 = (af ** 2).sum(1)[None, :]
                for p0 in range(0, pend.size, 8192):
                    pi = pend[p0:p0 + 8192]
                    p = xyz_sel[pi]
                    d = (p ** 2).sum(1)[:, None] + af2 - 2.0 * (p @ af.T)
                    j = d.argmin(axis=1)
                    best_a[p0:p0 + pi.size] = free[j]
                    best_d[p0:p0 + pi.size] = d[np.arange(pi.size), j]
                o = np.lexsort((best_d, best_a))
                po, co = pend[o], best_a[o]
                starts = np.flatnonzero(np.r_[True, co[1:] != co[:-1]])
                rank = np.arange(co.size, dtype=np.int64) - np.repeat(
                    starts, np.diff(np.r_[starts, co.size]))
                take = rank < (gsz - cap[co])
                if not take.any():
                    break
                owner[po[take]] = co[take]
                np.add.at(cap, co[take], 1)
                pend = po[~take]

        got = np.flatnonzero(owner >= 0)
        if got.size == 0:
            return slot_src
        # Slot order inside a group is distance-to-its-own-anchor rank, and the
        # live slots form a prefix -- both are relied on downstream (the presence
        # count prior, and the eval's per-group statistics).
        own_d = dst[got, 0].copy()
        moved = owner[got] != idx[got, 0]
        if moved.any():
            mi = got[moved]
            own_d[moved] = np.linalg.norm(xyz_sel[mi] - anch[owner[mi]], axis=1) ** 2
        o = np.lexsort((own_d, owner[got]))
        got, a_sorted = got[o], owner[got][o]
        starts = np.flatnonzero(np.r_[True, a_sorted[1:] != a_sorted[:-1]])
        rank = np.arange(got.size, dtype=np.int64) - np.repeat(
            starts, np.diff(np.r_[starts, got.size]))
        slot_src[a_sorted * gsz + rank] = got
        return slot_src

    def _local_kd_order(self, xyz: np.ndarray, n_groups: int) -> np.ndarray:
        """Permutation of ``arange(n_groups * group_size)`` making each run of
        ``group_size`` entries a spatially compact, equal-count group.

        kd runs only *inside* a block of ``partition_block`` Morton-consecutive
        groups, so the global Morton ordering of the blocks -- and hence prefix
        packing, full groups, and the cell -> Z-order-grid locality -- survives.

        The per-group indices are sorted back into Morton order before they are
        written. ``kd_equal_groups`` leaves each leaf sorted along that leaf's
        split axis, which makes the within-group slot layout monotone along one
        direction; the cheapest fit to a monotone target is a straight line, and
        the groups then reconstruct as streaks with no lateral extent. Measured:
        with the axis order the predicted groups reach sqrt(lam2/lam1) = 0.036
        against 0.544 in the ground truth. Only *which* points share a group is
        kd's job here -- the order inside it stays Morton.
        """
        gsz = self.group_size
        blk = self.partition_block if self.partition_block > 0 else n_groups
        out = np.empty(n_groups * gsz, dtype=np.int64)
        w = 0
        for g0 in range(0, n_groups, blk):
            g1 = min(g0 + blk, n_groups)
            lo, hi = g0 * gsz, g1 * gsz
            for part in kd_equal_groups(xyz[lo:hi], g1 - g0):
                out[w : w + part.shape[0]] = lo + np.sort(part)
                w += int(part.shape[0])
        return out

    @staticmethod
    def _template_slot_order(pts: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Reorder each group's points onto the fixed decoder template.

        The decoder emits ``R diag(s) (template + residual)``. Whichever GT point
        ends up in slot *i* is our choice -- the reconstruction is scored as a set
        -- so choose the assignment that makes the residual small. Per group:
        centre, whiten in the covariance eigenframe (canonical sign from the
        per-axis skew), then solve the 64x64 assignment against the template.

        Measured on val, residual = GT_local - template, and the reconstruction
        chamfer that a rank-k residual head can reach from it (GT's own point
        spacing is 0.191, i.e. 1.0x):

            slot order            mean|res|   rank-22 bound   chamfer   x spacing
            Morton (previous)         1.010           1.176     0.261      1.37
            optimal assignment        0.408           0.679     0.225      1.18

        Without this the residual head burns most of its 22 usable dimensions
        undoing an arbitrary permutation instead of describing shape, which is
        why the first folding run sat at 0.382 -- worse than its own 0.367
        envelope oracle. Greedy nearest-slot was also measured (rank-22 0.751,
        |res| 0.600) and is both worse and no faster than the exact solver.
        """
        from scipy.optimize import linear_sum_assignment

        g, gsz = pts.shape[0], pts.shape[1]
        tmpl = fibonacci_ball(gsz)
        c = pts - pts.mean(axis=1, keepdims=True)
        cov = np.einsum("gpi,gpj->gij", c, c) / float(gsz)
        w, v = np.linalg.eigh(cov)
        sd = np.sqrt(np.clip(w, 0.0, None))
        # A perfectly flat or collinear group would divide by ~0; floor the thin
        # axes against the fattest one so whitening stays finite.
        sd = np.maximum(sd, 1e-3 * sd[:, -1:].clip(min=1e-12))
        loc = np.einsum("gpi,gij->gpj", c, v) / sd[:, None, :]
        skew = np.einsum("gpj->gj", loc**3)
        sk = np.where(skew >= 0.0, 1.0, -1.0)
        # The decoder rebuilds the frame as R(axis-angle) @ diag(softplus(.)), which
        # is always a *proper* rotation with positive scales: det > 0. eigh's basis
        # and a free per-axis sign flip can just as easily land on a reflection, and
        # then no (R, s) the decoder can emit reproduces the group. Force det = +1 by
        # flipping the axis whose skew is closest to zero, i.e. the one whose sign was
        # least determined anyway.
        det = np.linalg.det(v) * sk.prod(axis=1)
        bad = det < 0.0
        if bad.any():
            j = np.abs(skew[bad]).argmin(axis=1)
            sk[np.nonzero(bad)[0], j] *= -1.0
        loc = loc * sk[:, None, :]
        order = np.empty((g, gsz), dtype=np.int64)
        for i in range(g):
            d = np.linalg.norm(tmpl[:, None, :] - loc[i][None, :, :], axis=-1)
            order[i] = linear_sum_assignment(d)[1]
        return np.take_along_axis(idx.reshape(g, gsz), order, axis=1).reshape(-1)

    def _template_reorder_full_groups(
        self, pos: np.ndarray, xyz_sel: np.ndarray, gsz: int, ng: int
    ) -> np.ndarray:
        """Hungarian-assign groups onto the folding template.

        Full cells use the historical all-256 assignment. When
        ``count_aware_template`` is on, partial cells are assigned to the
        recentered live prefix so packer and decoder share the same prior.
        """
        return self._template_reorder_groups(
            pos, xyz_sel, gsz, ng,
            partial=bool(getattr(self, "count_aware_template", False)),
        )

    def _template_reorder_groups(
        self, pos: np.ndarray, xyz_sel: np.ndarray, gsz: int, ng: int,
        partial: bool = False,
    ) -> np.ndarray:
        from scipy.optimize import linear_sum_assignment

        gpos = np.asarray(pos[: ng * gsz], dtype=np.int64).reshape(ng, gsz).copy()
        full = (gpos >= 0).all(axis=1)
        if np.any(full):
            idx = gpos[full].reshape(-1)
            xyzg = xyz_sel[idx].reshape(-1, gsz, 3)
            gpos[full] = self._template_slot_order(xyzg, idx).reshape(-1, gsz)
        if partial:
            for g in np.flatnonzero(~full):
                live = gpos[g] >= 0
                k = int(live.sum())
                if k <= 1:
                    continue
                idx = gpos[g][live]
                pts = xyz_sel[idx]
                c = pts - pts.mean(axis=0, keepdims=True)
                tmpl = fibonacci_prefix(gsz, k)
                d = np.linalg.norm(tmpl[:, None, :] - c[None, :, :], axis=-1)
                order = linear_sum_assignment(d)[1]
                new = np.full(gsz, -1, dtype=np.int64)
                new[:k] = idx[order]
                gpos[g] = new
        out = np.asarray(pos, dtype=np.int64).copy()
        out[: ng * gsz] = gpos.reshape(-1)
        return out

    def _pack_shared(self, selected_enc: np.ndarray, xyz_all: np.ndarray,
                     anchors) -> Tuple[np.ndarray, np.ndarray]:
        """One ownership plan: encoder keeps the full cell, target the live prefix."""
        gi = int(self.group_in)
        ng = int(self.num_groups)
        gsz = int(self.group_size)
        si = int(self.max_input_points)
        xyz_sel = xyz_all[selected_enc]
        pos_i = self._anchor_assign(xyz_sel, gi, ng, anchors)
        src_i = -np.ones(si, dtype=np.int64)
        liv = pos_i >= 0
        nslot = min(int(liv.shape[0]), ng * gi)
        src_i[:nslot][liv[:nslot]] = selected_enc[pos_i[:nslot][liv[:nslot]]]

        pos_t = -np.ones(ng * gsz, dtype=np.int64)
        for g in range(ng):
            cell = pos_i[g * gi:(g + 1) * gi]
            live_cell = cell[cell >= 0][:gsz]
            if live_cell.size:
                pos_t[g * gsz:g * gsz + live_cell.size] = live_cell
        if self.slot_sort == "template":
            pos_t = self._template_reorder_groups(
                pos_t, xyz_sel, gsz, ng,
                partial=bool(getattr(self, "count_aware_template", False)))
        slot_src = -np.ones(self.max_points, dtype=np.int64)
        live_t = pos_t >= 0
        slot_src[: ng * gsz][live_t] = selected_enc[pos_t[live_t]]
        return slot_src, src_i

    def _pool_for_scene(self, scene_key):
        vp = getattr(self, "view_pool", None)
        if isinstance(vp, dict):
            if scene_key and scene_key in vp:
                return vp[scene_key]
            if scene_key:
                for k, pool in vp.items():
                    if os.path.basename(str(k).rstrip("/")) == os.path.basename(str(scene_key).rstrip("/")):
                        return pool
            return next(iter(vp.values())) if len(vp) == 1 else None
        return vp

    def _allowed_pool_indices(self, pool) -> np.ndarray:
        if not pool:
            return np.zeros(0, dtype=np.int64)
        excl = getattr(self, "view_exclude", None) or set()
        return np.array([i for i in range(len(pool)) if i not in excl], dtype=np.int64)

    def _image_is_held(self, image_path, scene_key) -> bool:
        excl = getattr(self, "view_exclude", None)
        if not excl or not image_path:
            return False
        pool = self._pool_for_scene(scene_key)
        if not pool:
            return False
        for i in excl:
            if 0 <= int(i) < len(pool) and images_match(image_path, pool[int(i)]["image"]):
                return True
        return False

    def _pack_slots(self, selected: np.ndarray, xyz_sel: np.ndarray, cap=None, gsz=None, ng=None,
                    anchors=None):
        """Place ``selected`` indices into a ``max_points`` slot array.

        With ``slot_redistribute``, points are spread across all ``num_groups``
        so empty trailing groups no longer waste compact channels. Each group
        keeps at most ``group_size`` points; leftovers inside a group are pad.
        """
        s = int(cap) if cap else self.max_points
        gsz = int(gsz) if gsz else self.group_size
        ng = int(ng) if ng else self.num_groups
        slot_src = -np.ones(s, dtype=np.int64)
        n = int(selected.shape[0])
        if n == 0:
            return slot_src

        anch = self.scene_anchors if anchors is None else anchors
        if anch is not None and ng == anch.shape[0]:
            # Fixed anchors decide WHICH points share a cell (time-stable).
            # Within a *full* cell, slot order still has to match the folding
            # template. Distance-to-anchor rank was leaving residual capacity
            # to undo a permutation (HISTORY §3.3: Morton |res| 1.010 vs
            # template 0.408). Partial groups keep the live prefix so presence
            # still means "first k slots".
            pos = self._anchor_assign(xyz_sel, gsz, ng, anch)
            if self.slot_sort == "template":
                pos = self._template_reorder_full_groups(pos, xyz_sel, gsz, ng)
            live = pos >= 0
            slot_src[: ng * gsz][live] = selected[pos[live]]
            return slot_src

        if not self.slot_redistribute:
            k = min(n, s)
            n_full = k // gsz
            if n_full > 0:
                m = n_full * gsz
                xyz = xyz_sel[:m]
                idx = np.arange(m, dtype=np.int64)
                if self.partition_block > 0:
                    idx = self._local_kd_order(xyz, n_full)
                if self.slot_sort == "template":
                    idx = self._template_slot_order(xyz[idx].reshape(n_full, gsz, 3), idx)
                slot_src[:m] = selected[idx]
            # trailing partial group keeps the raw Morton order
            if k > n_full * gsz:
                slot_src[n_full * gsz : k] = selected[n_full * gsz : k]
            return slot_src

        if self.partition_mode == "kd":
            # kd decides *which* points share a group; slot order inside a group
            # stays Morton (``selected`` is Morton-sorted, so ascending local index
            # is Morton order). Ordering slots along the kd split axis instead makes
            # the within-group layout a near-monotone line, which the decoder can
            # guess from the centroid alone and which shows up as streak artifacts.
            groups = kd_equal_groups(xyz_sel, ng)
            for g, loc in enumerate(groups):
                k = min(int(loc.shape[0]), gsz)
                if k <= 0:
                    continue
                slot_src[g * gsz : g * gsz + k] = selected[np.sort(loc)[:k]]
            return slot_src

        # Morton-contiguous equal split (selected is already Morton-ordered).
        groups = morton_equal_groups(n, ng)
        for g, loc in enumerate(groups):
            k = min(int(loc.shape[0]), gsz)
            if k <= 0:
                continue
            slot_src[g * gsz : g * gsz + k] = selected[loc[:k]]
        return slot_src

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict:
        path = self.files[index]
        # Per-scene frame. Everything downstream in this method -- normalisation,
        # the augmentation's world map, the anchor assignment and the center/scale
        # handed to the render -- has to use THIS file's scene, not scene 0.
        # NOT `si`: the encoder-input block below already uses that name for
        # max_input_points and would silently overwrite the scene index. Caught by
        # the multi-scene verification, which reported scene=2097152.
        scene_idx = int(self.file_scene[index])
        center, scale = self.centers[scene_idx], self.scales[scene_idx]
        gs = load_npz_state(path)
        target_all, _ = gaussian_target_channels(gs, center, scale)
        n_original = target_all.shape[0]

        rng = np.random.default_rng((self.seed * 1000003 + self.epoch * 7919 + index) & 0x7FFFFFFF)
        xyz_norm = target_all[:, 0:3]
        cam_vec = gs.get("camera", None)
        world_xf = None
        # Fixed anchors follow the cloud through augmentation, in normalised space.
        anchors = self.scene_anchors_list[scene_idx]
        if self.augment:
            xyz_norm, target_all, world_xf, norm_xf = self._augment(
                xyz_norm, target_all, rng, center=center, scale=scale)
            if cam_vec is not None:
                cam_vec = transform_camera_vector(cam_vec, *world_xf)
            if anchors is not None:
                Ra, sa, sh = norm_xf
                anchors = ((anchors @ Ra.T) * sa + sh[None, :]).astype(np.float32)

        keep = np.ones(target_all.shape[0], dtype=bool)
        if self.drop_outside:
            keep = np.all(np.abs(xyz_norm) <= 1.0, axis=1)
        keep_idx = np.nonzero(keep)[0]          # kept row -> index in the npz arrays
        target_all = target_all[keep]
        xyz_norm = target_all[:, 0:3]
        gs_kept = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in gs.items()}
        valid_n = target_all.shape[0]
        if valid_n == 0:
            raise RuntimeError(f"all points dropped for {path}")

        importance = None
        order = None
        cache_hit = False
        shared_enc = None
        shared_on = bool(getattr(self, "shared_cell_owner", False)) and self.max_input_points > self.max_points
        if self.cache_slots and not self.augment:
            cached = self._slot_cache.get(path)
            if cached is not None:
                self._slot_cache.move_to_end(path)
                selected, slot_src = cached[0], cached[1]
                shared_enc = cached[2] if len(cached) > 2 else None
                cache_hit = True
        if not cache_hit:
            importance = self._importance(gs_kept, target_all, xyz_norm=xyz_norm)
            codes = morton3d_np(xyz_norm, bits=self.morton_bits)
            order = np.argsort(codes, kind="stable")
            if shared_on:
                sel_i = self._select(order, importance, xyz_norm, rng, cap=self.max_input_points)
                slot_src, shared_enc = self._pack_shared(sel_i, xyz_norm, anchors)
                selected = slot_src[slot_src >= 0]
            else:
                selected = self._select(order, importance, xyz_norm, rng)
                xyz_sel = xyz_norm[selected]
                slot_src = self._pack_slots(selected, xyz_sel, anchors=anchors)
            if self.cache_slots and not self.augment:
                packed = (
                    np.asarray(selected, dtype=np.int64),
                    np.asarray(slot_src, dtype=np.int64),
                )
                if shared_enc is not None:
                    packed = packed + (np.asarray(shared_enc, dtype=np.int64),)
                self._slot_cache[path] = packed
                while len(self._slot_cache) > self._slot_cache_max:
                    self._slot_cache.popitem(last=False)

        s = self.max_points
        target = np.zeros((s, self.target_dim), np.float32)
        input_feat = np.zeros((s, self.input_dim), np.float32)
        mask = np.zeros((s,), np.float32)
        # Which npz row each slot came from, -1 for empty. Attribute association
        # and render comparison must use this, not a nearest-neighbour lookup:
        # 0.16-0.17% of the Gaussians in these scenes are *exactly* coincident
        # (NN distance 0) and another ~0.5% sit closer than the 3e-5 float32
        # error of the normalise/denormalise round trip, so an NN map from the
        # selected subset back to the npz is only 0.991 injective. That is
        # harmless as an evaluation control but unusable as a training target,
        # and slot <-> Gaussian identity is exactly what the attribute stage needs.
        source_index = np.full((s,), -1, np.int64)
        live = slot_src >= 0
        used = int(live.sum())
        if used > 0:
            src = slot_src[live]
            tgt = target_all[src]
            density = self._local_density(xyz_norm)[src].reshape(-1, 1)
            # Keep the historical feature width for checkpoint/layout
            # compatibility, but this channel is deliberately constant.  It used
            # to contain pruning_score, which is not a deployable input.
            intrinsic_pad = np.zeros((used, 1), np.float32)
            # Slot / chunk ranks follow the packed layout (group-major), not the
            # raw selection order — otherwise redistribute breaks the cues.
            flat_pos = np.nonzero(live)[0].astype(np.float32)
            slot_rank = (flat_pos % self.group_size) / max(float(self.group_size - 1), 1.0)
            chunk_rank = (flat_pos // self.chunk_size) / max(
                float(np.ceil(self.max_points / self.chunk_size) - 1), 1.0
            )
            inp = np.concatenate(
                [tgt, intrinsic_pad, density, slot_rank.reshape(-1, 1), chunk_rank.reshape(-1, 1)],
                axis=1,
            ).astype(np.float32)
            target[live] = tgt
            input_feat[live] = inp
            mask[live] = 1.0
            source_index[live] = keep_idx[src]

        # Encoder input. A larger independent sample made z_compact describe a
        # different set than the decoder target, so a world-model cell had no
        # stable Gaussian identity. Keep the two identical unless the caller
        # explicitly asks for a bigger encoder capacity.
        enc_in = input_feat
        enc_mask = mask
        if self.max_input_points > self.max_points:
            si = int(self.max_input_points)
            gi = int(self.group_in)
            ngi = si // gi
            if shared_enc is not None:
                src_i = np.asarray(shared_enc, dtype=np.int64)
            else:
                if order is None or importance is None:
                    importance = self._importance(gs_kept, target_all, xyz_norm=xyz_norm)
                    codes = morton3d_np(xyz_norm, bits=self.morton_bits)
                    order = np.argsort(codes, kind="stable")
                sel_i = self._select(order, importance, xyz_norm, rng, cap=si)
                # Plain Morton fill, not the template/kd slot ordering the target uses.
                # Within-group slot ORDER is meaningless on the encoder side because the
                # group is pooled by attention before anything downstream sees it; the
                # template ordering exists so the decoder's slot i has a consistent
                # meaning, which is a property of the OUTPUT, not the input.
                src_i = -np.ones(si, dtype=np.int64)
                k_i = min(int(sel_i.shape[0]), si)
                if anchors is not None and ngi == anchors.shape[0]:
                    # Historical independent assign. Shared ownership (above) is
                    # the contract that keeps cell IDs identical; this branch is
                    # kept so old args.json eval stays bit-comparable.
                    pos_i = self._anchor_assign(
                        xyz_norm[sel_i], gi, ngi, anchors,
                        spill=int(getattr(self, "anchor_spill", 8)),
                    )
                    liv_i = pos_i >= 0
                    src_i[: ngi * gi][liv_i] = sel_i[pos_i[liv_i]]
                elif k_i > 0:
                    per = np.minimum(np.diff(np.linspace(0, k_i, ngi + 1).astype(np.int64)), gi)
                    pos = 0
                    for gidx in range(ngi):
                        c = int(per[gidx])
                        if c <= 0:
                            continue
                        src_i[gidx * gi : gidx * gi + c] = sel_i[pos : pos + c]
                        pos += c
            enc_in = np.zeros((si, self.input_dim), np.float32)
            enc_mask = np.zeros((si,), np.float32)
            liv = src_i >= 0
            if int(liv.sum()) > 0:
                s2 = src_i[liv]
                tg2 = target_all[s2]
                den2 = self._local_density(xyz_norm)[s2].reshape(-1, 1)
                intrinsic_pad2 = np.zeros((int(liv.sum()), 1), np.float32)
                fp = np.nonzero(liv)[0].astype(np.float32)
                sr2 = (fp % gi) / max(float(gi - 1), 1.0)
                cr2 = (fp // self.chunk_size) / max(float(np.ceil(si / self.chunk_size) - 1), 1.0)
                enc_in[liv] = np.concatenate(
                    [tg2, intrinsic_pad2, den2, sr2.reshape(-1, 1), cr2.reshape(-1, 1)], axis=1
                ).astype(np.float32)
                enc_mask[liv] = 1.0

        # 16-vector, or zeros when the frame has no camera. The render loss skips
        # any sample whose fx is 0, so a mixed dataset degrades gracefully instead
        # of crashing the step.
        if cam_vec is None:
            cam_vec = np.zeros(16, np.float32)

        # The real photograph this frame's camera was posed against, when a map is
        # available. The render loss otherwise compares against a *render of the
        # target Gaussians*, which is a proxy: measured on three held-out frames,
        # that proxy is only 21.75 dB away from the photograph itself, so the model
        # was being taught to reproduce the ground-truth Gaussians' own error.
        # Augmentation is safe here -- transform_camera_vector moves the camera by
        # the same world similarity, and the docstring's whole point is that the
        # render is unchanged, so the photograph still corresponds.
        photo = np.zeros(1, np.float32)
        scene_key = None
        pm = getattr(self, "photo_map", None)
        if pm:
            # FULL PATH first, basename second. Every scene names its snapshots
            # step_XXXXXX.npz, so with two scenes the basenames collide 100% and a
            # basename lookup would supervise truck's Gaussians with train's
            # photographs -- no error anywhere, just a model fitting the wrong
            # image. The basename form is kept so single-scene maps still work.
            ent = pm.get(path)
            if ent is None:
                ent = pm.get(os.path.abspath(path))
            if ent is None:
                ent = pm.get(os.path.basename(path))
            # {npz: path} (single scene) or {npz: {image, scene}} (multi-scene).
            # A miss is fatal in the multi-scene case, not cosmetic: without a
            # scene key the view pool falls back to "the only pool" and, with two
            # pools, to None -- so the sample would train against no photograph at
            # all and the render loss would quietly drop to the proxy reference.
            if ent is None:
                if len(getattr(self, "roots", [""])) > 1:
                    raise KeyError(
                        f"photo_map has no entry for {path}. With multiple scenes the "
                        f"map must be keyed by FULL PATH (tools/merge_assets.py does "
                        f"this); basenames collide because every scene names its "
                        f"snapshots step_XXXXXX.npz.")
                ip, scene_key = None, None
            else:
                ip = ent.get("image") if isinstance(ent, dict) else ent
                scene_key = ent.get("scene") if isinstance(ent, dict) else None
            if scene_key is None:
                names = getattr(self, "scene_names", None) or []
                if names and 0 <= scene_idx < len(names):
                    scene_key = names[scene_idx]
            if bool(getattr(self, "holdout_own_photo", False)) and ip and self._image_is_held(ip, scene_key):
                # Same exclude policy as extras: do not train on a held-out
                # photograph through the snapshot's own-photo path. Replace the
                # camera AND the image together; swapping only the pixels would
                # pair the wrong pose with the wrong photo.
                pool_own = self._pool_for_scene(scene_key)
                allowed_own = self._allowed_pool_indices(pool_own)
                if allowed_own.size and pool_own:
                    j = int(np.random.default_rng().choice(allowed_own))
                    e = pool_own[j]
                    ip = e.get("image")
                    cam_vec = np.asarray(e["cam"], np.float32)
                    if world_xf is not None:
                        cam_vec = transform_camera_vector(cam_vec, *world_xf)
                else:
                    ip = None
            if ip:
                try:
                    from PIL import Image
                    photo = (np.asarray(Image.open(ip).convert("RGB"), np.float32) / 255.0
                             ).transpose(2, 0, 1)
                except Exception:
                    photo = np.zeros(1, np.float32)

        # EXTRA REAL VIEWS. This dataset is 3000 snapshots of ONE scene's 3DGS
        # optimisation, so every one of the 301 photographs is a valid view of
        # every snapshot. Without this, each sample is supervised by exactly one
        # real image (plus jittered *renders of the target Gaussians*, which carry
        # no information the target does not already have).
        #
        # One view per sample is the whole problem. Measured directly: optimising
        # a scene's code against 4 views gains +3.71 dB on those 4 and loses
        # 0.93 dB on 4 held-out views. With one view the 3D appearance is wildly
        # under-determined, and under that uncertainty the loss-minimising answer
        # is the conditional mean -- which is exactly the washed-out render this
        # model produces. Per-scene methods (3DGS, Scaffold-GS, HAC) fit against
        # ~300 images per scene; this was fitting against 1.
        #
        # Any world augmentation applies to these cameras too: the same similarity
        # that moved the cloud moves every camera, and transform_camera_vector is
        # exactly that map.
        if scene_key is None:
            names = getattr(self, "scene_names", None) or []
            if names and 0 <= scene_idx < len(names):
                scene_key = names[scene_idx]
            else:
                scene_key = f"scene{scene_idx}"
        extra_imgs = np.zeros((0, 3, 1, 1), np.float32)
        extra_cams = np.zeros((0, 16), np.float32)
        # View selection must be fresh on EVERY visit, not per epoch. The sampling
        # rng above is seeded by (seed, epoch, index), and the loader runs with
        # persistent_workers=True, so worker copies never see set_epoch and each
        # sample would be supervised from one fixed set of viewpoints for the whole
        # run. Measured on a frozen model with the render budget held constant at
        # 800 renders: optimising attributes against 4 FIXED photographs moves
        # held-out PSNR by +0.02 dB, while drawing fresh photographs each step from
        # a pool of 64 moves it by +4.13 dB, and drawing ONE fresh photograph per
        # step (what 3DGS does) by +4.67 dB. What matters is how many distinct
        # views the optimisation sees in total, not how many it sees per step.
        # Training leaves extra_view_seed unset so each visit is a fresh draw.
        # Probes set it so two edge-weight runs see the same cameras.
        vrng = np.random.default_rng(getattr(self, "extra_view_seed", None))
        k = int(getattr(self, "extra_real_views", 0))
        vp = getattr(self, "view_pool", None)
        # Per-SCENE pools. A flat list would silently supervise one scene's
        # Gaussians with another scene's photographs the moment the dataset holds
        # more than one scene -- which is the intended direction for this model,
        # so the lookup has to be keyed from the start.
        pool = self._pool_for_scene(scene_key)
        if k > 0 and pool:
            from PIL import Image
            # Never train on a view the eval scores. Without this the held-out set
            # is drawn from the same pool the loader samples every step, so after a
            # few thousand steps every "held-out" photograph has been fitted and the
            # reported PSNR measures memorisation. An empty allowed set must stay
            # empty -- falling back to the full pool re-opens the leak.
            allowed = self._allowed_pool_indices(pool)
            js = (vrng.choice(allowed, size=min(k, allowed.size), replace=False)
                  if allowed.size else np.zeros(0, dtype=np.int64))
            ims, cvs = [], []
            keep_full = bool(getattr(self, "keep_extra_fullres", False))
            for j in js:
                e = pool[int(j)]
                try:
                    im = Image.open(e["image"]).convert("RGB")
                    if (not keep_full) and getattr(self, "view_downscale", 1) > 1:
                        d = int(self.view_downscale)
                        im = im.resize((im.width // d, im.height // d), Image.BILINEAR)
                    ims.append((np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1))
                    cv2 = np.asarray(e["cam"], np.float32)
                    if world_xf is not None:
                        cv2 = transform_camera_vector(cv2, *world_xf)
                    cvs.append(np.asarray(cv2, np.float32))
                except Exception:
                    pass
            if ims:
                extra_imgs = np.stack(ims); extra_cams = np.stack(cvs)

        return {
            # Canonical group centre shared by the 64-slot target and the
            # 144-point encoder set.  Their sampled centroids are different by
            # construction; using this fixed scene anchor prevents that sampling
            # difference from becoming an impossible decoder target.
            "group_anchor": torch.from_numpy(np.ascontiguousarray(
                anchors if anchors is not None
                else np.zeros((self.num_groups, 3), np.float32)
            )),
            "group_anchor_valid": torch.tensor(anchors is not None, dtype=torch.bool),
            "enc_input": torch.from_numpy(np.ascontiguousarray(enc_in)),
            "enc_mask": torch.from_numpy(np.ascontiguousarray(enc_mask)),
            "extra_imgs": torch.from_numpy(np.ascontiguousarray(extra_imgs)),
            "extra_cams": torch.from_numpy(np.ascontiguousarray(extra_cams)),
            "photo": torch.from_numpy(np.ascontiguousarray(photo)),
            "input": torch.from_numpy(input_feat),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "camera": torch.from_numpy(np.asarray(cam_vec, np.float32)),
            "source_index": torch.from_numpy(source_index),
            "num_points_original": torch.tensor(n_original, dtype=torch.int64),
            "num_points_valid": torch.tensor(valid_n, dtype=torch.int64),
            "num_points_used": torch.tensor(used, dtype=torch.int64),
            "center": torch.from_numpy(np.asarray(center, np.float32).copy()),
            "scale": torch.tensor(float(scale), dtype=torch.float32),
            "scene": torch.tensor(scene_idx, dtype=torch.int64),
            "scene_key": scene_key or f"scene{scene_idx}",
            "name": os.path.basename(path),
        }
