import numpy as np
import torch
import os, json, gc, psutil, time, io
from threading import Thread
from queue import Queue

# Replay writer: write plain uncompressed .npz files. (Zarr/Blosc support removed)
# Zarr/numcodecs support removed: we write plain .npz files only.
ZARR_AVAILABLE = False

# ===== helpers for safe delta & conversion =====
def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def _safe_delta(a, b):
    """shape가 (N,3) vs (N,1,3) 등일 때도 안전하게 뺄셈해서 numpy 반환"""
    a_np, b_np = _to_numpy(a), _to_numpy(b)
    try:
        return b_np - a_np
    except Exception:
        return b_np.squeeze() - a_np.squeeze()

# ============================================================
# 1️⃣ Snapshot : 기존 구조 + GPU→CPU 전송 최적화
# ============================================================

def snapshot_state(gaussians, viewpoint_cam, to_cpu=False):
    # Gather camera intrinsics robustly. Prefer explicit K if present,
    # otherwise look for fx/fy/cx/cy attributes. If none available,
    # fall back to a conservative estimation from image size (assume 60deg FOV).
    # Store intrinsics as Python floats (not tensors) so they are not
    # coerced to float16 in the generic tensor->half conversion step.
    fx = fy = cx = cy = None
    try:
        if hasattr(viewpoint_cam, "K") and getattr(viewpoint_cam, "K") is not None:
            K_src = viewpoint_cam.K
            # accept numpy array or tensor
            if isinstance(K_src, torch.Tensor):
                K_np = K_src.detach().cpu().numpy()
            else:
                K_np = np.array(K_src)
            fx = float(K_np[0, 0])
            fy = float(K_np[1, 1])
            cx = float(K_np[0, 2])
            cy = float(K_np[1, 2])
        else:
            # try common attribute names
            def _getf(name):
                v = getattr(viewpoint_cam, name, None)
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            fx = _getf('fx') or _getf('f') or _getf('focal')
            fy = _getf('fy') or fx
            cx = _getf('cx')
            cy = _getf('cy')

            # if still missing, try to infer from original_image size
            if fx is None:
                if hasattr(viewpoint_cam, 'original_image'):
                    try:
                        h, w = viewpoint_cam.original_image.shape[-2:]
                    except Exception:
                        w = getattr(viewpoint_cam, 'width', None)
                        h = getattr(viewpoint_cam, 'height', None)
                    if w is not None:
                        # assume 60 degree horizontal FOV as conservative default
                        import math
                        fx = (w / 2.0) / math.tan(math.radians(60.0) / 2.0)
                        fy = fy or fx
                        cx = cx or (w / 2.0)
                        cy = cy or (h / 2.0 if h is not None else w / 2.0)
                else:
                    # final fallback: small non-zero defaults
                    fx = fy = 1.0
                    cx = cy = 0.0
    except Exception:
        # Very conservative fallback in case of unexpected camera object
        fx = fy = 1.0
        cx = cy = 0.0

    R = torch.as_tensor(viewpoint_cam.R, device="cuda")
    T = torch.as_tensor(viewpoint_cam.T, device="cuda")

    # Ensure color saved includes full SH coefficients when present.
    color = gaussians.get_features_dc.clone()
    try:
        if color.ndim == 3:
            # prefer ordering where last dim is SH coeffs -> (N, F, C)
            if color.shape[1] <= 4 and color.shape[2] > 4:
                c_flat = color.transpose(1, 2).reshape(color.shape[0], -1)
            elif color.shape[1] > 4 and color.shape[2] <= 4:
                c_flat = color.transpose(1, 2).reshape(color.shape[0], -1)
            else:
                c_flat = color.transpose(1, 2).reshape(color.shape[0], -1)
            color_to_store = c_flat
        else:
            color_to_store = color
    except Exception:
        color_to_store = color

    # SH Rest Features 가져오기
    features_rest = gaussians.get_features_rest.clone()
    # --- 원본 이미지(raw data)는 저장하지 않음; 사용된 파일 경로만 검출하여 저장 ---
    image_path = None
    try:
        # original_image가 파일 경로(string/bytes)로 제공되는 경우 이를 사용
        img = getattr(viewpoint_cam, "original_image", None)
        if isinstance(img, (str, bytes)):
            image_path = str(img)

        # 아니면 일반적으로 사용되는 속성들에서 파일 경로를 찾아봄
        if image_path is None:
            for name in ("original_image_path", "image_path", "img_path", "img_name", "filename", "file_name", "path"):
                val = getattr(viewpoint_cam, name, None)
                if val is not None:
                    try:
                        image_path = str(val)
                        break
                    except Exception:
                        continue
    except Exception:
        image_path = None

    state = {
        "camera": {
            # store intrinsics as Python floats to avoid unintended float16 coercion
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "R": R, "T": T,
            "t": torch.as_tensor(
                getattr(viewpoint_cam, "trans", [0.0, 0.0, 0.0]), device="cuda"
            ),
            # 원본 이미지(raw data)는 저장하지 않음; 사용된 파일 경로(가능한 경우)만 저장
            "image_path": image_path,
        },
        "gaussians": {
            "xyz": gaussians.get_xyz.clone(),
            "scaling": gaussians.get_scaling.clone(),
            "color": color_to_store.clone(),
            "features_rest": features_rest,  # SH 계수 저장
            "opacity": gaussians.get_opacity.clone(),
            "rot": gaussians.get_rotation.clone(),
        },
        "ids": (
            gaussians.get_ids.clone()
            if hasattr(gaussians, "get_ids")
            else torch.arange(gaussians.get_xyz.shape[0], device="cuda")
        ),
    }

    if to_cpu:
        state = snapshot_state_obj_convert(state, to_cpu=True)

    return state


# ============================================================
# 2️⃣ Action Δ 계산
# ============================================================

def diff_action(state_t, state_tp1, dt: int = 1, mode: str = "aggregate"):
    """
    state_t   : 이전 '로그 시점' 스냅샷
    state_tp1 : 현재 '로그 시점' 스냅샷
    """
    ids_t  = _to_numpy(state_t["ids"])
    ids_tp = _to_numpy(state_tp1["ids"])

    # 공통/신규/소멸 id
    common_ids, idx_t, idx_tp = np.intersect1d(ids_t, ids_tp, return_indices=True)
    new_ids  = np.setdiff1d(ids_tp, ids_t)
    dead_ids = np.setdiff1d(ids_t, ids_tp)

    g_t, g_tp = state_t["gaussians"], state_tp1["gaussians"]

    d_xyz      = _safe_delta(g_t["xyz"][idx_t],     g_tp["xyz"][idx_tp])
    d_scaling  = _safe_delta(g_t["scaling"][idx_t], g_tp["scaling"][idx_tp])
    d_color    = _safe_delta(g_t["color"][idx_t],   g_tp["color"][idx_tp])
    d_opacity  = _safe_delta(g_t["opacity"][idx_t], g_tp["opacity"][idx_tp])
    d_rotation = _safe_delta(g_t["rot"][idx_t],     g_tp["rot"][idx_tp])

    # SH Rest Delta 계산
    f_rest_t = g_t.get("features_rest", None)
    f_rest_tp = g_tp.get("features_rest", None)
    
    d_features_rest = None
    if f_rest_t is not None and f_rest_tp is not None:
        d_features_rest = _safe_delta(f_rest_t[idx_t], f_rest_tp[idx_tp])

    if mode == "per_step" and dt > 0:
        d_xyz      = d_xyz / dt
        d_scaling  = d_scaling / dt
        d_color    = d_color / dt
        d_opacity  = d_opacity / dt
        d_rotation = d_rotation / dt
        if d_features_rest is not None:
            d_features_rest = d_features_rest / dt

    return {
        "continuous": {
            "ids": common_ids.tolist(),
            "d_xyz": d_xyz,
            "d_scaling": d_scaling,
            "d_color": d_color,
            "d_features_rest": d_features_rest,
            "d_opacity": d_opacity,
            "d_rotation": d_rotation,
        },
        "birth": new_ids.tolist(),
        "death": dead_ids.tolist(),
        "dt": int(dt),
        "mode": str(mode),
    }

def zero_action_from_state(state, dt: int = 0, mode: str = "aggregate"):
    """첫 로그(기준 없음)에서 쓰는 0-액션 생성기"""
    ids = _to_numpy(state["ids"])
    g = state["gaussians"]
    zeros_like = lambda x: np.zeros_like(_to_numpy(x))
    
    d_features_rest = None
    if "features_rest" in g:
        d_features_rest = zeros_like(g["features_rest"])

    return {
        "continuous": {
            "ids": ids.tolist(),
            "d_xyz": zeros_like(g["xyz"]),
            "d_scaling": zeros_like(g["scaling"]),
            "d_color": zeros_like(g["color"]),
            "d_features_rest": d_features_rest,
            "d_opacity": zeros_like(g["opacity"]),
            "d_rotation": zeros_like(g["rot"]),
        },
        "birth": ids.tolist(),          
        "death": [],
        "dt": int(dt),
        "mode": str(mode),
    }
    
    
# ============================================================
# 3️⃣ 보상 계산
# ============================================================

def compute_reward_simple(gt, pred, n_visible=0, pose_delta_norm=0.0):
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    l1 = float(np.abs(pred_np - gt_np).mean())
    psnr = -10.0 * np.log10(np.mean((pred_np - gt_np) ** 2) + 1e-8)

    return {
        "l1": l1,
        "psnr": float(psnr),
        "n_visible": n_visible,
        "pose_penalty": float(pose_delta_norm),
    }
    
def _count_gaussians(state) -> int:
    ids = state.get("ids", None)
    if ids is not None:
        if torch.is_tensor(ids): 
            return int(ids.numel())
        return int(np.asarray(ids).size)
    # fallback
    g = state["gaussians"]["xyz"]
    if torch.is_tensor(g): 
        return int(g.shape[0])
    return int(np.asarray(g).shape[0])

def compute_reward_quality_efficiency(
    *,
    loss_t: float,
    prev_loss_t: float | None,
    n_t: int,
    n_prev: int | None,
    quality_weight: float = 1.0,   
    eff_delta_weight: float = 0.1, 
    eff_abs_weight: float = 0.0,   
    n_ref: float | None = None,    
    squash_tanh_scale: float | None = 0.1,  
):
    q = 0.0 if prev_loss_t is None else (float(prev_loss_t) - float(loss_t))

    eff = 0.0
    if n_prev is not None:
        n_prev = float(n_prev)
        n_t = float(n_t)
        nref = float(n_ref) if n_ref is not None else max(1.0, n_prev if n_prev > 0 else n_t)
        dN = (n_t - n_prev) / nref
        absN = n_t / nref
        eff = - eff_delta_weight * dN - eff_abs_weight * absN

    r = quality_weight * q + eff
    r_unsquashed = float(r)
    if squash_tanh_scale is not None:
        r = float(np.tanh(r / float(squash_tanh_scale)))

    return {
        "scalar": float(r),     
        "quality": float(q),    
        "eff": float(eff),      
        "loss_t": float(loss_t),
        "loss_prev": (None if prev_loss_t is None else float(prev_loss_t)),
        "n_t": int(n_t),
        "n_prev": (None if n_prev is None else int(n_prev)),
        "unsquashed": r_unsquashed,
    }

# ============================================================
# 4️⃣ Async Replay Logger (Optimized: Compressed & Float16)
# ============================================================
class ReplayLogger:
    def __init__(self, outdir, flush_every=10, async_write=True):
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.flush_every = flush_every
        self.index = []
        self.async_write = async_write
        self.mem_print_interval = 5 

        if async_write:
            self.queue = Queue(maxsize=2)
            self.worker = Thread(target=self._writer, daemon=True)
            self.worker.start()

        self._last_mem_log = time.time()

    def _writer(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            fname, payload = item
            try:
                # Always write an uncompressed .npz file as the writer's behavior.
                try:
                    # fname expected to include .npz extension
                    np.savez(fname, **payload)
                except Exception as e:
                    print(f"[Writer Error - save npz] {fname}: {e}")
            except Exception as e:
                print(f"[Writer Error] {fname}: {e}")
            finally:
                self.queue.task_done()

    def _check_memory(self):
        now = time.time()
        if now - self._last_mem_log > self.mem_print_interval:
            mem = psutil.virtual_memory()
            self._last_mem_log = now

            if mem.percent >= 90:
                print(f"[WARN] Very high RAM usage ({mem.percent:.1f}%). Forcing cleanup...")
                gc.collect()
                torch.cuda.empty_cache()
            elif mem.percent >= 85:
                gc.collect()
                torch.cuda.empty_cache()


    def add_step(self, it, state_t, action_t, reward_t, events, meta):
        # GPU → CPU 변환
        # NOTE: `state_tp1` was removed from the signature and is not stored here.
        state_t = snapshot_state_obj_convert(state_t, to_cpu=True)
        action_t = snapshot_state_obj_convert(action_t, to_cpu=True)
        reward_t = snapshot_state_obj_convert(reward_t, to_cpu=True)

        payload = {
            "it": it,
            "state_t": state_t,
            "action_t": action_t,
            # "state_tp1": removed intentionally
            "reward_t": reward_t,
            "events": json.dumps(events).encode("utf-8"),
            "meta": json.dumps(meta).encode("utf-8"),
        }

        # Storage: always write uncompressed .npz files
        fname = os.path.join(self.outdir, f"step_{it:06d}.npz")

        if self.async_write:
            self.queue.put((fname, payload))
        else:
            # Synchronous write: write uncompressed .npz to disk
            try:
                np.savez(fname, **payload)
            except Exception as e:
                print(f"[Sync Write Error] {fname}: {e}")

        self.index.append({"it": it, "file": fname})
        if len(self.index) % self.flush_every == 0:
            with open(os.path.join(self.outdir, "index.jsonl"), "a") as f:
                for rec in self.index:
                    f.write(json.dumps(rec) + "\n")
            self.index = []

        self._check_memory()

    def close(self):
        if self.async_write:
            self.queue.put(None)
            self.worker.join()

# ============================================================
# 5️⃣ Helper : 안전한 GPU→CPU 변환 (Float16 적용)
# ============================================================

def snapshot_state_obj_convert(obj, to_cpu=False):
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        # [최적화 2] 저장 용량 절감을 위해 float32 -> float16 변환
        # 리플레이 및 인코더 학습 용도로는 정밀도 손실이 무시할 만한 수준입니다.
        if t.dtype == torch.float32:
            t = t.half()
            
        if to_cpu:
            t = t.to("cpu", non_blocking=True)
            torch.cuda.synchronize()
            return t.numpy()
        else:
            return t
    elif isinstance(obj, dict):
        return {k: snapshot_state_obj_convert(v, to_cpu=to_cpu) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [snapshot_state_obj_convert(v, to_cpu=to_cpu) for v in obj]
    else:
        return obj

# ============================================================
# 6️⃣ Compact replay dump (encoder/decoder training)
# ============================================================
# Official 3DGS dumps millions of Gaussians. The old ReplayLogger stored SH rest
# + action deltas into an *uncompressed* object-pickle npz. This path keeps the
# fields needed for encoder/decoder + later RL (activated xyz/scale/rot/opacity,
# SH-DC, camera, ids, reward), float16, zip-deflate level 9, arrays as
# separate npy members. SH-rest is omitted (~74% of late-file bytes; the current
# tokenizer renders at sh_degree=0). Action deltas are omitted — they duplicate
# the Gaussian fields and dominate disk.

def _as_f16_np(x):
    if isinstance(x, torch.Tensor):
        a = x.detach().float().contiguous().cpu().numpy()
    else:
        a = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(a, dtype=np.float16)


def snapshot_state_compact(gaussians, viewpoint_cam):
    """CPU numpy snapshot: activated attributes + SH-DC (no SH-rest)."""
    import math

    w = int(getattr(viewpoint_cam, "image_width", 0) or 0)
    h = int(getattr(viewpoint_cam, "image_height", 0) or 0)
    if w <= 0 or h <= 0:
        img = getattr(viewpoint_cam, "original_image", None)
        if img is not None and hasattr(img, "shape"):
            h, w = int(img.shape[-2]), int(img.shape[-1])

    fovx = float(getattr(viewpoint_cam, "FoVx"))
    fovy = float(getattr(viewpoint_cam, "FoVy"))
    fx = (w / 2.0) / math.tan(fovx / 2.0) if w > 0 else 1.0
    fy = (h / 2.0) / math.tan(fovy / 2.0) if h > 0 else fx

    color = gaussians.get_features_dc.detach()
    if color.ndim == 3:
        color = color.reshape(color.shape[0], -1)

    xyz = gaussians.get_xyz
    n = int(xyz.shape[0])
    if hasattr(gaussians, "get_ids"):
        ids = gaussians.get_ids.detach().reshape(-1).cpu().numpy().astype(np.uint32, copy=False)
    else:
        ids = np.arange(n, dtype=np.uint32)

    image_name = str(getattr(viewpoint_cam, "image_name", "") or "")
    image_path = None
    for name in ("original_image_path", "image_path"):
        val = getattr(viewpoint_cam, name, None)
        if val is not None:
            image_path = str(val)
            break

    return {
        "camera": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(w) * 0.5,
            "cy": float(h) * 0.5,
            "R": np.asarray(viewpoint_cam.R, dtype=np.float32),
            "T": np.asarray(viewpoint_cam.T, dtype=np.float32),
            "image_width": int(w),
            "image_height": int(h),
            "image_name": image_name,
            "image_path": image_path,
            "uid": int(getattr(viewpoint_cam, "uid", -1)),
        },
        "gaussians": {
            "xyz": _as_f16_np(xyz),
            "scaling": _as_f16_np(gaussians.get_scaling),
            "color": _as_f16_np(color),
            "opacity": _as_f16_np(gaussians.get_opacity),
            "rot": _as_f16_np(gaussians.get_rotation),
        },
        "ids": ids,
    }


def _savez_deflate9(path, arrays):
    """np.savez_compressed but zip deflate level 9 (smaller, slower writes)."""
    import zipfile
    import io

    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with zipfile.ZipFile(
        tmp, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True
    ) as zf:
        for k, v in arrays.items():
            buf = io.BytesIO()
            np.save(buf, np.asanyarray(v), allow_pickle=False)
            zf.writestr(k + ".npy", buf.getvalue())
    os.replace(tmp, path)


REWARD_KEYS = (
    "scalar",
    "quality",
    "eff",
    "loss_t",
    "loss_prev",
    "n_t",
    "n_prev",
    "unsquashed",
)


def pack_reward(reward_t):
    """Pack the replay reward dict into a fixed float32 vector (NaN = missing)."""
    if not reward_t:
        return np.full((len(REWARD_KEYS),), np.nan, dtype=np.float32)
    vals = []
    for k in REWARD_KEYS:
        v = reward_t.get(k) if isinstance(reward_t, dict) else None
        vals.append(np.nan if v is None else float(v))
    return np.asarray(vals, dtype=np.float32)


def unpack_reward(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    out = {}
    for i, k in enumerate(REWARD_KEYS):
        if i >= arr.size or np.isnan(arr[i]):
            out[k] = None
        elif k in ("n_t", "n_prev"):
            out[k] = int(arr[i])
        else:
            out[k] = float(arr[i])
    return out


class CompactReplayLogger:
    def __init__(self, outdir, flush_every=50, async_write=True, queue_size=2):
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.flush_every = flush_every
        self.index = []
        self.async_write = async_write
        self._writes = 0
        if async_write:
            self.queue = Queue(maxsize=queue_size)
            self.worker = Thread(target=self._writer, daemon=True)
            self.worker.start()

    def _writer(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            fname, payload = item
            try:
                _savez_deflate9(fname, payload)
            except Exception as e:
                print(f"[CompactWriter Error] {fname}: {e}", flush=True)
            finally:
                self.queue.task_done()

    def add_step(self, it, state_t, reward_t=None, meta=None):
        g = state_t["gaussians"]
        cam = state_t["camera"]
        cam_vec = np.concatenate([
            np.array([cam["fx"], cam["fy"], cam["cx"], cam["cy"]], np.float32),
            np.asarray(cam["R"], dtype=np.float32).reshape(-1),
            np.asarray(cam["T"], dtype=np.float32).reshape(-1),
        ]).astype(np.float32)
        payload = {
            "it": np.int32(it),
            "xyz": np.ascontiguousarray(g["xyz"], dtype=np.float16),
            "scaling": np.ascontiguousarray(g["scaling"], dtype=np.float16),
            "rot": np.ascontiguousarray(g["rot"], dtype=np.float16),
            "opacity": np.ascontiguousarray(g["opacity"], dtype=np.float16),
            "color": np.ascontiguousarray(g["color"], dtype=np.float16),
            "ids": np.ascontiguousarray(state_t["ids"], dtype=np.uint32),
            "cam": cam_vec,
            "image_name": np.asarray(str(cam.get("image_name", "") or "")),
            "image_wh": np.array(
                [int(cam.get("image_width", 0) or 0), int(cam.get("image_height", 0) or 0)],
                dtype=np.int32,
            ),
            "reward": pack_reward(reward_t),
        }
        if "features_rest" in g and g["features_rest"] is not None:
            payload["features_rest"] = np.ascontiguousarray(g["features_rest"], dtype=np.float16)
        fname = os.path.join(self.outdir, f"step_{int(it):06d}.npz")
        if self.async_write:
            self.queue.put((fname, payload))
        else:
            _savez_deflate9(fname, payload)

        n = int(payload["xyz"].shape[0])
        self.index.append({"it": int(it), "file": fname, "n": n})
        self._writes += 1
        if self._writes == 1 or (self._writes % 50) == 0:
            try:
                sz = os.path.getsize(fname) if os.path.isfile(fname) else -1
            except OSError:
                sz = -1
            print(
                f"[compact_npz] wrote it={int(it)} n={n} bytes={sz} -> {fname}",
                flush=True,
            )
        if len(self.index) >= self.flush_every:
            self.flush_index()
        return fname

    def flush_index(self):
        if not self.index:
            return
        idx_path = os.path.join(self.outdir, "index.jsonl")
        with open(idx_path, "a", encoding="utf-8") as f:
            for rec in self.index:
                f.write(json.dumps(rec) + "\n")
        self.index = []

    def close(self):
        if self.async_write:
            self.queue.put(None)
            self.worker.join()
        self.flush_index()


# ============================================================
# 7️⃣ 가시 가우시안 개수 추정
# ============================================================
def estimate_visible_gaussians(visibility_filter):
    if isinstance(visibility_filter, torch.Tensor):
        return int(visibility_filter.sum().item())
    elif isinstance(visibility_filter, np.ndarray):
        return int(np.sum(visibility_filter))
    else:
        return 0