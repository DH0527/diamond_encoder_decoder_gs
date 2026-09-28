"""
render_adapter.py

Lightweight adapter around gaussian-splatting's render(...) used in render.py
Provides helpers to render views into PyTorch tensors (N,C,H,W in [0,1]) and
simple loss functions so it can be integrated into training pipelines.

This file is intended to be imported by training scripts (e.g. a decoder trainer)
so they can call render_adapter.render_views(...) and compute image losses.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import torch
import torchvision

try:
    # The original project exposes a render function used by render.py
    from gaussian_renderer import render as gs_render
except Exception:
    gs_render = None


def _to_tensor_image(img, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert renderer output to a torch.FloatTensor in [0,1] with shape (C,H,W).

    Accepts either a numpy array or a torch tensor. The renderer commonly returns
    HxWxC arrays. We convert to CxHxW float in [0,1].
    """
    if isinstance(img, torch.Tensor):
        t = img
    else:
        t = torch.tensor(img)
    # ensure float
    t = t.to(dtype=torch.float32)
    # If last dim is channel
    if t.ndim == 3 and t.shape[-1] in (1, 3, 4):
        t = t.permute(2, 0, 1)
    # clamp / normalize if values appear to be in 0..255
    if t.max() > 2.0:
        t = t / 255.0
    if device is not None:
        t = t.to(device)
    return t


def render_view(view, gaussians, pipeline, background, use_trained_exp=False, separate_sh=False, device: Optional[torch.device] = None):
    """Render a single view using the gaussian-splatting renderer and return a CxHxW float tensor in [0,1].

    Parameters mirror the ones in gaussian-splatting/render.py. If the underlying
    renderer is unavailable (module import failed), an exception is raised.
    """
    if gs_render is None:
        raise RuntimeError("gaussian-splatting renderer (gs_render) is not available in PYTHONPATH")

    out = gs_render(view, gaussians, pipeline, background, use_trained_exp=use_trained_exp, separate_sh=separate_sh)
    # renderer returns a dict with key "render" (as seen in render.py)
    img = out["render"] if isinstance(out, dict) and "render" in out else out

    t = _to_tensor_image(img, device=device)
    return t


def render_views(views: Iterable, gaussians, pipeline, background, use_trained_exp=False, separate_sh=False, device: Optional[torch.device] = None) -> torch.Tensor:
    """Render an iterable of views and return a batched tensor (N,C,H,W) in [0,1].

    This function is small and synchronous (calls renderer for each view).
    """
    imgs: List[torch.Tensor] = []
    for v in views:
        imgs.append(render_view(v, gaussians, pipeline, background, use_trained_exp=use_trained_exp, separate_sh=separate_sh, device=device))
    # pad/stack to common spatial size if needed
    sizes = [tuple(img.shape[1:]) for img in imgs]
    if len(set(sizes)) != 1:
        # resize to the first size using torchvision (bilinear)
        H, W = sizes[0]
        resized = []
        for img in imgs:
            if (img.shape[1], img.shape[2]) != (H, W):
                img = torchvision.transforms.functional.resize(img, (H, W))
            resized.append(img)
        imgs = resized
    batch = torch.stack(imgs, dim=0)
    return batch


def image_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean-squared error between images. Expects tensors in [0,1], shape (N,C,H,W) or (C,H,W).

    If mask is provided, it should be broadcastable to (N,1,H,W) to weight pixels.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    assert pred.shape == target.shape, f"pred {pred.shape} target {target.shape}"
    loss = (pred - target).pow(2).mean(dim=1, keepdim=True)  # (N,1,H,W)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


if __name__ == "__main__":
    # Quick CLI to render and save images similarly to render.py but using this adapter.
    import argparse
    from arguments import ModelParams, PipelineParams, get_combined_args
    from scene import Scene
    from os import makedirs

    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    cfg = model.extract(args)
    pipe = pipeline.extract(args)
    scene = Scene(cfg, GaussianModel=__import__("gaussian_renderer").GaussianModel, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    out_dir = os.path.join(cfg.model_path, cfg.scene_name if hasattr(cfg, 'scene_name') else cfg.name, f"renders_adapter_{args.iteration}")
    makedirs(out_dir, exist_ok=True)

    # choose train/test as in render.py
    if not args.skip_train:
        views = scene.getTrainCameras()
        batch = render_views(views, scene.gaussians, pipe, background, use_trained_exp=cfg.train_test_exp, separate_sh=False, device=torch.device("cpu"))
        for i in range(batch.size(0)):
            torchvision.utils.save_image(batch[i], os.path.join(out_dir, f"train_{i:05d}.png"))
    if not args.skip_test:
        views = scene.getTestCameras()
        batch = render_views(views, scene.gaussians, pipe, background, use_trained_exp=cfg.train_test_exp, separate_sh=False, device=torch.device("cpu"))
        for i in range(batch.size(0)):
            torchvision.utils.save_image(batch[i], os.path.join(out_dir, f"test_{i:05d}.png"))
