"""Differentiable rasterisation of a Gaussian set, for render-space supervision.

Everything measured in this project so far has been point-space chamfer, so
whether the current 1.27x-point-spacing reconstruction *looks* acceptable has
never been checked. That answer gates several open decisions -- the Gaussian
budget K, whether attributes can be decoded rather than stored, and whether
attribute supervision should be parameter loss, render loss, or both.

The npz files carry full intrinsics and extrinsics per frame
(``state_t['camera']`` with fx, fy, cx, cy, R, T), so a frame can be rendered
from its own camera without the COLMAP dataset. There is no ground-truth
photograph in the npz, so the comparison this enables is
**render(original Gaussians) vs render(reconstructed Gaussians)** -- which is
exactly the "Original <-> Reconstruction" row the evaluation plan needs.

Conventions follow the reference implementation in
``/data/daeho/aabb/gaussian-splatting`` (``utils/graphics_utils.py``): R is
camera-to-world rotation stored row-major such that ``Rt[:3,:3] = R.T``, and T
is the world-to-camera translation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch


C0 = 0.28209479177387814          # the l=0 SH basis value


def sh_dc_to_rgb(dc):
    """The npz ``color`` field is the DC spherical-harmonic coefficient, not RGB.

    Verified from the raw arrays: it spans -2.3 to 10.1, which is the
    ``features_dc`` range of a 3DGS checkpoint. Feeding it to the rasteriser as
    precomputed colour without this conversion renders a flat grey fog.
    """
    return dc * C0 + 0.5


def rgb_to_sh_dc(rgb):
    return (rgb - 0.5) / C0


def _world_to_view(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    rt = np.zeros((4, 4), dtype=np.float64)
    rt[:3, :3] = np.asarray(R, np.float64).transpose()
    rt[:3, 3] = np.asarray(t, np.float64).reshape(3)
    rt[3, 3] = 1.0
    return rt.astype(np.float32)


def _projection(znear: float, zfar: float, fovx: float, fovy: float) -> torch.Tensor:
    tx, ty = math.tan(fovx * 0.5), math.tan(fovy * 0.5)
    top, right = ty * znear, tx * znear
    p = torch.zeros(4, 4)
    p[0, 0] = znear / right
    p[1, 1] = znear / top
    p[3, 2] = 1.0
    p[2, 2] = zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


class Camera:
    """Everything the rasteriser needs, derived from one npz ``camera`` dict."""

    def __init__(self, cam: Dict, device="cuda", znear: float = 0.01, zfar: float = 100.0,
                 downscale: int = 1):
        fx, fy = float(cam["fx"]), float(cam["fy"])
        cx, cy = float(cam["cx"]), float(cam["cy"])
        # cx / cy are the principal point, so the sensor is 2*cx by 2*cy when it
        # is centred -- which is what these captures use.
        w, h = int(round(cx * 2)), int(round(cy * 2))
        if downscale > 1:
            w, h = w // downscale, h // downscale
            fx, fy = fx / downscale, fy / downscale
        self.image_width, self.image_height = w, h
        self.FoVx = 2.0 * math.atan(w / (2.0 * fx))
        self.FoVy = 2.0 * math.atan(h / (2.0 * fy))
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)

        w2v = torch.from_numpy(_world_to_view(cam["R"], cam["T"])).to(device)
        self.world_view_transform = w2v.transpose(0, 1)          # rasteriser wants column-major
        proj = _projection(znear, zfar, self.FoVx, self.FoVy).to(device).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ proj
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.device = device


def perturb_camera_vector(cam_vec, rng, max_deg: float = 8.0, max_shift: float = 0.15):
    """A nearby synthetic viewpoint, for multi-view equivalence.

    One camera per frame cannot certify that two Gaussian sets are the same 3D
    object: everything the camera cannot see is unconstrained, and scale/opacity
    can absorb some of the geometric error that a single projection would
    otherwise expose. There is no second photograph, but there *is* a second
    render -- the target Gaussians can be rasterised from any pose -- so the
    supervision generalises to

        L = (1/M) * sum_m D( R(pred, C_m), R(target, C_m) )

    which is view equivalence rather than one-view agreement. This is also the
    form the canonicaliser needs later: "equivalent under K Gaussians" only means
    something if it holds from more than the one pose the frame was captured at.

    The perturbation orbits the existing camera about the scene it is already
    looking at, so the new pose keeps the same content in frame. Yaw/pitch about
    the camera's own axes plus a small translation in camera space.
    """
    v = np.asarray(cam_vec, np.float64).reshape(-1).copy()
    if v.shape[0] != 16 or v[0] == 0.0:
        return np.asarray(cam_vec, np.float32)
    R_cam, T = v[4:13].reshape(3, 3), v[13:16]
    ay, ax = np.deg2rad(rng.uniform(-max_deg, max_deg, 2))
    ca, sa = np.cos(ay), np.sin(ay)
    ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    cb, sb = np.cos(ax), np.sin(ax)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]])
    # Rotate the camera in its own frame; depth (T[2]) is the orbit radius, so
    # the rotation is applied about the point it is looking at.
    delta = rx @ ry
    R_new = R_cam @ delta.T
    T_new = delta @ T + rng.uniform(-max_shift, max_shift, 3) * np.array([1.0, 1.0, 0.25])
    return np.concatenate([v[:4], R_new.reshape(-1), T_new]).astype(np.float32)


def render_gaussians(
    xyz: torch.Tensor,          # (N, 3) world coordinates
    scaling: torch.Tensor,      # (N, 3) *linear* scale, not log
    rotation: torch.Tensor,     # (N, 4) quaternion, will be normalised
    opacity: torch.Tensor,      # (N, 1) in [0, 1], not logit
    colors: torch.Tensor,       # (N, 3) precomputed RGB, already sh_dc_to_rgb'd
    cam: Camera,
    bg: Optional[torch.Tensor] = None,
    scale_modifier: float = 1.0,
    return_radii: bool = False,
) -> torch.Tensor:
    """-> (3, H, W) image. Differentiable in every Gaussian argument.

    With ``return_radii``, also returns the per-Gaussian screen radius the
    rasteriser already computes. ``radii > 0`` marks the points this view can
    actually constrain, which is what the parameter anchor needs in order to stay
    out of the render's way where the render has something to say.

    Colour is passed precomputed rather than as SH coefficients: the rest-band
    SH in this dataset is small (per-channel range +-0.9 against +-10 for the DC
    term) and is exactly zero in early frames, so keeping the view-dependent
    path out of the loop costs almost nothing and keeps the attribute decoder
    simpler until view dependence is actually being learned.
    """
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    if bg is None:
        bg = torch.zeros(3, device=xyz.device)
    settings = GaussianRasterizationSettings(
        image_height=cam.image_height,
        image_width=cam.image_width,
        tanfovx=cam.tanfovx,
        tanfovy=cam.tanfovy,
        bg=bg,
        scale_modifier=scale_modifier,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=0,
        campos=cam.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasteriser = GaussianRasterizer(raster_settings=settings)
    screen = torch.zeros_like(xyz, requires_grad=True)
    q = rotation / rotation.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    out = rasteriser(
        means3D=xyz,
        means2D=screen,
        shs=None,
        colors_precomp=colors,
        opacities=opacity,
        scales=scaling,
        rotations=q,
        cov3D_precomp=None,
    )
    # out[1] is the per-Gaussian screen radius, already computed by the
    # rasteriser and normally discarded. radii > 0 means the Gaussian was
    # actually rasterised in this view, i.e. it is one of the points the
    # photometric loss can say anything about.
    return (out[0], out[1]) if return_radii else out[0]


def gaussians_from_target(target: torch.Tensor, layout: Dict[str, int], scene_scale: float):
    """Undo the packing in ``io_utils.gaussian_target_channels``.

    The target stores centred/normalised xyz, log scale, a quaternion, logit
    opacity and DC colour. Rendering needs world units, linear scale, a sigmoid
    opacity and RGB. ``center`` must be added back by the caller, which holds it.
    """
    a = layout
    xyz = target[..., a["xyz"] : a["xyz"] + 3] * scene_scale
    scaling = torch.exp(target[..., a["scale"] : a["scale"] + 3]) * scene_scale
    rot = target[..., a["rot"] : a["rot"] + 4]
    opacity = torch.sigmoid(target[..., a["opacity"] : a["opacity"] + 1])
    color = sh_dc_to_rgb(target[..., a["color"] : a["color"] + 3])
    return xyz, scaling, rot, opacity, color


def ssim_value(a: torch.Tensor, b: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Box-window SSIM, differentiable. (3,H,W) or (B,3,H,W) -> scalar tensor.

    A box window rather than the usual Gaussian one: it is a single avg_pool2d
    per moment instead of four separable convolutions, which matters because
    this runs inside the training step on a 977x544 image.
    """
    import torch.nn.functional as F

    if a.dim() == 3:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    a, b = a.clamp(0, 1), b.clamp(0, 1)
    pad = window // 2
    mu_a = F.avg_pool2d(a, window, 1, pad)
    mu_b = F.avg_pool2d(b, window, 1, pad)
    saa = F.avg_pool2d(a * a, window, 1, pad) - mu_a ** 2
    sbb = F.avg_pool2d(b * b, window, 1, pad) - mu_b ** 2
    sab = F.avg_pool2d(a * b, window, 1, pad) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.mean()


@torch.no_grad()
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean()
    return float(-10.0 * torch.log10(mse.clamp(min=1e-12)))


@torch.no_grad()
def ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11) -> float:
    return float(ssim_value(a, b, window))


def _edge_weight(ref_img: torch.Tensor, gain: float) -> torch.Tensor:
    """Per-pixel weight from the REFERENCE image's local gradient.

    L1 weights every pixel the same, so the flat majority decides the loss even
    though the model already matches it. Measured on the step_028630 own-camera
    render at 977x544: the flat half of the pixels carries 28% of the total L1 at
    a mean error of 0.035, while the top 5% by gradient carries 11% at 0.132 --
    four times the error on a twentieth of the weight. The optimiser can buy the
    same loss reduction more cheaply by smoothing the sky than by drawing the
    handrail, and that is what the renders show.

    Not cached. A previous pointer-keyed cache could return another image's
    weights after the allocator reused the address.
    """
    g = ref_img.mean(0, keepdim=True)
    gx = torch.zeros_like(g)
    gy = torch.zeros_like(g)
    gx[..., :, 1:-1] = (g[..., :, 2:] - g[..., :, :-2]) * 0.5
    gy[..., 1:-1, :] = (g[..., 2:, :] - g[..., :-2, :]) * 0.5
    grad = (gx * gx + gy * gy).sqrt()
    return 1.0 + float(gain) * grad / grad.mean().clamp(min=1e-6)


def _sobel(img: torch.Tensor) -> torch.Tensor:
    """Sobel gradients of an image, channels kept separate. (C,H,W) -> (2C,H,W)."""
    import torch.nn.functional as F
    c = img.shape[0]
    k = img.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = F.conv2d(img[None], k, padding=1, groups=c)
    gy = F.conv2d(img[None], k.transpose(2, 3), padding=1, groups=c)
    return torch.cat([gx, gy], dim=1)[0]


def photometric_loss(pred_img: torch.Tensor, ref_img: torch.Tensor, lam_dssim: float = 0.2,
                     edge_gain: float = 0.0, w_sobel: float = 0.0):
    """The 3DGS objective: ``(1 - lam) * L1 + lam * (1 - SSIM)``.

    ``ref_img`` is a render of the original Gaussians, not a photograph -- the
    npz has no image. That makes this an *equivalence* objective: the decoder is
    free to choose any Gaussians that look like the original from this view,
    which is exactly the freedom the attribute budget needs (storing the true
    attributes costs 8-112x the compact latent, matching their appearance costs
    nothing extra).

    ``w_sobel`` adds an L1 between the two images' Sobel gradients. This is a
    different statement from ``edge_gain``, which only decides *where* the L1 is
    spent: a blurred render that keeps the right local mean still scores well
    under a reweighted L1, because nothing in it asks for contrast. Matching the
    gradients asks for the edge itself, which blur cannot supply at any weight.

    Measured on C1/T16k step 70000, attribute decoder only, 14 supervised views,
    scored on 5 views that were never supervised (viewgen_mv.py):
        w_sobel  handrail crop   lettering crop   held-out mean
          0          21.89 -> 21.15   21.92 -> 21.01   18.99 -> 20.51
          1          21.89 -> 25.61   21.92 -> 24.08   19.57 -> 21.54
          5          21.89 -> 27.14   21.92 -> 26.56   19.40 -> 19.89
    Without the term the crops do not improve at all even though the supervised
    views do (19.67 -> 24.58), which is the blur the reweighted L1 permits. At 5
    the crops go further but the held-out gain collapses, so 1 is the operating
    point and the ramp is shared with the other detail gains.
    """
    if edge_gain > 0.0:
        w = _edge_weight(ref_img.detach(), edge_gain)
        l1 = ((pred_img - ref_img).abs() * w).sum() / (w.sum() * pred_img.shape[0]).clamp(min=1e-9)
    else:
        l1 = (pred_img - ref_img).abs().mean()
    if w_sobel > 0.0:
        l1 = l1 + float(w_sobel) * (_sobel(pred_img) - _sobel(ref_img.detach())).abs().mean()
    if lam_dssim <= 0.0:
        return l1, l1, l1.new_zeros(())
    d = 1.0 - ssim_value(pred_img, ref_img)
    return (1.0 - lam_dssim) * l1 + lam_dssim * d, l1, d

# ---------------------------------------------------------------------------
_VGG_FEAT = None


def vgg_perceptual(pred: torch.Tensor, target: torch.Tensor,
                   layers=(3, 8, 15)) -> torch.Tensor:
    """Feature-space L1 between two (3, H, W) images, VGG16 relu1_2/2_2/3_3.

    Why this term exists. The render objective was 0.8 * L1 + 0.2 * (1 - SSIM),
    and both are minimised by the conditional MEAN of the plausible appearances,
    which is exactly a blurred, low-contrast image. Measured on the trained model:
    predicted colour std is 0.237 against 0.300 for the per-scene optimum, and the
    share of colour variation *inside* a 64-point group is 14.9% against 74.7% for
    the ground truth. That is the regression-to-the-mean signature, not a bug.
    Raising the SSIM weight from 0.2 to 0.45 was tried and did not move either
    number (15.31 dB vs 16.13 dB baseline), which is what the restoration
    literature predicts: SSIM is still a distortion metric.

    A deep feature loss is the standard fix -- it rewards matching texture
    statistics rather than per-pixel means, so the model is no longer paid to
    hedge. Weights come from the local torch hub cache; nothing is downloaded.
    """
    global _VGG_FEAT
    if _VGG_FEAT is None:
        from torchvision.models import vgg16
        net = vgg16(weights="IMAGENET1K_V1").features[: max(layers) + 1]
        for q in net.parameters():
            q.requires_grad_(False)
        _VGG_FEAT = net.to(pred.device).eval()
    mean = pred.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = pred.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = torch.stack([(pred.clamp(0, 1) - mean) / std, (target.clamp(0, 1) - mean) / std])
    loss = x.new_zeros(())
    h = x
    for i, layer in enumerate(_VGG_FEAT):
        h = layer(h)
        if i in layers:
            loss = loss + (h[0] - h[1]).abs().mean()
    return loss / len(layers)

