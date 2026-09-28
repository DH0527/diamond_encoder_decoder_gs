"""Small reusable building blocks: MLP, Fourier features, attention, windows."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """Fixed log-spaced sin/cos features; optionally concatenates the raw input."""

    def __init__(self, in_dim: int, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_freqs = int(num_freqs)
        self.include_input = bool(include_input)
        freqs = 2.0 ** torch.arange(self.num_freqs, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = self.in_dim * (1 if include_input else 0) + self.in_dim * 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.unsqueeze(-1) * self.freqs.to(x.dtype)
        feats = [torch.sin(proj), torch.cos(proj)]
        out = torch.cat([f.flatten(-2) for f in feats], dim=-1)
        if self.include_input:
            out = torch.cat([x, out], dim=-1)
        return out


class MLP(nn.Module):
    """Plain MLP, with optional guards for heads that must not be able to die.

    ``in_norm`` normalises the input, so a head cannot be knocked out by a change
    of scale in whatever produces its input (e.g. the compact latent growing by
    two orders of magnitude while the std hinge pulls it off the floor).
    ``hidden_norm`` normalises every pre-activation, which keeps roughly half of
    the units on the positive side of the nonlinearity. ``leaky`` swaps GELU for
    a leaky ReLU, whose derivative is never exactly zero, so a saturated unit
    always retains a path back.

    Without these, an additive residual head sitting behind a good
    non-parametric shortcut can be driven into permanent saturation: the
    gradient of its output weight becomes exactly zero and it never recovers.
    """

    def __init__(
        self,
        dims: List[int],
        last_act: bool = False,
        dropout: float = 0.0,
        in_norm: bool = False,
        hidden_norm: bool = False,
        leaky: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        if in_norm:
            layers.append(nn.LayerNorm(dims[0]))
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or last_act:
                if hidden_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.LeakyReLU(0.1) if leaky else nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def residual_head(dims: List[int], dropout: float = 0.0) -> MLP:
    """An MLP for a zero-initialised additive residual branch."""
    return MLP(dims, dropout=dropout, in_norm=True, hidden_norm=True, leaky=True)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self attention + MLP with an optional key padding mask."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP([dim, dim * mlp_ratio, dim], dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        if key_padding_mask is not None:
            # A row whose keys are all masked would produce NaNs; keep such rows
            # attending to themselves and zero the update afterwards.
            all_masked = key_padding_mask.all(dim=-1, keepdim=True)
            kpm = key_padding_mask & ~all_masked
            a, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
            a = a.masked_fill(all_masked.unsqueeze(-1), 0.0)
        else:
            a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0, mlp_ratio: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP([dim, dim * mlp_ratio, dim], dropout=dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        h = self.norm_q(q)
        m = self.norm_kv(kv)
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=-1, keepdim=True)
            kpm = key_padding_mask & ~all_masked
            a, _ = self.attn(h, m, m, key_padding_mask=kpm, need_weights=False)
            a = a.masked_fill(all_masked.unsqueeze(-1), 0.0)
        else:
            a, _ = self.attn(h, m, m, need_weights=False)
        q = q + a
        return q + self.mlp(self.norm2(q))


class ResBlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        h = F.gelu(self.conv1(self.norm(x)))
        return x + self.conv2(h)


# ---------------------------------------------------------------------------
# windowed attention over a 2D cell grid
# ---------------------------------------------------------------------------


def window_partition(x: torch.Tensor, h: int, w: int, ws: int, shift: int = 0):
    """(B, h*w, d) -> (B * nwin, ws*ws, d). Pads the grid when it is not divisible."""
    b, n, d = x.shape
    x = x.view(b, h, w, d)
    if shift:
        x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
    pad_h = (ws - h % ws) % ws
    pad_w = (ws - w % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // ws, ws, wp // ws, ws, d)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, d)
    return x, (hp, wp, pad_h, pad_w)


def window_reverse(x: torch.Tensor, b: int, h: int, w: int, ws: int, meta, shift: int = 0):
    hp, wp, pad_h, pad_w = meta
    d = x.shape[-1]
    x = x.view(b, hp // ws, wp // ws, ws, ws, d).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, d)
    if pad_h or pad_w:
        x = x[:, :h, :w]
    if shift:
        x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
    return x.reshape(b, h * w, d)


class WindowSelfAttention(nn.Module):
    """Swin-style local self attention over the latent cell grid.

    Alternating layers use a half-window shift so information crosses window
    borders, as in the SLat encoder of TRELLIS.
    """

    def __init__(self, dim: int, heads: int, grid_hw, window: int = 8, dropout: float = 0.0, shift: bool = False):
        super().__init__()
        self.h, self.w = int(grid_hw[0]), int(grid_hw[1])
        self.ws = int(window)
        self.shift = self.ws // 2 if shift else 0
        self.block = SelfAttentionBlock(dim, heads=heads, dropout=dropout)
        self.pos = nn.Parameter(torch.zeros(1, self.ws * self.ws, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = x.shape[0]
        xw, meta = window_partition(x, self.h, self.w, self.ws, self.shift)
        kpm = None
        if valid is not None:
            vw, _ = window_partition(valid.unsqueeze(-1).to(x.dtype), self.h, self.w, self.ws, self.shift)
            kpm = vw.squeeze(-1) <= 0.5
        xw = self.block(xw + self.pos.to(xw.dtype), key_padding_mask=kpm)
        return window_reverse(xw, b, self.h, self.w, self.ws, meta, self.shift)


def neighbor_gather(x: torch.Tensor, h: int, w: int, radius: int = 1) -> torch.Tensor:
    """(B, h*w, d) -> (B, h*w, (2r+1)^2, d) with edge clamping."""
    b, n, d = x.shape
    grid = x.view(b, h, w, d).permute(0, 3, 1, 2)
    k = 2 * radius + 1
    patches = F.unfold(F.pad(grid, (radius,) * 4, mode="replicate"), kernel_size=k)
    patches = patches.view(b, d, k * k, h * w).permute(0, 3, 2, 1)
    return patches.contiguous()
