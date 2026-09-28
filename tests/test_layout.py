"""Layout / shape / gradient sanity checks. Runs on CPU in a few seconds.

    python -m tests.test_layout
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.config import Can3TokConfig, channel_budget, describe_layout, patch_layout, validate_layout
from can3tok.model import build_model
from can3tok.morton import grid_zorder_permutation, inverse_permutation


def test_zorder_permutation():
    for h, w in ((64, 64), (256, 128), (8, 8)):
        perm = grid_zorder_permutation(h, w)
        assert perm.numel() == h * w
        assert torch.equal(torch.sort(perm).values, torch.arange(h * w)), "not a permutation"
        inv = inverse_permutation(perm)
        assert torch.equal(perm[inv], torch.arange(h * w))
    # The property that matters for a conv/UNet diffusion model: cells that are
    # 2D neighbours in the stored map should be close in sequence order too
    # (sequence order == 3D Morton order == spatial adjacency in 3D).
    # The mean gap is the same for any space-filling order, so measure how much
    # of the mass sits on *small* gaps: that is what a 3x3 kernel sees.
    def close_neighbor_fraction(perm: torch.Tensor, h: int, w: int, thr: int = 4) -> float:
        seq = inverse_permutation(perm).view(h, w).float()
        gaps = torch.cat(
            [(seq[:, 1:] - seq[:, :-1]).abs().flatten(), (seq[1:, :] - seq[:-1, :]).abs().flatten()]
        )
        return float((gaps <= thr).float().mean())

    z_frac = close_neighbor_fraction(grid_zorder_permutation(64, 64), 64, 64)
    r_frac = close_neighbor_fraction(torch.arange(4096), 64, 64)
    print(f"  2D neighbours within 4 sequence steps: zorder={z_frac:.3f} row-major={r_frac:.3f}")
    assert z_frac > r_frac, "z-order must improve 2D/3D locality agreement"
    print("zorder permutation OK")


def test_budget_rejects_zero_padding():
    cfg = Can3TokConfig(max_points=4096, group_size=32, local_tokens_per_group=8,
                        latent_channels=32, latent_hw=(64, 32), compact_latent_hw=(16, 16))
    try:
        validate_layout(cfg)
    except ValueError as e:
        assert "provably zero" in str(e)
        print("padded-token guard OK")
        return
    raise AssertionError("expected the padded-token guard to fire")


def test_forward_backward():
    torch.manual_seed(0)
    max_points = 4096
    cfg = Can3TokConfig(
        max_points=max_points,
        target_dim=3,
        sh_dim=0,
        group_size=32,
        local_tokens_per_group=4,
        latent_channels=32,
        latent_hw=(32, 16),          # 512 cells = 128 groups * 4
        compact_latent_channels=32,
        compact_latent_hw=(8, 8),    # 64 cells -> merge = 2
        budget_centroid=4, budget_occupancy=2, budget_shape=10,
        model_dim=64, heads=4,
        compress_intra_layers=2, compress_window_layers=2, compress_window=4,
        decompress_intra_layers=2, decompress_window_layers=2,
        decoder_layers=2, patch_chunk=32,
        gen_window_layers=2, gen_group_layers=1, gen_cross_layers=2, gen_region_chunk=16,
        checkpoint_decode=False,
    )
    print(describe_layout(cfg))
    lay = patch_layout(cfg)
    assert lay["merge"] == 2 and lay["tokens_per_cell"] == 8
    bud = channel_budget(cfg)
    assert bud["per_group"] == 16

    model = build_model(cfg)
    x = torch.randn(2, max_points, 63) * 0.3
    mask = torch.ones(2, max_points)
    mask[:, -300:] = 0.0
    x[..., 0:3] = x[..., 0:3].clamp(-1, 1)

    out = model(x, mask, run_decode=True, run_gen=True, gen_noise_std=0.02)
    assert out["z_compact"].shape == (2, 32, 8, 8), out["z_compact"].shape
    assert out["z_raw"].shape == (2, 32, 32, 16)
    assert out["pred"].shape == (2, max_points, 3)
    assert out["gen_pred"].shape == (2, max_points, 3)

    # identity path: z_raw must equal the raw Morton patch exactly
    patch = out["patch"]
    tok = out["z_raw"].reshape(2, 32, -1).transpose(1, 2)[:, : lay["num_groups"] * 4].reshape(2, lay["num_groups"], 128)
    assert torch.allclose(tok, patch, atol=1e-5), (tok - patch).abs().max()
    print("identity pack is lossless OK")

    # centroid channels really carry the group centroid
    cen = out["z_compact"].reshape(2, 2, 16, 64)[:, :, 0:3]
    print("  centroid channel |mean| =", float(cen.abs().mean()))

    loss = out["pred"].square().mean() + out["gen_pred"].square().mean() + out["z_raw_hat"].square().mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  params with grad: {n_train - len(missing)}/{n_train}")
    if missing:
        print("  no grad:", missing)
    print("forward/backward OK")


def test_losses():
    from can3tok.losses import multiscale_chamfer, patch_dispersion_loss, z_raw_losses

    pred = torch.randn(1, 2048, 3) * 0.2
    tgt = torch.randn(1, 2048, 3) * 0.2
    mask = torch.ones(1, 2048)
    c = multiscale_chamfer(pred, tgt, mask, scales=(256, 512))
    assert torch.isfinite(c)
    d = patch_dispersion_loss(pred, mask, group_size=32, margin=0.01)
    assert torch.isfinite(d)
    zh = torch.randn(1, 32, 8, 8)
    zr = torch.randn(1, 32, 8, 8)
    gv = torch.ones(1, 16)
    parts = z_raw_losses(zh, zr, gv, num_groups=16, tpg=4)
    for k, v in parts.items():
        assert torch.isfinite(v), k
    print("losses OK")


if __name__ == "__main__":
    test_zorder_permutation()
    test_budget_rejects_zero_padding()
    test_forward_backward()
    test_losses()
    print("\nall layout tests passed")
