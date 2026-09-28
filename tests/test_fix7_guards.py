"""Regression tests for the fix_7 patches.

Each test pins one behaviour that fix_6 got wrong, and each one fails against
fix_6. See FIX7_NOTES.md for the measurements that motivated them.
"""
from __future__ import annotations

import torch

from can3tok.config import Can3TokConfig, channel_budget, patch_layout, validate_layout
from can3tok.losses import intra_group_chamfer, shape_channel_mask


# --------------------------------------------------------------------------
# 1. The latent regularisers must see the shape block only.
# --------------------------------------------------------------------------
def test_shape_mask_excludes_appearance_channels():
    # L16k layout: centroid 4 | occupancy 1 | shape 3 | appearance 8, per_group 16.
    m = shape_channel_mask(16, per_group=16, c_centroid=4, c_occupancy=1, c_shape=3)
    assert m.sum().item() == 3.0, "only the 3 shape channels are learned geometry"
    assert m[:5].sum().item() == 0.0, "anchors are excluded"
    assert m[5:8].sum().item() == 3.0, "the shape block is channels 5..7"
    assert m[8:].sum().item() == 0.0, "appearance must not be regularised as shape"

    # Without the bound this is the fix_6 behaviour: 11 channels, not 3.
    old = shape_channel_mask(16, per_group=16, c_centroid=4, c_occupancy=1)
    assert old.sum().item() == 11.0


def test_shape_mask_repeats_per_group_across_channels():
    m = shape_channel_mask(32, per_group=16, c_centroid=4, c_occupancy=1, c_shape=3)
    assert m.sum().item() == 6.0, "two groups of 3 shape channels"
    assert torch.equal(m[:16], m[16:])


# --------------------------------------------------------------------------
# 2. A degenerate GT group must not be able to dominate the radius term.
# --------------------------------------------------------------------------
def _radius(pred, target, mask, gsz):
    _, parts = intra_group_chamfer(
        pred, target, mask, gsz, scale=None, center=True, return_parts=True
    )
    return float(parts["radius"])


def test_radius_term_ignores_a_collapsed_ground_truth_group():
    torch.manual_seed(0)
    b, g, gsz = 1, 4, 8
    n = g * gsz
    target = torch.randn(b, n, 3)
    pred = target + 0.01 * torch.randn(b, n, 3)
    mask = torch.ones(b, n)

    healthy = _radius(pred, target, mask, gsz)

    # Collapse group 0's target onto a single point: its radius is exactly 0.
    degenerate = target.clone()
    degenerate[:, :gsz] = degenerate[:, 0:1]
    got = _radius(pred, degenerate, mask, gsz)

    # fix_6 divided by clamp(0, min=1e-6) here and returned ~1e5.
    assert got < 10.0 * max(healthy, 1e-3), (
        f"collapsed group inflated the radius term to {got} against a healthy "
        f"{healthy}; it should be dropped from the reduction, not clamped"
    )
    assert torch.isfinite(torch.tensor(got))


def test_radius_term_still_reports_a_shrunken_group():
    torch.manual_seed(0)
    b, g, gsz = 1, 4, 8
    n = g * gsz
    target = torch.randn(b, n, 3)
    mask = torch.ones(b, n)

    # Every predicted group is half the extent of its target. The guard above must
    # not have made the term blind to the defect it exists to catch.
    pred = target * 0.5
    shrunk = _radius(pred, target, mask, gsz)
    exact = _radius(target.clone(), target, mask, gsz)
    assert exact < 1e-4, "an exact match has no radius error"
    assert 0.4 < shrunk < 0.6, f"half-size groups should score ~0.5, got {shrunk}"


# --------------------------------------------------------------------------
# 3. The shape budget floor must reject the configuration that actually shipped.
# --------------------------------------------------------------------------
_L16K = dict(
    max_points=262144, group_size=256, local_tokens_per_group=32,
    latent_channels=32, latent_hw=(256, 128),
    compact_latent_channels=16, compact_latent_hw=(32, 32),
    budget_centroid=4, budget_occupancy=1, attr_pack_dim=11,
)


def test_validate_layout_rejects_the_l16k_shape_budget():
    cfg = Can3TokConfig(**_L16K, budget_shape=3, budget_appearance=8)
    # Sanity: this is the layout the run used, and it is arithmetically valid --
    # which is exactly why nothing caught it.
    bud = channel_budget(cfg)
    assert bud["per_group"] == 16
    try:
        validate_layout(cfg)
    except ValueError as e:
        assert "shape channels per point" in str(e)
    else:
        raise AssertionError("0.0117 shape channels per point must be rejected")


def test_validate_layout_accepts_the_rebalanced_budget():
    cfg = Can3TokConfig(**_L16K, budget_shape=7, budget_appearance=4)
    info = validate_layout(cfg)
    assert info["budget"]["per_group"] == 16, "same total latent, re-split"
    assert info["budget"]["shape"] == 7


def test_shape_budget_floor_can_be_waived_explicitly():
    cfg = Can3TokConfig(**_L16K, budget_shape=3, budget_appearance=8,
                        allow_low_shape_budget=True)
    validate_layout(cfg)


def test_working_reference_layout_is_not_flagged():
    # R7/R8: shape 19 over 64 points. The floor must not fire on a layout this
    # codebase has already reached its targets on.
    cfg = Can3TokConfig(
        max_points=262144, group_size=64, local_tokens_per_group=8,
        latent_channels=32, latent_hw=(256, 128),
        compact_latent_channels=32, compact_latent_hw=(64, 64),
        budget_centroid=4, budget_occupancy=1, budget_shape=19,
        budget_appearance=8, attr_pack_dim=11,
    )
    lay = patch_layout(cfg)
    ppc = lay["points_per_cell"] * max(lay["merge"], 1)
    assert 19 / ppc > 0.05, "reference layout should be clear of the warning band"
    validate_layout(cfg)


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    failed = 0
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        try:
            getattr(mod, name)()
            print(f"  PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print("all fix_7 guard tests passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
