"""Group-relative scale band and the R8H bias remapping."""
from argparse import Namespace
from types import SimpleNamespace

import torch
import torch.nn as nn

from can3tok.attr_decoder import AttributeDecoder
from can3tok.train import recenter_scale_band


def _decoder(down=2.5, up=1.5, bias=-2.559):
    head = nn.Linear(4, 3)
    nn.init.constant_(head.bias, bias)
    return SimpleNamespace(
        cfg=SimpleNamespace(
            attr_scale_cap_down=down,
            attr_scale_cap_up=up,
            attr_scale_group_base=True,
            attr_scale_log_cap=3.0,
        ),
        head_scale=head,
        scale_base_a=nn.Parameter(torch.tensor(0.7872)),
    )


def test_asymmetric_band_blocks_giant_log_scale():
    dec = _decoder()
    raw = torch.full((2, 8, 3), 50.0)
    extent = torch.tensor([0.01, 0.04])
    out = AttributeDecoder._bounded_log_scale(dec, raw, extent)
    base = 0.7872 * extent.clamp(min=1e-6).log().view(2, 1, 1) - 2.559
    assert torch.allclose(out, base + 1.5, atol=1e-4)


def test_asymmetric_band_allows_smaller_than_upper():
    dec = _decoder()
    raw = torch.full((2, 8, 3), -50.0)
    extent = torch.tensor([0.01, 0.04])
    out = AttributeDecoder._bounded_log_scale(dec, raw, extent)
    base = 0.7872 * extent.clamp(min=1e-6).log().view(2, 1, 1) - 2.559
    assert torch.allclose(out, base - 2.5, atol=1e-4)


def test_recenter_resets_absolute_log_bias():
    core = SimpleNamespace(attr_decoder=_decoder(bias=-7.58))
    args = Namespace(attr_scale_cap_down=2.5, attr_scale_cap_up=1.5)
    assert recenter_scale_band(core, None, args)
    assert abs(float(core.attr_decoder.head_scale.bias.mean()) + 2.559) < 1e-5


def test_recenter_leaves_group_relative_bias_alone():
    core = SimpleNamespace(attr_decoder=_decoder(bias=-2.559))
    args = Namespace(attr_scale_cap_down=2.5, attr_scale_cap_up=1.5)
    assert not recenter_scale_band(core, None, args)
    assert abs(float(core.attr_decoder.head_scale.bias.mean()) + 2.559) < 1e-5


def test_zero_weight_head_emits_group_median_not_upper_cap():
    """W~0 must start at the group-relative median, not saturate up=1.5."""
    dec = _decoder()
    nn.init.zeros_(dec.head_scale.weight)
    nn.init.constant_(dec.head_scale.bias, -2.559)
    h = torch.randn(2, 8, 4)
    raw = dec.head_scale(h)
    extent = torch.tensor([0.01, 0.04])
    out = AttributeDecoder._bounded_log_scale(dec, raw, extent)
    base = 0.7872 * extent.clamp(min=1e-6).log().view(2, 1, 1) - 2.559
    assert torch.allclose(out, base.expand_as(out), atol=1e-5)
