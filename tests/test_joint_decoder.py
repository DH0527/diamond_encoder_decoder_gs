import torch

from can3tok.config import Can3TokConfig
from can3tok.joint_decoder import JointGaussianRefiner


def _cfg():
    return Can3TokConfig(
        max_points=8,
        target_dim=14,
        sh_dim=0,
        group_size=4,
        compact_latent_channels=8,
        compact_latent_hw=(2, 1),
        budget_centroid=4,
        budget_occupancy=1,
        budget_shape=2,
        budget_appearance=1,
        joint_shared_decoder=True,
        joint_decoder_dim=16,
        joint_decoder_layers=1,
        joint_decoder_chunk=2,
        joint_nbr_window=1,
        heads=4,
        patch_chunk=2,
    )


def _inputs():
    geometry = torch.randn(1, 8, 14)
    render = torch.randn(1, 8, 14)
    render[..., :3] = geometry[..., :3] + 0.01
    render[..., 6:10] = torch.nn.functional.normalize(render[..., 6:10], dim=-1)
    presence = torch.randn(1, 8)
    code = torch.randn(1, 2, 3)
    scale = torch.rand(1, 2, 1) + 0.1
    return geometry, render, presence, code, scale


def test_joint_refiner_is_function_preserving_at_initialisation():
    module = JointGaussianRefiner(_cfg())
    geometry, render, presence, code, scale = _inputs()
    out = module(geometry, render, presence, code, scale)
    assert torch.equal(out["geometry"][..., :3], geometry[..., :3])
    assert torch.allclose(out["geometry"][..., 3:], render[..., 3:], atol=1e-7)
    assert torch.allclose(out["render"], render, atol=1e-7)
    assert torch.equal(out["presence"], presence)


def test_geometry_and_attribute_losses_reach_same_shared_code():
    module = JointGaussianRefiner(_cfg())
    geometry, render, presence, code, scale = _inputs()
    # Zero output heads intentionally delay the code gradient by one optimiser
    # step while preserving R8 exactly.  Activate representative joint heads as
    # they would be after that first step, then verify both objectives reach code.
    with torch.no_grad():
        module.head_xyz.weight.normal_(std=1e-3)
        module.head_scale.weight.normal_(std=1e-3)
        module.head_opacity.weight.normal_(std=1e-3)
        module.head_color.weight.normal_(std=1e-3)

    def code_grad(which):
        c = code.clone().requires_grad_(True)
        out = module(geometry, render, presence, c, scale)
        loss = (out["geometry"][..., :3].square().mean() if which == "geometry"
                else out["render"][..., 3:].square().mean())
        return torch.autograd.grad(loss, c)[0]

    g_geo = code_grad("geometry")
    g_attr = code_grad("attribute")
    assert g_geo.norm() > 0
    assert g_attr.norm() > 0
    # Both gradients cover the complete shared code, not disjoint shape/app slices.
    assert torch.all(g_geo.abs().sum(dim=(0, 1)) > 0)
    assert torch.all(g_attr.abs().sum(dim=(0, 1)) > 0)
