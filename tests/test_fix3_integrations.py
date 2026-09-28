import numpy as np
import torch

from can3tok.config import Can3TokConfig
from can3tok.data import ReplayGaussianDataset
from can3tok.model import Can3TokAE
from can3tok.losses import (group_centroid_loss, intra_group_chamfer,
                            intra_group_sinkhorn_loss, target_group_extent)
from can3tok.schedule import effective_weights
from can3tok.train import build_parser


def test_pruning_scores_affect_sampling_not_codec_features():
    ds = ReplayGaussianDataset.__new__(ReplayGaussianDataset)
    ds.layout = {"opacity": 0}
    ds.density_aware_sample = False
    ds.density_importance_weight = 0.0
    target = np.zeros((8, 1), dtype=np.float32)
    a = ds._importance({"pruning_scores": np.arange(8, dtype=np.float32)}, target)
    b = ds._importance({"pruning_scores": np.arange(8, dtype=np.float32)[::-1]}, target)
    assert not np.allclose(a, b)


def test_anchor_full_groups_follow_template_slot_sort():
    ds = ReplayGaussianDataset.__new__(ReplayGaussianDataset)
    ds.scene_anchors = np.array([[0, 0, 0], [4, 0, 0]], np.float32)
    ds.slot_redistribute = False
    ds.anchor_spill = 8
    ds.anchor_assignment = "spill"
    ds.group_size = 8
    ds.num_groups = 2
    ds.max_points = 16
    g0 = np.stack([np.linspace(-0.4, 0.4, 8), np.zeros(8), np.zeros(8)], axis=1).astype(np.float32)
    g1 = g0.copy()
    g1[:, 0] += 4.0
    xyz = np.concatenate([g0, g1], axis=0)
    selected = np.arange(16, dtype=np.int64)
    ds.slot_sort = "morton"
    dist_order = ds._pack_slots(selected, xyz, cap=16, gsz=8, ng=2, anchors=ds.scene_anchors)
    ds.slot_sort = "template"
    tmpl_order = ds._pack_slots(selected, xyz, cap=16, gsz=8, ng=2, anchors=ds.scene_anchors)
    assert int((dist_order >= 0).sum()) == 16
    assert int((tmpl_order >= 0).sum()) == 16
    assert not np.array_equal(dist_order, tmpl_order)


def test_capacitated_anchor_assignment_uses_global_free_slots():
    ds = ReplayGaussianDataset.__new__(ReplayGaussianDataset)
    ds.anchor_spill = 1
    ds.anchor_assignment = "capacitated"
    anchors = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]], np.float32)
    points = np.array([[i * 0.01, 0, 0] for i in range(8)], np.float32)
    slots = ds._anchor_assign(points, gsz=2, ng=4, anchors=anchors)
    assert int((slots >= 0).sum()) == 8
    assert [(slots[2 * i:2 * i + 2] >= 0).sum() for i in range(4)] == [2, 2, 2, 2]


def test_direct_joint_geometry_and_attributes_train_same_shared_head():
    cfg = Can3TokConfig(
        max_points=128, group_size=8, local_tokens_per_group=4,
        latent_channels=8, latent_hw=(8, 8), compact_latent_channels=32,
        compact_latent_hw=(2, 2), budget_centroid=4, budget_occupancy=1,
        budget_shape=2, budget_appearance=1, max_input_points=256,
        pool_dim=32, pool_queries=4, model_dim=32, heads=4,
        compress_mid_channels=16, compress_intra_layers=1,
        compress_window_layers=0, decompress_intra_layers=1,
        decompress_window_layers=0, decoder_layers=1, use_gen_branch=False,
        shared_free_head=True, joint_direct_decoder=True,
        joint_local_memory=True, joint_memory_layers=1,
        joint_decoder_dim=32, joint_decoder_layers=1, joint_decoder_chunk=4,
        target_dim=14, sh_dim=0, checkpoint_decode=False,
        # Plumbing test: asserts geometry and attributes reach the same shared
        # head. Capacity is irrelevant, so the shape-budget floor is waived.
        allow_low_shape_budget=True,
    )
    model = Can3TokAE(cfg)
    x, enc = torch.randn(1, 128, 7), torch.randn(1, 256, 7)
    x[..., 3:], enc[..., 3:] = 0, 0
    out = model(x, torch.ones(1, 128), enc_x=enc, enc_mask=torch.ones(1, 256), run_gen=False)
    assert out["pred"].data_ptr() == out["attr_pred"].data_ptr()
    geom = out["pred"][..., :3].square().mean()
    attr = out["attr_pred"][..., 3:].square().mean()
    (geom + attr).backward()
    assert model.compressor.head_shared.weight.grad.abs().sum() > 0
    assert model.direct_joint_decoder.code_proj.net[0].weight.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in model.direct_joint_decoder.local_proj.parameters()
               if p.grad is not None) > 0
    # F3B froze this module and ignored its T local tokens.  In the hierarchical
    # path the geometry objective must train the compact->local expansion.
    assert sum(p.grad.abs().sum() for p in model.decompressor.parameters()
               if p.grad is not None) > 0


def test_geometry_first_attribute_weight_ramp():
    args = build_parser().parse_args([
        "--root", "/tmp/data", "--out_dir", "/tmp/out",
        "--stage", "geometry", "--latent_end", "0", "--geo_end", "1000",
        "--attr_start", "1500", "--attr_loss_ramp_steps", "1000",
        "--render_start", "2500", "--attr_param_decay_steps", "1500",
        "--attr_param_floor", "0", "--w_scale", "1",
    ])
    assert effective_weights(1000, args)["w_scale"] == 0.0
    assert effective_weights(1500, args)["w_scale"] == 0.0
    assert 0.49 < effective_weights(2000, args)["w_scale"] < 0.51
    assert effective_weights(2500, args)["w_scale"] == 1.0


def test_fixed_anchor_is_shared_origin_and_translation_is_trainable():
    cfg = Can3TokConfig(
        max_points=16, group_size=4, local_tokens_per_group=2,
        latent_channels=8, latent_hw=(2, 4), compact_latent_channels=32,
        compact_latent_hw=(1, 1), budget_centroid=4, budget_occupancy=1,
        budget_shape=2, budget_appearance=1, model_dim=16, heads=4,
        compress_mid_channels=8, compress_intra_layers=0,
        compress_window_layers=0, decompress_intra_layers=0,
        decompress_window_layers=0, decoder_layers=0, use_gen_branch=False,
        shared_free_head=True, joint_direct_decoder=True,
        joint_local_memory=True, joint_memory_layers=1,
        joint_decoder_dim=16, joint_decoder_layers=0, joint_decoder_chunk=4,
        target_dim=14, sh_dim=0, checkpoint_decode=False,
        use_fixed_anchor_center=True,
    )
    model = Can3TokAE(cfg)
    x = torch.zeros(1, 16, 7)
    x[..., :3] = torch.randn(1, 16, 3) * 0.01 + 0.2
    anchor = torch.zeros(1, 4, 3)
    out = model(x, torch.ones(1, 16), group_anchor=anchor, run_gen=False)
    assert torch.allclose(out["decoded_centroid"], anchor, atol=1e-5)
    assert torch.allclose(out["group_translation_abs"], torch.zeros(()), atol=1e-7)
    loss = out["pred"][..., :3].mean()
    loss.backward()
    assert model.direct_joint_decoder.head_translation.weight.grad.abs().sum() > 0


def test_local_memory_changes_centered_group_shape_and_receives_gradient():
    cfg = Can3TokConfig(
        max_points=8, group_size=4, local_tokens_per_group=2,
        latent_channels=8, latent_hw=(2, 2), compact_latent_channels=16,
        compact_latent_hw=(1, 1), budget_centroid=4, budget_occupancy=1,
        budget_shape=2, budget_appearance=1, model_dim=16, heads=4,
        joint_direct_decoder=True, joint_local_memory=True,
        joint_decoder_dim=16, joint_decoder_layers=1, joint_memory_layers=1,
        joint_decoder_chunk=2, target_dim=14, sh_dim=0,
    )
    from can3tok.joint_decoder import DirectJointGaussianDecoder
    dec = DirectJointGaussianDecoder(cfg)
    centroid = torch.zeros(1, 2, 3)
    count = torch.full((1, 2), 4.0)
    scale = torch.ones(1, 2, 1)
    code = torch.zeros(1, 2, 3)
    local = torch.zeros(1, 2, 2, 8)
    local[:, 1] = torch.randn(1, 2, 8)
    local.requires_grad_(True)
    out = dec(centroid, count, scale, code, local_tokens=local)
    xyz = out["pred"][..., :3].reshape(1, 2, 4, 3)
    shape = xyz - xyz.mean(dim=2, keepdim=True)
    assert not torch.allclose(shape[:, 0], shape[:, 1], atol=1e-6)
    shape[:, 1].square().mean().backward()
    assert local.grad[:, 1].abs().sum() > 0


def test_structured_compact_queries_survive_to_point_decoder():
    cfg = Can3TokConfig(
        max_points=128, group_size=8, local_tokens_per_group=4,
        latent_channels=8, latent_hw=(8, 8), compact_latent_channels=40,
        compact_latent_hw=(2, 2), budget_centroid=4, budget_occupancy=1,
        budget_shape=4, budget_appearance=1, max_input_points=256,
        pool_dim=32, pool_queries=4, model_dim=32, heads=4,
        compress_mid_channels=16, compress_intra_layers=1,
        compress_window_layers=0, decompress_intra_layers=0,
        decompress_window_layers=0, decoder_layers=0, use_gen_branch=False,
        shared_free_head=True, structured_local_code=True,
        compact_global_dim=1, compact_local_tokens=2, compact_local_dim=2,
        joint_direct_decoder=True, joint_local_memory=True,
        joint_decoder_dim=32, joint_decoder_layers=1, joint_memory_layers=1,
        joint_decoder_chunk=4, target_dim=14, sh_dim=0,
        checkpoint_decode=False,
    )
    model = Can3TokAE(cfg)
    x, enc = torch.randn(1, 128, 7), torch.randn(1, 256, 7)
    x[..., 3:], enc[..., 3:] = 0, 0
    out = model(x, torch.ones(1, 128), enc_x=enc,
                enc_mask=torch.ones(1, 256), run_gen=False)
    xyz = out["pred"][..., :3].reshape(1, 16, 8, 3)
    centred = xyz - xyz.mean(dim=2, keepdim=True)
    centred.square().mean().backward()
    assert model.compressor.structured_queries.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in model.compressor.head_struct_local.parameters()
               if p.grad is not None) > 0
    assert sum(p.grad.abs().sum() for p in model.direct_joint_decoder.local_proj.parameters()
               if p.grad is not None) > 0
    assert not any(p.requires_grad for p in model.decompressor.parameters())


def test_centered_local_shape_and_absolute_centroid_are_separated():
    target = torch.randn(1, 8, 3) * 0.02
    pred = target + torch.tensor([0.1, -0.05, 0.03])
    mask = torch.ones(1, 8)
    scale = target_group_extent(target, mask, 4)
    local_ch = intra_group_chamfer(
        pred, target, mask, 4, scale=scale, center=True, chunk_groups=4)
    local_ot = intra_group_sinkhorn_loss(
        pred, target, mask, 4, scale=scale, center=True,
        epsilon=0.08, iterations=6, chunk_groups=4)
    centroid = group_centroid_loss(pred, target, mask, 4, scale=None)
    assert float(local_ch) < 1e-4
    assert float(local_ot) < 0.1
    assert float(centroid) > 0.1


def test_decode_compact_matches_training_codec_path():
    cfg = Can3TokConfig(
        max_points=128, group_size=8, local_tokens_per_group=4,
        latent_channels=8, latent_hw=(8, 8), compact_latent_channels=32,
        compact_latent_hw=(2, 2), budget_centroid=4, budget_occupancy=1,
        budget_shape=2, budget_appearance=1, model_dim=32, heads=4,
        compress_mid_channels=16, compress_intra_layers=1,
        compress_window_layers=0, decompress_intra_layers=1,
        decompress_window_layers=0, decoder_layers=1, use_gen_branch=False,
        joint_direct_decoder=False, target_dim=14, sh_dim=0,
        checkpoint_decode=False, shortcut_alpha=0.0, shortcut_alpha_eval=0.0,
        # Plumbing test: asserts that decode_compact and the training codec path
        # produce identical tensors. Capacity is irrelevant to that, and the
        # config is sized for speed, so the shape-budget floor is waived.
        allow_low_shape_budget=True,
    )
    model = Can3TokAE(cfg).eval()
    x = torch.randn(1, 128, 7)
    x[..., 3:] = 0
    mask = torch.ones(1, 128)
    out = model(x, mask, run_gen=False)
    rec = model.decode_compact(out["z_compact"].detach())
    assert rec["pred"].shape == out["pred"].shape
    assert torch.allclose(out["pred"], rec["pred"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out["z_raw_hat"], rec["z_raw_hat"], atol=1e-5, rtol=1e-5)


def test_structured_local_head_does_not_read_collapsed_mid():
    cfg = Can3TokConfig(
        max_points=128, group_size=8, local_tokens_per_group=4,
        latent_channels=8, latent_hw=(8, 8), compact_latent_channels=40,
        compact_latent_hw=(2, 2), budget_centroid=4, budget_occupancy=1,
        budget_shape=4, budget_appearance=1, model_dim=32, heads=4,
        compress_mid_channels=16, compress_intra_layers=1,
        compress_window_layers=0, decompress_intra_layers=0,
        decompress_window_layers=0, decoder_layers=0, use_gen_branch=False,
        shared_free_head=True, structured_local_code=True,
        compact_global_dim=1, compact_local_tokens=2, compact_local_dim=2,
        joint_direct_decoder=True, joint_local_memory=True,
        joint_decoder_dim=32, joint_decoder_layers=1, joint_memory_layers=1,
        joint_decoder_chunk=4, target_dim=14, sh_dim=0,
        checkpoint_decode=False,
    )
    model = Can3TokAE(cfg)
    first = next(p for p in model.compressor.head_struct_local.net.parameters()
                 if p.ndim == 2)
    assert first.shape[1] == cfg.model_dim


def test_invert_path_does_not_generate_xyz_from_a_shared_head():
    cfg = Can3TokConfig(
        max_points=128, group_size=8, local_tokens_per_group=4,
        latent_channels=8, latent_hw=(8, 8), compact_latent_channels=32,
        compact_latent_hw=(2, 2), budget_centroid=4, budget_occupancy=1,
        budget_shape=2, budget_appearance=1, model_dim=32, heads=4,
        compress_mid_channels=16, compress_intra_layers=1,
        compress_window_layers=0, decompress_intra_layers=1,
        decompress_window_layers=0, decoder_layers=1, use_gen_branch=False,
        folding_decode=False, target_dim=14, sh_dim=0,
        checkpoint_decode=False, residual_pack=True,
        shortcut_alpha=0.0, shortcut_alpha_eval=0.0,
        # Plumbing test: asserts the invert path has no shared xyz head. Capacity
        # is irrelevant, so the shape-budget floor is waived.
        allow_low_shape_budget=True,
    )
    model = Can3TokAE(cfg)
    assert model.decompressor.shape_xyz is None
    x = torch.randn(1, 128, 7)
    x[..., 3:] = 0
    mask = torch.ones(1, 128)
    out = model(x, mask, run_gen=False)
    rec = model.decode_compact(out["z_compact"].detach())
    assert torch.allclose(out["pred"], rec["pred"], atol=1e-5, rtol=1e-5)
    assert out["learned_xyz"].abs().mean() > 0
