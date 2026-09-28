"""Contracts from the 2026-09-15 audit (mdmd/12): eval, decode, packing, flags.

    python -m tests.test_audit_contracts
"""
from __future__ import annotations

import os
import sys
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from can3tok.config import Can3TokConfig
from can3tok.data import ReplayGaussianDataset, images_match
from can3tok.eval_utils import (
    EVAL_CONTRACT_VERSION,
    EvalView,
    eval_fingerprint,
    filter_eval_views,
    save_comparison_figure,
    scene_ids_match,
)
from can3tok.losses import _even_subset, _sample_spatial_indices, projected_histogram_loss
from can3tok.model import Can3TokAE
from can3tok.template import fibonacci_ball, fibonacci_prefix
from can3tok.train import build_config, build_parser


def test_scene_filter_drops_cross_scene_cameras():
    a = EvalView("train", "a.png", np.zeros(16, np.float32), None, 0)
    b = EvalView("truck", "b.png", np.ones(16, np.float32), None, 1)
    hit = filter_eval_views([a, b], "train")
    assert [v.scene_id for v in hit] == ["train"]
    assert filter_eval_views([a, b], "missing") == []
    assert scene_ids_match("foo/train", "train")


def test_eval_fingerprint_changes_with_metric_and_split():
    a = Namespace(score_metric="psnr", eval_val_indices="0,1", eval_pred_mask=0,
                  eval_view_force="", eval_view_count=8, eval_view_seed=1234)
    b = Namespace(score_metric="psnr_gap_worst", eval_val_indices="0,1", eval_pred_mask=0,
                  eval_view_force="", eval_view_count=8, eval_view_seed=1234)
    fa, fb = eval_fingerprint(a, [3, 26]), eval_fingerprint(b, [3, 26])
    assert fa["contract"] == EVAL_CONTRACT_VERSION
    assert fa != fb
    assert fa == eval_fingerprint(a, [26, 3])


def test_spatial_sampler_covers_the_sorted_tail():
    xyz = torch.zeros(100, 3)
    xyz[:, 0] = torch.linspace(-1, 1, 100)
    mask = torch.ones(100)
    idx = _sample_spatial_indices(xyz, mask, 60)
    assert int(idx.min()) <= 5
    assert int(idx.max()) >= 94
    sub = _even_subset(torch.arange(100), 10)
    assert int(sub[0]) == 0 and int(sub[-1]) == 99


def test_projected_histogram_respects_pred_mask():
    pred = torch.zeros(1, 16, 3)
    tgt = torch.zeros(1, 16, 3)
    pred[0, :8, 0] = torch.linspace(-0.5, 0.5, 8)
    tgt[0, :8, 0] = pred[0, :8, 0]
    mask = torch.zeros(1, 16)
    mask[0, :8] = 1
    pred_pad = pred.clone()
    pred_pad[0, 8:, 0] = 4.0
    live = projected_histogram_loss(pred, tgt, mask, samples=16, bins=16, pred_mask=mask)
    padded = projected_histogram_loss(pred_pad, tgt, mask, samples=16, bins=16, pred_mask=mask)
    assert torch.allclose(live, padded, atol=1e-5)


def test_prefix_template_is_zero_mean():
    full = fibonacci_ball(256)
    assert float(np.linalg.norm(full.mean(0))) < 1e-6
    for k in (32, 64, 80, 128, 192):
        pref = fibonacci_prefix(256, k)
        assert pref.shape == (k, 3)
        assert float(np.linalg.norm(pref.mean(0))) < 1e-6
        raw = full[:k]
        assert float(np.linalg.norm(raw.mean(0))) > 0.2


def test_shared_pack_keeps_one_owner_per_source():
    ds = ReplayGaussianDataset.__new__(ReplayGaussianDataset)
    ds.group_in = 8
    ds.num_groups = 4
    ds.group_size = 2
    ds.max_input_points = 32
    ds.max_points = 8
    ds.slot_sort = "morton"
    ds.count_aware_template = False
    ds.scene_anchors = None
    ds.anchor_spill = 4
    ds.anchor_assignment = "spill"
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(40, 3)).astype(np.float32) * 0.15
    anchors = rng.normal(size=(4, 3)).astype(np.float32) * 0.15
    selected = np.arange(40, dtype=np.int64)
    slot_src, src_i = ReplayGaussianDataset._pack_shared(ds, selected, xyz, anchors)
    tgt_ids = set(int(i) for i in slot_src if i >= 0)
    enc_ids = set(int(i) for i in src_i if i >= 0)
    assert tgt_ids <= enc_ids
    enc_owner = {int(src_i[s]): s // ds.group_in for s in range(src_i.size) if src_i[s] >= 0}
    for s, src in enumerate(slot_src):
        if src < 0:
            continue
        assert enc_owner[int(src)] == s // ds.group_size


def test_holdout_does_not_restore_the_full_pool():
    ds = ReplayGaussianDataset.__new__(ReplayGaussianDataset)
    ds.view_exclude = {0, 1, 2}
    pool = [{"image": f"/tmp/im{i}.png", "cam": np.zeros(16, np.float32)} for i in range(3)]
    ds.view_pool = {"train": pool}
    assert ReplayGaussianDataset._allowed_pool_indices(ds, pool).size == 0
    assert ReplayGaussianDataset._image_is_held(ds, "/tmp/im1.png", "train")
    assert not ReplayGaussianDataset._image_is_held(ds, "/tmp/other.png", "train")
    assert images_match("/tmp/im1.png", "/tmp/im1.png")


def test_old_args_json_keeps_structural_flags_off():
    parser = build_parser()
    # Parser default for a NEW run is on.
    fresh = parser.parse_args(["--root", "/tmp", "--out_dir", "/tmp/out"])
    assert int(fresh.normalize_pooler_xyz) == 1
    assert int(fresh.shared_cell_owner) == 1
    # A checkpoint Namespace from S/T/E1 does not have the keys.
    old = Namespace(max_points=128, group_size=8, local_tokens_per_group=4,
                    latent_channels=8, latent_hw=[8, 8], compact_latent_channels=16,
                    compact_latent_hw=[4, 4], budget_centroid=4, budget_occupancy=1,
                    budget_shape=2, budget_appearance=1, attr_pack_dim=0,
                    stage="xyz", uniform_budget=False, model_dim=32, heads=4,
                    dropout=0.0, encoder_residual=False, map_blocks=0,
                    compress_intra_layers=1, compress_window_layers=0,
                    compress_window=8, compress_mid_channels=8,
                    compress_wide_channels=0, compress_wide_layers=0,
                    decompress_intra_layers=1, decompress_window_layers=0,
                    decoder_layers=1, no_decoder_neighbor_context=False,
                    residual_scale=0.6, patch_chunk=32, no_gen_branch=True,
                    gen_window_layers=1, gen_group_layers=1, gen_cross_layers=1,
                    gen_region_chunk=8, gen_offset_scale=1.0,
                    no_gen_neighbor_context=False, latent_mode="ae",
                    no_latent_zorder=False, denoise_std=0.0,
                    no_checkpoint_decode=True, checkpoint_gen=False,
                    no_residual_pack=False, shortcut_alpha_init=0.0,
                    shortcut_alpha_eval=0.0, allow_padded_tokens=True)
    cfg = build_config(old, target_dim=3, sh_dim=0)
    assert cfg.normalize_pooler_xyz is False
    assert cfg.count_aware_template is False
    assert cfg.attr_slot_mask is False
    assert cfg.use_fixed_anchor_center is False


def test_decode_compact_matches_forward_including_attributes():
    cfg = Can3TokConfig(
        max_points=64, group_size=8, local_tokens_per_group=4,
        latent_channels=8, latent_hw=(8, 8), compact_latent_channels=8,
        compact_latent_hw=(4, 2), budget_centroid=4, budget_occupancy=1,
        budget_shape=2, budget_appearance=1, model_dim=32, heads=4,
        compress_mid_channels=8, compress_intra_layers=1,
        compress_window_layers=0, decompress_intra_layers=1,
        decompress_window_layers=0, decoder_layers=1, use_gen_branch=False,
        target_dim=14, sh_dim=0, checkpoint_decode=False,
        shortcut_alpha=0.0, shortcut_alpha_eval=0.0,
        allow_low_shape_budget=True, attr_decoder_layers=1, attr_decoder_dim=32,
        attr_pack_dim=4, folding_decode=False,
    )
    model = Can3TokAE(cfg).eval()
    with torch.no_grad():
        model.latent_scale.fill_(0.73)
    x = torch.randn(1, 64, 18)
    mask = torch.ones(1, 64)
    with torch.no_grad():
        out = model(x, mask, run_gen=False)
        rec = model.decode_compact(out["z_compact"])
        z_n = model.encode_compact(x, mask, normalized=True)
        rec_n = model.decode_compact(z_n, normalized=True)
    assert "attr_pred" in rec
    assert torch.allclose(out["pred"], rec["pred"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out["attr_pred"], rec["attr_pred"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out["pred"], rec_n["pred"], atol=1e-4, rtol=1e-4)
    raw_wrong = model.decode_compact(z_n, normalized=False)
    assert not torch.allclose(out["pred"], raw_wrong["pred"], atol=1e-3, rtol=1e-3)


def test_comparison_figure_allows_shorter_pred_cloud(tmp_path=None):
    import tempfile
    gt = np.zeros((70000, 3), np.float32)
    gt[:, 0] = np.linspace(0, 1, 70000)
    pred = np.zeros((50000, 3), np.float32)
    pred[:, 0] = np.linspace(0, 1, 50000)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        save_comparison_figure(path, gt, pred, max_points=60000, title="mismatch")
        assert os.path.isfile(path) and os.path.getsize(path) > 0
    finally:
        os.remove(path)


if __name__ == "__main__":
    mod = sys.modules[__name__]
    failed = 0
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        try:
            getattr(mod, name)()
            print(f"  PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print("all audit contract tests passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
