#!/usr/bin/env bash
# L17k: L16k's latent budget, re-split so the intra-group shape code is large
# enough to carry intra-group geometry.
#
#   bash scripts/launch_l17k_shape_rebalance.sh --detached
#   tail -F runs/L17k_*.log
#
# What this changes against runs/L16k_20260827_063453 (see FIX7_NOTES.md for the
# measurements behind each one):
#
#   budget_shape       3 -> 7   the 256 points in a group had 3 free scalars
#   budget_appearance  8 -> 4   between them; rel_offset_p50 measured 1.414
#   w_latent_decorr  5.0 -> 1.0 decorr now covers 7 shape channels, not 11
#   score_metric  psnr -> psnr_gap_worst   pooled PSNR hid a per-scene regression
#
# The total latent is unchanged at 16 x 32 x 32 = 16384 scalars, and per_group
# still sums to 16, so this is a re-split of the same budget rather than a
# bigger model. Numbers here are the full set: no layering onto another
# launcher, because the R7->R8->R9 chain relied on argparse last-wins and made
# the effective configuration readable only by running it.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
ROOT=${ROOT:-/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg,/data/daeho/aaaa_proj/seondo/speedy-splat/output/truck_colmap_seg_prune_scores/replay_seg}
GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
WORKERS=${WORKERS:-6}
MAX_STEPS=${MAX_STEPS:-64000}
TAG=${TAG:-L17k_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
EXTRA_ARGS=${EXTRA_ARGS:-}
# Deliberately not 29521: the L16k run may still be holding it.
MASTER_PORT=${MASTER_PORT:-29537}

BUDGET_SHAPE=${BUDGET_SHAPE:-7}
BUDGET_APPEARANCE=${BUDGET_APPEARANCE:-4}

export CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 ROOT="$ROOT" GPUS="$GPUS" NPROC="$NPROC" WORKERS="$WORKERS" \
    MAX_STEPS="$MAX_STEPS" TAG="$TAG" OUT="$OUT" LOG="$LOG" EXTRA_ARGS="$EXTRA_ARGS" \
    MASTER_PORT="$MASTER_PORT" TORCHRUN="$TORCHRUN" \
    BUDGET_SHAPE="$BUDGET_SHAPE" BUDGET_APPEARANCE="$BUDGET_APPEARANCE" \
    nohup setsid bash "$0" >>"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  mkdir -p "$OUT"; echo "$!" >"${OUT}/run.pid"
  echo "Detached PID=$!"
  echo "Log: $LOG"
  echo "Out: $OUT"
  exit 0
fi

mkdir -p "$OUT"
echo "launch L17k | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  budget/group: centroid=4 occupancy=1 shape=$BUDGET_SHAPE appearance=$BUDGET_APPEARANCE (sum=16)"
echo "  latent: 16 x 32 x 32 = 16384 scalars, 1024 groups of 256 points"
echo "  guards: loss_spike_mult=25 latent_scale_sync_every=50 score=psnr_gap_worst"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  --root "$ROOT" \
  --out_dir "$OUT" \
  --split_path assets/split_speedy_both.json \
  --stats_path assets/stats_speedy_train.json,assets/stats_speedy_truck.json \
  --scene_anchors assets/anchors_speedy_train_n1024.npy,assets/anchors_speedy_truck_n1024.npy \
  --photo_map assets/npz_to_image_speedy_both.json \
  --view_pool assets/view_pool_speedy_both.json \
  --num_val_files 300 \
  --min_snapshot_step 12000 \
  --stats_stride 50 \
  --stats_quantile 0.02 \
  --max_points 262144 \
  --max_input_points 2097152 \
  --drop_outside \
  --sample_mode stratified \
  --crop_prob 0.0 \
  --density_aware_sample \
  --density_importance_weight 0.35 \
  --no_slot_redistribute \
  --partition_mode morton \
  --partition_block 32 \
  --slot_sort template \
  --anchor_assignment spill \
  --anchor_spill 8 \
  --cache_slots \
  --augment \
  --aug_rot_deg 8.0 \
  --aug_scale_jitter 0.03 \
  --aug_shift 0.01 \
  --workers "$WORKERS" \
  --batch_size 1 \
  --chunk_size 64 \
  \
  --group_size 256 \
  --local_tokens_per_group 32 \
  --latent_channels 32 \
  --latent_hw 256 128 \
  --compact_latent_channels 16 \
  --compact_latent_hw 32 32 \
  --budget_centroid 4 \
  --budget_occupancy 1 \
  --budget_shape "$BUDGET_SHAPE" \
  --budget_appearance "$BUDGET_APPEARANCE" \
  --compact_global_dim 3 \
  --compact_local_tokens 4 \
  --compact_local_dim 6 \
  \
  --model_dim 448 \
  --heads 8 \
  --pool_dim 128 \
  --pool_queries 16 \
  --compress_intra_layers 3 \
  --compress_window_layers 2 \
  --compress_window 8 \
  --compress_mid_channels 256 \
  --decompress_intra_layers 3 \
  --decompress_window_layers 2 \
  --decoder_layers 10 \
  --residual_scale 0.6 \
  --patch_chunk 192 \
  --encoder_residual \
  --encoder_residual_start 0 \
  --pack_trainable \
  --folding_decode \
  --folding_aniso_log_cap 1.5 \
  --folding_res_cap 1.0 \
  --folding_res_start 0 \
  --folding_res_ramp_steps 400 \
  --folding_frame_channels 6 \
  --decoder_refine_start 400 \
  --decoder_refine_ramp_steps 800 \
  --shortcut_alpha_start 999999999 \
  --no_gen_branch \
  --stage geometry \
  \
  --attr_pack_dim 11 \
  --attr_pack_hidden 64 \
  --attr_decoder_layers 4 \
  --attr_decoder_dim 256 \
  --attr_cond xattn \
  --attr_local_pe 1 \
  --attr_read_shape 1 \
  --attr_nbr_window 1 \
  --attr_nudge_cap 0.15 \
  --attr_scale_log_cap 3.0 \
  --attr_scale_cap_down 3.0 \
  --attr_scale_cap_up 3.0 \
  --attr_scale_group_base 1 \
  --attr_start 1200 \
  --attr_loss_ramp_steps 1 \
  --attr_force_steps 1500 \
  --attr_anneal_steps 4000 \
  --attr_param_decay_steps 5000 \
  --attr_param_floor 0.5 \
  --attr_anchor_covered 0.25 \
  --attr_responsibility 1.0 \
  --attr_match_mode sinkhorn \
  --attr_detach_geometry 1 \
  --attr_detach_release 12000 \
  \
  --latent_end 0 \
  --late_decode_steps 0 \
  --geo_end 800 \
  --geo_weak_scale 0.35 \
  --detail_ramp_steps 100 \
  --gen_start 999999999 \
  --gen_end 999999999 \
  --late_codec_start 999999999 \
  --polish_start 0 \
  --latent_decay_steps 6000 \
  --diversity_decay_steps 8000 \
  \
  --w_chamfer 5.0 \
  --w_coverage 1200.0 \
  --chamfer_scales 4096,16384,65536 \
  --chamfer_scale_weights 0.2,0.3,0.5 \
  --balanced_chamfer \
  --coverage_samples 49152 \
  --w_intra_chamfer 3.0 \
  --w_intra_sinkhorn 4.0 \
  --w_group_centroid 2.0 \
  --w_shape_sinkhorn 1.5 \
  --w_p2g 3.0 \
  --w_radius 4.0 \
  --w_attr_set 3.0 \
  --w_presence 0.5 \
  --sinkhorn_epsilon 0.08 \
  --sinkhorn_iterations 6 \
  --sinkhorn_chunk 48 \
  --sinkhorn_attr_weight 2.0 \
  --intra_chamfer_chunk 96 \
  --attr_set_chunk 96 \
  --xyz_beta 0.002 \
  --hard_frac 0.05 \
  --hard_min_points 2048 \
  \
  --w_scale 1.0 \
  --w_rot 0.0 \
  --w_opacity 1.0 \
  --w_color 1.0 \
  --w_sh 0.0 \
  --w_cov3d 2.0 \
  \
  --w_latent_std 1.0 \
  --latent_std_floor 0.35 \
  --latent_std_ceil 3.0 \
  --w_latent_decorr 1.0 \
  \
  --w_render 0.0 \
  --w_render_attr 60.0 \
  --render_start 1200 \
  --render_ramp_steps 1500 \
  --render_presence 1 \
  --render_views 5 \
  --render_view_jitter_deg 8.0 \
  --render_downscale 2 \
  --render_downscale_start 8 \
  --render_downscale_steps 8000 \
  --render_lam_dssim 0.2 \
  --render_min_coverage 0.25 \
  --extra_real_views 4 \
  --view_downscale 2 \
  \
  --w_equiv 0.5 \
  --equiv_start 4000 \
  --equiv_ramp_steps 2000 \
  --equiv_every 4 \
  --equiv_rot_deg 10.0 \
  --w_distill 3.0 \
  \
  --lr 2e-4 \
  --lr_min 1e-5 \
  --lr_warmup_steps 500 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --loss_spike_mult 25.0 \
  --latent_scale_sync_every 50 \
  --amp bf16 \
  --seed 42 \
  --encoder_warmup_steps 0 \
  \
  --max_steps "$MAX_STEPS" \
  --log_every 50 \
  --eval_every 1000 \
  --eval_milestones 500,1000,2000,4000,6000,8000,12000,16000,20000,24000,28000,32000,40000,48000,56000,64000 \
  --eval_val_indices 0,20,40,60,184,204,224,244 \
  --eval_chamfer_samples 32768 \
  --eval_view_count 8 \
  --eval_view_seed 1234 \
  --score_metric psnr_gap_worst \
  --save_every 2000 \
  $EXTRA_ARGS
