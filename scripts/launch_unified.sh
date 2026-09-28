#!/usr/bin/env bash
# One continuous from-scratch run carrying every change this project measured.
#
# WHY 40000 STEPS, NOT 60000. The five-run warm-start chain
# (R8H->R8I->M4->N1->P1) spent 60000 steps to reach 19.20 dB. Reading its eval
# curve, three parts of that were not productive:
#   0-3000      PSNR 8.47 -> 8.62 with the render off. It jumped to 13.29 the
#               moment the render turned on at 3500. Reproduced independently in
#               N4, where codec_rmse moved -0.4% over 6000 render-free steps and
#               then -40% in the 5000 steps after the render started.
#   R8I         5000 steps for a NET LOSS: 16.48 -> 16.35.
#   4 restarts  each --init_from cost 1.2-5.1 dB and 1000-5000 steps of recovery,
#               ~10000 steps in total. A single continuous run pays none of it.
# 60000 - 3000 - 5000 - 10000 = 42000 productive steps, so 40000 continuous is
# the honest equivalent, with one cosine instead of five restarted ones.
#
# WHY THE LOSS BALANCE IS NOT THE CHAIN'S. The goal is render quality and
# geometric structure inside a fixed 262144-point budget, NOT reproducing the
# sampled subset point for point. Measured shares in the old objective:
# intra_chamfer 24% + group_centroid 23% + intra_sinkhorn 16% = 63% went to
# "match the per-anchor statistics of whichever 262144 points the sampler drew",
# against coverage at 5% (0.02% before its units were corrected). P2 cut those
# terms 5x and the centroid error IMPROVED. So they are cut here from the start.
#
# The structure terms that replace them are the ones that were verified to stop
# a specific measured collapse:
#   w_radius  the predicted group radius was 0.670x the ground truth's -- the
#             group shrinks toward its own centroid. M7 took it to 0.911x.
#   w_p2g     the median predicted point sat 0.315 group radii OFF the GT
#             surface. Needs ~3.0; at 0.5 the value did not move in 500 steps.
#   w_coverage rendering a random half of the GT Gaussians with GT positions and
#             GT attributes scores 19.74 dB against 48.39 for the full set, and
#             this model reaches only ~48% distinct points.
#   w_attr_set within a group the model emitted 64 near-identical Gaussians
#             (orientation diversity 2.3% of the GT's) because every attribute
#             target is position-keyed and blends co-located Gaussians. Adding a
#             permutation-invariant set distance took that to 15.8% and PSNR
#             17.80 -> 18.67.
# Deliberately OFF: w_intra_spacing pushed apart the near-coincident pairs the GT
# actually uses (its median within-group NN distance is 0.007 group radii), and
# w_splat_area penalised the top 20% of splat area while the whole distribution
# was 0.40x too small.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TAG:?TAG required}"
: "${DATA_ROOT:?DATA_ROOT required}"
ASSET="${ASSET:-}"                  # single-scene shorthand; or set SPLIT/STATS/ANCH/PHOTO/VIEWP
: "${GPUS:?GPUS required}"
NPROC="${NPROC:-1}"
MAX_STEPS="${MAX_STEPS:-40000}"
CELL_CAP="${CELL_CAP:-512}"         # per-anchor input cap; 4096 * CELL_CAP
MIN_SNAP="${MIN_SNAP:-12000}"
OUT="runs/${TAG}_$(date +%Y%m%d_%H%M%S)"
MIP=$(( 4096 * CELL_CAP ))

mkdir -p runs
# Multi-scene: SPLIT/PHOTO/VIEW are single merged files while STATS/ANCHORS are
# comma-separated, one per scene in --root order. Per-scene anchors are what makes
# two scenes share a 4096-cell latent without waste: measured on these two speedy
# scenes, one shared anchor set leaves 43.7% of train's cells with no points in
# them (truck 16.9%), and per-scene sets leave 0.0% for both.
SPLIT="${SPLIT:-assets/split_${ASSET}.json}"
STATS="${STATS:-assets/stats_${ASSET}.json}"
ANCH="${ANCH:-assets/anchors_${ASSET}.npy}"
PHOTO="${PHOTO:-assets/npz_to_image_${ASSET}.json}"
VIEWP="${VIEWP:-assets/view_pool_${ASSET}.json}"
ARGS="--root ${DATA_ROOT}
 --out_dir ${OUT}
 --split_path ${SPLIT}
 --stats_path ${STATS}
 --scene_anchors ${ANCH}
 --photo_map ${PHOTO}
 --view_pool ${VIEWP}
 --num_val_files 300 --min_snapshot_step ${MIN_SNAP}
 --max_points 262144 --max_input_points ${MIP} --pool_chunk -1
 --drop_outside --sample_mode stratified --crop_prob 0.0
 --density_aware_sample --no_slot_redistribute
 --partition_mode morton --partition_block 32 --slot_sort template --cache_slots
 --anchor_assignment spill --anchor_spill 8 --use_fixed_anchor_center 0
 --augment --aug_rot_deg 8.0 --aug_scale_jitter 0.03 --aug_shift 0.01
 --workers 6 --batch_size 1
 --chunk_size 64 --group_size 64 --local_tokens_per_group 8
 --latent_channels 32 --latent_hw 256 128
 --compact_latent_channels 32 --compact_latent_hw 64 64
 --budget_centroid 4 --budget_occupancy 1 --budget_shape 19 --budget_appearance 8
 --attr_pack_dim 11 --attr_pack_hidden 64 --pool_dim 128
 --attr_decoder_layers 4 --attr_decoder_dim 256 --attr_cond xattn --attr_local_pe 1
 --attr_scale_log_cap 3.0 --attr_scale_cap_down 3.0 --attr_scale_cap_up 3.0
 --attr_nbr_window 1 --attr_read_shape 1 --attr_nudge_cap 0.15
 --model_dim 448 --heads 8
 --compress_intra_layers 3 --compress_window_layers 2 --compress_window 8
 --compress_mid_channels 256 --decompress_intra_layers 3 --decompress_window_layers 2
 --decoder_layers 10 --residual_scale 0.6 --patch_chunk 512
 --folding_decode --folding_res_cap 1.0 --folding_frame_channels 6
 --folding_aniso_log_cap 1.5 --folding_res_start 0 --folding_res_ramp_steps 400
 --decoder_refine_start 400 --decoder_refine_ramp_steps 800
 --encoder_residual --encoder_residual_start 0 --pack_trainable
 --no_gen_branch --stage geometry
 --shared_free_head 0 --structured_local_code 0
 --joint_shared_decoder 0 --joint_direct_decoder 0
 --shortcut_alpha_start 999999999
 --gen_start 999999999 --gen_end 999999999 --late_codec_start 999999999
 --w_xyz 0 --w_xyz_mse 0 --w_xyz_hard 0 --w_xyz_residual 0 --w_xyz_residual_hard 0
 --w_chamfer 5.0 --chamfer_scales 4096,16384,65536 --chamfer_scale_weights 0.2,0.3,0.5
 --balanced_chamfer --intra_chamfer_chunk 1024 --detail_ramp_steps 100
 --w_group_centroid 0.3 --w_intra_chamfer 0.5 --w_intra_sinkhorn 0.8 --w_shape_sinkhorn 1.5
 --sinkhorn_epsilon 0.08 --sinkhorn_iterations 6 --sinkhorn_chunk 256
 --w_radius 4.0 --w_p2g 3.0 --w_intra_spacing 0.0
 --w_coverage 1200.0 --coverage_samples 49152 --w_presence 0.5
 --w_attr_set 3.0 --sinkhorn_attr_weight 2.0 --attr_match_mode sinkhorn
 --w_scale 1.0 --w_rot 0.0 --w_opacity 1.0 --w_color 1.0 --w_sh 0.0 --w_cov3d 2.0
 --attr_responsibility 1.0 --attr_anchor_covered 0.25
 --w_render_attr 60.0 --w_render 0 --w_splat_area 0.0 --render_presence 1
 --render_views 5 --extra_real_views 4 --view_downscale 2
 --render_lam_dssim 0.2 --render_min_coverage 0.25
 --w_latent_std 1.0 --latent_std_floor 0.35 --latent_std_ceil 3.0
 --w_latent_decorr 5.0
 --w_equiv 0.5 --equiv_ramp_steps 2000 --equiv_every 4 --equiv_rot_deg 10.0 --equiv_shape_weight 0.0
 --w_z_raw 0 --w_z_raw_mse 0 --w_z_hard 0 --w_z_token_var 0 --w_z_std_ratio 0
 --w_z_residual 0 --w_z_residual_hard 0 --w_z_intra_chamfer 0
 --w_learned_residual 0 --w_shape_direct 0
 --w_plane_chamfer 0 --w_proj_hist 0 --w_dispersion 0 --w_voxel_occ 0
 --w_teacher_cycle 0 --w_res_ratio 0 --w_kl 0
 --latent_end 0 --late_decode_steps 0 --geo_end 800 --geo_weak_scale 0.35
 --attr_start 1200 --attr_force_steps 1500 --attr_anneal_steps 4000
 --attr_param_decay_steps 5000 --attr_param_floor 0.5
 --attr_detach_geometry 1 --attr_detach_release 12000
 --render_start 1200 --render_ramp_steps 1500
 --render_downscale_start 8 --render_downscale 2 --render_downscale_steps 8000
 --equiv_start 4000 --polish_start 0
 --lr 2e-4 --lr_min 1e-5 --lr_warmup_steps 500
 --weight_decay 1e-4 --grad_clip 1.0 --amp bf16 --seed 42 --encoder_warmup_steps 0
 --score_metric psnr --eval_view_count 8 --eval_view_seed 1234
 --eval_val_indices 0,4,7,31,63,95,127,159
 --max_steps ${MAX_STEPS} --log_every 50 --eval_every 1000
 --eval_milestones 500,1000,2000,3000,4000,6000,8000,10000,13000,16000,20000,24000,28000,32000,36000,40000
 --save_every 2000 ${EXTRA_ARGS:-}"

echo "TAG=${TAG}  GPUS=${GPUS}  NPROC=${NPROC}  cell_cap=${CELL_CAP} (max_input_points=${MIP})"
echo "OUT=${OUT}"
LOG="${OUT}.log"
mkdir -p "${OUT}"

if [ "${NPROC}" -gt 1 ]; then
  LAUNCH="torchrun --standalone --nproc_per_node=${NPROC} --master_port=$((29500 + RANDOM % 400))"
else
  LAUNCH="python -u"
fi

cat > "${OUT}/launch_cmd.txt" <<EOF
CUDA_VISIBLE_DEVICES=${GPUS} ${LAUNCH} train.py ${ARGS}
EOF

nohup env CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 \
  /home/super/anaconda3/envs/can3tok/bin/${LAUNCH} train.py ${ARGS} > "${LOG}" 2>&1 &
echo $! > "${OUT}.pid"
echo "PID=$(cat ${OUT}.pid)"
echo "Log: ${LOG}"
