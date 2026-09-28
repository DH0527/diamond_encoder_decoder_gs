#!/usr/bin/env bash
# L1 = J1 with the two defects the folding audit found, and nothing else.
#
# Rendered images at J1 step 1000 show identical round blobs tiling the frame and
# no structure at all, against GT renders where the lettering and panels are
# legible. That is template_erank 5.5 against GT 101.9, made visible. The audit
# located it:
#
#   shape code (19ch)   |.| 0.921   erank  6.58 / 19
#   unit ball           |.| 0.655   identical for all 4096 groups
#   residual (192 out)  |.| 0.445   erank  5.33 / 192   <- capped by the code
#   ball + residual     |.| 0.620   erank  5.80
#   x aniso             |.| 7.976   erank  4.24         <- LOSES rank here
#   aniso spread        0.027 .. 1411.5
#
#  (1) --folding_aniso_log_cap 1.5
#      The geometric-mean normalisation fixes the product of the three axis
#      scales but not their ratio, so one axis ran to 1411x. That stage inflates
#      the offsets 13x and destroys 1.6 of the 5.8 effective rank: a few needle
#      groups become the huge smears in the render. 1.5 still allows a 4.5x
#      stretch per axis, which covers the data (GT lam2 0.757).
#
#  (2) --w_latent_decorr 0.5  (from 0.05)
#      The residual's rank (5.33 of 192) is capped by the CODE's rank (6.58 of
#      19), so the bottleneck is upstream of the decoder: the compressor is
#      emitting shape codes that span a third of their own width. The
#      off-diagonal correlation penalty attacks exactly that.
#
# Also keeps K1's two objective fixes, since they are cheap and measured:
# no parameter anchor, and a presence-gated render.
set -euo pipefail
cd "$(dirname "$0")/.."
export CAN3TOK_OUT="${CAN3TOK_OUT:-runs/L1_rank_$(date +%Y%m%d_%H%M%S)}"
export EXTRA_ARGS="--folding_aniso_log_cap 1.5 --w_latent_decorr 0.5 \
 --attr_param_floor 0.0 --render_presence 1 ${EXTRA_ARGS:-}"
export RUN_LABEL="L1 bounded anisotropy + shape-code decorrelation"
exec bash scripts/launch_j1_shared_joint.sh "${1:---foreground}"
