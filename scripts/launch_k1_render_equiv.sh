#!/usr/bin/env bash
# K1 = J1 with the two changes the objective implies, and nothing else.
#
# The stated goal is: encoder -> latent -> world model IMPROVES the latent ->
# decoder -> better Gaussians. Two consequences, both measured:
#
#  (1) attr_param_floor 0.5 -> 0.0
#      The parameter anchor pulls the decoder toward THIS snapshot's own
#      imperfect Gaussians. Measured: swapping in GT attributes costs -0.50 dB,
#      and using the GT group-median scale as the base costs -1.6 dB against a
#      fitted one. The snapshots' own photographic ceiling is 16.34 dB early and
#      20.13 dB late, so anchoring to them caps the decoder below the target.
#      Reconstruction ceiling at this budget is 11.03 dB; the render-equivalence
#      ceiling at the SAME budget is 21.43 dB. The anchor optimises for the wrong
#      one of those two.
#
#  (2) render_presence 1
#      Until now render_loss picked the prediction's Gaussians with the GT mask,
#      so `presence` never entered the render objective and putting a Gaussian in
#      an empty region was free. Measured on R8: GT mask 14.89 dB, predicted
#      presence 14.76, all-slots-present 11.85 -- the head works, it was just
#      never asked. This is also the precondition for a variable-count decoder,
#      which is the real fix for "45% of slots empty while 15.6% of points are
#      dropped" but is a refactor, not a flag.
#
# Everything else is byte-identical to J1, so K1 - J1 isolates these two.
set -euo pipefail
cd "$(dirname "$0")/.."
export CAN3TOK_OUT="${CAN3TOK_OUT:-runs/K1_render_equiv_$(date +%Y%m%d_%H%M%S)}"
export EXTRA_ARGS="--attr_param_floor 0.0 --render_presence 1 ${EXTRA_ARGS:-}"
export RUN_LABEL="K1 render-equivalent (no parameter anchor, presence-gated render)"
exec bash scripts/launch_j1_shared_joint.sh "${1:---foreground}"
