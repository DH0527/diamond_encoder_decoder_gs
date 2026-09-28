#!/usr/bin/env bash
# J0 control for J1.  Every schedule/loss/data option is identical to J1; only
# the shared joint refiner is disabled.  J1-J0 is therefore the architecture
# contribution, whereas J1-R8 would mix architecture, matching and polish.
set -euo pipefail
cd "$(dirname "$0")/.."

export CAN3TOK_OUT="${CAN3TOK_OUT:-runs/J0_ot_polish_control_$(date +%Y%m%d_%H%M%S)}"
export EXTRA_ARGS="--joint_shared_decoder 0 ${EXTRA_ARGS:-}"
export RUN_LABEL="J0 balanced-OT polish control (no joint decoder)"
exec bash scripts/launch_j1_shared_joint.sh "${1:---foreground}"
