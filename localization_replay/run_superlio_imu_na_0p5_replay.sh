#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 RUN_ID DEFAULT_PORT [app options...]" >&2
  exit 2
fi
RUN_ID="$1"
DEFAULT_PORT="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROSBAG_ROOT="${LOCALIZATION_REPLAY_ROSBAG_ROOT:-/workspace/rosbag}"
RUN_TIME="${RUN_ID:0:2}_${RUN_ID:2:2}_${RUN_ID:4:2}"

exec "${SCRIPT_DIR}/run_superlio_replay.sh" "${RUN_ID}" "${DEFAULT_PORT}" \
  --bag "${ROSBAG_ROOT}/superlio_imu_na_0p5_${RUN_ID}" \
  --source-bag "${ROSBAG_ROOT}/rosbag2_2026_08_04-${RUN_TIME}" \
  --glim-trajectory \
    "${ROSBAG_ROOT}/glim_rosbag2_2026_08_04-${RUN_TIME}/traj_imu.txt" \
  "$@"
