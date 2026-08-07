#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_localization_replay.sh ROSBAG_PATH [GLIM_PATH] [app options...]

Examples:
  ./run_localization_replay.sh \
    /workspace/rosbag/rosbag2_2026_08_06-07_23_29

  ./run_localization_replay.sh /path/to/rosbag /path/to/traj_imu.txt --port 8090

ROSBAG_PATH may be either rosbag2_YYYY_MM_DD-HH_MM_SS or a versioned
superlio_YYYY_MM_DD-HH_MM_SS_SUFFIX bag.
If GLIM_PATH is omitted, the matching glim_rosbag2_* directory beside the bag
is selected automatically. Only traj_imu.txt is needed. If it is absent, the
same command falls back to Super-LIO-only replay.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_PATH="$(realpath -e "$1")"
shift

if [[ ! -d "${BAG_PATH}" ]]; then
  echo "ROSBAG_PATH is not a directory: ${BAG_PATH}" >&2
  exit 2
fi

BAG_NAME="$(basename "${BAG_PATH}")"
if [[ "${BAG_NAME}" =~ ^(rosbag2|superlio)_([0-9]{4})_([0-9]{2})_([0-9]{2})-([0-9]{2})_([0-9]{2})_([0-9]{2})(_.+)?$ ]]; then
  RUN_DATE="${BASH_REMATCH[2]}_${BASH_REMATCH[3]}_${BASH_REMATCH[4]}"
  RUN_ID="${BASH_REMATCH[5]}${BASH_REMATCH[6]}${BASH_REMATCH[7]}"
  RUN_TIME="${BASH_REMATCH[5]}_${BASH_REMATCH[6]}_${BASH_REMATCH[7]}"
else
  echo "Cannot infer date/time from bag name: ${BAG_NAME}" >&2
  echo "Expected: rosbag2_YYYY_MM_DD-HH_MM_SS or superlio_YYYY_MM_DD-HH_MM_SS_SUFFIX" >&2
  exit 2
fi

DATA_ROOT="$(dirname "${BAG_PATH}")"
ORIGINAL_BAG="${DATA_ROOT}/rosbag2_${RUN_DATE}-${RUN_TIME}"
if [[ "${BAG_NAME}" == "$(basename "${ORIGINAL_BAG}")" ]]; then
  REPLAY_REFERENCE_BAG="${BAG_PATH}"
elif [[ -d "${ORIGINAL_BAG}" ]]; then
  REPLAY_REFERENCE_BAG="${ORIGINAL_BAG}"
else
  echo "Original replay-reference bag not found: ${ORIGINAL_BAG}" >&2
  echo "It is needed to remove replayed original /lio/odom poses from a versioned bag." >&2
  exit 2
fi
GLIM_NAME="glim_rosbag2_${RUN_DATE}-${RUN_TIME}"
GLIM_DIRECTORY="${DATA_ROOT}/${GLIM_NAME}"
GLIM_TRAJECTORY=""

if [[ $# -gt 0 && "$1" != --* ]]; then
  GLIM_INPUT="$(realpath -e "$1")"
  shift
  if [[ -f "${GLIM_INPUT}" ]]; then
    GLIM_TRAJECTORY="${GLIM_INPUT}"
  elif [[ -d "${GLIM_INPUT}" ]]; then
    GLIM_INPUT_NAME="$(basename "${GLIM_INPUT}")"
    for candidate in \
      "${GLIM_INPUT}/traj_imu.txt" \
      "${GLIM_INPUT}/tmp/${GLIM_INPUT_NAME}/traj_imu.txt"; do
      if [[ -f "${candidate}" ]]; then
        GLIM_TRAJECTORY="${candidate}"
        break
      fi
    done
  fi
else
  for candidate in \
    "${BAG_PATH}/traj_imu.txt" \
    "${GLIM_DIRECTORY}/traj_imu.txt" \
    "${GLIM_DIRECTORY}/tmp/${GLIM_NAME}/traj_imu.txt"; do
    if [[ -f "${candidate}" ]]; then
      GLIM_TRAJECTORY="${candidate}"
      break
    fi
  done
fi

APP_MODE=()
if [[ -n "${GLIM_TRAJECTORY}" && -f "${GLIM_TRAJECTORY}" ]]; then
  APP_MODE=(--glim-trajectory "${GLIM_TRAJECTORY}")
  echo "Mode: GLIM vs Super-LIO"
  echo "GLIM: ${GLIM_TRAJECTORY}"
else
  APP_MODE=(--superlio-only)
  echo "Mode: Super-LIO only (matching GLIM trajectory pair not found)"
fi

# Complete manual recordings carry their own robot/map context. Compact
# optimization bags intentionally contain only /lio/odom and must reuse the
# matching original bag for LowState, TF, and point clouds.
APP_CONTEXT=()
if /usr/bin/python3 - "${BAG_PATH}/metadata.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

metadata = yaml.safe_load(Path(sys.argv[1]).read_text())[
    "rosbag2_bagfile_information"
]
topics = {
    row["topic_metadata"]["name"]
    for row in metadata["topics_with_message_count"]
    if int(row["message_count"]) > 0
}
raise SystemExit(0 if {"/lowstate", "/tf", "/tf_static"} <= topics else 1)
PY
then
  APP_CONTEXT=(--use-bag-context)
  echo "Replay context: selected full bag"
else
  echo "Replay context: matching original bag (compact evaluation bag detected)"
fi

DEFAULT_PORT="${VISER_PORT:-8081}"
echo "Evaluated/replayed bag: ${BAG_PATH}"
echo "Spatial anchor: original source bag /lio/odom(t0) <-> GLIM traj_imu.txt(t0)"
echo "Replay-filter reference: ${REPLAY_REFERENCE_BAG}"
exec "${PROJECT_DIR}/run_superlio_replay.sh" "${RUN_ID}" "${DEFAULT_PORT}" \
  --run-date "${RUN_DATE}" \
  --data-root "${DATA_ROOT}" \
  --bag "${BAG_PATH}" \
  --source-bag "${REPLAY_REFERENCE_BAG}" \
  --spatial-anchor-bag "${REPLAY_REFERENCE_BAG}" \
  "${APP_CONTEXT[@]}" \
  "${APP_MODE[@]}" \
  "$@"
