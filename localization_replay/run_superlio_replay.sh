#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 RUN_ID DEFAULT_PORT [app options...]" >&2
  exit 2
fi

RUN_ID="$1"
DEFAULT_PORT="$2"
shift 2

if [[ ! "${RUN_ID}" =~ ^[0-9]{6}$ ]]; then
  echo "RUN_ID must be HHMMSS, got: ${RUN_ID}" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
ROSBAG_ROOT="${LOCALIZATION_REPLAY_ROSBAG_ROOT:-/workspace/rosbag}"

ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO_NAME}/setup.bash}"
if [[ -z "${AMENT_PREFIX_PATH:-}" && -f "${ROS_SETUP}" ]]; then
  set +u
  source "${ROS_SETUP}"
  set -u
fi

UNITREE_SETUP="${UNITREE_ROS_SETUP:-${HOME}/unitree_ros2/cyclonedds_ws/install/setup.bash}"
if [[ -f "${UNITREE_SETUP}" ]]; then
  set +u
  source "${UNITREE_SETUP}"
  set -u
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PROJECT_DIR}/setup_env.sh"
fi

if ! "${VENV_DIR}/bin/python" -c "import plotly, scipy, viser, yourdfpy" >/dev/null 2>&1; then
  "${PROJECT_DIR}/setup_env.sh"
fi

exec "${VENV_DIR}/bin/python" "${PROJECT_DIR}/superlio_063004_app.py" \
  --run-id "${RUN_ID}" \
  --data-root "${ROSBAG_ROOT}" \
  --port "${DEFAULT_PORT}" \
  "$@"
