#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

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

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the Python used by your ROS 2 installation." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo "Environment ready: ${VENV_DIR}"
echo "The virtual environment is local-only and is intentionally excluded from Git."
