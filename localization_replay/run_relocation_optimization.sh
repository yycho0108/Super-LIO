#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/superlio_ros2_env.sh"

if [[ ! -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  "${SCRIPT_DIR}/setup_env.sh"
fi

exec "${SCRIPT_DIR}/.venv/bin/python" \
  "${SCRIPT_DIR}/optimize_relocation.py" "$@"
