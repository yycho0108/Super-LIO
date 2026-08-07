#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_superlio_imu_na_0p5_replay.sh" 064922 8095 \
  --post-recovery-align-steps 10 \
  "$@"
