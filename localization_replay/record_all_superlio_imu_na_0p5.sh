#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for run_id in 062309 063004 064152 064922; do
  "${SCRIPT_DIR}/record_superlio_imu_na_0p5.sh" "${run_id}" "$@"
done
