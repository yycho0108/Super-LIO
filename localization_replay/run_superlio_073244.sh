#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PROJECT_DIR}/run_superlio_replay.sh" 073244 8091 \
  --run-date 2026_08_06 "$@"
