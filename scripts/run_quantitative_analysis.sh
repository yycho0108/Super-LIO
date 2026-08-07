#!/usr/bin/env bash

set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/dongjae/ws/install/setup.bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MAP_FRAME_BAG=/home/dongjae/Downloads/rosbag2_2026_08_02-12_41_25
REFERENCE_BAG=/home/dongjae/Downloads/rosbag2_2026_08_03-15_13_43
ESTIMATED_BAG=/home/dongjae/rosbag2_2026_08_03-22_09_23
IMU_TRAJECTORY=/home/dongjae/Downloads/glim_dump8/traj_imu.txt
PLOT_PYTHON=/home/dongjae/viz_env/bin/python
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$REPO_ROOT/analysis_results/$RUN_STAMP"

usage() {
  command printf '%s\n' \
    'Usage: run_quantitative_analysis.sh [options]' \
    '  --output-dir PATH       Output directory' \
    '  --map-frame-bag PATH    Bag defining the prebuilt-map frame' \
    '  --reference-bag PATH    Bag containing GLIM ground truth' \
    '  --estimated-bag PATH    Bag containing Super-LIO localization' \
    '  --imu-trajectory PATH   GLIM traj_imu.txt trajectory' \
    '  --help                  Show this help'
}

while (($#)); do
  case "$1" in
    --output-dir) OUTPUT_DIR=${2:?missing value for --output-dir}; shift 2 ;;
    --map-frame-bag) MAP_FRAME_BAG=${2:?missing value for --map-frame-bag}; shift 2 ;;
    --reference-bag) REFERENCE_BAG=${2:?missing value for --reference-bag}; shift 2 ;;
    --estimated-bag) ESTIMATED_BAG=${2:?missing value for --estimated-bag}; shift 2 ;;
    --imu-trajectory) IMU_TRAJECTORY=${2:?missing value for --imu-trajectory}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) command printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

for required in \
    "$MAP_FRAME_BAG/metadata.yaml" \
    "$REFERENCE_BAG/metadata.yaml" \
    "$ESTIMATED_BAG/metadata.yaml" \
    "$IMU_TRAJECTORY" \
    "$PLOT_PYTHON"; do
  if [[ ! -e "$required" ]]; then
    command printf 'Missing required artifact: %s\n' "$required" >&2
    exit 66
  fi
done

mkdir -p "$OUTPUT_DIR"
ros2 run g1_localization_tools quantitative_analysis \
  --map-frame-bag "$MAP_FRAME_BAG" \
  --reference-bag "$REFERENCE_BAG" \
  --estimated-bag "$ESTIMATED_BAG" \
  --imu-trajectory "$IMU_TRAJECTORY" \
  --output-dir "$OUTPUT_DIR"
"$PLOT_PYTHON" "$SCRIPT_DIR/plot_quantitative_analysis.py" \
  --input-dir "$OUTPUT_DIR"

command printf 'Report: %s\n' "$OUTPUT_DIR/summary.md"
