#!/usr/bin/env bash

set -eo pipefail

export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
ROS_WORKSPACE=$(cd -- "$REPO_ROOT/../.." && pwd)
source "$ROS_WORKSPACE/install/setup.bash"
G1_WORKSPACE=/home/dongjae/ws/src/g1_real
source "$G1_WORKSPACE/install/setup.bash"
export PYTHONPATH="$G1_WORKSPACE/src:${PYTHONPATH:-}"
set -u

BAG=/home/dongjae/rosbag2_2026_08_03-22_09_23
MAP="$REPO_ROOT/src/super_lio/map_real_20260802/dump7_xyzi_clean.pcd"
REFERENCE_BAG=/home/dongjae/Downloads/rosbag2_2026_08_03-15_13_43
MAP_FRAME_BAG=/home/dongjae/Downloads/rosbag2_2026_08_02-12_41_25
RVIZ_CONFIG="$REPO_ROOT/src/super_lio/rviz/glim_super_lio_comparison.rviz"
RATE=1.0
DURATION=0
LOOP=false
WITH_RVIZ=true
START_PAUSED=false

usage() {
  command printf '%s\n' \
    'Usage: show_glim_super_lio_comparison.sh [options]' \
    '  --rate RATE          Playback rate (default: 1.0)' \
    '  --duration SEC       Stop after wall-clock seconds; 0 means full bag' \
    '  --loop               Loop the bag until Ctrl-C' \
    '  --start-paused       Open RViz with bag playback paused' \
    '  --no-rviz            Headless verification mode' \
    '  --bag PATH           Override recorded Super-LIO bag' \
    '  --map PATH           Override fixed prebuilt PCD map' \
    '  --reference-bag PATH Override bag containing GLIM reference' \
    '  --map-frame-bag PATH Override bag defining the prebuilt-map frame' \
    '  --help               Show this text'
}

while (($#)); do
  case "$1" in
    --rate) RATE=${2:?missing value for --rate}; shift 2 ;;
    --duration) DURATION=${2:?missing value for --duration}; shift 2 ;;
    --loop) LOOP=true; shift ;;
    --start-paused) START_PAUSED=true; shift ;;
    --no-rviz) WITH_RVIZ=false; shift ;;
    --bag) BAG=${2:?missing value for --bag}; shift 2 ;;
    --map) MAP=${2:?missing value for --map}; shift 2 ;;
    --reference-bag) REFERENCE_BAG=${2:?missing value for --reference-bag}; shift 2 ;;
    --map-frame-bag) MAP_FRAME_BAG=${2:?missing value for --map-frame-bag}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) command printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

python3 - "$RATE" "$DURATION" <<'PY'
import math
import sys
for name, text, positive in (
        ('rate', sys.argv[1], True), ('duration', sys.argv[2], False)):
    value = float(text)
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        raise SystemExit(f'invalid --{name}: {text}')
PY

MAP_PUBLISHER="$ROS_WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/map_publisher"
OVERLAY="$ROS_WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/trajectory_overlay"
ROBOT_VISUALIZER="$ROS_WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/g1_robot_visualizer"
for required in "$BAG/metadata.yaml" "$MAP" \
                "$REFERENCE_BAG/metadata.yaml" \
                "$MAP_FRAME_BAG/metadata.yaml" "$RVIZ_CONFIG" \
                "$MAP_PUBLISHER" "$OVERLAY" "$ROBOT_VISUALIZER"; do
  if [[ ! -e "$required" ]]; then
    command printf 'Missing required file: %s\n' "$required" >&2
    command printf 'Build with: cd %s && colcon build --symlink-install --packages-select g1_localization_tools\n' \
      "$ROS_WORKSPACE" >&2
    exit 66
  fi
done

RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/g1_trajectory_comparison_$RUN_STAMP"
mkdir -p "$LOG_DIR"
command printf 'Comparison logs: %s\n' "$LOG_DIR"

declare -a CHILD_PIDS=()
CLEANED=false
start_child() {
  local name=$1
  shift
  stdbuf -oL -eL "$@" >"$LOG_DIR/$name.log" 2>&1 &
  CHILD_PIDS+=("$!")
}
cleanup() {
  [[ "$CLEANED" == true ]] && return
  CLEANED=true
  set +e
  local pid
  for pid in "${CHILD_PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -INT "$pid" 2>/dev/null
  done
  for pid in "${CHILD_PIDS[@]}"; do
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

start_child map_publisher "$MAP_PUBLISHER" --ros-args \
  -p "map_path:=$MAP" -p frame_id:=world
# The overlay derives one fixed T_map_glim from a single common sensor-time
# pose in MAP_FRAME_BAG and REFERENCE_BAG.  The evaluated Super-LIO output is
# never used for frame calibration or time fitting.
start_child trajectory_overlay "$OVERLAY" --ros-args \
  -p "reference_bag_path:=$REFERENCE_BAG" \
  -p "map_frame_bag_path:=$MAP_FRAME_BAG" \
  -p estimated_topic:=/comparison/input_odom \
  -p clean_estimated_topic:=/comparison/super_lio_localization_odom \
  -p world_frame:=world
start_child robot_state_publisher ros2 launch bidex_ros state_pub.launch.py
start_child g1_robot_visualizer "$ROBOT_VISUALIZER" --ros-args \
  -p lowstate_topic:=/comparison/input_lowstate \
  -p odometry_topic:=/comparison/super_lio_localization_odom

RVIZ_PID=''
if [[ "$WITH_RVIZ" == true ]]; then
  start_child rviz rviz2 -d "$RVIZ_CONFIG"
  RVIZ_PID=${CHILD_PIDS[4]}
fi

sleep 2
bag_command=(
  ros2 bag play "$BAG"
  --rate "$RATE"
  --topics /lio/odom /lowstate /livox/lidar
  --remap /lio/odom:=/comparison/input_odom \
          /lowstate:=/comparison/input_lowstate \
          /livox/lidar:=/comparison/input_lidar
)
[[ "$LOOP" == true ]] && bag_command+=(--loop)
[[ "$START_PAUSED" == true ]] && bag_command+=(--start-paused)

set +e
if [[ "$DURATION" == 0 || "$DURATION" == 0.0 ]]; then
  "${bag_command[@]}" 2>&1 | tee "$LOG_DIR/bag_player.log"
  BAG_RESULT=${PIPESTATUS[0]}
else
  timeout --signal=INT --kill-after=5s "$DURATION" \
    "${bag_command[@]}" 2>&1 | tee "$LOG_DIR/bag_player.log"
  BAG_RESULT=${PIPESTATUS[0]}
fi
set -e
if [[ $BAG_RESULT -ne 0 && $BAG_RESULT -ne 124 ]]; then
  command printf 'Bag playback failed with exit code %s\n' "$BAG_RESULT" >&2
  exit "$BAG_RESULT"
fi

command printf '%s\n' \
  'Playback finished. The fixed map and completed trails remain in RViz.' \
  'Close RViz or press Ctrl-C in this terminal to stop.'
if [[ -n "$RVIZ_PID" ]]; then
  set +e
  wait "$RVIZ_PID"
  set -e
fi
