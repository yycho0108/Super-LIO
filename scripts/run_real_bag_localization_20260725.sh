#!/usr/bin/env bash

# Source ROS before nounset: Humble's generated setup files reference optional
# shell variables that are not guaranteed to be defined.
#conda deactivate 2>/dev/null || true
#source /opt/ros/humble/setup.bash
#source /home/dongjae/Super-LIO/install/setup.bash
#set -euo pipefail

CONFIG=$WORKSPACE/src/super_lio/config/relocation_real_20260725.yaml

MAP_DIR=$WORKSPACE/src/super_lio/map_real_20260725
MAP=/gate/dump6_xyzi_flip.pcd
SOURCE_MAP=/gate/dump6_flip.pcd

BAG_DIR=$WORKSPACE/src/super_lio/rosbag2_2026_07_25-14_09_35
BAG_DB=$BAG_DIR/rosbag2_2026_07_25-14_09_35_0.db3

WORKSPACE=~/ws/src/Super-LIO
PCD_TOOL=$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/pcd_xyzi_tool
RVIZ_CONFIG=$WORKSPACE/src/super_lio/rviz/relocation_real_20260725.rviz

START_OFFSET=0
DURATION=60
PLAYBACK_RATE=1.0
WITH_RVIZ=false

# No independently surveyed Super-LIO initial pose is stored with this bag.
# This neutral placeholder is intentionally not copied from the GLIM reference.
INIT_POSE='[0.0,0.0,0.0,0.0,0.0,0.0]'

usage() {
  command printf '%s\n' \
    'Usage: run_real_bag_localization_20260725.sh [options]' \
    '  --headless                 Do not start RViz (default)' \
    '  --rviz                     Start RViz with the saved config' \
    '  --rate RATE                Bag playback rate (default: 1.0)' \
    '  --duration SECONDS         Wall-clock run time; 0 plays to bag end' \
    '  --start-offset SECONDS     Bag offset (default: 0)' \
    '  --init-pose CSV            [x,y,z,roll,pitch,yaw] parameter value' \
    '                              (default is a neutral placeholder; set manually)' \
    '  --help                     Show this text'
}

while (($#)); do
  case "$1" in
    --headless) WITH_RVIZ=false; shift ;;
    --rviz) WITH_RVIZ=true; shift ;;
    --rate) PLAYBACK_RATE=${2:?missing value for --rate}; shift 2 ;;
    --duration) DURATION=${2:?missing value for --duration}; shift 2 ;;
    --start-offset) START_OFFSET=${2:?missing value for --start-offset}; shift 2 ;;
    --init-pose) INIT_POSE=${2:?missing value for --init-pose}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) command printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

python3 - "$PLAYBACK_RATE" "$DURATION" "$START_OFFSET" "$INIT_POSE" <<'PY'
import ast
import math
import sys

for label, value, allow_zero in zip(
        ('rate', 'duration', 'start-offset'), sys.argv[1:4],
        (False, True, True)):
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        raise SystemExit(f'invalid --{label}: {value}')

try:
    init_pose = ast.literal_eval(sys.argv[4])
except (SyntaxError, ValueError) as error:
    raise SystemExit(f'invalid --init-pose: {sys.argv[4]}') from error
if (not isinstance(init_pose, (list, tuple)) or len(init_pose) != 6 or
        not all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in init_pose)):
    raise SystemExit('--init-pose must contain six finite numbers')
PY

required_files=(
  "$BAG_DIR/metadata.yaml"
  "$BAG_DB"
  "$CONFIG"
  "$SOURCE_MAP"
  "$PCD_TOOL"
  "$WORKSPACE/install/super_lio/lib/super_lio/relocation_node"
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/policy_odom_bridge"
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/map_publisher"
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/reference_visualizer"
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/localization_evaluator"
)
for required in "${required_files[@]}"; do
  if [[ ! -e "$required" ]]; then
    command printf 'Missing required build/data artifact: %s\n' "$required" >&2
    command printf 'Build with: cd %s && colcon build --symlink-install\n' "$WORKSPACE" >&2
    exit 66
  fi
done

# dump6.pcd has XYZ/RGB/normal fields but no intensity. Create only a separate
# XYZI map. pcd_xyzi_tool copies each float XYZ coordinate without any
# rotation, translation, or scaling and assigns intensity=0.0F.
if [[ ! -e "$MAP" ]]; then
  mkdir -p "$MAP_DIR"
  TEMP_MAP=$(mktemp "$MAP_DIR/.dump6_xyzi.pcd.XXXXXX")
  if ! "$PCD_TOOL" convert "$SOURCE_MAP" "$TEMP_MAP"; then
    command rm -f -- "$TEMP_MAP"
    exit 65
  fi
  command mv -- "$TEMP_MAP" "$MAP"
fi

if pgrep -x relocation_node >/dev/null 2>&1 ||
   pgrep -f '^python3 .*/(policy_odom_bridge|localization_evaluator|map_publisher|reference_visualizer)( |$)' >/dev/null 2>&1 ||
   pgrep -f '^ros2 bag play ' >/dev/null 2>&1; then
  command printf '%s\n' 'A related localization or bag process is already running; refusing to mix ROS graphs.' >&2
  pgrep -af 'relocation_node|policy_odom_bridge|localization_evaluator|map_publisher|reference_visualizer|ros2 bag play' >&2 || true
  exit 73
fi

RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$WORKSPACE/test_logs/real_bag_20260725/$RUN_STAMP
mkdir -p "$LOG_DIR"
command printf 'Log directory: %s\n' "$LOG_DIR"

declare -a CHILD_PIDS=()
declare -a CHILD_NAMES=()
CLEANED=false

start_child() {
  local child_name=$1
  local child_log=$2
  shift 2
  stdbuf -oL -eL "$@" >"$child_log" 2>&1 &
  CHILD_PIDS+=("$!")
  CHILD_NAMES+=("$child_name")
}

stop_children() {
  [[ "$CLEANED" == true ]] && return
  CLEANED=true
  set +e
  local index child_pid
  for ((index=${#CHILD_PIDS[@]}-1; index>=0; index--)); do
    child_pid=${CHILD_PIDS[index]}
    kill -0 "$child_pid" 2>/dev/null && kill -INT "$child_pid" 2>/dev/null
  done
  for ((index=${#CHILD_PIDS[@]}-1; index>=0; index--)); do
    child_pid=${CHILD_PIDS[index]}
    for _ in 1 2 3 4 5 6 7 8; do
      kill -0 "$child_pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$child_pid" 2>/dev/null; then
      command printf 'SIGINT timeout for %s (PID %s); sending SIGTERM\n' "${CHILD_NAMES[index]}" "$child_pid" >&2
      kill -TERM "$child_pid" 2>/dev/null
    fi
    wait "$child_pid" 2>/dev/null
  done
  set -e
}
trap stop_children EXIT INT TERM

start_child super_lio "$LOG_DIR/super_lio.log" \
  "$WORKSPACE/install/super_lio/lib/super_lio/relocation_node" \
  --ros-args --params-file "$CONFIG" \
  -p "lio.relocation.init_pose:=$INIT_POSE"
LIO_PID=${CHILD_PIDS[0]}
start_child policy_bridge "$LOG_DIR/policy_bridge.log" \
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/policy_odom_bridge"
start_child map_publisher "$LOG_DIR/map_publisher.log" \
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/map_publisher" \
  --ros-args -p "map_path:=$MAP" -p frame_id:=world
start_child reference_visualizer "$LOG_DIR/reference_visualizer.log" \
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/reference_visualizer"
start_child evaluator "$LOG_DIR/evaluator.log" \
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/localization_evaluator" \
  --ros-args -p "output_dir:=$LOG_DIR" -p "map_path:=$MAP" \
  -p "playback_rate:=$PLAYBACK_RATE"

if [[ "$WITH_RVIZ" == true ]]; then
  start_child rviz "$LOG_DIR/rviz.log" rviz2 -d "$RVIZ_CONFIG"
fi

for _ in $(seq 1 80); do
  grep -q 'Load map success' "$LOG_DIR/super_lio.log" && break
  if ! kill -0 "$LIO_PID" 2>/dev/null; then
    command printf '%s\n' 'Super-LIO exited before loading the map.' >&2
    exit 70
  fi
  sleep 0.25
done
if ! grep -q 'Load map success' "$LOG_DIR/super_lio.log"; then
  command printf '%s\n' 'Timed out waiting for map load.' >&2
  exit 70
fi

# /glim_ros/odom_corrected is isolated by remapping and is consumed only by
# the reference visualizer/evaluator. Super-LIO subscribes only to raw
# /livox/lidar and /livox/imu through its YAML configuration.
bag_command=(
  ros2 bag play "$BAG_DIR"
  --disable-keyboard-controls
  --rate "$PLAYBACK_RATE"
  --start-offset "$START_OFFSET"
  --topics /livox/lidar /livox/imu /glim_ros/odom_corrected
  --remap /glim_ros/odom_corrected:=/reference/glim_odom
)

set +e
if [[ "$DURATION" == 0 || "$DURATION" == 0.0 ]]; then
  "${bag_command[@]}" </dev/null >"$LOG_DIR/bag_player.log" 2>&1
  BAG_RESULT=$?
else
  timeout --signal=INT --kill-after=8s "$DURATION" "${bag_command[@]}" \
    </dev/null >"$LOG_DIR/bag_player.log" 2>&1
  BAG_RESULT=$?
fi
set -e
if [[ $BAG_RESULT -ne 0 && $BAG_RESULT -ne 124 ]]; then
  command printf 'rosbag player failed with exit code %s\n' "$BAG_RESULT" >&2
  exit 71
fi

sleep 2
stop_children
trap - EXIT INT TERM

required_logs=(
  'Load map success'
  'Map size: 13467'
  'Map init success'
  'INIT start'
  'Global ICP Converged Succeed'
  'KF init done'
  'Map init done'
)
for expected in "${required_logs[@]}"; do
  if ! grep -q "$expected" "$LOG_DIR/super_lio.log"; then
    command printf 'Missing required initialization log: %s\n' "$expected" >&2
    exit 72
  fi
done

set +e
python3 - "$LOG_DIR/summary.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as stream:
    report = json.load(stream)
topics = report['topics']
errors = []
rate_limits = {
    'livox_lidar': (25.0, 35.0),
    'livox_imu': (170.0, 220.0),
    'lio_odom': (25.0, 35.0),
    'lio_robo_odom': (170.0, 220.0),
    'lio_imu_odom': (170.0, 220.0),
    'policy_odom_50hz': (45.0, 55.0),
}
for name, (minimum, maximum) in rate_limits.items():
    topic = topics.get(name, {})
    rate = topic.get('header_rate_hz', 0.0)
    print(f'{name}: {rate:.3f} Hz, valid={topic.get("valid_count", 0)}')
    if not minimum <= rate <= maximum:
        errors.append(f'{name} rate {rate:.3f} outside [{minimum}, {maximum}]')
    for field in ('nonfinite_count', 'timestamp_nonmonotonic_count',
                  'position_jump_over_1m_count', 'stale_gap_count'):
        if topic.get(field, 0):
            errors.append(f'{name} {field}={topic[field]}')

overlap = report.get('cloud_map_overlap', {})
within = overlap.get('fraction_within_0_50m', 0.0)
print(f'cloud-map fraction within 0.50 m: {within:.3f}')
if within < 0.60:
    errors.append(f'cloud-map overlap within 0.50 m is only {within:.3f}')

comparison = report.get('comparisons', {}).get('lio_odom')
if comparison:
    planar_p95 = comparison['planar_error_m']['p95']
    yaw_p95 = comparison['yaw_error_deg']['p95']
    print(f'GLIM reference p95: planar={planar_p95:.3f} m, yaw={yaw_p95:.3f} deg')
    if planar_p95 > 2.0:
        errors.append(f'GLIM-reference planar p95 {planar_p95:.3f} m > 2 m')
    if yaw_p95 > 30.0:
        errors.append(f'GLIM-reference yaw p95 {yaw_p95:.3f} deg > 30 deg')

if errors:
    print('QUALITY VERDICT: FAIL', file=sys.stderr)
    for error in errors:
        print(f'  - {error}', file=sys.stderr)
    raise SystemExit(2)
print('QUALITY VERDICT: PASS')
PY
VERDICT=$?
set -e

if pgrep -x relocation_node >/dev/null 2>&1 ||
   pgrep -f '^python3 .*/(policy_odom_bridge|localization_evaluator|map_publisher|reference_visualizer)( |$)' >/dev/null 2>&1 ||
   pgrep -f '^ros2 bag play ' >/dev/null 2>&1; then
  command printf '%s\n' 'Orphan localization process detected after cleanup.' >&2
  pgrep -af 'relocation_node|policy_odom_bridge|localization_evaluator|map_publisher|reference_visualizer|ros2 bag play' >&2 || true
  exit 74
fi

command printf 'Completed. Logs: %s\n' "$LOG_DIR"
exit "$VERDICT"
