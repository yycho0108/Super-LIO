#!/usr/bin/env bash

conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source /home/dongjae/Super-LIO/install/setup.bash
set -euo pipefail

WORKSPACE=/home/dongjae/Super-LIO
CONFIG=$WORKSPACE/src/super_lio/config/relocation_real_20260725.yaml
MAP=$WORKSPACE/src/super_lio/map_real_20260725/dump6.pcd
RVIZ_CONFIG=$WORKSPACE/src/super_lio/rviz/relocation_real_20260722.rviz
WITH_RVIZ=false
INIT_POSE='[-0.10,0.10,0.0,0.0,0.0,-127.0]'

usage() {
  command printf '%s\n' \
    'Usage: run_live_super_lio_localization.sh [--rviz|--headless] [--init-pose CSV]' \
    'This script never starts rosbag or the Livox driver.'
}
while (($#)); do
  case "$1" in
    --rviz) WITH_RVIZ=true; shift ;;
    --headless) WITH_RVIZ=false; shift ;;
    --init-pose) INIT_POSE=${2:?missing value for --init-pose}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) command printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

for artifact in \
  "$CONFIG" "$MAP" \
  "$WORKSPACE/install/super_lio/lib/super_lio/relocation_node" \
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/policy_odom_bridge" \
  "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/map_publisher"; do
  [[ -e "$artifact" ]] || { command printf 'Missing artifact: %s\n' "$artifact" >&2; exit 66; }
done

if pgrep -x relocation_node >/dev/null 2>&1; then
  command printf '%s\n' 'A relocation_node is already running.' >&2
  exit 73
fi

command printf '%s\n' \
  'Livox driver is intentionally separate. Configure MID360_config.json IPs, then run in another terminal:' \
  "  ros2 launch $WORKSPACE/src/livox_ros_driver2/launch_ROS2/msg_MID360_launch.py" \
  'Required live input contract: /livox/lidar and /livox/imu'

LOG_DIR=$WORKSPACE/test_logs/live_localization/$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
declare -a PIDS=()

stop_live() {
  set +e
  local pid
  for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && kill -INT "$pid" 2>/dev/null; done
  for pid in "${PIDS[@]}"; do
    for _ in 1 2 3 4 5 6 7 8; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
  done
}
trap stop_live EXIT INT TERM

stdbuf -oL -eL "$WORKSPACE/install/super_lio/lib/super_lio/relocation_node" \
  --ros-args --params-file "$CONFIG" -p "lio.relocation.init_pose:=$INIT_POSE" \
  >"$LOG_DIR/super_lio.log" 2>&1 & PIDS+=("$!")
stdbuf -oL -eL "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/policy_odom_bridge" \
  >"$LOG_DIR/policy_bridge.log" 2>&1 & PIDS+=("$!")
stdbuf -oL -eL "$WORKSPACE/install/g1_localization_tools/lib/g1_localization_tools/map_publisher" \
  --ros-args -p "map_path:=$MAP" -p frame_id:=world \
  >"$LOG_DIR/map_publisher.log" 2>&1 & PIDS+=("$!")
if [[ "$WITH_RVIZ" == true ]]; then
  rviz2 -d "$RVIZ_CONFIG" >"$LOG_DIR/rviz.log" 2>&1 & PIDS+=("$!")
fi

command printf 'Live localization running. Logs: %s\n' "$LOG_DIR"
command printf '%s\n' \
  'Policy output: /localization/policy_odom_50hz' \
  'Safety diagnostics: /localization/status' \
  'Press Ctrl-C to stop all processes started by this script.'
wait "${PIDS[0]}"
