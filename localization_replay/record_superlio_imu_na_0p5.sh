#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_ID [--rate RATE] [--output-root DIR] [--dry-run]" >&2
  exit 2
fi

RUN_ID="$1"
shift
if [[ ! "${RUN_ID}" =~ ^(062309|063004|064152|064922)$ ]]; then
  echo "RUN_ID must be one of: 062309 063004 064152 064922" >&2
  exit 2
fi

PLAY_RATE=1.0
PLAYBACK_DELAY=3.0
ROSBAG_ROOT="${LOCALIZATION_REPLAY_ROSBAG_ROOT:-/workspace/rosbag}"
OUTPUT_ROOT="${ROSBAG_ROOT}"
DRY_RUN=false
while (($#)); do
  case "$1" in
    --rate) PLAY_RATE="${2:?missing value for --rate}"; shift 2 ;;
    --playback-delay)
      PLAYBACK_DELAY="${2:?missing value for --playback-delay}"
      shift 2
      ;;
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h)
      echo "usage: $0 RUN_ID [--rate RATE] [--output-root DIR] [--dry-run]"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

python3 - "${PLAY_RATE}" "${PLAYBACK_DELAY}" <<'PY'
import math
import sys

for name, raw, positive in (
    ("rate", sys.argv[1], True),
    ("playback-delay", sys.argv[2], False),
):
    value = float(raw)
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        raise SystemExit(f"invalid --{name}: {raw}")
PY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/superlio_ros2_env.sh"

RUN_TIME="${RUN_ID:0:2}_${RUN_ID:2:2}_${RUN_ID:4:2}"
INPUT_BAG="${ROSBAG_ROOT}/rosbag2_2026_08_04-${RUN_TIME}"
OUTPUT_BAG="${OUTPUT_ROOT}/superlio_imu_na_0p5_${RUN_ID}"
CONFIG="${REPO_ROOT}/src/super_lio/config/relocation_real_20260802_imu_na_0p5.yaml"
INSTALLED_LAUNCH="$(ros2 pkg prefix bidex_ros)/share/bidex_ros/launch/slam_bag_rviz_launch.py"

for required in "${INPUT_BAG}/metadata.yaml" "${CONFIG}" "${INSTALLED_LAUNCH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 66
  fi
done
if ! rg -q 'lio\.sensor\.imu_na: 0\.5([[:space:]]|$)' "${CONFIG}"; then
  echo "imu_na=0.5 not found in ${CONFIG}" >&2
  exit 65
fi
if ! rg -q 'super_lio_reloc_config_path' "${INSTALLED_LAUNCH}"; then
  echo "installed bidex_ros launch is stale; rebuild bidex_ros first" >&2
  echo "cd /home/dongjae/ws && colcon build --symlink-install --packages-select bidex_ros" >&2
  exit 65
fi
if [[ -e "${OUTPUT_BAG}" ]]; then
  echo "output already exists; refusing to overwrite: ${OUTPUT_BAG}" >&2
  exit 73
fi

RECORD_TOPICS=(
  /lio/odom
)
LAUNCH_ARGS=(
  "bag_path:=${INPUT_BAG}"
  slam_backend:=super_lio_reloc
  launch_rviz:=false
  use_sim_time:=true
  "play_rate:=${PLAY_RATE}"
  "playback_delay:=${PLAYBACK_DELAY}"
  "super_lio_reloc_config_path:=${CONFIG}"
  isolate_replayed_outputs:=true
  shutdown_when_bag_ends:=true
  launch_aux_nodes:=false
)

echo "input : ${INPUT_BAG}"
echo "output: ${OUTPUT_BAG}"
echo "config: ${CONFIG} (imu_na=0.5)"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, RMW=${RMW_IMPLEMENTATION}"
if [[ "${DRY_RUN}" == true ]]; then
  printf 'record command:'
  printf ' %q' ros2 bag record -o "${OUTPUT_BAG}" "${RECORD_TOPICS[@]}"
  printf '\nlaunch command:'
  printf ' %q' ros2 launch bidex_ros slam_bag_rviz_launch.py "${LAUNCH_ARGS[@]}"
  printf '\n'
  exit 0
fi

LOG_DIR="/tmp/superlio_imu_na_0p5_${RUN_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
RECORDER_PID=''
cleanup() {
  if [[ -n "${RECORDER_PID}" ]] && kill -0 "${RECORDER_PID}" 2>/dev/null; then
    kill -INT "${RECORDER_PID}" 2>/dev/null || true
    wait "${RECORDER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ros2 bag record -o "${OUTPUT_BAG}" "${RECORD_TOPICS[@]}" \
  >"${LOG_DIR}/record.log" 2>&1 &
RECORDER_PID=$!
sleep 2
if ! kill -0 "${RECORDER_PID}" 2>/dev/null; then
  echo "ros2 bag record failed to start; see ${LOG_DIR}/record.log" >&2
  wait "${RECORDER_PID}" || true
  exit 70
fi

set +e
ros2 launch bidex_ros slam_bag_rviz_launch.py "${LAUNCH_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/launch.log"
LAUNCH_STATUS=${PIPESTATUS[0]}
set -e
sleep 2
cleanup
RECORDER_PID=''
if [[ ${LAUNCH_STATUS} -ne 0 ]]; then
  echo "launch failed with ${LAUNCH_STATUS}; partial bag kept at ${OUTPUT_BAG}" >&2
  exit "${LAUNCH_STATUS}"
fi

python3 - "${OUTPUT_BAG}/metadata.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

metadata_path = Path(sys.argv[1])
metadata = yaml.safe_load(metadata_path.read_text())["rosbag2_bagfile_information"]
counts = {
    entry["topic_metadata"]["name"]: int(entry["message_count"])
    for entry in metadata["topics_with_message_count"]
}
required = ("/lio/odom",)
missing = [topic for topic in required if counts.get(topic, 0) == 0]
if missing:
    raise SystemExit(f"recording validation failed, empty topics: {missing}")
print("recorded topic counts:")
for topic in required:
    print(f"  {topic}: {counts[topic]:,}")
PY

{
  echo "run_id=${RUN_ID}"
  echo "input_bag=${INPUT_BAG}"
  echo "output_bag=${OUTPUT_BAG}"
  echo "imu_na=0.5"
  echo "config=${CONFIG}"
  echo "play_rate=${PLAY_RATE}"
  echo "ros_domain_id=${ROS_DOMAIN_ID}"
  sha256sum "${CONFIG}"
} >"${OUTPUT_BAG}/recording_manifest.txt"

echo "completed: ${OUTPUT_BAG}"
echo "logs: ${LOG_DIR}"
