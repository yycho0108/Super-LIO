#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  record_relocation_trial.sh --bag INPUT --output OUTPUT --config YAML [options]

Options:
  --rate RATE              rosbag playback rate (default: 1.0)
  --start-offset SEC       rosbag start offset (default: 0.0)
  --playback-delay SEC     subscriber warm-up delay (default: 3.0)
  --domain-id ID           isolated ROS domain (default: 42)
  --log-dir DIR            launch/record log directory
  --launch-file FILE       frozen slam replay launch file
  --super-lio-prefix DIR   frozen super_lio install prefix
  --basic-prefix DIR       frozen basic install prefix
  --dry-run                print commands without running them

The input bag's old localization outputs are remapped by the launch file, so
the recorded /lio/odom contains only the newly computed relocation result.
EOF
}

INPUT_BAG=''
OUTPUT_BAG=''
CONFIG=''
PLAY_RATE=1.0
START_OFFSET=0.0
PLAYBACK_DELAY=3.0
DOMAIN_ID=42
LOG_DIR=''
LAUNCH_FILE=''
SUPER_LIO_PREFIX=''
BASIC_PREFIX=''
DRY_RUN=false

while (($#)); do
  case "$1" in
    --bag) INPUT_BAG="${2:?missing value for --bag}"; shift 2 ;;
    --output) OUTPUT_BAG="${2:?missing value for --output}"; shift 2 ;;
    --config) CONFIG="${2:?missing value for --config}"; shift 2 ;;
    --rate) PLAY_RATE="${2:?missing value for --rate}"; shift 2 ;;
    --start-offset) START_OFFSET="${2:?missing value for --start-offset}"; shift 2 ;;
    --playback-delay) PLAYBACK_DELAY="${2:?missing value for --playback-delay}"; shift 2 ;;
    --domain-id) DOMAIN_ID="${2:?missing value for --domain-id}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?missing value for --log-dir}"; shift 2 ;;
    --launch-file) LAUNCH_FILE="${2:?missing value for --launch-file}"; shift 2 ;;
    --super-lio-prefix) SUPER_LIO_PREFIX="${2:?missing value for --super-lio-prefix}"; shift 2 ;;
    --basic-prefix) BASIC_PREFIX="${2:?missing value for --basic-prefix}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${INPUT_BAG}" || -z "${OUTPUT_BAG}" || -z "${CONFIG}" ]]; then
  usage >&2
  exit 2
fi

INPUT_BAG="$(realpath -e "${INPUT_BAG}")"
CONFIG="$(realpath -e "${CONFIG}")"
OUTPUT_PARENT="$(dirname "${OUTPUT_BAG}")"
mkdir -p "${OUTPUT_PARENT}"
OUTPUT_PARENT="$(realpath -e "${OUTPUT_PARENT}")"
OUTPUT_BAG="${OUTPUT_PARENT}/$(basename "${OUTPUT_BAG}")"

if [[ ! -f "${INPUT_BAG}/metadata.yaml" ]]; then
  echo "missing rosbag metadata: ${INPUT_BAG}/metadata.yaml" >&2
  exit 66
fi
if [[ -e "${OUTPUT_BAG}" ]]; then
  echo "output already exists; refusing to overwrite: ${OUTPUT_BAG}" >&2
  exit 73
fi

/usr/bin/python3 - "${PLAY_RATE}" "${START_OFFSET}" "${PLAYBACK_DELAY}" "${DOMAIN_ID}" <<'PY'
import math
import sys

rate, offset, delay = map(float, sys.argv[1:4])
domain = int(sys.argv[4])
if not math.isfinite(rate) or rate <= 0.0:
    raise SystemExit(f"invalid --rate: {sys.argv[1]}")
if not math.isfinite(offset) or offset < 0.0:
    raise SystemExit(f"invalid --start-offset: {sys.argv[2]}")
if not math.isfinite(delay) or delay < 0.0:
    raise SystemExit(f"invalid --playback-delay: {sys.argv[3]}")
if not 0 <= domain <= 232:
    raise SystemExit(f"--domain-id must be in [0, 232], got {domain}")
PY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUPERLIO_REPLAY_DOMAIN_ID="${DOMAIN_ID}"
source "${SCRIPT_DIR}/superlio_ros2_env.sh"

if [[ -n "${SUPER_LIO_PREFIX}" ]]; then
  SUPER_LIO_PREFIX="$(realpath -e "${SUPER_LIO_PREFIX}")"
  export AMENT_PREFIX_PATH="${SUPER_LIO_PREFIX}:${AMENT_PREFIX_PATH:-}"
  export LD_LIBRARY_PATH="${SUPER_LIO_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${BASIC_PREFIX}" ]]; then
  BASIC_PREFIX="$(realpath -e "${BASIC_PREFIX}")"
  export AMENT_PREFIX_PATH="${BASIC_PREFIX}:${AMENT_PREFIX_PATH:-}"
  export LD_LIBRARY_PATH="${BASIC_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "${LAUNCH_FILE}" ]]; then
  LAUNCH_FILE="$(ros2 pkg prefix bidex_ros)/share/bidex_ros/launch/slam_bag_rviz_launch.py"
else
  LAUNCH_FILE="$(realpath -e "${LAUNCH_FILE}")"
fi
if [[ ! -f "${LAUNCH_FILE}" ]]; then
  echo "launch not found: ${LAUNCH_FILE}" >&2
  exit 66
fi
if ! rg -q 'super_lio_reloc_config_path' "${LAUNCH_FILE}" || \
   ! rg -q 'isolate_replayed_outputs' "${LAUNCH_FILE}"; then
  echo "launch lacks config override/output isolation: ${LAUNCH_FILE}" >&2
  exit 65
fi

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="/tmp/relocation_trial_$(basename "${OUTPUT_BAG}")_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_DIR}"

RECORD_TOPICS=(/lio/odom)
LAUNCH_ARGS=(
  "bag_path:=${INPUT_BAG}"
  slam_backend:=super_lio_reloc
  launch_rviz:=false
  launch_aux_nodes:=false
  use_sim_time:=true
  "play_rate:=${PLAY_RATE}"
  "start_offset:=${START_OFFSET}"
  "playback_delay:=${PLAYBACK_DELAY}"
  "super_lio_reloc_config_path:=${CONFIG}"
  isolate_replayed_outputs:=true
  shutdown_when_bag_ends:=true
)

echo "input : ${INPUT_BAG}"
echo "output: ${OUTPUT_BAG}"
echo "config: ${CONFIG}"
echo "launch: ${LAUNCH_FILE}"
echo "super_lio prefix: $(ros2 pkg prefix super_lio)"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, RMW=${RMW_IMPLEMENTATION}"
if [[ "${DRY_RUN}" == true ]]; then
  printf 'record command:'
  printf ' %q' ros2 bag record -o "${OUTPUT_BAG}" "${RECORD_TOPICS[@]}"
  printf '\nlaunch command:'
  printf ' %q' ros2 launch "${LAUNCH_FILE}" "${LAUNCH_ARGS[@]}"
  printf '\nlogs: %s\n' "${LOG_DIR}"
  exit 0
fi

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
ros2 launch "${LAUNCH_FILE}" "${LAUNCH_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/launch.log"
LAUNCH_STATUS=${PIPESTATUS[0]}
set -e
sleep 2
cleanup
RECORDER_PID=''
if [[ ${LAUNCH_STATUS} -ne 0 ]]; then
  echo "launch failed with ${LAUNCH_STATUS}; partial bag kept: ${OUTPUT_BAG}" >&2
  exit "${LAUNCH_STATUS}"
fi

/usr/bin/python3 - "${OUTPUT_BAG}/metadata.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
metadata = yaml.safe_load(path.read_text())["rosbag2_bagfile_information"]
counts = {
    row["topic_metadata"]["name"]: int(row["message_count"])
    for row in metadata["topics_with_message_count"]
}
if counts.get("/lio/odom", 0) < 2:
    raise SystemExit(
        f"recording validation failed: /lio/odom count={counts.get('/lio/odom', 0)}"
    )
print(f"recorded /lio/odom: {counts['/lio/odom']:,}")
PY

{
  echo "input_bag=${INPUT_BAG}"
  echo "output_bag=${OUTPUT_BAG}"
  echo "config=${CONFIG}"
  echo "launch_file=${LAUNCH_FILE}"
  echo "super_lio_prefix=$(ros2 pkg prefix super_lio)"
  echo "play_rate=${PLAY_RATE}"
  echo "start_offset=${START_OFFSET}"
  echo "ros_domain_id=${ROS_DOMAIN_ID}"
  sha256sum "${CONFIG}"
  sha256sum "${LAUNCH_FILE}"
  sha256sum "$(ros2 pkg prefix super_lio)/lib/super_lio/relocation_node"
} >"${OUTPUT_BAG}/recording_manifest.txt"

echo "completed: ${OUTPUT_BAG}"
echo "logs: ${LOG_DIR}"
