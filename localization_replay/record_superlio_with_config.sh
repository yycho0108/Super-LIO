#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CONFIG="/home/dongjae/ws/src/Super-LIO/src/super_lio/config/relocation_real_20260802.yaml"

usage() {
  cat <<EOF
Usage:
  ./record_superlio_with_config.sh OUTPUT_BAG [CONFIG_YAML] [--dry-run]

The config defaults to:
  ${DEFAULT_CONFIG}
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

OUTPUT_ARGUMENT="$1"
shift
CONFIG_ARGUMENT="${DEFAULT_CONFIG}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG_ARGUMENT="$1"
  shift
fi

DRY_RUN=false
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 was not found; source the same ROS environment as the launch terminal." >&2
  exit 69
fi

OUTPUT_PARENT="$(realpath -e "$(dirname "${OUTPUT_ARGUMENT}")")"
OUTPUT_BAG="${OUTPUT_PARENT}/$(basename "${OUTPUT_ARGUMENT}")"
CONFIG_YAML="$(realpath -e "${CONFIG_ARGUMENT}")"
SNAPSHOT_NAME="super_lio_config_snapshot.yaml"

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Config is not a file: ${CONFIG_YAML}" >&2
  exit 66
fi
if [[ -e "${OUTPUT_BAG}" ]]; then
  echo "Output already exists; refusing to overwrite: ${OUTPUT_BAG}" >&2
  exit 73
fi

TOPICS=(
  /lowstate
  /lowcmd
  /livox/imu
  /livox/lidar
  /secondary_imu
  /tf
  /tf_static
  /glim_ros/odom_corrected
  /odometry/odom
  /glim_ros/odom
  /augmented_map
  /lio/odom
  /lio/cloud_world
)

echo "output bag: ${OUTPUT_BAG}"
echo "config    : ${CONFIG_YAML}"
echo "snapshot  : ${OUTPUT_BAG}/${SNAPSHOT_NAME}"
echo "sha256    : $(sha256sum "${CONFIG_YAML}" | awk '{print $1}')"

if [[ "${DRY_RUN}" == true ]]; then
  printf 'record command:'
  printf ' %q' ros2 bag record -o "${OUTPUT_BAG}" "${TOPICS[@]}"
  printf '\n'
  exit 0
fi

# Freeze the file before recording starts. The source YAML may safely be edited
# for the next experiment after this point without changing this snapshot.
SNAPSHOT_DIR="$(mktemp -d /tmp/super_lio_config_snapshot.XXXXXX)"
cp -p "${CONFIG_YAML}" "${SNAPSHOT_DIR}/${SNAPSHOT_NAME}"

install_snapshot() {
  if [[ -d "${OUTPUT_BAG}" && -f "${SNAPSHOT_DIR}/${SNAPSHOT_NAME}" ]]; then
    cp -p "${SNAPSHOT_DIR}/${SNAPSHOT_NAME}" \
      "${OUTPUT_BAG}/${SNAPSHOT_NAME}"
  fi
}

cleanup() {
  install_snapshot || true
  rm -rf "${SNAPSHOT_DIR}"
}
trap cleanup EXIT

# ros2 bag creates OUTPUT_BAG itself, so install the already-frozen snapshot as
# soon as that directory appears while recording continues in the foreground.
(
  for _ in $(seq 1 300); do
    if [[ -d "${OUTPUT_BAG}" ]]; then
      cp -p "${SNAPSHOT_DIR}/${SNAPSHOT_NAME}" \
        "${OUTPUT_BAG}/${SNAPSHOT_NAME}"
      exit 0
    fi
    sleep 0.1
  done
  echo "Timed out waiting for rosbag output directory: ${OUTPUT_BAG}" >&2
  exit 1
) &
SNAPSHOT_INSTALLER_PID=$!

set +e
ros2 bag record -o "${OUTPUT_BAG}" "${TOPICS[@]}"
RECORD_STATUS=$?
set -e

wait "${SNAPSHOT_INSTALLER_PID}" || true
install_snapshot
echo "saved config snapshot: ${OUTPUT_BAG}/${SNAPSHOT_NAME}"
exit "${RECORD_STATUS}"
