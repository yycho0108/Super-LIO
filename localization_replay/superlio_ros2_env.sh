#!/usr/bin/env bash

# Shared ROS 2 environment for deterministic offline Super-LIO replay.
_SUPERLIO_RESTORE_NOUNSET=false
case $- in
  *u*) _SUPERLIO_RESTORE_NOUNSET=true; set +u ;;
esac
source /opt/ros/humble/setup.bash
source /home/dongjae/ws/install/setup.bash
# bidex_ros currently needs a standard (non-editable) setuptools install on
# this machine; expose that package metadata to its generated entry points.
_BIDEX_SITE=/home/dongjae/ws/install/bidex_ros/local/lib/python3.10/dist-packages
if [[ -d "${_BIDEX_SITE}" ]]; then
  export PYTHONPATH="${_BIDEX_SITE}:${PYTHONPATH:-}"
fi
unset _BIDEX_SITE
if [[ "${_SUPERLIO_RESTORE_NOUNSET}" == true ]]; then
  set -u
fi
unset _SUPERLIO_RESTORE_NOUNSET

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
  <NetworkInterface name="lo" priority="default" multicast="default" />
</Interfaces></General></Domain></CycloneDDS>'

# Keep offline replay isolated from live robot and unrelated ROS processes.
export ROS_DOMAIN_ID="${SUPERLIO_REPLAY_DOMAIN_ID:-42}"
