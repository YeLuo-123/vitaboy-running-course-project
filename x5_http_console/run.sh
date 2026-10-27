#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source /opt/ros/humble/setup.bash
source /home/wheeltec/wheeltec_ros2/install/setup.bash
source /home/wheeltec/robocup_ws/install/setup.bash
# Ubuntu's SciPy 1.8 expects the distro NumPy 1.21, not the globally installed 1.26.
export PYTHONPATH="/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec /usr/bin/python3 server.py "$@"
