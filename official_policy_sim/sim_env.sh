#!/usr/bin/env bash
# Source only for this simulation shell; never modify ~/.bashrc or robot env.
ROOT="$HOME/honor_deploy_ws"
unset CYCLONEDDS_URI PYTHONHOME
source /opt/ros/humble/setup.bash
source "$ROOT/sdk_install/setup.bash"
export ROS_DOMAIN_ID=117 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT/official_policy_sim/localhost_dds.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=egl
