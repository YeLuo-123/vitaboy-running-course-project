#!/usr/bin/env bash
# Source only: use the existing robot SDK without changing its installation.
source /opt/ros/humble/setup.bash
source /opt/iceoryx_cyclone_dds/install/setup.bash
source /opt/build_dist/common/share/interaction_msgs/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/opt/iceoryx_cyclone_dds/config/cyclonedds.xml
unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
