-- Initial 2D Cartographer configuration for the RoboCup home scene.
--
-- Before running this file:
--   1. Start a lidar driver publishing sensor_msgs/LaserScan on /scan.
--   2. Provide base_footprint -> laser and base_footprint -> gyro_link TFs.
--   3. Verify that /odom and /imu/data_raw timestamps are synchronized.
--
-- The first bringup uses the wheel odometry as an input and lets Cartographer
-- build the map->odom relation. The odom TF bridge is therefore disabled by
-- cartographer.launch.py when Cartographer owns the localization transform.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  map_frame = "map",
  tracking_frame = "gyro_link",
  published_frame = "odom_combined",
  odom_frame = "odom_combined",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = true,

  use_pose_extrapolator = true,
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,

  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,

  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 0.01,
  trajectory_publish_period_sec = 0.03,

  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,
}

MAP_BUILDER.use_trajectory_builder_2d = true
MAP_BUILDER.num_background_threads = 4

TRAJECTORY_BUILDER_2D.min_range = 0.12
TRAJECTORY_BUILDER_2D.max_range = 12.0
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1
TRAJECTORY_BUILDER_2D.use_imu_data = true
TRAJECTORY_BUILDER_2D.min_z = -0.2
TRAJECTORY_BUILDER_2D.max_z = 0.5
TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.025
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 90
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05

POSE_GRAPH.optimize_every_n_nodes = 90

return options
