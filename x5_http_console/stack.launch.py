from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node, SetRemap
from ament_index_python.packages import get_package_share_directory as share


def setup(context):
    mode = LaunchConfiguration('mode').perform(context)
    def include(package, file, **args):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(Path(share(package))/'launch'/file)), launch_arguments=args.items())
    params = str(Path(share('robocup_bringup'))/'config/nav2_params.yaml')
    lidar_params = str(Path(share('lslidar_driver'))/'params/lidar_uart_ros2/lsn10p.yaml')
    nodes = [include('turn_on_wheeltec_robot', 'base_serial.launch.py'),
             LifecycleNode(package='lslidar_driver', executable='lslidar_driver_node',
                           name='lslidar_driver_node', namespace='', output='screen', emulate_tty=True,
                           parameters=[lidar_params], respawn=True, respawn_delay=5.0),
             include('robocup_bringup', 'robot_bringup.launch.py', start_odom_tf='true'),
             Node(package='tf2_ros', executable='static_transform_publisher', name='x5_laser_tf', arguments=['0.03163','0.00009','0.09502','0','0','0','base_footprint','laser']),
             Node(package='tf2_ros', executable='static_transform_publisher', name='x5_imu_tf', arguments=['0','0','0','0','0','0','base_footprint','gyro_link'])]
    if mode == 'mapping':
        nodes += [include('robocup_bringup', 'cartographer.launch.py', configuration_directory=str(Path(__file__).parent/'config')),
            GroupAction([SetRemap(src='cmd_vel',dst='/cmd_vel_nav'), include('nav2_bringup','navigation_launch.py',params_file=params,use_composition='False',autostart='true')]),
            Node(package='nav2_velocity_smoother',executable='velocity_smoother',name='velocity_smoother',parameters=[params],remappings=[('cmd_vel','/cmd_vel_nav'),('cmd_vel_smoothed','/x5/nav_cmd')]),
            Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='lifecycle_manager_velocity',parameters=[{'autostart':True,'node_names':['velocity_smoother']}])]
    else:
        nodes += [include('robocup_bringup','navigation.launch.py',map=LaunchConfiguration('map').perform(context),cmd_vel_topic='/x5/nav_cmd')]
    return nodes


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('mode',default_value='mapping'),DeclareLaunchArgument('map',default_value=''),OpaqueFunction(function=setup)])
