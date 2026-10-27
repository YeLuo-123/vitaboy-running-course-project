"""Online SLAM map->odom, G1 planner/tracker, VitaBoy dynamics bridge."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def robot_description(context):
    filename = LaunchConfiguration('description').perform(context)
    if not filename:
        return []
    path = Path(filename).resolve()
    root = ET.parse(path).getroot()
    children = {joint.find('child').get('link') for joint in root.findall('joint')}
    roots = [link.get('name') for link in root.findall('link') if link.get('name') not in children]
    if roots != ['pelvis_link']:
        raise ValueError('Expected the actual VitaBoy pelvis-rooted URDF')
    for mesh in root.findall('.//mesh'):
        relative = mesh.get('filename')
        if relative.startswith('package:'):
            raise ValueError('Resolve package mesh paths explicitly before launch')
        mesh_path = (path.parent/relative).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh.set('filename', mesh_path.as_uri())
    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'use_sim_time':True, 'robot_description':ET.tostring(root,encoding='unicode')}]),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['--frame-id','base_link','--child-frame-id','pelvis_link'],
             parameters=[{'use_sim_time':True}])]


def navigation_nodes(context):
    running=LaunchConfiguration('running').perform(context).lower()=='true'
    limits=(2.4,.12,.6) if running else (.35,.15,.35)
    common={'use_sim_time':True}
    return [
        Node(package='vitaboy_navigation',executable='bridge',output='screen',
             parameters=[common,{'running_mode':running}]),
        Node(package='vitaboy_navigation',executable='planner',output='screen',
             parameters=[common,{'running_mode':running,'inflation_radius_m':.95 if running else .65,
                                 'allow_unknown':False,'replan_period':1.0}]),
        Node(package='vitaboy_navigation',executable='tracker',output='screen',
             parameters=[common,{'running_mode':running,'vx_limit':limits[0],
                                 'vy_limit':limits[1],'wz_limit':limits[2],
                                 'avoid_radius':6.0 if running else .9,
                                 'emergency_distance':.8 if running else .50,
                                 'goal_tolerance':.30 if running else .18,
                                 'waypoint_tolerance':.35 if running else .22}])]


def generate_launch_description():
    share = get_package_share_directory('vitaboy_navigation')
    slam_share = get_package_share_directory('slam_toolbox')
    common = {'use_sim_time': True}
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('description', default_value=''),
        DeclareLaunchArgument('running', default_value='false'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(share,'config','navigation.rviz')),
        DeclareLaunchArgument('slam_config', default_value=os.path.join(share,'config','slam.yaml')),
        OpaqueFunction(function=robot_description),
        OpaqueFunction(function=navigation_nodes),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
            slam_share, 'launch', 'online_async_launch.py')),
            launch_arguments={'use_sim_time': 'true', 'slam_params_file': LaunchConfiguration('slam_config')}.items()),
        Node(package='rviz2', executable='rviz2', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', LaunchConfiguration('rviz_config')], parameters=[common]),
    ])
