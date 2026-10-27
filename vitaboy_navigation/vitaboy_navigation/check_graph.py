"""Exercise the real ROS graph. No fabricated map or kinematic robot motion."""
import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException
from vitaboy_navigation.protocol import transform_planar, yaw


class Check(Node):
    def __init__(self):
        super().__init__('vitaboy_integration_check')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.maps = 0; self.occupied = 0; self.known = 0
        self.last_map = None; self.trajectory = []
        self.scan_times = []; self.odom_times = []; self.odom = None
        self.path = []; self.goal = None
        self.path_events = 0
        self.goal_pub = self.create_publisher(PoseStamped, '/g1pilot/goal', 10)
        self.trail_pub=self.create_publisher(RosPath,'/vitaboy/trajectory',10)
        self.goals_pub=self.create_publisher(MarkerArray,'/vitaboy/demo_goals',10)
        self.demo_goals=[];self.completed=0
        self.create_subscription(OccupancyGrid, '/map', self.on_map,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry, '/lidar_odometry/pose_fixed', self.on_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.create_subscription(RosPath, '/g1pilot/path', self.on_path, 10)
        self.tf = Buffer(); self.listener = TransformListener(self.tf, self)

    def on_map(self, message):
        self.last_map = message
        self.maps += 1
        self.occupied = sum(x >= 50 for x in message.data)
        self.known = sum(x >= 0 for x in message.data)

    def on_odom(self, message):
        self.odom = message
        self.odom_times.append(message.header.stamp.sec+message.header.stamp.nanosec/1e9)
        if len(self.odom_times) % 5 == 0:
            position = self.position()
            if position is not None:
                self.trajectory.append([self.odom_times[-1], *position])
                trail=RosPath();trail.header.frame_id='map';trail.header.stamp=message.header.stamp
                for _,x,y in self.trajectory:
                    pose=PoseStamped();pose.header=trail.header
                    pose.pose.position.x=x;pose.pose.position.y=y;pose.pose.position.z=.03
                    pose.pose.orientation.w=1.;trail.poses.append(pose)
                self.trail_pub.publish(trail)

    def on_scan(self, message):
        self.scan_times.append(message.header.stamp.sec+message.header.stamp.nanosec/1e9)

    def on_path(self, message):
        self.path = message.poses
        self.path_events += 1

    def position(self):
        if self.odom is None: return None
        try:
            t = self.tf.lookup_transform('map', 'odom', Time()).transform
        except TransformException:
            return None
        p = self.odom.pose.pose.position; q = t.rotation
        return transform_planar(p.x, p.y, 0., t.translation.x, t.translation.y,
                                yaw([q.w, q.x, q.y, q.z]))[:2]

    def publish_goal(self, goal):
        message = PoseStamped(); message.header.frame_id = 'map'
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y = goal
        message.pose.orientation.w = 1.
        self.goal_pub.publish(message)
        markers=MarkerArray()
        for i,(x,y) in enumerate(self.demo_goals):
            m=Marker();m.header=message.header;m.ns='demo_goals';m.id=i
            m.type=Marker.SPHERE;m.action=Marker.ADD;m.pose.orientation.w=1.
            m.pose.position.x=x;m.pose.position.y=y;m.pose.position.z=.15
            m.scale.x=m.scale.y=m.scale.z=.35;m.color.a=1.
            m.color.g=1. if i<self.completed else .55;m.color.r=0. if i<self.completed else 1.
            markers.markers.append(m)
            label=Marker();label.header=message.header;label.ns='demo_goal_labels';label.id=i
            label.type=Marker.TEXT_VIEW_FACING;label.action=Marker.ADD;label.pose.orientation.w=1.
            label.pose.position.x=x;label.pose.position.y=y+.5;label.pose.position.z=.4
            label.scale.z=.45;label.color.r=label.color.g=label.color.b=label.color.a=1.
            label.text=f'Goal {i+1}';markers.markers.append(label)
        self.goals_pub.publish(markers)


def frequency(values):
    if len(values) < 2 or values[-1] <= values[0]: return 0.
    return (len(values)-1)/(values[-1]-values[0])


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--goal-timeout', type=float, default=65.)
    parser.add_argument('--goals', default='[[0.8,0.0],[0.5,1.2],[2.8,0.0]]')
    parser.add_argument('--goal-tolerance',type=float,default=.18)
    parser.add_argument('--hold-seconds',type=float,default=0.)
    parser.add_argument('--reject-goal',default='[1.8,1.4]')
    parser.add_argument('--status-file',type=Path)
    cli, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args); node = Check(); results = []
    node.demo_goals=json.loads(cli.goals)
    def status(label):
        if cli.status_file:
            temporary=cli.status_file.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(label=label,goals=node.demo_goals,completed=node.completed)))
            temporary.replace(cli.status_file)
    report = dict(slam_executed=False, navigation_passed=False, oracle_odometry=True,
                  goals=results, backend='ROS2 slam_toolbox + G1-Nav algorithms + VitaBoy MuJoCo')
    try:
        deadline = time.monotonic()+35
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            if node.known > 100 and node.occupied > 0 and node.position() is not None:
                break
        report['slam_executed'] = bool(node.maps and node.occupied > 0 and node.position() is not None)
        report['bootstrap_occupied_cells'] = node.occupied
        if not report['slam_executed']: raise RuntimeError('SLAM map or map->odom unavailable')
        for goal in node.demo_goals:
            status(f'Goal {len(results)+1}/{len(node.demo_goals)} ({goal[0]:.1f}, {goal[1]:.1f}) | online map')
            start = time.monotonic(); deadline = start+cli.goal_timeout
            initial = node.position(); last_send = -math.inf; minimum = math.inf
            reached = False; saw_path = False
            while time.monotonic() < deadline:
                if time.monotonic()-last_send > 2.:
                    node.publish_goal(goal); last_send = time.monotonic()
                rclpy.spin_once(node, timeout_sec=.02)
                saw_path |= bool(node.path and math.dist(
                    (node.path[-1].pose.position.x,node.path[-1].pose.position.y),goal)<.1)
                current = node.position()
                if current is not None:
                    minimum = min(minimum, math.dist(current, goal))
                    if math.dist(current, goal) <= cli.goal_tolerance:
                        reached = True; break
            result = dict(goal=goal, start=initial, reached=reached, saw_path=saw_path,
                          minimum_error_m=minimum if math.isfinite(minimum) else None,
                          wall_duration_s=time.monotonic()-start, final=node.position())
            results.append(result); print(json.dumps(result), flush=True)
            if not reached: break
            node.completed+=1
            status(f'Goal {node.completed}/{len(node.demo_goals)} reached | settling')
            deadline=time.monotonic()+cli.hold_seconds
            while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.02)
        report['navigation_passed'] = bool(len(results) == len(json.loads(cli.goals)) and
                                           all(x['reached'] and x['saw_path'] for x in results))
        if report['navigation_passed']:
            before = node.path_events
            status('All goals reached | reject occupied goal and stop')
            node.publish_goal(json.loads(cli.reject_goal))
            deadline = time.monotonic()+5.
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.02)
                if node.path_events > before and not node.path: break
            report['occupied_or_unknown_goal_rejected'] = bool(node.path_events > before and not node.path)
            deadline = time.monotonic()+2.
            samples = []
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.02)
                if node.odom is not None and deadline-time.monotonic() < .7:
                    velocity = node.odom.twist.twist.linear
                    samples.append(math.hypot(velocity.x, velocity.y))
            report['final_stop_mean_mps'] = sum(samples)/len(samples) if samples else None
            report['stopped_after_rejected_goal'] = bool(samples and report['final_stop_mean_mps'] < .08)
            report['navigation_passed'] &= bool(report['occupied_or_unknown_goal_rejected'] and report['stopped_after_rejected_goal'])
    except Exception as error:
        report['error'] = str(error)
    finally:
        report.update(map_messages=node.maps, known_cells=node.known, occupied_cells=node.occupied,
                      odometry_hz_sim=frequency(node.odom_times), scan_hz_sim=frequency(node.scan_times))
        report['goal_tolerance_m']=cli.goal_tolerance
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)
        if node.last_map:
            m = node.last_map
            snapshot = dict(resolution=m.info.resolution, width=m.info.width, height=m.info.height,
                            origin=[m.info.origin.position.x, m.info.origin.position.y], data=list(m.data),
                            trajectory_map=node.trajectory)
            cli.output.with_suffix('.map.json').write_text(json.dumps(snapshot))
        node.destroy_node(); rclpy.shutdown()
    if not report['navigation_passed']: raise SystemExit(1)


if __name__ == '__main__': main()
