"""Local-only ROS interface for the official VitaBoy MuJoCo controller."""
import json
import math
import socket
import time
import uuid

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from builtin_interfaces.msg import Time as TimeMessage
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock as ClockMessage
from sensor_msgs.msg import Joy, JointState, LaserScan
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster
from vitaboy_navigation.protocol import VERSION, LIMITS, RUNNING_LIMITS, decode_state, finite_vector, yaw


def stamp(seconds):
    nanos = round(seconds*1e9)
    return TimeMessage(sec=nanos//1000000000, nanosec=nanos % 1000000000)


class Bridge(Node):
    def __init__(self):
        super().__init__('vitaboy_bridge')
        self.declare_parameter('running_mode', False)
        envelope=RUNNING_LIMITS if self.get_parameter('running_mode').value else LIMITS
        self.declare_parameter('max_vx', envelope[0])
        self.declare_parameter('max_vy', envelope[1])
        self.declare_parameter('max_wz', envelope[2])
        self.limits = tuple(float(self.get_parameter(name).value) for name in ('max_vx','max_vy','max_wz'))
        if any(not 0 < x <= cap for x, cap in zip(self.limits, envelope)):
            raise ValueError('Bridge limits exceed configured simulation envelope')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', 15101)); self.socket.setblocking(False)
        self.session = uuid.uuid4().hex; self.sequence = 0
        self.command = (0., 0., 0.)
        self.last_command = self.last_state = -math.inf
        self.last_seq = -1; self.last_scan = -1.
        self.clock_pub = self.create_publisher(ClockMessage, '/clock', 10)
        self.odom_pub = self.create_publisher(Odometry, '/lidar_odometry/pose_fixed', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.enable_pub = self.create_publisher(Bool, '/g1pilot/auto_enable', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Joy, '/g1pilot/auto_joy', self.on_joy, 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_twist, 10)
        # A steady clock bootstraps /clock and expires commands even if simulation stops.
        self.timer = self.create_timer(.02, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def set_command(self, values):
        try:
            values = finite_vector(values, 3)
        except ValueError:
            return
        self.command = tuple(max(-cap, min(cap, x)) for x, cap in zip(values, self.limits))
        self.last_command = time.monotonic()

    def on_twist(self, message):
        self.set_command([message.linear.x, message.linear.y, message.angular.z])

    def on_joy(self, message):
        if len(message.axes) < 3:
            return
        # Same negative-axis convention as G1-Nav; scale matches tracker limits.
        self.set_command([-message.axes[1]*self.limits[0],
                          -message.axes[0]*self.limits[1],
                          -message.axes[2]*self.limits[2]])

    def send(self, values):
        self.sequence += 1
        packet = dict(schema=VERSION, robot='vitaboy', session=self.session,
                      seq=self.sequence, command=list(values))
        self.socket.sendto(json.dumps(packet, allow_nan=False).encode(), ('127.0.0.1', 15100))

    def tick(self):
        now = time.monotonic(); latest = None
        while True:
            try:
                packet, source = self.socket.recvfrom(32768)
            except BlockingIOError:
                break
            if source[0] != '127.0.0.1':
                continue
            try:
                state = decode_state(packet)
                if state['seq'] > self.last_seq:
                    latest = state; self.last_seq = state['seq']
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
        if latest:
            self.last_state = now
            self.publish_state(latest)
        healthy = now-self.last_state <= .25
        self.enable_pub.publish(Bool(data=healthy))
        values = self.command if healthy and now-self.last_command <= .25 else (0., 0., 0.)
        self.send(values)

    def publish_state(self, state):
        timestamp = stamp(state['sim_time'])
        self.clock_pub.publish(ClockMessage(clock=timestamp))
        x, y, z = state['position']; qw, qx, qy, qz = state['quaternion_wxyz']
        heading = yaw(state['quaternion_wxyz']); c, s = math.cos(heading/2), math.sin(heading/2)
        odom = Odometry(); odom.header.stamp = timestamp; odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        odom.pose.pose.position.x = x; odom.pose.pose.position.y = y
        odom.pose.pose.orientation.w = c; odom.pose.pose.orientation.z = s
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = state['twist_body'][:2]
        odom.twist.twist.angular.z = state['twist_body'][5]
        self.odom_pub.publish(odom)
        transform = TransformStamped(); transform.header = odom.header
        transform.child_frame_id = 'base_footprint'
        transform.transform.translation.x = x; transform.transform.translation.y = y
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(transform)
        body = TransformStamped(); body.header.stamp = timestamp
        body.header.frame_id = 'base_footprint'; body.child_frame_id = 'base_link'
        body.transform.translation.z = z
        body.transform.rotation.w = c*qw+s*qz
        body.transform.rotation.x = c*qx+s*qy
        body.transform.rotation.y = c*qy-s*qx
        body.transform.rotation.z = c*qz-s*qw
        self.tf.sendTransform(body)
        laser = TransformStamped(); laser.header.stamp = timestamp
        laser.header.frame_id = 'base_footprint'; laser.child_frame_id = 'base_scan'
        laser.transform.translation.z = float(state['scan']['height'])
        laser.transform.rotation.w = 1.
        self.tf.sendTransform(laser)
        joints = JointState(); joints.header.stamp = timestamp
        joints.name = state['joint_names']; joints.position = state['joint_position']
        self.joint_pub.publish(joints)
        scan = state['scan']
        if scan['stamp'] > self.last_scan:
            self.last_scan = scan['stamp']
            message = LaserScan(); message.header.stamp = stamp(scan['stamp'])
            message.header.frame_id = 'base_scan'; message.angle_min = -math.pi
            message.angle_increment = 2*math.pi/360
            message.angle_max = message.angle_min+359*message.angle_increment
            message.scan_time = 1/15; message.time_increment = 0.
            message.range_min = .1; message.range_max = float(scan.get('range_max',8.))
            message.ranges = [float(x) if x is not None else math.inf for x in scan['ranges']]
            self.scan_pub.publish(message)

    def destroy_node(self):
        try:
            self.send((0., 0., 0.))
        finally:
            self.socket.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args); node = Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok(): raise
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
