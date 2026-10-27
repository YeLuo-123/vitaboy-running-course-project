"""G1 planning/tracking algorithms with explicit SLAM frame and freshness handling."""
import math
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from vitaboy_navigation.planning import astar_grid

import rclpy
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Joy
from tf2_ros import Buffer, TransformListener, TransformException
from vitaboy_navigation._g1.dijkstra_planner import DijkstraPlanner
from vitaboy_navigation._g1.nav2point import Nav2Point
from vitaboy_navigation.protocol import (LIMITS, transform_planar, yaw, approach_command,
                                        brake_forward, running_approach_command, brake_running)


class Frames:
    def init_frames(self):
        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_pose_wall = -math.inf
        self.latest_odom = None

    def pose_in(self, message, frame):
        if not frame or frame == message.header.frame_id:
            p, q = message.pose.pose.position, message.pose.pose.orientation
            return p.x, p.y, yaw([q.w, q.x, q.y, q.z])
        transform = self.tf_buffer.lookup_transform(frame, message.header.frame_id, Time())
        age = self.get_clock().now().nanoseconds-(transform.header.stamp.sec*10**9+transform.header.stamp.nanosec)
        if age > 500000000:
            raise ValueError('Stale map/odom transform')
        p, q = message.pose.pose.position, message.pose.pose.orientation
        t, r = transform.transform.translation, transform.transform.rotation
        return transform_planar(p.x, p.y, yaw([q.w, q.x, q.y, q.z]),
                                t.x, t.y, yaw([r.w, r.x, r.y, r.z]))

    def refresh_pose(self, frame):
        if self.latest_odom is None or time.monotonic()-self.last_pose_wall > .35:
            self.have_pose = False; return False
        try:
            self.px, self.py, self.pyaw = self.pose_in(self.latest_odom, frame)
            self.have_pose = True; return True
        except (TransformException, ValueError):
            self.have_pose = False; return False

    def cb_odom(self, message):
        self.latest_odom = message; self.last_pose_wall = time.monotonic()


class Planner(Frames, DijkstraPlanner):
    def __init__(self):
        super().__init__(); self.init_frames()
        self.destroy_subscription(self.sub_map)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_map = self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value, self.cb_map, qos)
        self.declare_parameter('running_mode',False)
        self.running=bool(self.get_parameter('running_mode').value)

    def inflate_occupancy(self,occ,w,h,r_cells,occ_th):
        # Same circular footprint and unknown-cell precedence as the G1 loops,
        # without blocking TF/odometry callbacks for seconds on a larger map.
        if r_cells<=0:return occ[:]
        grid=np.asarray(occ,dtype=np.int16).reshape(h,w)
        occupied=grid>=occ_th
        result=np.zeros_like(grid)
        if occupied.any():result[distance_transform_edt(~occupied)<=r_cells]=100
        result[grid<0]=-1
        return result.ravel().tolist()

    def dijkstra(self,start,goal):
        # A verified straight route needs no graph expansion. Large running
        # maps use A* to keep ROS TF/odometry callbacks timely.
        if self.grid_line_clear(start[:2],goal):return [start[:2],goal]
        if self.running:
            threshold=int(self.get_parameter('occ_threshold').value)
            unknown=not bool(self.get_parameter('allow_unknown').value)
            blocked=[v>=threshold or (v<0 and unknown) for v in self.occ_inf]
            return astar_grid(blocked,self.w,self.h,start,goal,float(self.get_parameter('turn_cost_gain').value))
        return super().dijkstra(start,goal)

    def periodic_replan(self):
        # Replan with current map->odom even if map pixels did not change.
        if self.goal is not None:
            self.plan_to_goal()

    def grid_line_clear(self, a, b):
        # Supercover: a shortcut touching an occupied cell corner is also rejected.
        x,y=a; endx,endy=b
        nx,ny=abs(endx-x),abs(endy-y)
        sx=1 if endx>x else -1;sy=1 if endy>y else -1
        ix=iy=0
        if self.is_occ(x,y): return False
        while ix<nx or iy<ny:
            decision=(1+2*ix)*ny-(1+2*iy)*nx
            if decision==0:
                if self.is_occ(x+sx,y) or self.is_occ(x,y+sy): return False
                x+=sx;y+=sy;ix+=1;iy+=1
            elif decision<0:
                x+=sx;ix+=1
            else:
                y+=sy;iy+=1
            if self.is_occ(x,y): return False
        return True

    def plan_to_goal(self):
        if self.map is None or not self.refresh_pose(self.map_frame):
            self.reject_path('Waiting for SLAM map and fresh map->odom transform')
            return
        if self.goal and self.goal[2] != self.map_frame:
            self.reject_path('Goal frame differs from SLAM map frame')
            return
        super().plan_to_goal()


class Tracker(Frames, Nav2Point):
    def __init__(self):
        super().__init__(); self.init_frames()
        self.scan_wall = -math.inf; self.enable_wall = -math.inf
        self.path_wall = -math.inf
        self.declare_parameter('running_mode', False)
        self.running = bool(self.get_parameter('running_mode').value)

    def cb_scan(self, message):
        if message.header.frame_id != 'base_scan':
            self.scan_points = []; return
        self.scan_wall = time.monotonic()
        super().cb_scan(message)

    def cb_auto_enable(self, message):
        self.enable_wall = time.monotonic()
        super().cb_auto_enable(message)

    def cb_path(self, message):
        self.path_wall = time.monotonic()
        super().cb_path(message)

    def stop(self):
        message = Joy(); message.axes = [0.]*8; message.buttons = [0]*14
        message.header.stamp = self.get_clock().now().to_msg()
        self.pub_joy.publish(message)

    def apply_local_avoidance(self, vx, vy, wz):
        # The G1 sum-of-beam repulsion grows with scan density and stalls this policy.
        # Keep tangential forward motion; cap closing speed to each observed surface.
        # Global inflated-map planning chooses the route; this layer brakes locally.
        vx = max(0., min(self.vx_lim, vx))
        brake=brake_running if self.running else brake_forward
        vx = brake(vx, self.scan_points, self.emergency_distance)
        return vx, 0., max(-self.wz_lim,min(self.wz_lim,wz))

    def loop(self):
        # Called by the inherited timer only after __init__ has completed.
        now = time.monotonic()
        if (not self.auto_enabled or now-self.enable_wall > .35 or
                now-self.scan_wall > .35 or now-self.path_wall > 3. or
                not self.path or not self.refresh_pose(self.path_frame)):
            self.stop(); return
        self.x, self.y, self.yaw = self.px, self.py, self.pyaw
        goal_distance = math.dist((self.x,self.y), self.path[-1])
        if goal_distance <= self.goal_tol:
            self.stop(); return
        # Retain the G1 global path and avoidance, adapt tracking to VitaBoy's walking deadzone.
        nearest = min(range(len(self.path)), key=lambda i: math.dist((self.x,self.y),self.path[i]))
        index = nearest
        lookahead = max(.7, min(2.2, .8+abs(self.latest_odom.twist.twist.linear.x)*.6)) if self.running else .40
        while index < len(self.path)-1 and math.dist((self.x,self.y),self.path[index]) < lookahead:
            index += 1
        wx, wy = self.path[index]
        self.publish_wp_marker(wx, wy)
        approach=running_approach_command if self.running else approach_command
        vx, vy, wz = approach(wx-self.x, wy-self.y, self.yaw, goal_distance,
                             (self.vx_lim,self.vy_lim,self.wz_lim))
        vx, vy, wz = self.apply_local_avoidance(vx, vy, wz)
        message = Joy(); message.header.stamp = self.get_clock().now().to_msg()
        message.axes = [max(-1.,min(1.,-vy/self.vy_lim)),
                        max(-1.,min(1.,-vx/self.vx_lim)),
                        max(-1.,min(1.,-wz/self.wz_lim)), 0.,0.,0.,0.,0.]
        message.buttons = [0]*14; message.buttons[8] = 1
        self.pub_joy.publish(message)


def spin(cls, args):
    rclpy.init(args=args); node = cls()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok(): raise
    finally:
        if isinstance(node, Tracker) and rclpy.ok(): node.stop()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


def planner_main(args=None): spin(Planner, args)
def tracker_main(args=None): spin(Tracker, args)
