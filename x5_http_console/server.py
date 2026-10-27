#!/usr/bin/env python3
"""LAN HTTP console; ROS callbacks and commands serialized on one executor."""
import argparse
import base64
import concurrent.futures
import json
import math
import os
from pathlib import Path
import queue
import secrets
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml
from scipy import ndimage as ndi
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import ParticleCloud
from std_srvs.srv import Empty
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from tf2_ros import Buffer, TransformListener
from core import FREE_LIMIT, valid_name, world_to_cell, frontier_goal
from areas import load_areas, save_area, set_target, delete_area, validate_target
from speech import Speaker

ROOT = Path(__file__).resolve().parent
MAPS = ROOT/'maps'
LOGS = ROOT/'logs'
for directory in (MAPS, LOGS): directory.mkdir(exist_ok=True)


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


class Console(Node):
    def __init__(self, dry_run=False):
        super().__init__('x5_http_console')
        self.dry_run = dry_run
        self.commands = queue.Queue()
        self.mode = 'idle'; self.process = None; self.logfile = None
        self.active_map = None
        self.last_stop = 0.
        self.last_wall_clock = time.time(); self.last_monotonic_clock = time.monotonic()
        self.stack_started = 0.; self.recovery_started = 0.; self.recovery_sequence = 0
        self.recovery_busy = False
        self.grid = None; self.info = None; self.map_time = 0.; self.map_version = 0
        self.obstacle_distance = None; self.scan_match = None; self.scan_match_time = 0.
        self.scan_time = 0.; self.odom_time = 0.; self.lease = 0.
        self.odom_linear = 0.; self.odom_angular = 0.; self.power_voltage = None
        self.base_linear = 0.; self.base_angular = 0.; self.base_command_time = 0.
        self.armed = False; self.exploring = False; self.excluded = []; self.owner = None
        self.manual = False; self.manual_time = 0.; self.manual_twist = Twist()
        self.goal_handle = None; self.pending = False; self.generation = 0
        self.goal = None; self.goal_time = 0.; self.goal_status = 'idle'
        self.distance_remaining = None; self.next_explore = 0.; self.empty_count = 0
        self.pose = None; self.plan = []; self.message = '等待启动建图或加载地图'
        self.amcl_covariance = None; self.amcl_time = 0.; self.amcl_estimate = None; self.initial_pose = None
        self.amcl_history = []; self.amcl_particles = []; self.localization_epoch = 0
        self.last_odom_motion = 0.; self.sampling_until = 0.; self.sampling_next = 0.
        self.sampling_count = 0; self.sampling_future = None
        self.speaker = Speaker()
        self.route_steps = []; self.route_index = -1; self.route_phase = 'idle'
        self.route_due = 0.; self.route_note = ''
        self.initialized = False; self.last_nav = 0.; self.nav_twist = Twist()
        self.tf = Buffer(); self.listener = TransformListener(self.tf, self)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_amcl_pose, 10)
        self.create_subscription(ParticleCloud, '/particle_cloud', self.on_particle_cloud, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Float32, '/PowerVoltage', self.on_voltage, 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_base_command, 10)
        self.create_subscription(RosPath, '/plan', self.on_plan, 1)
        self.create_subscription(Twist, '/x5/nav_cmd', self.on_velocity, 10)
        self.vel = self.create_publisher(Twist, '/x5/test_cmd' if dry_run else '/x5/gated_cmd', 10)
        self.initial_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.nomotion_client = self.create_client(Empty, '/request_nomotion_update')
        self.action = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.lifecycle = {
            name: (self.create_client(GetState, f'/{name}/get_state'),
                   self.create_client(ChangeState, f'/{name}/change_state'))
            for name in ('map_server', 'amcl')
        }
        self.create_timer(.1, self.tick)
        self.create_timer(2., self.explore_tick)
        self.create_timer(2., self.recover_localization)
        self.create_timer(.25, self.route_tick)
        self.create_timer(.25, self.sampling_tick)

    def on_map(self, msg):
        if msg.info.width*msg.info.height != len(msg.data) or msg.header.frame_id != 'map': return
        self.grid = np.array(msg.data, dtype=np.int8)
        self.info = {'width':msg.info.width,'height':msg.info.height,'resolution':msg.info.resolution,
                     'origin':[msg.info.origin.position.x,msg.info.origin.position.y,yaw(msg.info.origin.orientation)]}
        self.map_time = time.monotonic(); self.map_version += 1
        obstacles = self.grid.reshape(msg.info.height,msg.info.width) >= 65
        self.obstacle_distance = (ndi.distance_transform_edt(~obstacles)*msg.info.resolution
                                  if np.any(obstacles) else None)
        self.scan_match = None; self.scan_match_time = 0.

    def on_amcl_pose(self, msg):
        if self.mode != 'navigation' or not self.initialized or msg.header.frame_id != 'map': return
        covariance = msg.pose.covariance
        self.amcl_covariance = [max(0.,float(covariance[i])) for i in (0,7,35)]
        self.amcl_time = time.monotonic()
        pose=msg.pose.pose
        self.amcl_estimate=[pose.position.x,pose.position.y,yaw(pose.orientation)]
        if not self.amcl_history or self.amcl_time-self.amcl_history[-1][0]>=.45:
            sx,sy,sa=(math.sqrt(value) for value in self.amcl_covariance)
            self.amcl_history.append([self.amcl_time,*self.amcl_estimate,sx,sy,math.degrees(sa)])
            self.amcl_history=self.amcl_history[-180:]

    def on_particle_cloud(self, msg):
        if self.mode != 'navigation' or not self.initialized or msg.header.frame_id != 'map': return
        particles=msg.particles
        step=max(1,math.ceil(len(particles)/240))
        self.amcl_particles=[[p.pose.position.x,p.pose.position.y] for p in particles[::step][:240]]

    def sampling_active(self):
        return self.mode == 'navigation' and self.sampling_until > time.monotonic()

    def start_sampling(self):
        if self.mode != 'navigation' or not self.initialized or self.grid is None:
            raise ValueError('请先加载地图并设置初始位姿')
        if self.armed or self.pending or self.goal_handle or self.manual or self.exploring:
            raise ValueError('请先停止运动任务，再静止采样')
        if abs(self.odom_linear)>.015 or abs(self.odom_angular)>.03:
            raise ValueError('请保持小车静止再采样')
        if not self.nomotion_client.service_is_ready():
            raise ValueError('AMCL 静止更新服务尚未就绪')
        self.sampling_until=time.monotonic()+12
        self.sampling_next=time.monotonic()
        self.sampling_count=0;self.sampling_future=None
        self.localization_epoch+=1
        self.amcl_history=[];self.amcl_particles=[]
        self.message='正在静止采样：AMCL 将用新雷达扫描更新定位，不会驱动车轮'

    def sampling_tick(self):
        if self.sampling_until and time.monotonic()>=self.sampling_until:
            self.sampling_until=0.
            self.message='静止采样结束：请核对地图箭头和实体小车是否在同一房间、朝向是否一致'
        if not self.sampling_active(): return
        if (self.armed or time.monotonic()-self.scan_time>2 or time.monotonic()-self.odom_time>2
                or abs(self.odom_linear)>.015 or abs(self.odom_angular)>.03):
            self.sampling_until=0.
            self.message='静止采样已停止：小车移动或传感器数据中断'
            return
        now=time.monotonic()
        if now<self.sampling_next or not self.nomotion_client.service_is_ready():return
        if self.sampling_future is not None:
            if not self.sampling_future.done():return
            try:self.sampling_future.result()
            except Exception as exc:
                self.sampling_until=0.;self.message='静止采样失败：'+str(exc);return
        self.sampling_future=self.nomotion_client.call_async(Empty.Request())
        self.sampling_count+=1;self.sampling_next=now+1

    def localization_uncertainty(self):
        if self.mode != 'navigation' or not self.initialized or self.pose is None or self.amcl_covariance is None:
            return {'fresh':False,'age_sec':None,'estimate':self.amcl_estimate,'sigma_x':None,'sigma_y':None,
                    'sigma_yaw_deg':None,'scan_match':self.scan_match,'sampling':self.sampling_active(),
                    'sample_count':self.sampling_count,'uncertain':True}
        age=time.monotonic()-self.amcl_time
        fresh=age<3
        sx,sy,sa=(math.sqrt(value) for value in self.amcl_covariance)
        moving_without_update=age>3 and self.last_odom_motion>self.amcl_time and time.monotonic()-self.last_odom_motion<3
        active_goal_without_update=self.armed and age>5 and time.monotonic()-self.goal_time>8
        map_mismatch=(self.scan_match is not None and time.monotonic()-self.scan_match_time<2
                      and self.scan_match<.70)
        return {'fresh':fresh,'age_sec':round(age,1),'estimate':self.amcl_estimate,'sigma_x':sx,'sigma_y':sy,
                'sigma_yaw_deg':math.degrees(sa),'scan_match':self.scan_match,
                'sampling':self.sampling_active(),'sample_count':self.sampling_count,
                'uncertain':self.sampling_active() or moving_without_update or active_goal_without_update or map_mismatch
                            or max(sx,sy)>.35 or math.degrees(sa)>25}

    def on_scan(self, msg):
        ranges=np.asarray(msg.ranges,dtype=np.float64)
        valid=np.isfinite(ranges)&(ranges>=max(.2,msg.range_min))&(ranges<=min(6.,msg.range_max))
        if not np.any(valid): return
        self.scan_time = time.monotonic()
        if (self.mode!='navigation' or not self.initialized or self.pose is None
                or self.obstacle_distance is None or self.info is None): return
        try: laser=self.tf.lookup_transform('base_footprint',msg.header.frame_id,Time()).transform
        except Exception: return
        indices=np.flatnonzero(valid)[::3]
        if len(indices)<30:return
        angle=msg.angle_min+indices*msg.angle_increment
        distance=ranges[indices]
        x,y,heading=self.pose
        laser_heading=heading+yaw(laser.rotation)
        x+=math.cos(heading)*laser.translation.x-math.sin(heading)*laser.translation.y
        y+=math.sin(heading)*laser.translation.x+math.cos(heading)*laser.translation.y
        endpoint_x=x+distance*np.cos(laser_heading+angle)
        endpoint_y=y+distance*np.sin(laser_heading+angle)
        ox,oy,map_yaw=self.info['origin']; resolution=self.info['resolution']
        dx,dy=endpoint_x-ox,endpoint_y-oy
        col=np.floor((math.cos(map_yaw)*dx+math.sin(map_yaw)*dy)/resolution).astype(int)
        row=np.floor((-math.sin(map_yaw)*dx+math.cos(map_yaw)*dy)/resolution).astype(int)
        inside=(col>=0)&(col<self.info['width'])&(row>=0)&(row<self.info['height'])
        matches=np.zeros(len(indices),dtype=bool)
        matches[inside]=self.obstacle_distance[row[inside],col[inside]]<.15
        score=float(np.mean(matches))
        self.scan_match=score if self.scan_match is None else .7*self.scan_match+.3*score
        self.scan_match_time=self.scan_time

    def on_odom(self, msg):
        self.odom_time = time.monotonic()
        self.odom_linear = msg.twist.twist.linear.x
        self.odom_angular = msg.twist.twist.angular.z
        if abs(self.odom_linear)>.015 or abs(self.odom_angular)>.03:self.last_odom_motion=self.odom_time
    def on_voltage(self, msg): self.power_voltage = float(msg.data)
    def on_base_command(self, msg):
        self.base_linear = msg.linear.x; self.base_angular = msg.angular.z
        self.base_command_time = time.monotonic()
    def on_plan(self, msg): self.plan = [[p.pose.position.x,p.pose.position.y] for p in msg.poses][::2]
    def on_velocity(self, msg): self.nav_twist = msg; self.last_nav = time.monotonic()

    def ready(self):
        now = time.monotonic()
        return (self.mode in ('mapping','navigation') and self.process is not None and self.process.poll() is None
                and now-self.scan_time < 1.5 and now-self.odom_time < 1.5 and self.pose is not None
                and self.grid is not None and self.action.server_is_ready()
                and (self.mode != 'navigation' or self.initialized)
                and (self.mode != 'mapping' or now-self.map_time < 5))

    def manual_ready(self):
        now = time.monotonic()
        return (self.mode == 'mapping' and self.process is not None and self.process.poll() is None
                and now-self.scan_time < 1.5 and now-self.odom_time < 1.5
                and self.pose is not None and self.grid is not None and now-self.map_time < 5)

    def diagnostics(self):
        now = time.monotonic()
        return {
            'stack_ok': self.process is not None and self.process.poll() is None,
            'scan_ok': now-self.scan_time < 1.5,
            'odom_ok': now-self.odom_time < 1.5,
            'map_ok': self.grid is not None and (self.mode != 'mapping' or now-self.map_time < 5),
            'pose_ok': self.pose is not None,
            'nav_ok': self.action.server_is_ready(),
            'initialpose_ok': self.initial_pub.get_subscription_count() > 0,
            'localized': self.initialized if self.mode == 'navigation' else True,
        }

    def tick(self):
        wall_now,steady_now=time.time(),time.monotonic()
        clock_jump=abs((wall_now-self.last_wall_clock)-(steady_now-self.last_monotonic_clock))>2
        self.last_wall_clock,self.last_monotonic_clock=wall_now,steady_now
        if clock_jump and self.mode == 'navigation':
            self.stop_stack()
            self.message='系统时钟刚发生跳变，导航已停止；等待时钟同步后重新加载地图并设置初始位姿'
            return
        try:
            tf = self.tf.lookup_transform('map','base_footprint',Time())
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec/1e9
            age = self.get_clock().now().nanoseconds/1e9-stamp
            self.pose = [tf.transform.translation.x,tf.transform.translation.y,yaw(tf.transform.rotation)] if abs(age) < 2 else None
        except Exception: self.pose = None
        if self.process and self.process.poll() is not None and self.mode != 'error':
            self.stop(); self.mode = 'error'; self.message = 'ROS 启动进程已退出，请检查日志'
        now = time.monotonic()
        if (self.mode == 'navigation' and self.grid is not None and not self.initialized
                and self.message.startswith('正在启动 ROS')):
            self.message = '地图已加载，请在地图上设置机器人当前的初始位姿'
        if self.mode == 'mapping' and self.manual_ready() and self.message.startswith('正在启动 ROS'):
            self.message = 'Cartographer 地图就绪：可启用键盘控制'
        if self.ready() and (self.message.startswith('正在启动 ROS') or self.message.startswith('Cartographer 地图就绪')):
            self.message = '建图和导航就绪：可启用键盘控制或自主探索' if self.mode == 'mapping' else '导航就绪：可以在地图上选择目标'
        if self.armed and (now-self.lease > 3 or not (self.manual_ready() if self.manual else self.ready())
                           or (self.mode == 'navigation' and self.localization_uncertainty()['uncertain'])):
            self.stop(); self.message = '已停止：控制页面断连或传感器/定位未就绪、不稳定'
        if self.manual:
            self.vel.publish(self.manual_twist if now-self.manual_time < .4 else Twist())
        elif self.armed and now-self.last_nav < .5:
            v = Twist()
            v.linear.x = max(-.15,min(.15,self.nav_twist.linear.x))
            v.linear.y = max(-.10,min(.10,self.nav_twist.linear.y))
            v.angular.z = max(-.4,min(.4,self.nav_twist.angular.z))
            self.vel.publish(v)
        elif self.mode != 'idle': self.vel.publish(Twist())
        for _ in range(8):
            try: task, data, future = self.commands.get_nowait()
            except queue.Empty: break
            if not future.set_running_or_notify_cancel(): continue
            try: future.set_result(self.dispatch(task,data))
            except Exception as exc:
                self.get_logger().warning(str(exc)); future.set_exception(exc)

    def stop(self):
        self.armed = False; self.exploring = False; self.manual = False; self.generation += 1
        self.sampling_until = 0.
        if self.goal_handle: self.goal_handle.cancel_goal_async()
        self.goal_handle = None; self.pending = False
        self.goal_status = 'stopped'; self.vel.publish(Twist()); self.nav_twist = Twist(); self.last_nav = 0
        self.manual_twist = Twist(); self.manual_time = 0.
        self.owner = None; self.distance_remaining = None; self.plan = []
        if self.route_phase in ('announcing','running','dwell'):
            self.route_phase='stopped'; self.route_note='任务已取消'
        self.speaker.cancel()

    def manual_start(self, client):
        if not self.manual_ready():
            raise ValueError('请先启动 Cartographer 建图并等待传感器和地图就绪')
        if self.exploring or self.pending or self.goal_handle:
            raise ValueError('请先停止自主探索或导航任务')
        if self.manual and self.owner != client:
            raise ValueError('键盘控制正由另一页面使用')
        self.owner = client; self.manual = True; self.armed = True
        self.lease = time.monotonic(); self.manual_time = 0.; self.manual_twist = Twist()
        self.message = '键盘控制已启用：按住 W/S 前后、A/D 左右转；松开即停'

    def manual_drive(self, data):
        if not self.manual or not self.armed or data.get('_client') != self.owner:
            raise ValueError('请先在当前页面启用键盘控制')
        if not self.manual_ready():
            self.stop(); raise ValueError('传感器或定位未就绪，已停止运动')
        forward, turn = data.get('forward'), data.get('turn')
        if type(forward) is not int or forward not in (-1, 0, 1):
            raise ValueError('前后方向无效')
        if type(turn) is not int or turn not in (-1, 0, 1):
            raise ValueError('转向方向无效')
        command = Twist()
        command.linear.x = .12 if forward > 0 else (-.10 if forward < 0 else 0.)
        command.angular.z = .35 * turn
        self.manual_twist = command
        self.manual_time = self.lease = time.monotonic()

    def stop_stack(self):
        self.stop()
        if self.process:
            try:
                os.killpg(self.process.pid,signal.SIGINT)
                self.process.wait(timeout=7)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGTERM)
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid,signal.SIGKILL); self.process.wait(timeout=2)
            except ProcessLookupError: pass
            self.process = None
        if self.logfile: self.logfile.close(); self.logfile = None
        self.mode = 'idle'; self.grid = None; self.info = None; self.plan = []; self.pose = None
        self.obstacle_distance = None; self.scan_match = None; self.scan_match_time = 0.
        self.amcl_covariance = None; self.amcl_time = 0.;self.amcl_estimate=None;self.initial_pose=None
        self.amcl_history=[];self.amcl_particles=[];self.sampling_count=0
        self.active_map = None
        self.initialized = False; self.tf.clear()
        self.last_stop = time.monotonic()
        self.recovery_sequence += 1; self.recovery_busy = False

    def start_stack(self, mode, map_name=None):
        if self.mode != 'idle': raise ValueError('请先停止当前模式，再切换模式')
        if time.time()<1700000000:
            raise ValueError('小车系统时钟尚未同步，请联网等待正确时间后再启动 ROS')
        owned_nodes = {'wheeltec_robot','wheeltec_robot_node','cartographer_node','amcl','controller_server','lslidar_driver_node'}
        conflicts = owned_nodes.intersection(name for name, _ in self.get_node_names_and_namespaces())
        # DDS discovery can briefly retain nodes after ros2 launch exits.
        if conflicts and time.monotonic()-self.last_stop < 15:
            deadline = time.monotonic()+9
            while conflicts and time.monotonic() < deadline:
                time.sleep(.25)
                conflicts = owned_nodes.intersection(name for name, _ in self.get_node_names_and_namespaces())
        if conflicts: raise ValueError('检测到 ROS 节点仍存在（关闭模式后请等数秒重试）：'+', '.join(sorted(conflicts)))
        args = ['ros2','launch',str(ROOT/'stack.launch.py'),'mode:='+mode]
        if mode == 'navigation':
            name = valid_name(map_name)
            path = MAPS/name/'map.yaml'
            if not path.is_file(): raise ValueError('地图不存在')
            args.append('map:='+str(path))
        self.logfile = open(LOGS/'stack.log','a',buffering=1)
        self.logfile.write('\n=== '+time.strftime('%Y-%m-%d %H:%M:%S')+' '+mode+' ===\n')
        self.process = subprocess.Popen(args,stdout=self.logfile,stderr=subprocess.STDOUT,start_new_session=True)
        self.mode = mode; self.message = '正在启动 ROS，请等待地图、传感器和定位就绪'
        self.active_map = name if mode == 'navigation' else None
        self.amcl_covariance = None; self.amcl_time = 0.;self.amcl_estimate=None;self.initial_pose=None
        self.scan_match = None; self.scan_match_time = 0.
        self.amcl_history=[];self.amcl_particles=[];self.sampling_count=0;self.sampling_until=0.
        self.route_steps=[];self.route_index=-1;self.route_phase='idle';self.route_note=''
        self.stack_started = time.monotonic(); self.recovery_busy = False
        self.excluded = []; self.empty_count = 0; self.goal = None
        self.goal_status = 'idle'; self.scan_time = 0.; self.odom_time = 0.

    def recover_localization(self):
        """Complete Nav2 localization activation if its startup manager stalled."""
        now = time.monotonic()
        if (self.mode != 'navigation' or self.process is None or self.process.poll() is not None
                or now-self.stack_started < 12):
            return
        if self.recovery_busy and now-self.recovery_started < 15: return
        if now-self.recovery_started < 10: return
        self.recovery_started = now; self.recovery_busy = True
        self.recovery_sequence += 1
        self._recover_node(0, self.recovery_sequence)

    def _recover_node(self, index, sequence):
        if sequence != self.recovery_sequence or self.mode != 'navigation': return
        names = ('map_server', 'amcl')
        if index == len(names):
            self.recovery_busy = False
            return
        name = names[index]
        state_client, change_client = self.lifecycle[name]
        if not state_client.service_is_ready():
            self._recover_node(index+1, sequence)
            return
        future = state_client.call_async(GetState.Request())
        def got_state(result):
            if sequence != self.recovery_sequence or self.mode != 'navigation': return
            try: state = result.result().current_state.id
            except Exception as exc:
                self.get_logger().warning(f'{name} 状态查询失败：{exc}')
                self._recover_node(index+1, sequence); return
            if state != State.PRIMARY_STATE_INACTIVE or not change_client.service_is_ready():
                self._recover_node(index+1, sequence); return
            request = ChangeState.Request()
            request.transition.id = Transition.TRANSITION_ACTIVATE
            change = change_client.call_async(request)
            def activated(done):
                if sequence != self.recovery_sequence or self.mode != 'navigation': return
                try:
                    if done.result().success:
                        self.get_logger().info(f'{name} 已自动激活')
                    else:
                        self.get_logger().warning(f'{name} 自动激活失败')
                except Exception as exc:
                    self.get_logger().warning(f'{name} 自动激活异常：{exc}')
                self._recover_node(index+1, sequence)
            change.add_done_callback(activated)
        future.add_done_callback(got_state)

    def send_goal(self, x, y, theta):
        if not all(math.isfinite(v) for v in (x,y,theta)): raise ValueError('坐标必须是有限数值')
        if not self.ready(): raise ValueError('导航尚未就绪：检查地图、雷达、里程计及初始位姿')
        if self.mode == 'navigation' and self.localization_uncertainty()['uncertain']:
            raise ValueError('定位不可靠：请核对激光与地图匹配度、实际位置和车头朝向，等待定位收敛')
        if self.pending or self.goal_handle: raise ValueError('已有导航任务，请先停止任务')
        col,row = world_to_cell(x,y,self.info)
        if not (0<=col<self.info['width'] and 0<=row<self.info['height']): raise ValueError('目标超出地图')
        value = self.grid[row*self.info['width']+col]
        if not 0<=value<=FREE_LIMIT: raise ValueError('目标必须位于已知空闲区域')
        self.armed = True
        if not self.exploring: self.lease = time.monotonic()
        self.goal = [x,y,theta]; self.goal_time = time.monotonic(); self.goal_status = 'pending'
        request = NavigateToPose.Goal(); request.pose.header.frame_id = 'map'
        request.pose.header.stamp = self.get_clock().now().to_msg()
        request.pose.pose.position.x = x; request.pose.pose.position.y = y
        request.pose.pose.orientation.z = math.sin(theta/2); request.pose.pose.orientation.w = math.cos(theta/2)
        generation = self.generation; self.pending = True
        future = self.action.send_goal_async(request, feedback_callback=self.feedback)
        future.add_done_callback(lambda f:self.accepted(f,generation))

    def feedback(self, msg): self.distance_remaining = float(msg.feedback.distance_remaining)

    def accepted(self, future, generation):
        try: handle = future.result()
        except Exception as exc:
            if generation == self.generation:
                self.pending = False; self.goal_status = 'failed'; self.message = str(exc); self.armed = False
                if self.route_phase == 'running': self.route_fail('导航目标发送失败：'+str(exc))
            return
        if generation != self.generation:
            if handle.accepted: handle.cancel_goal_async()
            return
        self.pending = False
        if not handle.accepted:
            self.goal_status = 'rejected'; self.armed = False
            if self.goal: self.excluded.append(self.goal[:2])
            if self.route_phase == 'running': self.route_fail('导航目标被 Nav2 拒绝')
            return
        self.goal_handle = handle; self.goal_status = 'executing'
        handle.get_result_async().add_done_callback(lambda f:self.finished(f,generation))

    def finished(self, future, generation):
        if generation != self.generation: return
        self.goal_handle = None
        try: status = future.result().status
        except Exception: status = 6
        self.goal_status = {4:'succeeded',5:'canceled',6:'failed'}.get(status,'failed')
        if self.goal: self.excluded.append(self.goal[:2])
        self.next_explore = time.monotonic()+3
        self.vel.publish(Twist())
        if not self.exploring: self.armed = False
        if self.route_phase == 'running':
            if status != 4:
                self.route_fail('导航到 '+self.route_steps[self.route_index]['name']+' 失败，任务已停止')
            else:
                current=self.route_steps[self.route_index]['name']
                if self.route_index+1 == len(self.route_steps):
                    announcement=f'已到达{current}，任务完成'
                    self.route_phase='complete';self.route_note=announcement;self.owner=None
                    self.message=announcement
                    try:self.speaker.say(announcement)
                    except Exception as exc:self.message=announcement+'；播报失败：'+str(exc)
                else:
                    following=self.route_steps[self.route_index+1]['name']
                    announcement=f'已到达{current}，现在前往{following}'
                    self.route_phase='dwell';self.route_due=0.
                    self.route_note=announcement;self.message=announcement+'；停留至少 3 秒'
                    try:self.speaker.say(announcement)
                    except Exception as exc:self.route_fail('语音播报失败：'+str(exc))

    def route_fail(self, reason):
        self.stop()
        self.route_phase='failed';self.route_note=reason;self.message=reason

    def start_route(self, ids, client):
        if self.mode != 'navigation' or not self.active_map or not self.ready():
            raise ValueError('请先加载地图、设置初始位姿并等待导航就绪')
        if self.localization_uncertainty()['uncertain']:
            raise ValueError('定位仍不稳定，请重新设置初始位姿，暂不能开始多点导航')
        if self.manual or self.exploring or self.pending or self.goal_handle or self.armed:
            raise ValueError('请先停止当前运动任务')
        if self.route_phase in ('announcing','running','dwell'):
            raise ValueError('已有多点导航任务')
        if (not isinstance(ids,list) or not 1<=len(ids)<=30
                or any(not isinstance(identifier,str) for identifier in ids)
                or len(ids)!=len(set(ids))):
            raise ValueError('请选择 1–30 个不重复的区域目标点')
        areas={area['id']:area for area in load_areas(MAPS/self.active_map)}
        steps=[]
        for identifier in ids:
            if not isinstance(identifier,str) or identifier not in areas:
                raise ValueError('路线包含不存在的区域')
            area=areas[identifier]
            if area.get('target') is None: raise ValueError(area['name']+' 尚未设置目标点')
            target=validate_target(area['target'],area['polygon'],self.info,self.grid)
            steps.append({'id':identifier,'name':area['name'],'target':target})
        if not self.speaker.available(): raise ValueError('小车离线语音播报不可用')
        self.route_steps=steps;self.route_index=0;self.route_phase='announcing'
        self.route_note='自主导航开始，目前前往'+steps[0]['name']
        self.owner=client;self.lease=time.monotonic();self.message=self.route_note
        try:self.speaker.say(self.route_note)
        except Exception:
            self.route_phase='failed';self.owner=None
            raise

    def route_tick(self):
        if self.route_phase not in ('announcing','running','dwell'): return
        if time.monotonic()-self.lease>3 or not self.ready():
            self.route_fail('页面断连或导航未就绪，多点任务已停止')
            return
        if self.route_phase == 'running': return
        speech=self.speaker.status()
        if speech['busy']: return
        if speech['error']:
            self.route_fail('语音播报失败：'+speech['error'])
            return
        if self.route_phase == 'dwell':
            if not self.route_due:
                self.route_due=time.monotonic()+3
                return
            if time.monotonic()<self.route_due:return
            self.route_index+=1
        target=self.route_steps[self.route_index]['target']
        try:
            self.send_goal(target['x'],target['y'],target['yaw'])
            self.route_phase='running'
            self.message='正在前往'+self.route_steps[self.route_index]['name']
        except Exception as exc:
            self.route_fail('无法前往'+self.route_steps[self.route_index]['name']+'：'+str(exc))

    def explore_tick(self):
        if not self.exploring: return
        if time.monotonic()-self.lease > 3 or not self.ready(): return
        if self.goal_handle or self.pending:
            if time.monotonic()-self.goal_time > 90:
                goal = self.goal
                self.stop()
                if goal: self.excluded.append(goal[:2])
                self.message = '探索目标超时，已停止；可重新开始探索'
            return
        if time.monotonic() < self.next_explore: return
        goal = frontier_goal(self.grid,self.info,self.pose,self.excluded)
        if goal:
            self.empty_count = 0; self.send_goal(*goal); self.message = '正在自主探索未知区域边界'
        else:
            self.empty_count += 1
            self.message = '暂未找到可达前沿，正在复查'
            if self.empty_count >= 4:
                self.stop(); self.message = '探索结束：未发现可达新前沿，可保存地图；不代表所有区域均已覆盖'

    def save_map(self, name):
        name = valid_name(name)
        if self.mode != 'mapping' or self.grid is None or time.monotonic()-self.map_time >= 5:
            raise ValueError('请先启动建图并等待实时地图更新')
        if self.armed: raise ValueError('请先停止小车，再保存地图')
        directory = MAPS/name
        if directory.exists(): raise ValueError('地图名称已存在，请使用新名称')
        # Snapshot serialization; directory rename exposes a complete pair atomically.
        tmp = MAPS/('.tmp-'+secrets.token_hex(6)); tmp.mkdir()
        a = self.grid.reshape(self.info['height'],self.info['width'])
        # Nav2 raw mode preserves Cartographer's -1/0..100 occupancy values.
        # Trinary thresholds discarded its common free value (about 31), and
        # the old 205 "unknown" pixel was read back as free with free_thresh=.25.
        pixels = a.astype(np.uint8)
        with (tmp/'map.pgm').open('wb') as f:
            f.write(f'P5\n{self.info["width"]} {self.info["height"]}\n255\n'.encode()); f.write(np.flipud(pixels).tobytes())
        metadata = {'image':'map.pgm','mode':'raw','resolution':self.info['resolution'],'origin':self.info['origin'],'negate':0,'occupied_thresh':.65,'free_thresh':.25}
        (tmp/'map.yaml').write_text(yaml.safe_dump(metadata),encoding='utf-8')
        tmp.rename(directory)
        self.message = '地图已保存：'+name
        return {'name':name,'path':str(directory/'map.yaml')}

    def dispatch(self, task, data):
        if task == 'status':
            diagnostics = self.diagnostics()
            return {'mode':self.mode,'ready':self.ready(),'manual_ready':self.manual_ready(),'armed':self.armed,'exploring':self.exploring,'manual':self.manual,'pose':self.pose,
                'scan_ok':diagnostics['scan_ok'],'odom_ok':diagnostics['odom_ok'],'map_version':self.map_version,
                'goal':self.goal,'goal_status':self.goal_status,'distance_remaining':self.distance_remaining,
                'message':self.message,'plan':self.plan,'dry_run':self.dry_run,'checks':diagnostics,
                'drive':{'command_linear':self.base_linear if time.monotonic()-self.base_command_time<1 else 0.,
                         'command_angular':self.base_angular if time.monotonic()-self.base_command_time<1 else 0.,
                         'measured_linear':self.odom_linear if diagnostics['odom_ok'] else None,
                         'measured_angular':self.odom_angular if diagnostics['odom_ok'] else None,
                         'power_voltage':self.power_voltage},
                'active_map':self.active_map,
                'localization':self.localization_uncertainty(),
                'route':{'phase':self.route_phase,'index':self.route_index,'steps':[{'id':step['id'],'name':step['name']} for step in self.route_steps],
                         'note':self.route_note},'speech':self.speaker.status()}
        if task == 'map':
            if self.grid is None: return {'info':None}
            return {'info':self.info,'version':self.map_version,'data':base64.b64encode(self.grid.tobytes()).decode()}
        if task == 'localization':
            return {'epoch':self.localization_epoch,'history':[[round(t-self.amcl_history[0][0],2),*values]
                    for t,*values in self.amcl_history] if self.amcl_history else [],
                    'particles':self.amcl_particles,'initial_pose':self.initial_pose,
                    'quality':self.localization_uncertainty()}
        if task == 'maps': return {'maps':sorted(p.parent.name for p in MAPS.glob('*/map.yaml') if not p.parent.name.startswith('.'))}
        if task == 'areas':
            map_name=valid_name(data.get('name'))
            if not (MAPS/map_name/'map.yaml').is_file(): raise ValueError('地图不存在')
            return {'name':map_name,'areas':load_areas(MAPS/map_name)}
        if task == 'heartbeat':
            if self.owner is None or data.get('_client') == self.owner: self.lease = time.monotonic()
        elif task == 'mapping/start': self.start_stack('mapping')
        elif task == 'navigation/start': self.start_stack('navigation',data.get('name'))
        elif task == 'localization/sample': self.start_sampling()
        elif task == 'mode/stop': self.stop_stack(); self.message = '当前模式已关闭'
        elif task == 'stop': self.stop(); self.message = '已停止运动及自主探索'
        elif task == 'manual/start': self.manual_start(data.get('_client'))
        elif task == 'manual/drive': self.manual_drive(data)
        elif task == 'manual/stop':
            if self.manual:
                if data.get('_client') != self.owner:
                    raise ValueError('键盘控制正由另一页面使用')
                self.stop(); self.message = '键盘控制已结束，小车已停止'
        elif task in ('areas/save','areas/target','areas/delete'):
            if self.mode != 'navigation' or self.active_map != data.get('name') or self.info is None or self.grid is None:
                raise ValueError('请先加载要标注的地图')
            if self.route_phase in ('announcing','running','dwell'):
                raise ValueError('多点导航进行中，不能修改区域')
            directory=MAPS/self.active_map
            if task == 'areas/save':
                return {'area':save_area(directory,data,self.info,self.grid)}
            if task == 'areas/target':
                return {'area':set_target(directory,data.get('id'),data.get('target'),self.info,self.grid)}
            return delete_area(directory,data.get('id'))
        elif task == 'route/start': self.start_route(data.get('ids'),data.get('_client'))
        elif task == 'route/stop':
            if self.route_phase in ('announcing','running','dwell'):
                self.stop();self.message='多点导航已取消'
        elif task == 'explore/start':
            if self.mode != 'mapping' or not self.ready(): raise ValueError('请先启动建图并等待导航就绪')
            if self.goal_handle or self.pending or self.manual: raise ValueError('请先停止键盘控制或当前导航任务')
            self.owner = data.get('_client'); self.exploring = True; self.armed = True; self.lease = time.monotonic(); self.empty_count = 0
        elif task == 'goal':
            if self.exploring: raise ValueError('请先停止自主探索')
            if self.manual: raise ValueError('请先结束键盘控制')
            if self.route_phase in ('announcing','running','dwell'): raise ValueError('请先停止多点导航任务')
            self.owner = data.get('_client')
            self.send_goal(float(data['x']),float(data['y']),float(data.get('yaw',0)))
        elif task == 'initialpose':
            if self.mode != 'navigation': raise ValueError('仅在已加载地图的导航模式下设置初始位姿')
            x,y,theta = float(data['x']),float(data['y']),float(data.get('yaw',0))
            if not all(math.isfinite(v) for v in (x,y,theta)): raise ValueError('坐标必须为有限数值')
            if self.grid is None or self.info is None: raise ValueError('请等待地图加载完成')
            if self.initial_pub.get_subscription_count() == 0: raise ValueError('定位节点尚未就绪，请稍后重试')
            col,row = world_to_cell(x,y,self.info)
            if not (0 <= col < self.info['width'] and 0 <= row < self.info['height']):
                raise ValueError('初始位置超出地图')
            if not 0 <= self.grid[row*self.info['width']+col] <= FREE_LIMIT:
                raise ValueError('初始位置必须位于已知空闲区域')
            self.stop()
            msg = PoseWithCovarianceStamped(); msg.header.frame_id = 'map'; msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x=x; msg.pose.pose.position.y=y
            msg.pose.pose.orientation.z=math.sin(theta/2); msg.pose.pose.orientation.w=math.cos(theta/2)
            # The operator selects the robot centre and heading on the map.
            # Keep particles near that room instead of spreading them across adjacent rooms.
            msg.pose.covariance[0]=.04; msg.pose.covariance[7]=.04; msg.pose.covariance[35]=.03
            self.amcl_covariance = None; self.amcl_time = 0.
            self.scan_match = None; self.scan_match_time = 0.
            self.initial_pub.publish(msg); self.initialized=True; self.message='初始位姿已发送，请确认地图位置和朝向'
            self.amcl_estimate=None;self.amcl_history=[];self.amcl_particles=[];self.initial_pose=[x,y,theta]
            self.sampling_count=0;self.sampling_until=0.
            try:self.start_sampling()
            except ValueError as exc:self.message='初始位姿已发送；静止采样暂不可用：'+str(exc)
            self.amcl_history.insert(0,[time.monotonic(),x,y,theta,.2,.2,math.degrees(math.sqrt(.03))])
        elif task == 'maps/save': return self.save_map(data.get('name'))
        else: raise ValueError('未知接口')
        return {'ok':True}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8080); parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(); rclpy.init(); node=Console(args.dry_run)
    executor=SingleThreadedExecutor(); executor.add_node(node)
    spin=threading.Thread(target=executor.spin,daemon=True); spin.start()
    token=secrets.token_urlsafe(32)
    guard = None
    if not args.dry_run:
        guard = subprocess.Popen(['/usr/bin/python3',str(ROOT/'velocity_guard.py')])
    def command(task,data):
        f=concurrent.futures.Future(); node.commands.put((task,data,f))
        try: return f.result(timeout=20)
        except concurrent.futures.TimeoutError: f.cancel(); raise ValueError('操作超时，请检查状态')
    class Handler(BaseHTTPRequestHandler):
        def reply(self,code,body,kind='application/json; charset=utf-8'):
            payload=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(code); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY'); self.end_headers()
            try:self.wfile.write(payload)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            parsed=urlparse(self.path);path=parsed.path
            if path=='/': self.reply(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
            elif path=='/api/session': self.reply(200,{'token':token,'client':secrets.token_urlsafe(18)})
            elif path in ('/api/status','/api/map','/api/maps','/api/localization'):
                try:self.reply(200,command(path[5:],{}))
                except Exception as e:self.reply(503,{'error':str(e)})
            elif path=='/api/areas':
                try:self.reply(200,command('areas',{'name':parse_qs(parsed.query).get('name',[None])[0]}))
                except Exception as e:self.reply(400,{'error':str(e)})
            else:self.reply(404,{'error':'不存在'})
        def do_POST(self):
            # Same-origin token prevents drive-by cross-site commands; LAN is the trust boundary.
            if not secrets.compare_digest(self.headers.get('X-Control-Token',''),token):
                return self.reply(403,{'error':'控制令牌无效，请刷新页面'})
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'): return self.reply(403,{'error':'跨域控制已拒绝'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8192: raise ValueError('请求体大小无效')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict): raise ValueError('需要 JSON 对象')
                data['_client'] = self.headers.get('X-Control-Client','api')
                path=urlparse(self.path).path
                allowed={'heartbeat','mapping/start','navigation/start','localization/sample','mode/stop','stop','manual/start','manual/drive','manual/stop','explore/start','goal','initialpose','maps/save','areas/save','areas/target','areas/delete','route/start','route/stop'}
                if not path.startswith('/api/') or path[5:] not in allowed: return self.reply(404,{'error':'不存在'})
                self.reply(200,command(path[5:],data))
            except Exception as e:self.reply(400,{'error':str(e)})
        def log_message(self,fmt,*args):
            if not any(path in str(args) for path in ('/api/status','/api/heartbeat','/api/map','/api/manual/drive')):
                super().log_message(fmt,*args)
    http=ThreadingHTTPServer(('0.0.0.0',args.port),Handler)
    def shutdown(*_):
        threading.Thread(target=http.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown); signal.signal(signal.SIGINT,shutdown)
    print(f'X5 console listening on :{args.port}, dry_run={args.dry_run}',flush=True)
    try:http.serve_forever()
    finally:
        http.server_close()
        try: command('mode/stop',{})
        except Exception: node.stop_stack()
        if guard:
            guard.send_signal(signal.SIGINT)
            try: guard.wait(timeout=3)
            except subprocess.TimeoutExpired: guard.kill()
        executor.shutdown(timeout_sec=3); spin.join(timeout=3)
        node.destroy_node(); rclpy.shutdown()

if __name__=='__main__':main()
