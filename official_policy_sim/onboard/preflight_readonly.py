"""Observe live hardware and DDS ownership. No command publishers or action clients."""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.convert import message_to_ordereddict
from interaction_msgs.msg import LowState, RobotState
from sensor_msgs.msg import Imu
from sys_monitor_msgs.msg import MotorMonitor

# Native joint-name/ID mapping verified against the factory MMI joints_config.csv.
# Body order matches the pinned official V1 configuration (29 body + 2 head).
EXPECTED = (list(range(10006, 10000, -1)) + list(range(20006, 20000, -1))
            + [50001, 50002, 50003] + list(range(30007, 30000, -1))
            + list(range(40007, 40000, -1)) + [60001, 60002])


def evaluate(last, counts, ages, links_first, links_last, publishers, subscribers, exclusive):
    failures = []
    for key in ('low_state', 'imu', 'monitor', 'robot_state'):
        if counts.get(key, 0) < 2 or ages.get(key, 999) > .5:
            failures.append(f'{key}: missing or stale feedback')
    low = last.get('low_state', {})
    motors = low.get('motor_state', [])
    ids = [m['joint_id'] for m in motors]
    if ids != EXPECTED:
        failures.append('joint ID array does not match the verified 31-joint order')
    if low.get('mode_pr') != 0 or low.get('mode_machine') != 0:
        failures.append('LowState is not real-hardware PR/normal mode')
    if last.get('robot_state', {}).get('is_sim') is not False:
        failures.append('RobotState does not confirm real hardware')
    errors = [m['joint_id'] for m in motors if m['error_code'] != 0]
    if errors:
        failures.append(f'motor error codes: {errors}')
    for m in motors:
        values = [m['pos_fb'], m['vel_fb'], m['tau_fb']] + list(m['temp_fb'])
        if not all(math.isfinite(v) for v in values):
            failures.append(f'nonfinite motor feedback: {m["joint_id"]}')
        if max(m['temp_fb'], default=0) >= 75:
            failures.append(f'motor temperature >=75C: {m["joint_id"]}')
    imu = last.get('imu', {})
    q = imu.get('orientation', {})
    q_values = [q.get(k, float('nan')) for k in ('x', 'y', 'z', 'w')]
    imu_values = q_values + [imu.get(group, {}).get(k, float('nan'))
                            for group in ('angular_velocity', 'linear_acceleration') for k in ('x', 'y', 'z')]
    if not all(math.isfinite(v) for v in imu_values) or not .8 < sum(v*v for v in q_values) < 1.2:
        failures.append('invalid IMU quaternion/vector data')
    if imu.get('orientation_covariance', [-1])[0] < 0:
        failures.append('IMU orientation unavailable')
    deltas = {}
    for jid in EXPECTED:
        a, b = links_first.get(jid), links_last.get(jid)
        if a is None or b is None:
            failures.append(f'missing motor link counters: {jid}')
            continue
        ds, dr = (b[0]-a[0]) % 2**32, (b[1]-a[1]) % 2**32
        deltas[str(jid)] = {'send_delta': ds, 'read_delta': dr}
        if ds == 0 or dr == 0 or ds > 100000 or dr > 100000:
            failures.append(f'motor link is not continuously responding: {jid} (send={ds}, read={dr})')
        elif dr / ds < .9:
            failures.append(f'motor response ratio below 90%: {jid} ({dr}/{ds})')
    if not any(s['node'] == 'controller_node' for s in subscribers):
        failures.append('native controller_node command subscriber not discovered')
    if exclusive and publishers:
        failures.append('another /xlab/hr/low_cmd publisher is still present; do not start a second controller')
    return failures, deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclusive', action='store_true', help='also require zero existing command publishers')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = Node('honor_keyboard_preflight_readonly', enable_rosout=False, start_parameter_services=False)
    last, counts, times, first_links, last_links = {}, {}, {}, {}, {}
    def receive(key, msg):
        last[key] = message_to_ordereddict(msg)
        counts[key] = counts.get(key, 0) + 1
        times[key] = time.monotonic()
        if key == 'monitor':
            for m in msg.link_state:
                first_links.setdefault(m.joint_id, (m.send_counter, m.read_counter))
                last_links[m.joint_id] = (m.send_counter, m.read_counter)
    subscriptions = []
    for key, topic, msg_type in [('low_state', '/xlab/hr/low_state', LowState),
                                  ('imu', '/xlab/hr/body_upper_imu', Imu),
                                  ('monitor', '/xlab/hr/motor_monitor', MotorMonitor),
                                  ('robot_state', '/xlab/hr/robot_state', RobotState)]:
        subscriptions.append(node.create_subscription(msg_type, topic, lambda m, k=key: receive(k, m), qos_profile_sensor_data))
    started = time.monotonic()
    while time.monotonic() - started < 6:
        rclpy.spin_once(node, timeout_sec=.05)
    now = time.monotonic()
    ages = {k: now-v for k,v in times.items()}
    def endpoints(items):
        return [{'node': p.node_name, 'namespace': p.node_namespace, 'type': p.topic_type} for p in items]
    pubs = endpoints(node.get_publishers_info_by_topic('/xlab/hr/low_cmd'))
    subs = endpoints(node.get_subscriptions_info_by_topic('/xlab/hr/low_cmd'))
    failures, deltas = evaluate(last, counts, ages, first_links, last_links, pubs, subs, args.exclusive)
    report = {'captured_at': datetime.now().astimezone().isoformat(), 'read_only': True, 'commands_sent': 0,
              'passed': not failures, 'exclusive_check': args.exclusive, 'failures': failures,
              'counts': counts, 'receive_age_seconds': ages, 'command_publishers': pubs,
              'command_subscribers': subs, 'motor_link_deltas': deltas, 'last': last,
              'scope': 'short-window checks only; not certification of balance, calibration, physical support or emergency stop'}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('passed', 'exclusive_check', 'counts', 'command_publishers', 'failures')}, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
