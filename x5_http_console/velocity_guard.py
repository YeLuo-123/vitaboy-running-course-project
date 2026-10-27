#!/usr/bin/env python3
"""Independent output watchdog: no new command for 0.4s => zero velocity."""
import signal
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def main():
    rclpy.init()
    node=Node('x5_velocity_guard')
    pub=node.create_publisher(Twist,'/cmd_vel',10)
    state={'time':0.,'command':Twist()}
    def receive(msg):state.update(time=time.monotonic(),command=msg)
    node.create_subscription(Twist,'/x5/gated_cmd',receive,10)
    def tick():
        age=time.monotonic()-state['time']
        if age<2.5:pub.publish(state['command'] if age<.4 else Twist())
    node.create_timer(.05,tick)
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if rclpy.ok():
            for _ in range(3):pub.publish(Twist());time.sleep(.05)
            node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
