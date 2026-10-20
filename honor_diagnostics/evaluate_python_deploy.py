"""Headless CPU MuJoCo test of the repository's actual Python deployment path."""
import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--policy', required=True)
p.add_argument('--speed', type=float, default=.6)
p.add_argument('--seconds', type=float, default=20.)
p.add_argument('--output', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent.parent / 'honor_rl_mjlab'
policy_path = Path(a.policy).expanduser().resolve()
output_path = Path(a.output).expanduser().resolve()
os.chdir(root)
sys.path.insert(0, str(root / 'deploy'))
from common.config import load_config
from common.policy import OnnxPolicy
from inference.mujoco_robot import MuJoCoRobot
from fsm.loco.loco import LocomotionState

robot = MuJoCoRobot(load_config(root/'deploy/config.toml'), launch_viewer=False)
loco = LocomotionState()
loco.policy_path = policy_path
loco.policy = OnnxPolicy(policy_path, observation_size=960)
loco._validate_metadata()
# Start at the training nominal pose. No fixed-pose stabilization, mocap anchor,
# xfrc support, viewer, keyboard input, ROS, or real-robot connection is involved.
robot.data.qpos[robot.qpos_indices] = loco.default_angles
mujoco.mj_forward(robot.model, robot.data)
loco.enter(robot.get_state())
command = np.array([a.speed, 0., 0.], dtype=np.float32)
velocities = []
heights = []
failed = False
duration = 0.
for i in range(round(a.seconds / robot.control_dt)):
    target = loco.step(robot.get_state(), command)
    assert not target.stabilize_base
    robot.send_command(target)
    state = robot.get_state()
    duration = (i+1)*robot.control_dt
    if state.base_height < .5 or state.projected_gravity[2] > -.7071:
        failed = True
        break
    if duration > 1.:
        # mj_objectVelocity(..., flg_local=1) uses the inertial-frame orientation
        # for a body, which need not match the pelvis link frame in this asset.
        # Free-joint translation velocity is world-frame; rotate explicitly into
        # the same pelvis link frame used by the training observations.
        rotation = robot.data.xmat[robot.pelvis_body_id].reshape(3, 3)
        linear = rotation.T @ robot.data.qvel[:3]
        angular_world = rotation @ robot.data.qvel[3:6]
        v = np.r_[angular_world, linear]
        velocities.append(v)
        heights.append(state.base_height)
v = np.array(velocities)
result = dict(policy=str(policy_path), speed_command=a.speed, horizon_s=a.seconds,
    simulated_s=duration, failed=failed, stabilize_base=False,
    mean_velocity_body=v.mean(0).tolist() if len(v) else None,
    vx_mean=float(v[:,3].mean()) if len(v) else None,
    vx_mae=float(np.abs(v[:,3]-a.speed).mean()) if len(v) else None,
    yaw_rate_abs_mean=float(np.abs(v[:,2]).mean()) if len(v) else None,
    minimum_base_height=min(heights) if heights else None)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
robot.close()
