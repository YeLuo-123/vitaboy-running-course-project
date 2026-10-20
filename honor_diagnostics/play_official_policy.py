"""Serve the unchanged bundled locomotion ONNX in a loopback-only simulator."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
import viser
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--port', type=int, default=8090)
p.add_argument('--speed', type=float, default=.3)
a = p.parse_args()
root = Path(__file__).resolve().parent.parent / 'honor_rl_mjlab'
os.chdir(root)
configure_torch_backends()
cfg = load_env_cfg('VitaBoy-Velocity-Flat', play=True)
cfg.scene.num_envs = 1
cfg.scene.terrain.terrain_type = 'plane'
cfg.scene.terrain.terrain_generator = None
cfg.curriculum.clear()
cfg.episode_length_s = 120.
cfg.seed = 42
cfg.viewer.max_extra_envs = 0
for name in ('push_robot', 'add_base_mass', 'physics_material'):
    cfg.events.pop(name, None)
cfg.events['reset_base'].params['pose_range'] = {}
cfg.events['reset_base'].params['velocity_range'] = {}
cfg.events['reset_robot_joints'].params['position_range'] = (1., 1.)
for group in cfg.observations.values():
    group.enable_corruption = False
cmd_cfg = cfg.commands['base_velocity']
cmd_cfg.heading_command = False
cmd_cfg.ranges.heading = None
cmd_cfg.rel_standing_envs = 0.
cmd_cfg.ranges.lin_vel_x = (-.3, .6)
cmd_cfg.ranges.lin_vel_y = (-.3, .3)
cmd_cfg.ranges.ang_vel_z = (-.6, .6)
cmd_cfg.resampling_time_range = (1000., 1000.)
raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
env = RslRlVecEnvWrapper(raw)
policy_file = root/'deploy/checkpoints/loco.onnx'
session = ort.InferenceSession(str(policy_file), providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
server = viser.ViserServer(host='127.0.0.1', port=a.port,
                           label='VitaBoy official locomotion - SIMULATION')
telemetry = server.gui.add_html('<b>Official locomotion policy · simulation only</b>')
status_file = Path(__file__).with_name('official_viewer_status.json')
last_status = 0.


def policy(obs):
    global last_status
    now = time.monotonic()
    if now-last_status >= 1.:
        robot = raw.scene['robot']
        cmd = raw.command_manager.get_command('base_velocity')[0].tolist()
        vel = robot.data.root_link_lin_vel_b[0].tolist()
        status = dict(policy=str(policy_file), simulation_only=True,
            command=cmd, velocity_body=vel,
            base_height=float(robot.data.root_link_pos_w[0,2]),
            step=int(raw.common_step_counter), updated_unix=time.time())
        status_file.write_text(json.dumps(status, indent=2))
        telemetry.content = (f'<b>Official locomotion · SIMULATION</b><br/>'
            f'Command vx: {cmd[0]:.2f} m/s · actual vx: {vel[0]:.2f} m/s<br/>'
            'Controls → Commands → Base_velocity: sliders; Zero stops movement.')
        last_status = now
    action = session.run(None, {input_name: obs['actor'].cpu().numpy()})[0]
    return torch.as_tensor(action, device=raw.device)


class OfficialViewer(ViserPlayViewer):
    def setup(self):
        super().setup()
        command = raw.command_manager.get_term('base_velocity')
        command._joystick_enabled.value = True
        for slider, value in zip(command._joystick_sliders, (a.speed, 0., 0.), strict=True):
            slider.value = value
        command.vel_command_b[0] = torch.tensor([a.speed, 0., 0.], device=raw.device)
        # Start immediately; the browser can pause or zero the command.
        self._is_paused = False
        self._pause_button.label = 'Pause'
        print(f'OFFICIAL_VIEWER_READY http://127.0.0.1:{a.port}', flush=True)


try:
    OfficialViewer(env, policy, frame_rate=30., viser_server=server).run()
finally:
    env.close()
    server.stop()
