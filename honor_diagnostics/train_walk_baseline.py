"""Isolated walking experiment using the installed official model and PPO runner.

This adds no robot connection, changes no tracked source, and does not resume the
failed policy. The exported ONNX includes the observation normalizer.
"""
from __future__ import annotations

import argparse
import os
import sys
import re
from datetime import datetime
from pathlib import Path


def configure_baseline(env, agent):
    env.scene.terrain.terrain_type = 'plane'
    env.scene.terrain.terrain_generator = None
    env.curriculum.clear()
    for name in ('push_robot', 'add_base_mass', 'physics_material'):
        env.events.pop(name, None)
    env.events['reset_base'].params['pose_range'] = {'yaw': (-.1, .1)}
    env.events['reset_base'].params['velocity_range'] = {}
    env.events['reset_robot_joints'].params['position_range'] = (.95, 1.05)
    command = env.commands['base_velocity']
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = .05
    command.ranges.lin_vel_x = (.25, .65)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    command.resampling_time_range = (5., 8.)
    command.debug_vis = False
    # Retain all 29 joints, the original nominal pose, PD, 0.25 action scale,
    # 50 Hz policy rate, and term-major ten-frame history.
    weights = {
        'track_lin_vel_xy_exp': 3., 'track_ang_vel_z_exp': .5,
        'lin_vel_z_l2': -.25, 'ang_vel_xy_l2': -.05,
        'energy': -1.e-4, 'action_rate_l2': -.005,
        'body_orientation_l2': -1., 'flat_orientation_l2': -.5,
        'termination_penalty': -50., 'feet_air_time': .8,
        'fly': -.3, 'joint_deviation_hip': -.1,
        'joint_deviation_arms': -.15, 'joint_deviation_legs': -.01,
    }
    for name, weight in weights.items():
        env.rewards[name].weight = weight
    agent.actor.obs_normalization = True
    agent.critic.obs_normalization = True
    agent.actor.distribution_cfg['init_std'] = .5
    agent.algorithm.learning_rate = 3.e-4
    agent.algorithm.entropy_coef = .003
    agent.num_steps_per_env = 32
    agent.experiment_name = 'vita_boy_walk_diagnostic'
    return env, agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--iterations', type=int, default=1500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-interval', type=int, default=250)
    p.add_argument('--resume', help='Checkpoint from a previous isolated walking experiment')
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent / 'honor_rl_mjlab'
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import mjlab.tasks
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from scripts.train import TrainConfig, run_train
    env, agent = configure_baseline(load_env_cfg('VitaBoy-Velocity-Flat'),
                                    load_rl_cfg('VitaBoy-Velocity-Flat'))
    env.scene.num_envs = args.num_envs
    agent.max_iterations = args.iterations
    agent.save_interval = args.save_interval
    agent.seed = args.seed
    agent.run_name = f'plane_forward_seed{args.seed}'
    if args.resume:
        checkpoint = Path(args.resume).expanduser().resolve()
        experiment = (root / 'logs/rsl_rl' / agent.experiment_name).resolve()
        if not checkpoint.is_file() or checkpoint.parent.parent != experiment:
            raise ValueError('--resume must be an existing checkpoint under ' + str(experiment))
        agent.resume = True
        agent.load_run = re.escape(checkpoint.parent.name)
        agent.load_checkpoint = re.escape(checkpoint.name)
    log_dir = Path('logs/rsl_rl') / agent.experiment_name / (
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + agent.run_name)
    print('DIAGNOSTIC_RUN', log_dir, flush=True)
    cfg = TrainConfig(env=env, agent=agent)
    run_train('VitaBoy-Velocity-Flat', cfg, log_dir)


if __name__ == '__main__':
    main()
