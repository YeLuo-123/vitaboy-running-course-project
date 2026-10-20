"""Fixed-command, first-episode evaluation; never connects to a real robot."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import mjlab.tasks  # registers installed task entry points
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint')
    p.add_argument('--onnx')
    p.add_argument('--baseline', action='store_true', help='Match isolated baseline config and normalizers')
    p.add_argument('--video', help='Optional first-command simulation video path')
    p.add_argument('--verify-onnx', help='Compare exported ONNX against checkpoint actions on rollout observations')
    p.add_argument('--task', default='VitaBoy-Velocity-Flat')
    p.add_argument('--profile', choices=['original', 'clean'], default='clean')
    p.add_argument('--speeds', nargs='+', type=float, default=[0., .3, .6])
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    configure_torch_backends()
    if a.verify_onnx:
        # Compare IEEE float32 inference with ONNX Runtime float32, avoiding
        # CUDA TF32 approximation as a confounder in export verification.
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cfg = load_env_cfg(a.task, play=True)
    agent = load_rl_cfg(a.task)
    if a.baseline:
        from train_walk_baseline import configure_baseline
        configure_baseline(cfg, agent)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = 42
    cfg.episode_length_s = a.seconds
    cfg.commands['base_velocity'].debug_vis = False
    cmd_cfg = cfg.commands['base_velocity']
    cmd_cfg.heading_command = False
    cmd_cfg.ranges.heading = None
    cmd_cfg.rel_standing_envs = 0.
    cmd_cfg.resampling_time_range = (1000., 1000.)
    cmd_cfg.ranges.lin_vel_y = (0., 0.)
    cmd_cfg.ranges.ang_vel_z = (0., 0.)
    if a.profile == 'clean':
        cfg.scene.terrain.terrain_type = 'plane'
        cfg.scene.terrain.terrain_generator = None
        for key in ['push_robot', 'add_base_mass', 'physics_material']:
            cfg.events.pop(key, None)
        cfg.events['reset_base'].params['pose_range'] = {}
        cfg.events['reset_base'].params['velocity_range'] = {}
        cfg.events['reset_robot_joints'].params['position_range'] = (1., 1.)
        for group in cfg.observations.values():
            group.enable_corruption = False
    cmd_cfg.ranges.lin_vel_x = (a.speeds[0], a.speeds[0])
    if a.video:
        cfg.viewer.width, cfg.viewer.height = 640, 480
    raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode='rgb_array' if a.video else None)
    vec = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    if a.checkpoint:
        runner_cls = load_runner_cls(a.task) or MjlabOnPolicyRunner
        runner = runner_cls(vec, asdict(agent), device='cuda:0')
        runner.load(a.checkpoint, load_cfg={'actor': True}, strict=True, map_location='cuda:0')
        policy = runner.get_inference_policy(device='cuda:0')
    elif a.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
        obs_name = sess.get_inputs()[0].name
        def policy(obs):
            x = obs['actor'].cpu().numpy()
            # Official export uses a fixed batch of one.
            return torch.as_tensor(np.concatenate([sess.run(None, {obs_name: row[None]})[0] for row in x]), device='cuda:0')
    else:
        def policy(obs):
            return torch.zeros((a.num_envs, raw.action_manager.total_action_dim), device='cuda:0')
    verifier = None
    if a.verify_onnx:
        if not a.checkpoint:
            p.error('--verify-onnx requires --checkpoint')
        import onnxruntime as ort
        verifier = ort.InferenceSession(a.verify_onnx, providers=['CPUExecutionProvider'])
    results = []
    writer = None
    if a.video:
        import imageio.v2 as imageio
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(a.video, fps=25)
    with torch.inference_mode():
        for speed in a.speeds:
            cmd_cfg.ranges.lin_vel_x = (speed, speed)
            obs, _ = vec.reset()
            robot = raw.scene['robot']
            foot_ids = [robot.body_names.index(n) for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
            initial_pos = robot.data.root_link_pos_w.clone()
            alive = torch.ones(a.num_envs, dtype=torch.bool, device='cuda:0')
            first_life = torch.full((a.num_envs,), a.seconds, device='cuda:0')
            sum_vel = torch.zeros(3, device='cuda:0')
            sum_error = torch.zeros((), device='cuda:0')
            sum_yaw_rate = torch.zeros((), device='cuda:0')
            sum_yaw_rate_abs = torch.zeros((), device='cuda:0')
            contacts_total = torch.zeros(3, device='cuda:0')
            count = 0
            stance_speed_sum = 0.
            stance_count = 0
            minimum_height = float('inf')
            action_abs = 0.
            action_max = 0.
            failure_count = 0
            onnx_max_error = 0.
            displacement = torch.zeros(a.num_envs, device='cuda:0')
            steps = round(a.seconds / raw.step_dt)
            for i in range(steps):
                action = policy(obs)
                if verifier is not None and i in (0, 1, 10, 100, 300, 500):
                    exported = verifier.run(None, {verifier.get_inputs()[0].name: obs['actor'][:1].cpu().numpy()})[0]
                    onnx_max_error = max(onnx_max_error, float(np.abs(exported - action[:1].cpu().numpy()).max()))
                obs, reward, dones, extra = vec.step(action)
                if writer is not None and speed == a.speeds[0] and i % 2 == 0:
                    writer.append_data(raw.render())
                relative_height = robot.data.root_link_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                # Add independent posture checks so a low crouch or non-torso
                # ground support cannot pass solely because torso contact is absent.
                posture_fail = (relative_height < .50) | (robot.data.projected_gravity_b[:, 2] > -.7071)
                failed = raw.reset_terminated.bool() | posture_fail
                newly_failed = alive & failed
                first_life[newly_failed] = (i + 1) * raw.step_dt
                failure_count += int(raw.reset_terminated.sum())
                valid = alive & ~dones.bool() & ~posture_fail
                pos = robot.data.root_link_pos_w
                displacement[valid] = (pos - initial_pos)[valid, 0]
                if i >= round(1. / raw.step_dt) and valid.any():
                    vel = robot.data.root_link_lin_vel_b[valid]
                    sum_vel += vel.sum(0)
                    sum_error += (vel[:, 0] - speed).abs().sum()
                    yaw_rate = robot.data.root_link_ang_vel_w[valid, 2]
                    sum_yaw_rate += yaw_rate.sum()
                    sum_yaw_rate_abs += yaw_rate.abs().sum()
                    force = raw.scene['feet_ground_contact'].data.force[valid]
                    foot_contact = force.norm(dim=-1) > 1.
                    foot_count = foot_contact.sum(-1)
                    contacts_total += torch.stack([(foot_count == k).sum() for k in (0, 1, 2)])
                    foot_speed = robot.data.body_link_lin_vel_w[:, foot_ids, :2][valid].norm(dim=-1)
                    stance_speed_sum += float((foot_speed * foot_contact).sum())
                    stance_count += int(foot_contact.sum())
                    minimum_height = min(minimum_height, float(relative_height[valid].min()))
                    count += int(valid.sum())
                    action_abs += float(action[valid].abs().sum())
                    action_max = max(action_max, float(action[valid].abs().max()))
                alive &= ~failed
                if i % 250 == 0:
                    print('PROGRESS', speed, i, 'alive', int(alive.sum()), flush=True)
            result = dict(speed_command=speed, profile=a.profile, task=a.task,
                checkpoint=a.checkpoint or a.onnx or 'zero', num_envs=a.num_envs,
                horizon_s=a.seconds, first_episode_survival_rate=float(alive.float().mean()),
                first_episode_mean_lifetime_s=float(first_life.mean()),
                first_episode_min_lifetime_s=float(first_life.min()),
                failures_all_episodes=failure_count,
                velocity_mean_after_1s=(sum_vel / max(1,count)).tolist(),
                vx_mae_after_1s=float(sum_error / max(1,count)),
                yaw_rate_mean_after_1s=float(sum_yaw_rate / max(1,count)),
                yaw_rate_abs_mean_after_1s=float(sum_yaw_rate_abs / max(1,count)),
                valid_samples_after_1s=count,
                foot_contact_fraction_none_single_double=(contacts_total / max(1,count)).tolist(),
                stance_foot_link_horizontal_speed_mean=stance_speed_sum/max(1,stance_count),
                minimum_base_height_after_1s=minimum_height if count else None,
                action_abs_mean=action_abs / max(1,count*29), action_abs_max=action_max,
                onnx_max_abs_action_error=onnx_max_error if verifier is not None else None,
                first_episode_world_x_displacement_mean=float(displacement.mean()))
            results.append(result)
            print('RESULT', json.dumps(result), flush=True)
            Path(a.output).parent.mkdir(parents=True, exist_ok=True)
            Path(a.output).write_text(json.dumps(results, indent=2))
    if writer is not None:
        writer.close()
    vec.close()


if __name__ == '__main__':
    main()
