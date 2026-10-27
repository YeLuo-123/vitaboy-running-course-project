"""Opt-in transition curriculum; baseline task and evaluation protocol stay fixed.

Half of the environments retain the original running command distribution. The
other half exercise stationary starts, running turns, braking and settling.
Only command targets and rewards change: no base state or contact is modified.
"""
from dataclasses import dataclass
from copy import deepcopy
import torch

from run_config import SmoothRunningCommand, SmoothRunningCommandCfg, make_config
from mjlab.managers import RewardTermCfg


class TransitionRunningCommand(SmoothRunningCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sequence_mask = torch.arange(self.num_envs, device=self.device) % 2 == 0
        self.sequence_speed = torch.full((self.num_envs,), 2.2, device=self.device)
        self.sequence_turn = torch.full((self.num_envs,), .3, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if hasattr(self, 'sequence_speed'):
            self.sequence_speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(2., 2.6)
            magnitude = torch.empty(len(env_ids), device=self.device).uniform_(.24, .38)
            sign = torch.where(torch.rand(len(env_ids), device=self.device) < .5, -1., 1.)
            self.sequence_turn[env_ids] = magnitude * sign

    def _update_command(self):
        if self.external_target is None and hasattr(self, 'sequence_mask'):
            # Episode time is used only to select commands, never to reset state.
            t = self._env.episode_length_buf * self._env.step_dt
            moving = ((t >= 2.) & (t < 19.)) | ((t >= 25.) & (t < 34.))
            first_turn = (t >= 11.) & (t < 19.)
            second_turn = (t >= 25.) & (t < 34.)
            sequence = torch.zeros_like(self.target)
            sequence[:, 0] = self.sequence_speed * moving
            sequence[:, 2] = self.sequence_turn * (first_turn.float() - second_turn.float())
            self.target[self.sequence_mask] = sequence[self.sequence_mask]
        super()._update_command()


@dataclass(kw_only=True)
class TransitionRunningCommandCfg(SmoothRunningCommandCfg):
    def build(self, env):
        return TransitionRunningCommand(self, env)


def settled_stance_reward(env):
    command = env.command_manager.get_command('base_velocity')
    stopped = command.abs().amax(-1) < .04
    d = env.scene['robot'].data
    speed_sq = d.root_link_lin_vel_b[:, :2].square().sum(-1)
    yaw_sq = d.root_link_ang_vel_b[:, 2].square()
    return torch.exp(-speed_sq / .08**2 - yaw_sq / .12**2) * stopped


def stopped_drift_penalty(env):
    stopped = env.command_manager.get_command('base_velocity').abs().amax(-1) < .04
    return env.scene['robot'].data.root_link_lin_vel_b[:, :2].norm(dim=-1).clamp(max=1.) * stopped


def persistent_yaw_tracking(env):
    """Use body-frame angular velocity, matching commands and evaluation."""
    actual = env.scene['robot'].data.root_link_ang_vel_b[:, 2]
    target = env.command_manager.get_command('base_velocity')[:, 2]
    if not hasattr(env, '_transition_yaw_actual'):
        env._transition_yaw_actual = torch.zeros_like(actual)
        env._transition_yaw_target = torch.zeros_like(target)
    resetting = env.episode_length_buf <= 1
    env._transition_yaw_actual[resetting] = actual[resetting]
    env._transition_yaw_target[resetting] = target[resetting]
    alpha = env.step_dt / (.2 + env.step_dt)
    env._transition_yaw_actual += (actual - env._transition_yaw_actual) * alpha
    env._transition_yaw_target += (target - env._transition_yaw_target) * alpha
    return torch.exp(-((env._transition_yaw_actual - env._transition_yaw_target) / .12).square())


def make_transition_config(stage, num_envs, seed):
    cfg = make_config(stage, num_envs, True, seed, focus_running=True,
                      refine_stability=True, precision_running=True)
    old = cfg.commands['base_velocity']
    cfg.commands['base_velocity'] = TransitionRunningCommandCfg(
        entity_name='robot', resampling_time_range=(7., 14.), rel_standing_envs=.2,
        heading_command=False, debug_vis=False, speed_curriculum=False,
        focus_running=True, ranges=deepcopy(old.ranges))
    cfg.episode_length_s = 40.
    cfg.rewards['mean_yaw_tracking'] = RewardTermCfg(func=persistent_yaw_tracking, weight=10.)
    cfg.rewards['settled_stance'] = RewardTermCfg(func=settled_stance_reward, weight=3.)
    cfg.rewards['stopped_drift'] = RewardTermCfg(func=stopped_drift_penalty, weight=-3.)
    return cfg
