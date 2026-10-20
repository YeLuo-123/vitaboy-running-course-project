"""View an isolated walking checkpoint with the matching training configuration."""
import argparse
import os
import sys
from pathlib import Path

from train_walk_baseline import configure_baseline

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', required=True)
p.add_argument('--viewer', choices=['native', 'viser'], default='viser')
p.add_argument('--num-envs', type=int, default=1)
a = p.parse_args()
root = Path(__file__).resolve().parent.parent / 'honor_rl_mjlab'
os.chdir(root)
sys.path.insert(0, str(root))
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls, register_mjlab_task
from scripts.play import PlayConfig, run_play

task = 'VitaBoy-Velocity-Flat'
env, agent = configure_baseline(load_env_cfg(task), load_rl_cfg(task))
play_env, _ = configure_baseline(load_env_cfg(task, play=True), load_rl_cfg(task))
for group in play_env.observations.values():
    group.enable_corruption = False
register_mjlab_task(task_id='VitaBoy-Walk-Diagnostic', env_cfg=env,
    play_env_cfg=play_env, rl_cfg=agent, runner_cls=load_runner_cls(task))
run_play('VitaBoy-Walk-Diagnostic', PlayConfig(checkpoint_file=a.checkpoint,
    num_envs=a.num_envs, viewer=a.viewer))
