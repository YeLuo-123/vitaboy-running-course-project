"""Read saved configuration and TensorBoard events without loading YAML objects."""
import argparse
import json
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument('--log-root', default='logs/rsl_rl/vita_boy_velocity')
p.add_argument('--output', required=True)
a = p.parse_args()
runs = []
for run in sorted(Path(a.log_root).iterdir()):
    if not (run / 'params/env.yaml').is_file():
        continue
    env = yaml.load((run/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    agent = yaml.load((run/'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    ea = EventAccumulator(str(run), size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags()['scalars']
    values = {}
    for tag in tags:
        events = ea.Scalars(tag)
        series = {}
        for step in [100, 500, 1000, 2000, 3000, 5000, 7500, events[-1].step]:
            if step > events[-1].step:
                continue
            window = [e.value for e in events if max(0, step-49) <= e.step <= step]
            if window:
                series[str(step)] = sum(window)/len(window)
        values[tag] = {'last_step': events[-1].step, 'last': events[-1].value,
                       'mean_over_last_up_to_50_iterations_at': series}
    record = dict(run=str(run), envs=int(env['scene']['num_envs']),
        steps_per_env=int(agent['num_steps_per_env']), configured_iterations=int(agent['max_iterations']),
        policy_dt=float(env['sim']['mujoco']['timestep'])*int(env['decimation']),
        terrain=env['scene']['terrain'], command=env['commands']['base_velocity'],
        reward_weights={n:float(v['weight']) for n,v in env['rewards'].items()},
        action_config=env['actions'], scalar_summary=values)
    runs.append(record)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(runs, indent=2))
print('Saved', a.output)
