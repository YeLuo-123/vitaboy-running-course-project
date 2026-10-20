"""Evidence-only summary: a failed rollout cannot pass through low error alone."""
import json
from pathlib import Path
import numpy as np
from run_config import ROOT

rows=[];cpu=[]
for path in sorted((ROOT/'results').glob('*.json')):
    if '.trajectory.' in path.name:continue
    r=json.loads(path.read_text())
    if 'completion_fraction' not in r:continue
    if 'stage' not in r:
        cpu.append(dict(name=path.stem,**r));continue
    mean=lambda key:float(np.mean(r[key])) if key in r else None
    rows.append(dict(name=path.stem,stage=r['stage'],scenario=r['scenario'],seed=r['seed'],
        duration_s=r['requested_duration_s'],envs=r['num_envs'],completion=r['completion_fraction'],
        actual_speed_mps=r['steady_speed_mean_mps'],command_mps=r['commanded_speed_mps'],
        stopped_mps=mean('stopped_speed_mps') if sum(r.get('stop_measurement_samples',[])) else None,yaw_mae_rads=mean('yaw_rate_mae_rads'),
        mechanical_J_per_m=mean('mechanical_cost_per_meter_jpm'),
        flight_fraction=mean('flight_fraction_steady_each_env'),passes_speed_and_completion=r['passes_speed_and_completion']))
(ROOT/'results/summary.json').write_text(json.dumps(dict(evaluations=rows,cpu_deployment=cpu,
    energy_is_battery_measurement=False,hardware_evaluated=False),indent=2))
for r in rows:print(r['name'],f"complete={r['completion']:.1%}",f"speed={r['actual_speed_mps']:.3f}",f"J/m={r['mechanical_J_per_m']:.1f}")
