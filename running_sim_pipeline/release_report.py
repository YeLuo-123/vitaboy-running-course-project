"""Build an acceptance record from explicit test files; never drops failed trials."""
import argparse,hashlib,json,math
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['checkpoint','policy','cpu','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=Path,help='Optional additional GPU 600s endurance test of the same checkpoint')
    p.add_argument('--transitions',nargs='+',type=Path,required=True)
    p.add_argument('--terrain',nargs='+',type=Path,required=True)
    a=p.parse_args();cp_sha=sha(a.checkpoint);policy_sha=sha(a.policy)
    records={}
    def read(path,expected_sha):
        data=json.loads(path.read_text())
        if data.get('policy_file_sha256')!=expected_sha:raise ValueError(f'Policy hash mismatch/missing: {path}')
        records[str(path)]=dict(sha256=sha(path),policy_sha256=expected_sha)
        return data
    cpu=read(a.cpu,policy_sha);gpu=read(a.gpu,cp_sha) if a.gpu else None
    turns=[read(x,cp_sha) for x in a.transitions];rough=[read(x,cp_sha) for x in a.terrain]
    if len({x['seed'] for x in rough})!=len(rough):raise ValueError('Terrain seeds must be distinct')
    count=sum(x['num_envs'] for x in rough)
    passed=sum(sum(t>=x['requested_duration_s']-1e-5 for t in x['lifetimes_s']) for x in rough)
    # Wilson interval is descriptive, not a certification: parallel trials share terrain.
    z=1.96;fraction=passed/count;denom=1+z*z/count
    center=(fraction+z*z/(2*count))/denom
    margin=z*math.sqrt(fraction*(1-fraction)/count+z*z/(4*count*count))/denom
    checks={
        'cpu_nominal_600s_actual_mean_ge_2':bool(cpu['passes_speed_and_duration'] and cpu['requested_duration_s']>=600 and cpu['whole_trial_forward_speed_mps']>=2.),
        'flat_and_randomized_transitions':all(x['passes_transition_checks'] for x in turns) and {'flat','robust'}.issubset({x['stage'] for x in turns}),
        'terrain_each_seed_ge_95_percent_and_2mps':all(x['passes_speed_and_completion'] and x['stage']=='rough' and x['observation_noise'] for x in rough),
    }
    if gpu is not None:
        checks['additional_gpu_nominal_600s_actual_mean_ge_2']=bool(gpu['passes_speed_and_completion'] and gpu['requested_duration_s']>=600 and min(gpu['whole_trial_forward_speed_each_env_mps'])>=2.)
    result=dict(schema_version=1,checkpoint=dict(path=str(a.checkpoint),sha256=cp_sha),
        policy=dict(path=str(a.policy),sha256=policy_sha),checks=checks,
        all_listed_simulation_checks_passed=all(checks.values()),
        terrain=dict(successes=passed,trials=count,completion_fraction=fraction,
            descriptive_wilson_95_interval=[center-margin,center+margin],
            seeds=[x['seed'] for x in rough],duration_each_s=[x['requested_duration_s'] for x in rough],
            steady_speed_each_seed_mps=[x['steady_speed_mean_mps'] for x in rough],
            completion_each_seed=[x['completion_fraction'] for x in rough],
            caveat='Two synthetic surfaces per seed; outcomes can correlate. These runs do not prove 600s terrain endurance.'),
        cpu_endurance=dict(duration_s=cpu['simulated_duration_s'],steady_speed_mps=cpu['steady_speed_mean_mps'],
            whole_trial_forward_speed_mps=cpu['whole_trial_forward_speed_mps'],distance_m=cpu['distance_m'],
            absolute_mechanical_cost_jpm=cpu['mechanical_cost_per_meter_jpm']),
        evidence=records,hardware_tested=False,battery_endurance_verified=False,
        hardware_calibration_completed=False,calibrated_thermal_model=False,
        amp_ablation_completed=False,
        limitations=['600s endurance is on nominal flat ground; rough tests are separate short trials.',
            'Mechanical work is sampled at 50Hz and excludes battery, motor efficiency, heat and electronics.',
            'CPU MuJoCo and MuJoCo-Warp share the physics family; this is deployment/numerical validation, not a different physical engine.',
            'Simulation asset mass and 150Nm peak capability require hardware identification; manual headline values do not establish per-joint continuous limits.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
