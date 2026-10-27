"""Frozen-protocol validation of one candidate selected on development seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from run_transition_experiment import Experiment, ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    x = Experiment(args.run)
    candidate_dir = x.run/'candidate'
    candidate_dir.mkdir(exist_ok=False)
    checkpoint = candidate_dir/'policy.pt'
    shutil.copy2(args.checkpoint, checkpoint)
    policy = candidate_dir/'policy.onnx'
    x.write('selection.json', dict(source_checkpoint=str(args.checkpoint), sha256=sha(checkpoint),
        basis='Selected using development seeds only. Final seeds never used for selection.'))
    baseline = x.run/'baseline/averaged_4400_4600.pt'
    nominal = x.evaluate('candidate_final_nominal', checkpoint, 'flat', 'transitions',
                         105260, 16, 40, ['--export-onnx', str(policy)])
    turns, legacy_terrain, new_terrain, paired_terrain, paired_turns = [], [], [], [], []
    turns.append(x.evaluate('candidate_legacy_transitions', checkpoint, 'robust', 'transitions', 718281, 32, 40))
    for seed in x.plan['final_new_seeds']:
        turns.append(x.evaluate(f'candidate_final_transitions_{seed}', checkpoint, 'robust', 'transitions', seed, 32, 40))
        paired_turns.append(x.evaluate(f'baseline_final_transitions_{seed}', baseline, 'robust', 'transitions', seed, 32, 40))
        new_terrain.append(x.evaluate(f'candidate_final_terrain_{seed}', checkpoint, 'rough', 'straight', seed, 128, 25))
        paired_terrain.append(x.evaluate(f'baseline_final_terrain_{seed}', baseline, 'rough', 'straight', seed, 128, 25))
    for seed in x.plan['legacy_terrain_seeds']:
        legacy_terrain.append(x.evaluate(f'candidate_legacy_terrain_{seed}', checkpoint, 'rough', 'straight', seed, 128, 25))
    cpu_path = x.run/'results/candidate_cpu_600s.json'
    x.execute('candidate_cpu_600s', [str(ROOT/'evaluate_cpu.py'), '--policy', str(policy),
                                  '--duration', '600', '--speed', '2.4', '--output', str(cpu_path)])
    cpu = json.loads(cpu_path.read_text())
    load_result=lambda row:json.loads(Path(row['file']).read_text())
    pairs=list(zip(turns[1:],paired_turns))+list(zip(new_terrain,paired_terrain))
    paired_physics=all(load_result(t)['physics_variation_snapshot']==load_result(b)['physics_variation_snapshot'] for t,b in pairs)
    capacity_ok=True
    for row in [nominal]+turns+legacy_terrain+new_terrain:
        c=load_result(row)['numerical_capacity']
        capacity_ok &= c['peak_total_contacts']<c['total_contact_capacity'] and c['peak_constraints_per_world']<c['constraint_capacity_per_world']
    preserve = x.plan['preservation_gates']
    successes = lambda rows: sum(round(r['completion']*128) for r in rows)
    checks = dict(
        nominal_transitions=nominal['transitions_pass'],
        all_randomized_transitions=all(t['transitions_pass'] for t in turns),
        legacy_terrain_each_seed=all(t['speed_pass'] for t in legacy_terrain),
        legacy_terrain_success_count_preserved=successes(legacy_terrain)>=preserve['legacy_terrain_successes_min_of_384'],
        legacy_terrain_speed_preserved=all(t['speed']>=minimum for t, minimum in zip(legacy_terrain,preserve['legacy_terrain_each_seed_speed_min_mps'])),
        new_terrain_each_seed=all(t['speed_pass'] for t in new_terrain),
        new_terrain_paired_success_preserved=successes(new_terrain)>=successes(paired_terrain),
        new_terrain_paired_speed_preserved=all(t['speed']>=b['speed']-.05 for t,b in zip(new_terrain,paired_terrain)),
        cpu_600s_speed_preserved=cpu['passes_speed_and_duration'] and cpu['whole_trial_forward_speed_mps']>=preserve['cpu_whole_trial_speed_min_mps'],
        same_checkpoint=all(t['policy_sha256']==sha(checkpoint) for t in [nominal]+turns+legacy_terrain+new_terrain),
        cpu_same_export=cpu['policy_file_sha256']==sha(policy),
        paired_randomization_snapshots_equal=paired_physics,
        physics_buffer_capacity_sufficient=bool(capacity_ok),
        onnx_export_error_below_1e3=load_result(nominal)['onnx_max_abs_action_error']<1e-3,
        baseline_policy_unchanged=sha(ROOT/'artifacts/running_policy.onnx')==x.plan['baseline_policy_sha256'])
    report=dict(experiment=x.plan['id'], checks=checks, all_checks_passed=all(checks.values()),
        candidate_checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint)),
        candidate_policy=dict(path=str(policy), sha256=sha(policy)),
        nominal=nominal, randomized_transitions=turns, baseline_paired_transitions=paired_turns,
        legacy_terrain=legacy_terrain, new_terrain=new_terrain, baseline_paired_terrain=paired_terrain,
        cpu_endurance=cpu, legacy_terrain_successes=successes(legacy_terrain),
        new_terrain_successes=successes(new_terrain), paired_baseline_terrain_successes=successes(paired_terrain),
        evaluation_protocol='Unchanged evaluate.py and evaluate_cpu.py; original thresholds and stationary initialization.',
        limitations=['Synthetic terrain, shared surfaces per seed; not outdoor reliability certification.',
                     'No hardware, battery, thermal or 600s rough-terrain validation.',
                     'Single training seed and one optimization round; no AMP ablation.'])
    x.write('validation_report.json', report)
    x.write('status.json', dict(phase='validation_complete',running=False,passed=report['all_checks_passed']))


if __name__ == '__main__':
    main()
