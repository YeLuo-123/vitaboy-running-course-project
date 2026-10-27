"""Reproducible, sequential GPU experiment. Never overwrites released policy.

Run in the existing Linux training environment. The experiment directory must
already contain plan.json and an immutable baseline snapshot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def compact(path):
    r = json.loads(Path(path).read_text())
    return dict(file=str(path), policy_sha256=r['policy_file_sha256'],
                completion=r['completion_fraction'], speed=r['steady_speed_mean_mps'],
                stop=max(r.get('stopped_speed_mps', [0.])),
                bias=max(abs(x) for row in r.get('turn_mean_bias_by_direction_rads', [[0.]]) for x in row),
                filtered_mae=max(r.get('turn_filtered_yaw_rate_mae_rads', [0.])),
                raw_mae=max(r.get('turn_phase_yaw_rate_mae_rads', [0.])),
                action_rate_rms=max(r.get('action_rate_rms_per_s', [0.])),
                tilt=max(r.get('base_tilt_rms_rad', [0.])),
                transitions_pass=r.get('passes_transition_checks', False),
                speed_pass=r['passes_speed_and_completion'])


class Experiment:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.plan = json.loads((self.run / 'plan.json').read_text())
        self.env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4',
                        MUJOCO_GL='egl', PYTHONUNBUFFERED='1')

    def write(self, name, value):
        target = self.run / name
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(target)

    def execute(self, label, argv):
        log = self.run / (label + '.log')
        started = time.time()
        with log.open('x') as output:
            child = subprocess.Popen([sys.executable] + [str(x) for x in argv],
                                     cwd=ROOT.parent / 'honor_rl_mjlab', env=self.env,
                                     stdout=output, stderr=subprocess.STDOUT)
            self.write('status.json', dict(phase=label, pid=child.pid, driver_pid=os.getpid(),
                                           started_at=started, argv=argv, running=True))
            rc = child.wait()
        self.write(label + '.execution.json', dict(argv=argv, returncode=rc,
                                                  elapsed_s=time.time()-started))
        if rc:
            self.write('status.json', dict(phase=label, running=False, failed=True, returncode=rc))
            raise RuntimeError(f'{label} failed, see {log}')

    def evaluate(self, label, checkpoint, stage, scenario, seed, envs, duration, extra=()):
        out = self.run / 'results' / (label + '.json')
        if out.exists():
            raise FileExistsError(out)
        self.execute(label, [str(ROOT/'evaluate.py'), '--checkpoint', str(checkpoint),
            '--stage', stage, '--scenario', scenario, '--seed', str(seed),
            '--num-envs', str(envs), '--duration', str(duration), '--speed', '2.2',
            '--noise', '--output', str(out)] + list(extra))
        return compact(out)

    def develop(self):
        source = ROOT / self.plan['source_checkpoint']
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        logdir = ROOT / 'logs' / self.plan['id']
        if logdir.exists():
            raise FileExistsError(logdir)
        base = self.run / 'baseline' / 'averaged_4400_4600.pt'
        common = [str(ROOT/'train.py'), '--stage', 'rough', '--expert-source', 'dynamic',
                  '--transition-focus', '--fixed-learning-rate', str(self.plan['training']['lr']),
                  '--num-envs', str(self.plan['training']['envs']), '--seed', '42',
                  '--resume', str(source), '--save-interval', str(self.plan['training'].get('save_interval',200))]
        if self.plan['training'].get('preserve_speed'):
            common += ['--transition-preserve-speed']
        if self.plan['training'].get('retention_teacher'):
            common += ['--retention-teacher',str(ROOT/self.plan['training']['retention_teacher']),
                       '--retention-weight',str(self.plan['training'].get('retention_weight',1.))]
        self.execute('smoke', common + ['--iterations',str(self.plan['training'].get('smoke_updates',2)), '--log-dir', str(self.run/'smoke')])
        d = self.plan['development']
        def transition_tests(prefix, cp):
            seeds=d.get('transition_seeds',[d['transitions_seed']])
            rows=[self.evaluate(f'{prefix}_transitions'+(f'_{s}' if len(seeds)>1 else ''),
                               cp,'robust','transitions',s,32,40) for s in seeds]
            if len(rows)==1:return rows[0]
            summary=dict(files=[r['file'] for r in rows],policy_sha256=rows[0]['policy_sha256'])
            for k in ['stop','bias','filtered_mae','raw_mae','tilt','action_rate_rms']:summary[k]=max(r[k] for r in rows)
            for k in ['completion','speed']:summary[k]=min(r[k] for r in rows)
            for k in ['transitions_pass','speed_pass']:summary[k]=all(r[k] for r in rows)
            return summary
        baseline = dict(
            transitions=transition_tests('baseline_dev',base),
            terrain=self.evaluate('baseline_dev_terrain', base, 'rough', 'straight', d['terrain_seed'], 128, 25))
        self.write('baseline_development.json', baseline)
        self.execute('training', common + ['--iterations', str(self.plan['training']['updates']),
                                          '--log-dir', str(logdir)])
        candidates = []
        for step in self.plan.get('candidate_steps',[4800,5000,5200,5399]):
            checkpoint = logdir / f'model_{step}.pt'
            t = transition_tests(f'candidate_{step}_dev',checkpoint)
            r = self.evaluate(f'candidate_{step}_dev_terrain', checkpoint,
                             'rough', 'straight', d['terrain_seed'], 128, 25)
            candidates.append(dict(checkpoint=str(checkpoint), transitions=t, terrain=r))
            self.write('development_candidates.json', candidates)
        self.write('status.json', dict(phase='development_complete', running=False,
                                      source_checkpoint_sha256=source_hash, candidates=len(candidates)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    Experiment(args.run).develop()


if __name__ == '__main__':
    main()
