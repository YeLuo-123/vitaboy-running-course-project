"""Audit actual logged updates and optimizer settings, not checkpoint labels."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    plan=json.loads((args.run/'plan.json').read_text())
    log=root/'logs'/plan['id']
    ea=EventAccumulator(str(log),size_guidance={'scalars':0});ea.Reload()
    events=ea.Scalars('Perf/collection_time')
    last=log/f'model_{events[-1].step}.pt'
    state=torch.load(last,weights_only=False,map_location='cpu')
    record=dict(formal_updates=len(events),first_iteration_label=events[0].step,last_iteration_label=events[-1].step,
        logged_wall_span_s=events[-1].wall_time-events[0].wall_time,
        training_process=json.loads((args.run/'training.execution.json').read_text()),
        num_envs=plan['training']['envs'],steps_per_env_per_update=24,policy_hz=50,
        control_transitions=len(events)*plan['training']['envs']*24,
        source_checkpoint=plan['source_checkpoint'],
        source_checkpoint_sha256=hashlib.sha256((root/plan['source_checkpoint']).read_bytes()).hexdigest(),
        optimizer_lrs=[g['lr'] for g in state['optimizer_state_dict']['param_groups']],
        amp_expert_sha256=state['amp']['expert_provenance']['sha256'],smoke_updates_excluded=plan['training'].get('smoke_updates',2),
        hardware_execution=False)
    assert record['formal_updates']==plan['training']['updates']
    assert all(abs(lr-plan['training']['lr'])<1e-10 for lr in record['optimizer_lrs'])
    (args.run/'training_audit.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
