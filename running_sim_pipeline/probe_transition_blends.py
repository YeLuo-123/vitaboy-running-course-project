"""Evaluate two explicit actor interpolations on development data only.

Interpolated checkpoints are inference-only. Their PPO optimizer is never used.
"""
import argparse
import json
from pathlib import Path
from run_transition_experiment import Experiment, ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--first',type=Path,required=True)
    p.add_argument('--second',type=Path,required=True)
    args=p.parse_args()
    x=Experiment(args.run)
    out=x.run/'blends';out.mkdir(exist_ok=False)
    records=[]
    for weight in [.4,.65]:
        label=f'blend_{round(100*weight):03d}'
        cp=out/(label+'.pt')
        x.execute(label+'_create',[str(ROOT/'average_actor.py'),'--first',str(args.first),
            '--second',str(args.second),'--second-weight',str(weight),'--output',str(cp)])
        turns=[x.evaluate(f'{label}_dev_transitions_{s}',cp,'robust','transitions',s,32,40)
               for s in x.plan['development']['transition_seeds']]
        terrain=x.evaluate(f'{label}_dev_terrain',cp,'rough','straight',x.plan['development']['terrain_seed'],128,25)
        summary=dict(policy_sha256=turns[0]['policy_sha256'],files=[t['file'] for t in turns])
        for k in ['stop','bias','filtered_mae','raw_mae','tilt','action_rate_rms']:summary[k]=max(t[k] for t in turns)
        for k in ['completion','speed']:summary[k]=min(t[k] for t in turns)
        summary['transitions_pass']=all(t['transitions_pass'] for t in turns)
        records.append(dict(checkpoint=str(cp),second_weight=weight,transitions=summary,terrain=terrain))
        x.write('development_blends.json',records)
    x.write('status.json',dict(phase='blend_development_complete',running=False))


if __name__=='__main__':
    main()
