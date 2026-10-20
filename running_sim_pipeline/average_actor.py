"""Explicit inference-only actor interpolation; preserve both source checkpoints."""
import argparse,hashlib,json
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True)
p.add_argument('--second',type=Path,required=True);p.add_argument('--second-weight',type=float,default=.6)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if not 0<a.second_weight<1:raise ValueError('Interpolation weight must be in (0,1)')
x=torch.load(a.first,weights_only=False,map_location='cpu')
y=torch.load(a.second,weights_only=False,map_location='cpu')
assert 'actor_state_dict' in x and x['actor_state_dict'].keys()==y['actor_state_dict'].keys()
for k,v in y['actor_state_dict'].items():
    u=x['actor_state_dict'][k];assert u.shape==v.shape
    if torch.is_floating_point(v):y['actor_state_dict'][k]=(1-a.second_weight)*u+a.second_weight*v
    else:assert torch.equal(u,v),k
sources=[dict(path=str(z),sha256=hashlib.sha256(z.read_bytes()).hexdigest()) for z in [a.first,a.second]]
y['inference_only_actor_interpolation']=dict(sources=sources,second_weight=a.second_weight,
    warning='Critic/optimizer are from second source and are NOT a valid continuation state for averaged actor. Evaluate actor only.')
a.output.parent.mkdir(exist_ok=True,parents=True);torch.save(y,a.output)
a.output.with_suffix('.provenance.json').write_text(json.dumps(y['inference_only_actor_interpolation'],indent=2))
print(a.output)
