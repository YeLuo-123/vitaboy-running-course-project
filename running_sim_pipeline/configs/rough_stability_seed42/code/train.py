"""PPO + LS-AMP running, warm-started from the official velocity actor."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from run_config import ROOT,REPO,make_config,agent_config
from amp import AmpWrapper
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


class AmpRunner(MjlabOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        self.reset_amp_on_load=kwargs.pop('reset_amp_on_load',False)
        super().__init__(*args,**kwargs)
        original=self.alg.update
        def update():
            losses=original();losses.update(self.env.amp.update());return losses
        self.alg.update=update

    def save(self,path,infos=None):
        super().save(path,infos)
        state=torch.load(path,weights_only=False,map_location=self.device)
        state['amp']=self.env.amp.state_dict();torch.save(state,path)

    def load(self,path,*args,**kwargs):
        result=super().load(path,*args,**kwargs)
        state=torch.load(path,weights_only=False,map_location=self.device)
        if 'amp' not in state:raise ValueError('Resume checkpoint has no AMP state')
        if not self.reset_amp_on_load:self.env.amp.load_state_dict(state['amp'])
        return result


def official_warm_start(runner,env):
    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort
    actor=runner.alg.actor;state=actor.state_dict()
    path=REPO/'deploy/checkpoints/loco.onnx'
    weights={p.name:numpy_helper.to_array(p).copy() for p in onnx.load(str(path)).graph.initializer}
    for key,value in weights.items():
        assert key in state and tuple(value.shape)==tuple(state[key].shape),key
        state[key]=torch.as_tensor(value,device=runner.device,dtype=state[key].dtype)
    actor.load_state_dict(state,strict=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    obs=env.get_observations();x=obs['actor'][:8]
    with torch.inference_mode():actual=actor(obs)[:8].cpu().numpy()
    expected=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in x.cpu().numpy()])
    error=float(np.max(np.abs(expected-actual)));assert error<.01,error
    return dict(source=str(path),max_absolute_action_error=error,actor_input_dim=x.shape[-1],critic_fresh=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['flat','robust','rough'],default='flat')
    p.add_argument('--iterations',type=int,default=1000);p.add_argument('--num-envs',type=int,default=2048)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume');p.add_argument('--amp-weight',type=float,default=.4)
    p.add_argument('--save-interval',type=int,default=250);p.add_argument('--log-dir',type=Path)
    p.add_argument('--reset-amp',action='store_true',help='Explicitly reset discriminator/replay when changing expert dataset')
    p.add_argument('--expert-source',choices=['kinematic','dynamic'])
    p.add_argument('--focus-running',action='store_true',help='More sustained high-speed straight commands; keep standing/turn samples')
    p.add_argument('--refine-stability',action='store_true',help='Swing clearance, lateral foot margin, tighter yaw tracking and fall cost')
    args=p.parse_args();configure_torch_backends()
    cfg=make_config(args.stage,args.num_envs,True,args.seed,focus_running=args.focus_running,refine_stability=args.refine_stability);agent=agent_config()
    agent.seed=args.seed;agent.max_iterations=args.iterations;agent.save_interval=args.save_interval
    log=args.log_dir or ROOT/'logs'/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.stage)
    log.mkdir(parents=True,exist_ok=True)
    source=args.expert_source or ('dynamic' if args.stage=='rough' else 'kinematic')
    raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env=AmpWrapper(raw,args.amp_weight,source)
    runner=AmpRunner(env,asdict(agent),str(log),device='cuda:0',reset_amp_on_load=args.reset_amp)
    if args.resume:
        runner.load(str(Path(args.resume).resolve()),map_location='cuda:0')
    else:
        warm=official_warm_start(runner,env)
        (log/'warm_start_validation.json').write_text(json.dumps(warm,indent=2))
    manifest=dict(stage=args.stage,num_envs=args.num_envs,seed=args.seed,additional_iterations=args.iterations,
        resume=args.resume,amp_weight=args.amp_weight,expert=env.amp.provenance,
        algorithm=('PPO + LS-AMP legacy kinematic experts; expert-only gradient penalty' if source=='kinematic' else
                   'PPO + LS-AMP dynamics experts; +/-0.8 label smoothing, replay, interpolated gradient penalty'),
        expert_source=source,
        focus_running=args.focus_running,
        refine_stability=args.refine_stability,
        reset_amp_on_resume=args.reset_amp,
        actuator_control_hz=200,policy_hz=50,hardware_execution=False)
    (log/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (log/'code').mkdir(exist_ok=True)
    for filename in ['train.py','amp.py','run_config.py']:
        shutil.copy2(ROOT/filename,log/'code'/filename)
    dump_yaml(log/'env.yaml',asdict(cfg));dump_yaml(log/'agent.yaml',asdict(agent))
    (ROOT/'active_run.json').write_text(json.dumps(dict(log_dir=str(log),pid=os.getpid(),stage=args.stage),indent=2))
    print('RUNNING_TRAIN_LOG',str(log),flush=True)
    runner.learn(args.iterations,init_at_random_ep_len=True)
    raw.close()


if __name__=='__main__':main()
