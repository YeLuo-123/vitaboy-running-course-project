"""Train the official Tracking task on a retargeted running clip, in isolation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys

from tracking_common import ROOT, MOTION, configure_tracking


def import_official_actor(actor, onnx_path, device):
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort
    import torch
    from tensordict import TensorDict
    model=onnx.load(str(onnx_path))
    values={x.name:numpy_helper.to_array(x).copy() for x in model.graph.initializer}
    state=actor.state_dict()
    for name,value in values.items():
        if name.startswith('mlp.'):
            if name not in state or tuple(value.shape)!=tuple(state[name].shape):
                raise ValueError('Official actor architecture mismatch: '+name)
            state[name]=torch.as_tensor(value,device=device,dtype=state[name].dtype)
    divisor=next(n.input[1] for n in model.graph.node if n.op_type=='Div')
    mean=values['obs_normalizer._mean']
    std=np.maximum(values[divisor]-actor.obs_normalizer.eps,1.e-8)
    for name,value in [('obs_normalizer._mean',mean),('obs_normalizer._std',std),('obs_normalizer._var',std**2)]:
        state[name]=torch.as_tensor(value,device=device,dtype=state[name].dtype)
    state['obs_normalizer.count'].fill_(1000000)
    actor.load_state_dict(state,strict=True)
    actor.eval()
    rng=np.random.default_rng(42)
    x=(mean+rng.normal(size=(32,154))*values[divisor]).astype(np.float32)
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=['CPUExecutionProvider'])
    expected=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in x])
    with torch.inference_mode():
        actual=actor(TensorDict({'actor':torch.as_tensor(x,device=device)},batch_size=[32])).cpu().numpy()
    error=float(np.max(np.abs(actual-expected)))
    if error>0.01: raise RuntimeError(f'Imported actor failed parity: {error}')
    actor.train()
    return dict(source=str(onnx_path),max_abs_action_error=error,
                note='MLP and normalizer imported from official ONNX; critic and optimizer initialized fresh.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs',type=int,default=2048)
    p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-interval',type=int,default=250)
    p.add_argument('--from-scratch',action='store_true')
    p.add_argument('--resume')
    a=p.parse_args()
    repo=ROOT.parent/'honor_rl_mjlab'
    os.chdir(repo); sys.path.insert(0,str(repo))
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import mjlab.tasks
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls,register_mjlab_task
    from scripts.train import TrainConfig,run_train
    cfg=configure_tracking(load_env_cfg('VitaBoy-Tracking'),training=True)
    cfg.scene.num_envs=a.num_envs
    agent=load_rl_cfg('VitaBoy-Tracking')
    agent.seed=a.seed
    agent.actor.distribution_cfg['init_std']=.25
    agent.algorithm.learning_rate=1.e-4
    agent.algorithm.entropy_coef=.001
    agent.max_iterations=a.iterations
    agent.save_interval=a.save_interval
    agent.experiment_name='vita_boy_run_tracking'
    agent.run_name=f'lafan_run2_s4_seed{a.seed}'
    log_dir=Path('logs/rsl_rl')/agent.experiment_name/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+agent.run_name)
    log_dir.mkdir(parents=True,exist_ok=True)
    shutil.copy2(MOTION.parent/'manifest.json',log_dir/'retarget_manifest.json')
    base=load_runner_cls('VitaBoy-Tracking')
    if a.resume:
        cp=Path(a.resume).expanduser().resolve()
        if not cp.is_file() or cp.parent.parent!=(repo/'logs/rsl_rl'/agent.experiment_name).resolve():
            raise ValueError('Resume must be a checkpoint in this isolated experiment')
        agent.resume=True; agent.load_run=re.escape(cp.parent.name); agent.load_checkpoint=re.escape(cp.name)
    class RunRunner(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if not a.from_scratch and not a.resume:
                result=import_official_actor(self.alg.actor,repo/'deploy/checkpoints/tracking.onnx',self.device)
                (log_dir/'warm_start_validation.json').write_text(json.dumps(result,indent=2))
                print('WARM_START',json.dumps(result),flush=True)
    register_mjlab_task(task_id='VitaBoy-Run-Tracking',env_cfg=cfg,
        play_env_cfg=configure_tracking(load_env_cfg('VitaBoy-Tracking',play=True)),
        rl_cfg=agent,runner_cls=RunRunner)
    print('RUN_TRACKING_LOG',log_dir,flush=True)
    run_train('VitaBoy-Run-Tracking',TrainConfig(env=cfg,agent=agent,motion_file=str(MOTION)),log_dir)


if __name__=='__main__': main()
