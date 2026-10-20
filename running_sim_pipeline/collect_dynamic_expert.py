"""Collect finite, physically simulated Tracking rollouts for AMP experts."""
import sys,json,hashlib
import numpy as np
import torch
from dataclasses import asdict
from run_config import ROOT,WS
sys.path.insert(0,str(WS/'motion_pipeline'))
from tracking_common import configure_tracking
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from amp import state_features

cfg=configure_tracking(load_env_cfg('VitaBoy-Tracking',play=True),jitter=True)
cfg.scene.num_envs=32;cfg.seed=20261020
raw=ManagerBasedRlEnv(cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw)
runner=load_runner_cls('VitaBoy-Tracking')(env,asdict(load_rl_cfg('VitaBoy-Tracking')),device='cuda:0')
checkpoint=WS/'motion_pipeline/artifacts/model_500.pt'
runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
policy=runner.get_inference_policy(device='cuda:0');cmd=raw.command_manager.get_term('motion')
feet=raw.scene['robot'].find_bodies(('left_ankle_roll_link','right_ankle_roll_link'))[0]
pairs=[];velocity=[];wraps=0;terminated=0
with torch.inference_mode():
    obs,_=env.reset()
    for i in range(int(cmd.motion.time_step_total)-2):
        before=state_features(raw,feet);cursor=cmd.time_steps.clone()
        obs,_,done,_=env.step(policy(obs));after=state_features(raw,feet)
        wraps+=int(((cmd.time_steps<cursor)&~done.bool()).sum());terminated+=int(done.sum())
        if i>25:
            pairs.append(torch.cat([before,after],-1)[~done.bool()].cpu().numpy())
            velocity.append(float(raw.scene['robot'].data.root_link_lin_vel_b[:,0].mean()))
assert wraps==0 and terminated==0,(wraps,terminated)
data=np.concatenate(pairs);assert data.shape[1]==148 and np.isfinite(data).all()
out=ROOT/'assets/dynamic_amp_expert.npz';out.parent.mkdir(parents=True,exist_ok=True)
np.savez(out,transitions=data,fps=50.)
report=dict(source='Actual dynamics states of validated retargeted-running Tracking policy',
    tracking_checkpoint=str(checkpoint),tracking_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    retargeted_reference=str(cfg.commands['motion'].motion_file),source_seed=cfg.seed,num_rollouts=32,
    simulated_duration_each_s=(int(cmd.motion.time_step_total)-2)*raw.step_dt,
    transitions=len(data),mean_forward_speed_mps=float(np.mean(velocity)),feature_dim=74,
    wraps=wraps,terminated=terminated,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    note='Reference-state initialization is used ONLY for collecting the teacher clip. Command-driven students always start stationary.')
(ROOT/'assets/dynamic_amp_expert.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));raw.close()
