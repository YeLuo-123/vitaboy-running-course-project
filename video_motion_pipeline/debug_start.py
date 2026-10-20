import sys,os,json
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path('../video_motion_pipeline').resolve()))
import torch,mjlab.tasks
from tracking_common import configure_tracking,ROOT
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
cfg=configure_tracking(load_env_cfg('VitaBoy-Tracking',play=True));cfg.scene.num_envs=1
raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');vec=RslRlVecEnvWrapper(raw)
runner=load_runner_cls('VitaBoy-Tracking')(vec,asdict(load_rl_cfg('VitaBoy-Tracking')),device='cuda:0')
run=Path((ROOT/'outputs/training_run.txt').read_text().strip());runner.load(str(run/'model_200.pt'),load_cfg={'actor':True},strict=True,map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
with torch.inference_mode():
 obs,_=vec.reset();robot=raw.scene['robot'];cmd=raw.command_manager.get_term('motion')
 print('INITIAL',robot.data.root_link_lin_vel_w[0].tolist(),robot.data.root_link_ang_vel_w[0].tolist(),'max_qvel',float(robot.data.joint_vel.abs().max()))
 for i in range(20):
  action=policy(obs);obs,_,done,_=vec.step(action)
  print('STEP',i,'done',bool(done[0]),'height',float(robot.data.root_link_pos_w[0,2]),'maxaction',float(action.abs().max()),'qerr',float((robot.data.joint_pos-cmd.joint_pos).square().mean().sqrt()))
  if bool(done[0]):
   print('REASONS',{n:bool(raw.termination_manager.get_term(n)[0]) for n in raw.termination_manager.active_terms});break
vec.close()
