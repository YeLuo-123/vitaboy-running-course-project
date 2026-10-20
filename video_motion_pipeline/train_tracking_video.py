"""Fine tune the official Tracking actor on the video, in its own experiment."""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from tracking_common import ROOT,MOTION,configure_tracking

parser=argparse.ArgumentParser()
parser.add_argument('--iterations',type=int,default=600)
parser.add_argument('--num-envs',type=int,default=2048)
parser.add_argument('--seed',type=int,default=42)
args=parser.parse_args()
repo=ROOT.parent/'honor_rl_mjlab'
os.chdir(repo);sys.path.insert(0,str(repo))
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ.setdefault('OMP_NUM_THREADS','4')
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls,register_mjlab_task
from scripts.train import TrainConfig,run_train
helper_spec=importlib.util.spec_from_file_location('_actor_import',ROOT.parent/'motion_pipeline/train_tracking_run.py')
helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
cfg=configure_tracking(load_env_cfg('VitaBoy-Tracking'),training=True)
cfg.scene.num_envs=args.num_envs
agent=load_rl_cfg('VitaBoy-Tracking');agent.seed=args.seed
agent.actor.distribution_cfg['init_std']=.25
agent.algorithm.learning_rate=1e-4;agent.algorithm.entropy_coef=.001
agent.max_iterations=args.iterations;agent.save_interval=100
agent.experiment_name='vita_boy_video_tracking';agent.run_name=f'user_video_seed{args.seed}'
log_dir=repo/'logs/rsl_rl'/agent.experiment_name/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+agent.run_name)
log_dir.mkdir(parents=True,exist_ok=True)
shutil.copy2(MOTION.parent/'manifest.json',log_dir/'retarget_manifest.json')
base=load_runner_cls('VitaBoy-Tracking')
class VideoRunner(base):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        result=helper.import_official_actor(self.alg.actor,repo/'deploy/checkpoints/tracking.onnx',self.device)
        (log_dir/'warm_start_validation.json').write_text(json.dumps(result,indent=2))
register_mjlab_task(task_id='VitaBoy-Video-Tracking',env_cfg=cfg,
                   play_env_cfg=configure_tracking(load_env_cfg('VitaBoy-Tracking',play=True)),rl_cfg=agent,runner_cls=VideoRunner)
(ROOT/'outputs/training_run.txt').write_text(str(log_dir))
print('VIDEO_TRACKING_LOG',log_dir,flush=True)
run_train('VitaBoy-Video-Tracking',TrainConfig(env=cfg,agent=agent,motion_file=str(MOTION)),log_dir)
