"""Interactive finite-clip dynamics replay for this video's Tracking actor."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import torch
import viser
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer
from tracking_common import ROOT,configure_tracking

parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--port',type=int,default=8090)
args=parser.parse_args()
os.chdir(ROOT.parent/'honor_rl_mjlab')
configure_torch_backends()
cfg=configure_tracking(load_env_cfg('VitaBoy-Tracking',play=True))
cfg.scene.num_envs=1;cfg.seed=123
raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
env=RslRlVecEnvWrapper(raw)
runner=load_runner_cls('VitaBoy-Tracking')(env,asdict(load_rl_cfg('VitaBoy-Tracking')),device='cuda:0')
runner.load(args.checkpoint,load_cfg={'actor':True},strict=True,map_location='cuda:0')
actor=runner.get_inference_policy(device='cuda:0')
cmd=raw.command_manager.get_term('motion')
server=viser.ViserServer(host='127.0.0.1',port=args.port,label='VitaBoy - user video - Tracking')
info=server.gui.add_html('<b>用户视频动作 · Tracking 实验回放</b><br/>本检查点可完成片段，尚未通过跟踪精度验收。点击“播放 / 重新播放”；片段末尾自动暂停。')
viewer=None

def policy(obs):
    if int(cmd.time_steps[0])>=int(cmd.motion.time_step_total)-2:
        viewer._is_paused=True
        viewer._pause_button.label='Play';viewer._pause_button.disabled=True;viewer._step_button.disabled=True
        info.content='<b>视频动作片段已完成，自动暂停。</b><br/>点击“播放 / 重新播放”重置后再看一次。当前模型尚未通过跟踪精度验收；三栏 MP4 展示运动学重定向参考。'
    return actor(obs)

class VideoViewer(ViserPlayViewer):
    def setup(self):
        super().setup();self._is_paused=True;self._pause_button.label='Play'
        replay=server.gui.add_button('播放 / 重新播放')
        @replay.on_click
        def _(_event):
            self._pause_button.disabled=False;self._step_button.disabled=False
            self.request_reset()
            if self._is_paused:self.request_toggle_pause()
            info.content='<b>正在执行用户视频的 Tracking 实验策略。</b><br/>未使用虚拟支撑。当前模型尚未通过跟踪精度验收；片段末尾自动暂停。'
        print(f'VIDEO_TRACKING_VIEWER_READY http://127.0.0.1:{args.port}',flush=True)

try:
    viewer=VideoViewer(env,policy,frame_rate=30.,viser_server=server)
    viewer.run()
finally:
    env.close();server.stop()
