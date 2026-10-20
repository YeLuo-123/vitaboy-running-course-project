"""Interactive finite-clip Tracking replay. Pauses before any reference reset."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import torch
import viser
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer
from tracking_common import ROOT,configure_tracking

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--port',type=int,default=8091)
p.add_argument('--checkpoint',default=str(ROOT/'artifacts/model_500.pt'))
a=p.parse_args()
os.chdir(ROOT.parent/'honor_rl_mjlab')
configure_torch_backends()
cfg=configure_tracking(load_env_cfg('VitaBoy-Tracking',play=True))
cfg.scene.num_envs=1; cfg.seed=123
raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
env=RslRlVecEnvWrapper(raw)
runner=load_runner_cls('VitaBoy-Tracking')(env,asdict(load_rl_cfg('VitaBoy-Tracking')),device='cuda:0')
runner.load(a.checkpoint,load_cfg={'actor':True},strict=True,map_location='cuda:0')
actor=runner.get_inference_policy(device='cuda:0')
cmd=raw.command_manager.get_term('motion')
server=viser.ViserServer(host='127.0.0.1',port=a.port,label='VitaBoy retargeted running - Tracking')
info=server.gui.add_html('<b>元气仔跑步动作 · Tracking 仿真</b><br/>约 3.92 秒独立片段。点击“播放 / 重新播放”；末尾自动暂停，不循环传送机器人。')
viewer=None


def policy(obs):
    cursor=int(cmd.time_steps[0])
    if cursor>=int(cmd.motion.time_step_total)-2:
        viewer._is_paused=True
        viewer._pause_button.label='Play'
        viewer._pause_button.disabled=True
        viewer._step_button.disabled=True
        info.content='<b>片段完成，已自动暂停。</b><br/>点击“播放 / 重新播放”重置后再看一次。当前是短片段跟踪验证，尚非连续长跑。'
    return actor(obs)


class RunViewer(ViserPlayViewer):
    def setup(self):
        super().setup()
        self._is_paused=True
        self._pause_button.label='Play'
        replay=server.gui.add_button('播放 / 重新播放')
        @replay.on_click
        def _(_event):
            self._pause_button.disabled=False
            self._step_button.disabled=False
            self.request_reset()
            if self._is_paused: self.request_toggle_pause()
            info.content='<b>正在运行 Tracking 策略。</b><br/>参考速度约 1.82 m/s。片段末尾自动暂停；未使用虚拟支撑。'
        print(f'TRACKING_VIEWER_READY http://127.0.0.1:{a.port}',flush=True)


try:
    viewer=RunViewer(env,policy,frame_rate=30.,viser_server=server)
    viewer.run()
finally:
    env.close(); server.stop()
