"""Render a kinematic reference for inspection; this is not a policy test."""
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw

p=argparse.ArgumentParser()
p.add_argument('--name',default='lafan_run2_s4_geometry')
a=p.parse_args()
root=Path(__file__).resolve().parent
folder=root/'data/retargeted'/a.name
motion=np.load(folder/'retargeted_qpos.npz')
spec=mujoco.MjSpec.from_file(str(root.parent/'honor_rl_mjlab/src/whole_body_control/assets/robots/vita_boy/xml/main.xml'))
spec.worldbody.add_geom(name='reference_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[30,30,.1],rgba=[.25,.3,.35,1])
spec.worldbody.add_light(pos=[0,-4,6],dir=[0,1,-1])
model=spec.compile(); data=mujoco.MjData(model)
camera=mujoco.MjvCamera(); camera.type=mujoco.mjtCamera.mjCAMERA_FREE
camera.distance=2.7; camera.azimuth=90; camera.elevation=-12
renderer=mujoco.Renderer(model,height=480,width=640)
sheet=Image.new('RGB',(640*3,510*2),'white'); draw=ImageDraw.Draw(sheet)
snapshots=[15,45,75,105,135,165]
with imageio.get_writer(folder/'reference_kinematic.mp4',fps=50) as writer:
 for i,q in enumerate(motion['qpos']):
  data.qpos[:]=q; mujoco.mj_forward(model,data)
  camera.lookat[:]=q[:3]+[0,0,-.1]
  renderer.update_scene(data,camera=camera)
  frame=renderer.render(); writer.append_data(frame)
  if i in snapshots:
   k=snapshots.index(i); x=(k%3)*640; y=(k//3)*510
   sheet.paste(Image.fromarray(frame),(x,y+30))
   draw.text((x+8,y+8),f'Kinematic reference only | t={i/50:.2f}s',fill='black')
sheet.save(folder/'reference_frames.png')
renderer.close()
print('RENDERED',folder,flush=True)
