"""Visual source/target comparison. Both panels are kinematic references."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json
import sys
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent; WS=ROOT.parent
sys.path.insert(0,str(WS/'GMR_source'))
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

folder=ROOT/'data/retargeted/lafan_run2_s4_geometry'
manifest=json.loads((folder/'manifest.json').read_text())
source=ROOT/'data/raw'/manifest['source_file']
frames,_=load_bvh_file(str(source)); bvh=read_bvh(str(source))
st,en=manifest['source_frames_zero_based_half_open']
origin=np.array(frames[st]['Hips'][0]); origin[2]=0.
delta=np.array(frames[en-1]['Hips'][0])-np.array(frames[st]['Hips'][0])
alignment=Rotation.from_euler('z',-np.arctan2(delta[1],delta[0]))
positions=np.array([[alignment.apply(np.array(f[n][0])-origin) for n in bvh.bones] for f in frames[st:en]])
spec=mujoco.MjSpec.from_string('<mujoco><visual><global offwidth="640" offheight="480"/></visual><worldbody/></mujoco>')
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[30,30,.1],rgba=[.25,.3,.35,1])
spec.worldbody.add_light(pos=[0,-4,6],dir=[0,1,-1])
segments=[]
for j,parent in enumerate(bvh.parents):
 if parent<0: continue
 geom=spec.worldbody.add_geom(name=f'human_{j}',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
     size=[.025,.1,0],rgba=[.2,.65,.95,1],contype=0,conaffinity=0)
 segments.append((j,int(parent)))
spec.worldbody.add_geom(name='head',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.085,0,0],rgba=[.3,.7,1,1],contype=0,conaffinity=0)
model=spec.compile(); data=mujoco.MjData(model)
camera=mujoco.MjvCamera(); camera.type=mujoco.mjtCamera.mjCAMERA_FREE
camera.distance=3.25; camera.azimuth=90; camera.elevation=-12
renderer=mujoco.Renderer(model,height=480,width=640)
robot_reader=imageio.get_reader(folder/'reference_kinematic.mp4')
count=len(np.load(folder/'retargeted_qpos.npz')['qpos'])
sheet=Image.new('RGB',(1280,512*3),'white'); sheet_draw=ImageDraw.Draw(sheet)
with imageio.get_writer(folder/'human_to_vita_retarget.mp4',fps=50) as writer:
 for i in range(count):
  f=i*.02*30; j=min(int(f),len(positions)-1); k=min(j+1,len(positions)-1); blend=f-j
  xyz=positions[j]*(1-blend)+positions[k]*blend
  for child,parent in segments:
   gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'human_{child}')
   vec=xyz[child]-xyz[parent]; length=np.linalg.norm(vec)
   model.geom_pos[gid]=(xyz[child]+xyz[parent])/2
   model.geom_size[gid,1]=length/2
   rot,_=Rotation.align_vectors([vec/length],[[0,0,1]])
   model.geom_quat[gid]=rot.as_quat(scalar_first=True)
  model.geom_pos[model.geom('head').id]=xyz[bvh.bones.index('Head')]
  mujoco.mj_forward(model,data)
  camera.lookat[:]=xyz[0]+[0,0,-.1]
  renderer.update_scene(data,camera=camera)
  canvas=Image.new('RGB',(1280,512),'white')
  canvas.paste(Image.fromarray(renderer.render()),(0,30))
  canvas.paste(Image.fromarray(robot_reader.get_data(i)),(640,30))
  draw=ImageDraw.Draw(canvas)
  draw.text((8,8),f'LAFAN1 human source | original timing | t={i/50:.2f}s',fill='black')
  draw.text((648,8),'VitaBoy GMR target | kinematic reference, not dynamics proof',fill='black')
  writer.append_data(np.array(canvas))
  if i in [25,85,145]: sheet.paste(canvas,(0,[25,85,145].index(i)*512))
sheet.save(folder/'human_to_vita_frames.png')
robot_reader.close(); renderer.close()
print('SOURCE_TARGET_VIDEO',folder/'human_to_vita_retarget.mp4',flush=True)
