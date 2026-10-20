"""Matched-camera inspection of the worst original self-collisions."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from self_collision import ROOT,ASSET,ArmCollisionChecker

folder=ROOT/'outputs/vita_boy_collision_free'
before=np.load(ROOT/'outputs/vita_boy/retargeted_qpos.npz')['qpos']
after=np.load(folder/'retargeted_qpos.npz')['qpos']
spec=mujoco.MjSpec.from_file(str(ASSET))
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[10,10,.1],rgba=[.25,.29,.32,1])
spec.worldbody.add_light(pos=[2,-3,5],dir=[0,1,-1])
model=spec.compile();model.vis.global_.offwidth=640;model.vis.global_.offheight=512
data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=512,width=640)
checker=ArmCollisionChecker(model);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.azimuth=0;cam.elevation=-10;cam.distance=1.9
sheet=Image.new('RGB',(1280,3*552),'white');draw=ImageDraw.Draw(sheet)
for row,t in enumerate([27.8,34.24,42.52]):
    i=round(t*50)
    forward=Rotation.from_quat(before[i,3:7],scalar_first=True).apply([1,0,0])
    cam.azimuth=float(np.degrees(np.arctan2(forward[1],forward[0]))+20)
    for col,(label,motion) in enumerate([('BEFORE',before),('AFTER',after)]):
        data.qpos[:]=motion[i];mujoco.mj_forward(model,data)
        cam.lookat[:]=data.body('torso_link').xpos+[0,0,.03]
        renderer.update_scene(data,cam)
        sheet.paste(Image.fromarray(renderer.render()),(col*640,row*552+40))
        distance=checker.distances(motion[i]).min()*1000
        draw.text((col*640+12,row*552+12),f'{label} | t={t:.2f}s | minimum gap {distance:.1f} mm',fill='black')
sheet.save(folder/'self_collision_before_after.jpg',quality=95)
renderer.close()
