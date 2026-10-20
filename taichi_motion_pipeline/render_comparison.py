"""Aspect-preserving, synchronized Tai Chi source/human/robot video."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse
import json
import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,default=ROOT/'outputs/vita_boy')
folder=parser.parse_args().folder
sk=np.load(ROOT/'outputs/human_skeleton.npz')
motion=np.load(folder/'retargeted_qpos.npz')
names=sk['joint_names'].tolist();parents=sk['parents']
positions=sk['positions'];quats=sk['quaternions_wxyz'];fps=float(sk['fps'])
offset=R.from_euler('x',-90,degrees=True)*R.from_euler('z',-90,degrees=True)
rootrot=R.from_quat(quats[0,0],scalar_first=True)*offset
heading=rootrot.apply([1,0,0]);alignment=R.from_euler('z',-np.arctan2(heading[1],heading[0]))
origin=positions[0,0].copy();origin[2]=0
positions=alignment.apply((positions-origin).reshape(-1,3)).reshape(positions.shape)
ground=np.percentile(positions[:,[names.index('left_ankle'),names.index('right_ankle')],2],5)-.07
positions[:,:,2]-=ground
# The video has continuous ground support; display the human with the same
# floor-alignment prior as the robot. The raw skeleton remains unchanged.
ankle_min=positions[:,[names.index('left_ankle'),names.index('right_ankle')],2].min(1)
positions[:,:,2]+=(.07-ankle_min)[:,None]
hs=mujoco.MjSpec.from_string('<mujoco><visual><global offwidth="640" offheight="512"/></visual><worldbody/></mujoco>')
hs.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[20,20,.1],rgba=[.3,.35,.4,1])
hs.worldbody.add_light(pos=[2,-4,6],dir=[0,1,-1])
segments=[(j,int(parents[j])) for j in range(1,22)]
for j,parent in segments:
    b=hs.worldbody.add_body(name=f'bone_body_{j}',mocap=True)
    b.add_geom(name=f'bone_{j}',type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=[.027,.1,0],rgba=[.2,.65,.95,1],contype=0,conaffinity=0)
b=hs.worldbody.add_body(name='head_body',mocap=True)
b.add_geom(name='head',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.085,0,0],rgba=[.3,.7,1,1],contype=0,conaffinity=0)
hm=hs.compile();hd=mujoco.MjData(hm)
rs=mujoco.MjSpec.from_file(str(ROOT.parent/'honor_rl_mjlab/src/whole_body_control/assets/robots/vita_boy/xml/main.xml'))
rs.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[20,20,.1],rgba=[.3,.35,.4,1])
rs.worldbody.add_light(pos=[2,-4,6],dir=[0,1,-1])
rm=rs.compile();rm.vis.global_.offheight=512;rd=mujoco.MjData(rm)
hr=mujoco.Renderer(hm,height=512,width=512);rr=mujoco.Renderer(rm,height=512,width=512)
camh=mujoco.MjvCamera();camr=mujoco.MjvCamera()
for cam in [camh,camr]:
    cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.azimuth=0;cam.elevation=-12;cam.distance=3.3
camr.distance=2.8
reader=iio.get_reader(ROOT/'input/user_video.mp4')
roi=json.loads((ROOT/'inspection/display_roi.json').read_text())['xyxy']
qpos=motion['qpos'];snapshots=np.linspace(0,len(qpos)-1,8).astype(int).tolist()
sheet=Image.new('RGB',(1536,544*len(snapshots)),'white')
with iio.get_writer(folder/'taichi_to_vita_comparison.mp4',fps=50,codec='libx264',quality=8) as writer, \
     iio.get_writer(folder/'taichi_vita_kinematic.mp4',fps=50,codec='libx264',quality=8) as robot_writer:
    for i,q in enumerate(qpos):
        f=i/50*fps;j=min(int(f),len(positions)-1);k=min(j+1,len(positions)-1);t=f-j
        xyz=positions[j]*(1-t)+positions[k]*t
        for child,parent in segments:
            g=hm.geom(f'bone_{child}').id;v=xyz[child]-xyz[parent];length=np.linalg.norm(v)
            mid=hm.body(f'bone_body_{child}').mocapid[0]
            hd.mocap_pos[mid]=(xyz[child]+xyz[parent])/2;hm.geom_size[g,1]=length/2
            rot,_=R.align_vectors([v/max(length,1e-9)],[[0,0,1]]);hd.mocap_quat[mid]=rot.as_quat(scalar_first=True)
        hd.mocap_pos[hm.body('head_body').mocapid[0]]=xyz[names.index('head')]+[0,0,.05]
        mujoco.mj_forward(hm,hd);camh.lookat[:]=xyz[0]+[0,0,.05];hr.update_scene(hd,camera=camh)
        rd.qpos[:]=q;mujoco.mj_forward(rm,rd);camr.lookat[:]=q[:3]+[0,0,.05];rr.update_scene(rd,camera=camr)
        human_img=hr.render();robot_img=rr.render()
        canvas=Image.new('RGB',(1536,544),'white')
        source=Image.fromarray(reader.get_data(min(round(f),len(positions)-1))).crop(roi)
        source=ImageOps.contain(source,(512,512))
        canvas.paste(source,((512-source.width)//2,32+(512-source.height)//2))
        canvas.paste(Image.fromarray(human_img),(512,32));canvas.paste(Image.fromarray(robot_img),(1024,32))
        d=ImageDraw.Draw(canvas)
        d.text((8,10),f'Tai Chi input | fixed crop | t={i/50:.2f}s',fill='black')
        d.text((524,10),'GVHMR human + ground alignment',fill='black');d.text((1036,10),'GMR VitaBoy | kinematic reference',fill='black')
        writer.append_data(np.asarray(canvas));robot_writer.append_data(robot_img)
        if i in snapshots:sheet.paste(canvas,(0,snapshots.index(i)*544))
        if i%250==0:print('RENDER_FRAME',i,len(qpos),flush=True)
sheet.save(folder/'comparison_frames.jpg')
hr.close();rr.close();reader.close()
print('RENDERED',folder/'taichi_to_vita_comparison.mp4',flush=True)
