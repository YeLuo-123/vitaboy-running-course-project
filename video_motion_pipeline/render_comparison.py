"""Render source video / GVHMR skeleton / GMR VitaBoy, with shared timing."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parent
folder=ROOT/'outputs/vita_boy'
sk=np.load(ROOT/'outputs/human_skeleton.npz')
motion=np.load(folder/'retargeted_qpos.npz')
names=sk['joint_names'].tolist(); parents=sk['parents']
positions=sk['positions']; quats=sk['quaternions_wxyz'];fps=float(sk['fps'])
offset=R.from_euler('x',-90,degrees=True)*R.from_euler('z',-90,degrees=True)
rootrot=R.from_quat(quats[0,0],scalar_first=True)*offset
heading=rootrot.apply([1,0,0]); alignment=R.from_euler('z',-np.arctan2(heading[1],heading[0]))
origin=positions[0,0].copy();origin[2]=0
positions=alignment.apply((positions-origin).reshape(-1,3)).reshape(positions.shape)
ground=np.percentile(positions[:,[names.index('left_ankle'),names.index('right_ankle')],2],5)-.07
positions[:,:,2]-=ground
hs=mujoco.MjSpec.from_string('<mujoco><visual><global offwidth="640" offheight="512"/></visual><worldbody/></mujoco>')
hs.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[10,10,.1],rgba=[.22,.27,.32,1])
hs.worldbody.add_light(pos=[2,-4,6],dir=[0,1,-1])
segments=[(j,int(parents[j])) for j in range(1,22)]
for j,parent in segments:
    bone=hs.worldbody.add_body(name=f'bone_body_{j}',mocap=True)
    bone.add_geom(name=f'bone_{j}',type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=[.027,.1,0],rgba=[.2,.65,.95,1],contype=0,conaffinity=0)
head=hs.worldbody.add_body(name='head_body',mocap=True)
head.add_geom(name='head',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.085,0,0],rgba=[.3,.7,1,1],contype=0,conaffinity=0)
hm=hs.compile();hd=mujoco.MjData(hm)
rs=mujoco.MjSpec.from_file(str(ROOT.parent/'honor_rl_mjlab/src/whole_body_control/assets/robots/vita_boy/xml/main.xml'))
rs.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[10,10,.1],rgba=[.22,.27,.32,1])
rs.worldbody.add_light(pos=[2,-4,6],dir=[0,1,-1])
rm=rs.compile();rm.vis.global_.offheight=512;rd=mujoco.MjData(rm)
hr=mujoco.Renderer(hm,height=512,width=496);rr=mujoco.Renderer(rm,height=512,width=496)
camh=mujoco.MjvCamera();camr=mujoco.MjvCamera()
for cam in [camh,camr]:
    cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.azimuth=0;cam.elevation=-12;cam.distance=3.3
camr.distance=2.8
reader=iio.get_reader(ROOT/'input/user_video.mp4')
qpos=motion['qpos'];snapshots=np.linspace(0,len(qpos)-1,6).astype(int).tolist()
sheet=Image.new('RGB',(1280,544*6),'white')
with iio.get_writer(folder/'video_to_vita_comparison.mp4',fps=50,codec='libx264',quality=8) as writer:
    for i,q in enumerate(qpos):
        f=i/50*fps;j=min(int(f),len(positions)-1);k=min(j+1,len(positions)-1);t=f-j
        xyz=positions[j]*(1-t)+positions[k]*t
        for child,parent in segments:
            g=hm.geom(f'bone_{child}').id;vec=xyz[child]-xyz[parent];length=np.linalg.norm(vec)
            mocap=hm.body(f'bone_body_{child}').mocapid[0]
            hd.mocap_pos[mocap]=(xyz[child]+xyz[parent])/2;hm.geom_size[g,1]=length/2
            rot,_=R.align_vectors([vec/max(length,1e-9)],[[0,0,1]]);hd.mocap_quat[mocap]=rot.as_quat(scalar_first=True)
        hd.mocap_pos[hm.body('head_body').mocapid[0]]=xyz[names.index('head')]+[0,0,.05]
        mujoco.mj_forward(hm,hd);camh.lookat[:]=xyz[0]+[0,0,.05];hr.update_scene(hd,camera=camh)
        rd.qpos[:]=q;mujoco.mj_forward(rm,rd);camr.lookat[:]=q[:3]+[0,0,.05];rr.update_scene(rd,camera=camr)
        canvas=Image.new('RGB',(1280,544),'white')
        original=Image.fromarray(reader.get_data(min(round(f),len(positions)-1))).resize((288,512))
        canvas.paste(original,(0,32));canvas.paste(Image.fromarray(hr.render()),(288,32));canvas.paste(Image.fromarray(rr.render()),(784,32))
        d=ImageDraw.Draw(canvas);d.text((8,10),f'Input video | t={i/50:.2f}s',fill='black');d.text((300,10),'GVHMR reconstructed human',fill='black');d.text((796,10),'GMR VitaBoy | kinematic reference',fill='black')
        writer.append_data(np.asarray(canvas))
        if i in snapshots:sheet.paste(canvas,(0,snapshots.index(i)*544))
sheet.save(folder/'comparison_frames.jpg')
hr.close();rr.close();reader.close()
print('RENDERED',folder/'video_to_vita_comparison.mp4')
