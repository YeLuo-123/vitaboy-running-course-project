"""Inspect reconstruction confidence and source/robot anatomical directions.

These metrics quantify retargeting relative to GVHMR, not video ground truth.
"""
from pathlib import Path
import json
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

ROOT=Path(__file__).resolve().parent
sk=np.load(ROOT/'outputs/human_skeleton.npz')
parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,default=ROOT/'outputs/vita_boy')
folder=parser.parse_args().folder
motion=np.load(folder/'retargeted_qpos.npz')
qpos=motion['qpos'];names=sk['joint_names'].tolist();pos=sk['positions'];fps=float(sk['fps'])
rootrot=R.from_quat(sk['quaternions_wxyz'][0,0],scalar_first=True)*R.from_euler('x',-90,degrees=True)*R.from_euler('z',-90,degrees=True)
heading=rootrot.apply([1,0,0]);align=R.from_euler('z',-np.arctan2(heading[1],heading[0]))
pos=align.apply(pos.reshape(-1,3)).reshape(pos.shape)
model=mujoco.MjModel.from_xml_path(str(ROOT.parent/'honor_rl_mjlab/src/whole_body_control/assets/robots/vita_boy/xml/main.xml'))
data=mujoco.MjData(model);results={}
curves={}
ts=np.arange(len(pos))/fps;tt=np.arange(len(qpos))/50
for side in ['left','right']:
    for tag,h1,h2,r1,r2 in [('upper_arm','shoulder','elbow','shoulder_yaw_link','elbow_link'),
                            ('forearm','elbow','wrist','elbow_link','wrist_yaw_link')]:
        sv=pos[:,names.index(side+'_'+h2)]-pos[:,names.index(side+'_'+h1)]
        source=np.column_stack([np.interp(tt,ts,sv[:,k]) for k in range(3)])
        robot=[]
        for q in qpos:
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            robot.append(data.body(side+'_'+r2).xpos-data.body(side+'_'+r1).xpos)
        robot=np.array(robot)
        cos=np.sum(source*robot,axis=1)/(np.linalg.norm(source,axis=1)*np.linalg.norm(robot,axis=1))
        angle=np.rad2deg(np.arccos(np.clip(cos,-1,1)))
        key=side+'_'+tag;curves[key]=angle
        results[key]=dict(mean_deg=float(angle.mean()),p95_deg=float(np.percentile(angle,95)),max_deg=float(angle.max()))
manifest=json.loads((folder/'manifest.json').read_text())
report=dict(anatomical_direction_error_to_GVHMR=results,kinematic_manifest=manifest,
            note='Reconstruction has no 3D ground truth; these errors measure GMR fidelity to estimated human joints, not true human pose accuracy.')
(folder/'quality_report.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for name,curve in curves.items():axes[0,0].plot(tt,curve,label=name)
axes[0,0].set(ylabel='Arm direction error (deg)',xlabel='Time (s)');axes[0,0].legend(fontsize=8)
axes[0,1].plot(tt,motion['sole_clearance'][:,0],label='left sole');axes[0,1].plot(tt,motion['sole_clearance'][:,1],label='right sole');axes[0,1].axhline(0,c='gray',ls='--');axes[0,1].set(ylabel='Sole clearance (m)',xlabel='Time (s)');axes[0,1].legend()
axes[1,0].plot(tt,qpos[:,2]);axes[1,0].set(ylabel='Robot root height (m)',xlabel='Time (s)')
vel=np.gradient(qpos[:,7:],.02,axis=0)
axes[1,1].plot(tt,np.max(np.abs(vel),axis=1));axes[1,1].set(ylabel='Largest joint speed (rad/s)',xlabel='Time (s)')
fig.suptitle('GVHMR -> GMR -> VitaBoy: kinematic inspection (no dynamics claim)')
fig.savefig(folder/'motion_quality.png',dpi=160);plt.close(fig)
print(json.dumps(results,indent=2))
