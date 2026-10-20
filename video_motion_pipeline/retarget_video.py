"""Retarget GVHMR SMPL-X global motion to the actual 29-DoF VitaBoy.

Run with the existing mjlab Python; reconstruction is isolated in another env.
The intermediate skeleton is numeric NPZ, avoiding cross-environment pickles.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WS = ROOT.parent
sys.path[:0] = [str(WS/'motion_pipeline/vendor'), str(WS/'GMR_source')]
import mujoco
import mink
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp
from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton', type=Path, default=ROOT/'outputs/human_skeleton.npz')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/vita_boy')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sk = np.load(args.skeleton)
    names = sk['joint_names'].tolist()
    positions, quats = sk['positions'], sk['quaternions_wxyz']
    fps = float(sk['fps'])
    assert len(positions) >= 10 and np.isfinite(positions).all()
    asset = WS/'honor_rl_mjlab/src/whole_body_control/assets/robots/vita_boy/xml/main.xml'
    model = mujoco.MjModel.from_xml_path(str(asset))
    model.opt.timestep = 1/fps
    data = mujoco.MjData(model)
    joints = [model.joint(i).name for i in range(1,model.njnt)]
    assert len(joints) == 29

    # SMPL-X zero pose has straight arms out sideways. Calibrate each VitaBoy
    # link in that SAME anatomical pose, including its +x forearm axis.
    tpose = model.qpos0.copy()
    for side,sign in [('left',1),('right',-1)]:
        tpose[7+joints.index(side+'_shoulder_roll_joint')] = sign*np.pi/2
        tpose[7+joints.index(side+'_elbow_joint')] = np.pi/2
    data.qpos[:] = tpose
    mujoco.mj_forward(model,data)
    human_zero = R.from_euler('x',90,degrees=True)
    target_zero = R.from_euler('z',-90,degrees=True)
    pairs = {'pelvis_link':'pelvis', 'torso_link':'spine3'}
    for side in ['left','right']:
        pairs.update({side+'_hip_yaw_link':side+'_hip',
                      side+'_knee_link':side+'_knee',
                      side+'_ankle_roll_link':side+'_ankle',
                      side+'_shoulder_yaw_link':side+'_shoulder',
                      side+'_elbow_link':side+'_elbow',
                      side+'_wrist_yaw_link':side+'_wrist'})
    offsets={body:(human_zero.inv()*target_zero*R.from_matrix(data.body(body).xmat.reshape(3,3))).as_quat(scalar_first=True).tolist()
             for body in pairs}
    config = dict(robot_root_name='pelvis_link',human_root_name='pelvis',
                  ground_height=0.,human_height_assumption=float(sk['human_height']),
                  use_ik_match_table1=True,use_ik_match_table2=True,
                  human_scale_table={name:(.78 if any(x in name for x in ['shoulder','elbow','wrist']) else .84)
                                     for name in pairs.values()})
    for phase in [1,2]:
        table={}
        for body,human in pairs.items():
            poscost,rotcost=0.,15.
            if body=='pelvis_link':poscost,rotcost=(300. if phase==2 else 0.),500.
            elif body=='torso_link':rotcost=150.
            elif 'ankle' in body:poscost,rotcost=150.,20.
            elif 'knee' in body:poscost=15. if phase==2 else 0.
            elif 'shoulder' in body:rotcost=1.
            elif 'elbow' in body:poscost,rotcost=80.,1.
            elif 'wrist' in body:poscost,rotcost=120.,.2
            table[body]=[human,poscost,rotcost,[0.,0.,0.],offsets[body]]
        config['ik_match_table'+str(phase)]=table
    config_file=out/'smplx_to_vita_boy.json'
    config_file.write_text(json.dumps(config,indent=2))
    ROBOT_XML_DICT['vita_boy_video']=asset
    IK_CONFIG_DICT['smplx']['vita_boy_video']=config_file
    retargeter=GeneralMotionRetargeting('smplx','vita_boy_video',
                    actual_human_height=float(sk['human_height']),verbose=False,solver='daqp',use_velocity_limit=False)
    retargeter.model.opt.timestep=1/fps
    origin=positions[0,names.index('pelvis')].copy();origin[2]=0.
    root_rotation=R.from_quat(quats[0,names.index('pelvis')],scalar_first=True)*R.from_quat(offsets['pelvis_link'],scalar_first=True)
    heading=root_rotation.apply([1,0,0]); yaw=np.arctan2(heading[1],heading[0])
    alignment=R.from_euler('z',-yaw)
    aligned_pos=alignment.apply((positions-origin).reshape(-1,3)).reshape(positions.shape)
    aligned_quat=(alignment*R.from_quat(quats.reshape(-1,4),scalar_first=True)).as_quat(scalar_first=True).reshape(quats.shape)
    human_frames=[{name:(aligned_pos[i,j],aligned_quat[i,j]) for j,name in enumerate(names)} for i in range(len(positions))]
    init=model.qpos0.copy()
    init[:3]=aligned_pos[0,names.index('pelvis')]*.84
    init[3:7]=(alignment*root_rotation).as_quat(scalar_first=True)
    for j,name in enumerate(joints):
        init[7+j]=(-.312 if 'hip_pitch' in name else .669 if 'knee' in name else -.363 if 'ankle_pitch' in name
                   else .6 if 'elbow' in name else .2 if 'shoulder_pitch' in name else .2 if name=='left_shoulder_roll_joint'
                   else -.2 if name=='right_shoulder_roll_joint' else 0.)
    retargeter.configuration.update(init)
    posture_cost=np.zeros(model.nv)
    for j,name in enumerate(joints):
        if 'wrist' in name:posture_cost[6+j]=.5 if 'roll' in name else 2.
        elif 'shoulder_yaw' in name:posture_cost[6+j]=.05
    posture=mink.PostureTask(retargeter.model,cost=posture_cost)
    posture.set_target(init)
    retargeter.tasks1.append(posture);retargeter.tasks2.append(posture)
    for _ in range(50):retargeter.retarget(human_frames[0])
    arm_ids=[7+j for j,n in enumerate(joints) if any(t in n for t in ['shoulder','elbow','wrist'])]
    def fit_cost(q,frame):
        retargeter.configuration.update(q)
        d=retargeter.configuration.data;targets=retargeter.scaled_human_data
        cost=0.
        for side in ['left','right']:
            for tag,body in [('elbow','elbow_link'),('wrist','wrist_yaw_link')]:
                cost+=float(np.sum((d.body(side+'_'+body).xpos-targets[side+'_'+tag][0])**2))
            src=frame[side+'_elbow'][0]-frame[side+'_shoulder'][0]
            dst=d.body(side+'_elbow_link').xpos-d.body(side+'_shoulder_yaw_link').xpos
            cost+=.01*(1-np.clip(src@dst/np.linalg.norm(src)/np.linalg.norm(dst),-1,1))
        return cost
    poses=[]
    for i,frame in enumerate(human_frames):
        previous=retargeter.configuration.data.qpos.copy()
        seeds=[previous]
        for overhead in [False,True]:
            seed=previous.copy();seed[arm_ids]=init[arm_ids]
            for side,sign in [('left',1),('right',-1)]:
                seed[7+joints.index(side+'_elbow_joint')]=1.3
                if overhead:
                    seed[7+joints.index(side+'_shoulder_pitch_joint')]=-2.7
                    seed[7+joints.index(side+'_shoulder_roll_joint')]=sign*.8
            seeds.append(seed)
        candidates=[]
        for seed in seeds:
            retargeter.configuration.update(seed)
            q=retargeter.retarget(frame)
            cost=fit_cost(q,frame)+.0002*float(np.mean((q[arm_ids]-previous[arm_ids])**2))
            candidates.append((cost,q))
        _,q=min(candidates,key=lambda c:c[0])
        retargeter.configuration.update(q);poses.append(q)
        if i%30==0:print('RETARGET_FRAME',i,flush=True)
    raw=np.array(poses)
    np.savez(out/'raw_gmr_qpos.npz',qpos=raw,fps=fps,joint_names=joints)
    smooth=raw.copy()
    smooth[:,:3]=savgol_filter(raw[:,:3],5,3,axis=0)
    smooth[:,7:]=savgol_filter(raw[:,7:],7,3,axis=0)
    ts=np.arange(len(raw))/fps; tt=np.arange(0,ts[-1]+1e-8,.02)
    qpos=np.zeros((len(tt),model.nq))
    for j in list(range(3))+list(range(7,model.nq)):qpos[:,j]=np.interp(tt,ts,smooth[:,j])
    qpos[:,3:7]=Slerp(ts,R.from_quat(smooth[:,3:7],scalar_first=True))(tt).as_quat(scalar_first=True)
    limits=model.jnt_range[1:];before=qpos[:,7:].copy()
    qpos[:,7:]=np.clip(qpos[:,7:],limits[:,0]+1e-4,limits[:,1]-1e-4)
    feet=[[model.geom(f'{side}_foot{k}_collision').id for k in range(1,10)] for side in ['left','right']]
    def soles(q):
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        return [min(data.geom_xpos[g,2]-(np.abs(data.geom_xmat[g].reshape(3,3)[2])@model.geom_size[g]
                       if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX else model.geom_size[g,0]) for g in ids) for ids in feet]
    clearance=np.array([soles(q) for q in qpos])
    shift=.003-float(np.percentile(clearance.min(1),2))
    qpos[:,2]+=shift;clearance+=shift
    # Correct morphology-dependent sole offsets only during inferred source
    # support. Keep the source vertical jump trajectory through flight phases.
    source_min=aligned_pos[:,[names.index('left_ankle'),names.index('right_ankle')],2].min(1)
    floor=np.percentile(source_min,5)
    support=np.interp(tt,ts,source_min)<floor+.035
    correction=np.interp(tt,tt[support],(.003-clearance.min(1))[support])
    correction=savgol_filter(correction,7,3)
    qpos[:,2]+=correction;clearance+=correction[:,None]
    assert np.isfinite(qpos).all()
    np.savez(out/'retargeted_qpos.npz',qpos=qpos,fps=50,joint_names=joints,sole_clearance=clearance)
    np.savetxt(out/'video_motion.csv',np.column_stack([qpos[:,:3],qpos[:,[4,5,6,3]],qpos[:,7:]]),delimiter=',',fmt='%.9f')
    velocities=np.gradient(qpos[:,7:],.02,axis=0)
    report=dict(source='GVHMR smpl_params_global',source_skeleton_sha256=hashlib.sha256(args.skeleton.read_bytes()).hexdigest(),
                source_frames=len(raw),source_fps=fps,output_frames=len(tt),output_fps=50,duration_s=float(tt[-1]),
                calibration='SMPL-X neutral T-pose to actual VitaBoy T-pose; per-link constant orientation offsets',
                global_ground_shift_m=shift,contact_correction_range_m=[float(correction.min()),float(correction.max())],
                root_height_range_m=[float(qpos[:,2].min()),float(qpos[:,2].max())],
                minimum_sole_clearance_m=float(clearance.min()),clipped_values=int(np.count_nonzero(before!=qpos[:,7:])),
                max_joint_velocity_rads=float(np.abs(velocities).max()),joint_velocity_p99_rads=float(np.percentile(np.abs(velocities),99)),
                joint_names=joints,scope='Kinematic reference only; dynamic Tracking is a separate validation.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
