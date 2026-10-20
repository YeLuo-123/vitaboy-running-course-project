"""Retarget a publisher BVH clip to the actual VitaBoy MJCF with GMR/Mink."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WS = ROOT.parent
sys.path[:0] = [str(ROOT/'vendor'), str(WS/'GMR_source')]
import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp
from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.utils.lafan1 import load_bvh_file

p = argparse.ArgumentParser()
p.add_argument('--source', default='run2_subject4.bvh')
p.add_argument('--start', type=int, default=6940, help='zero based, inclusive')
p.add_argument('--end', type=int, default=7060, help='zero based, exclusive')
p.add_argument('--name', default='lafan_run2_s4_geometry')
a = p.parse_args()
out = ROOT/'data/retargeted'/a.name
out.mkdir(parents=True, exist_ok=True)
source = ROOT/'data/raw'/a.source
frames, human_height = load_bvh_file(str(source))
cfg = json.loads((WS/'GMR_source/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json').read_text())
cfg['robot_root_name'] = 'pelvis_link'
cfg['human_height_assumption'] = human_height
for body in cfg['human_scale_table']:
    cfg['human_scale_table'][body] = .75 if any(x in body for x in ('Arm','Hand')) else .84
for table in ('ik_match_table1','ik_match_table2'):
    cfg[table]['pelvis_link'] = cfg[table].pop('pelvis')
    # The VitaBoy hip-yaw origin lies farther down the thigh than the human
    # hip center. Track its orientation, without equating those positions.
    for side in ('left','right'):
        cfg[table][side+'_hip_yaw_link'][1] = 0.
    for body, entry in cfg[table].items():
        entry[4] = [1.,0.,0.,0.]
        if body == 'pelvis_link': entry[1:3] = [300. if table.endswith('2') else 0.,500.]
        elif body == 'torso_link': entry[1:3] = [0.,150.]
        elif 'ankle' in body: entry[1:3] = [100.,15.]
        elif 'hip_yaw' in body: entry[1:3] = [0.,15.]
        elif 'knee' in body: entry[1:3] = [15. if table.endswith('2') else 0.,15.]
        else: entry[1:3] = [0.,8. if 'wrist' not in body else 2.]
config_file = out/'bvh_lafan1_to_vita_boy.json'
config_file.write_text(json.dumps(cfg, indent=2))
asset = WS/'honor_rl_mjlab/src/whole_body_control/assets/robots/vita_boy/xml/main.xml'
ROBOT_XML_DICT['vita_boy_course'] = asset
IK_CONFIG_DICT['bvh_lafan1']['vita_boy_course'] = config_file
retargeter = GeneralMotionRetargeting('bvh_lafan1','vita_boy_course', verbose=False,
                                     solver='daqp', use_velocity_limit=False)
model = retargeter.model
model.opt.timestep = 1/30
joint_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
assert len(joint_names) == 29, joint_names
qinit = model.qpos0.copy()
for j,name in enumerate(joint_names):
    value = -.312 if 'hip_pitch' in name else .669 if 'knee' in name else -.363 if 'ankle_pitch' in name else .6 if 'elbow' in name else .2 if 'shoulder_pitch' in name else .2 if name=='left_shoulder_roll_joint' else -.2 if name=='right_shoulder_roll_joint' else 0.
    qinit[7+j] = value
retargeter.configuration.update(qinit)
positions = np.array([f['Hips'][0] for f in frames])
delta = positions[a.end-1,:2]-positions[a.start,:2]
yaw = np.arctan2(delta[1],delta[0])
alignment = Rotation.from_euler('z',-yaw)
origin = positions[a.start].copy(); origin[2] = 0.
start = max(0,a.start-60)
qposes = []
errors = []

def normalized(v):
    return v/max(np.linalg.norm(v),1.e-9)

def frame_from_z(z,y):
    z=normalized(z); y=normalized(y-z*(y@z)); x=normalized(np.cross(y,z))
    return Rotation.from_matrix(np.column_stack([x,y,z])).as_quat(scalar_first=True)

def frame_from_x(x,y):
    x=normalized(x); y=normalized(y-x*(y@x)); z=normalized(np.cross(x,y))
    return Rotation.from_matrix(np.column_stack([x,y,z])).as_quat(scalar_first=True)

def calibrated_frames(h):
    # Match anatomical segment directions to VitaBoy's own link axes. Its
    # forearm extends along +x at zero angle; copying G1's forearm offsets
    # (a different zero-pose geometry) produces incorrect elbow/shoulder IK.
    pos=lambda n: np.asarray(h[n][0])
    lateral=normalized(pos('LeftUpLeg')-pos('RightUpLeg'))
    lateral[2]=0.; lateral=normalized(lateral)
    root_q=frame_from_z(np.array([0.,0.,1.]),lateral)
    h['Hips'][1]=root_q
    h['Spine2'][1]=frame_from_z(pos('Spine2')-pos('Hips'),pos('LeftArm')-pos('RightArm'))
    for side in ('Left','Right'):
        h[side+'UpLeg'][1]=frame_from_z(pos(side+'UpLeg')-pos(side+'Leg'),lateral)
        h[side+'Leg'][1]=frame_from_z(pos(side+'Leg')-pos(side+'Foot'),lateral)
        foot_q=Rotation.from_quat(frame_from_x(pos(side+'Toe')-pos(side+'Foot'),lateral),scalar_first=True)
        h[side+'FootMod'][1]=(foot_q*Rotation.from_euler('y',-.25)).as_quat(scalar_first=True)
        h[side+'Arm'][1]=frame_from_z(pos(side+'Arm')-pos(side+'ForeArm'),lateral)
        h[side+'ForeArm'][1]=frame_from_x(pos(side+'Hand')-pos(side+'ForeArm'),lateral)
        h[side+'Hand'][1]=h[side+'ForeArm'][1].copy()
    return h

for i in range(start,a.end):
    human = {name:[alignment.apply(np.asarray(pos)-origin),
                  (alignment*Rotation.from_quat(quat,scalar_first=True)).as_quat(scalar_first=True)]
             for name,(pos,quat) in frames[i].items()}
    human=calibrated_frames(human)
    if i == start:
        initial=qinit.copy()
        initial[:3]=human['Hips'][0]*cfg['human_scale_table']['Hips']
        initial[3:7]=human['Hips'][1]
        retargeter.configuration.update(initial)
    q = retargeter.retarget(human)
    if i >= a.start:
        qposes.append(q)
        errors.append([retargeter.error1(),retargeter.error2()])
    if i % 30 == 0: print('IK_FRAME',i,flush=True)
qposes = np.array(qposes)
np.savez(out/'raw_gmr_qpos.npz',qpos=qposes, fps=30, joint_names=joint_names, errors=errors)
# Smooth a short temporal window, then resample without changing playback time.
qposes[:,:3] = savgol_filter(qposes[:,:3],5,3,axis=0)
qposes[:,7:] = savgol_filter(qposes[:,7:],5,3,axis=0)
times = np.arange(len(qposes))/30
target_t = np.arange(0,times[-1]+1.e-9,.02)
result = np.zeros((len(target_t),model.nq))
for j in list(range(3))+list(range(7,model.nq)):
    result[:,j] = np.interp(target_t,times,qposes[:,j])
result[:,3:7] = Slerp(times,Rotation.from_quat(qposes[:,3:7],scalar_first=True))(target_t).as_quat(scalar_first=True)
limits = model.jnt_range[1:]
before_clip = result[:,7:].copy()
result[:,7:] = np.clip(result[:,7:],limits[:,0]+1.e-4,limits[:,1]-1.e-4)
data = mujoco.MjData(model)
foot_ids = [[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'{side}_foot{k}_collision') for k in range(1,10)] for side in ['left','right']]
def soles(q):
    data.qpos[:] = q; mujoco.mj_forward(model,data)
    values=[]
    for ids in foot_ids:
        lowest=[]
        for gid in ids:
            z=data.geom_xpos[gid,2]
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                z -= np.abs(data.geom_xmat[gid].reshape(3,3)[2]) @ model.geom_size[gid]
            else: z -= model.geom_size[gid,0]
            lowest.append(z)
        values.append(min(lowest))
    return values
clearance = np.array([soles(q) for q in result])
ground_shift = .002-float(np.percentile(clearance.min(1),2))
result[:,2] += ground_shift
clearance += ground_shift
# Morphology changes ankle/sole offsets. Clean only the source-supported
# phases; interpolate the correction through source flight phases, preserving
# their relative height instead of pinning a foot to the floor at every frame.
source_feet = np.array([[min(f[side+'Foot'][0][2],f[side+'Toe'][0][2])
                        for side in ('Left','Right')] for f in frames])
human_floor = float(np.percentile(source_feet,1))
local_feet = source_feet[a.start:a.end]
human_min = np.interp(target_t,times,local_feet.min(1))
support = human_min < human_floor+.025
correction = np.interp(target_t,target_t[support],(.002-clearance.min(1))[support])
correction = savgol_filter(correction,7,3)
result[:,2] += correction
clearance += correction[:,None]
# CSV contract: xyz, quaternion xyzw, 29 joint positions in named natural order.
csv = np.column_stack([result[:,:3],result[:,[4,5,6,3]],result[:,7:]])
np.savetxt(out/'run.csv',csv,delimiter=',',fmt='%.9f')
np.savez(out/'retargeted_qpos.npz',qpos=result,fps=50,joint_names=joint_names,sole_clearance=clearance)
vel=np.gradient(result[:,7:],.02,axis=0)
speed=np.linalg.norm(result[-1,:2]-result[0,:2])/target_t[-1]
manifest=dict(source_file=a.source,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    source_url='https://github.com/ubisoft/ubisoft-laforge-animation-dataset',
    source_frames_zero_based_half_open=[a.start,a.end],source_fps=30,output_fps=50,
    duration_s=float(target_t[-1]),root_net_speed_mps=float(speed),
    playback_time_scale=1.,root_height_range=result[:,2].min().item(),
    root_height_max=result[:,2].max().item(),global_ground_shift_m=ground_shift,
    contact_cleanup_root_shift_range_m=[float(correction.min()),float(correction.max())],
    source_flight_fraction_above_2_5cm=float((~support).mean()),
    minimum_sole_clearance_m=float(clearance.min()),
    kinematic_flight_fraction_above_1cm=float((clearance>.01).all(1).mean()),
    limit_clipped_values=int(np.count_nonzero(before_clip!=result[:,7:])),
    maximum_abs_joint_velocity_rads=float(np.max(np.abs(vel))),
    joint_velocity_abs_p99_rads=float(np.percentile(np.abs(vel),99)),
    joint_names=joint_names,ik_config=str(config_file),robot_asset=str(asset),
    note='Kinematic retarget only. Flight and feasibility must be validated dynamically.')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('RETARGET_RESULT',json.dumps(manifest),flush=True)
