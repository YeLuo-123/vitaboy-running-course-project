"""Validate exported running actor through the official Python deployment state."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent; REPO=ROOT.parent/'honor_rl_mjlab'
sys.path.insert(0,str(REPO/'deploy'))
from common.config import load_config,load_state_config,joint_array
from common.policy import OnnxPolicy,read_metadata
from common.robot_interface import JOINT_NAMES
from inference.mujoco_robot import MuJoCoRobot
from fsm.tracking.tracking import TrackingState
from fsm.tracking.motion import Motion

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--policy',required=True)
p.add_argument('--motion',default=str(ROOT/'data/retargeted/lafan_run2_s4_geometry/run.npz'))
p.add_argument('--output',required=True)
p.add_argument('--video')
a=p.parse_args()
model_path=Path(a.policy).resolve(); motion_path=Path(a.motion).resolve()
reference=np.load(motion_path,allow_pickle=False)
config=load_state_config(REPO/'deploy/fsm/tracking/config.yaml')
robot=MuJoCoRobot(load_config(REPO/'deploy/config.toml'),launch_viewer=False)
asset_model=mujoco.MjModel.from_xml_path(str(REPO/'src/whole_body_control/assets/robots/vita_boy/xml/main.xml'))
all_names=[mujoco.mj_id2name(asset_model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,asset_model.nbody)]
assert len(all_names)==reference['body_pos_w'].shape[1]
metadata=read_metadata(model_path)
assert tuple(metadata['joint_names'].split(','))==JOINT_NAMES
selected=metadata['body_names'].split(',')
ref_ids=[all_names.index(n) for n in selected]
body_ids=[mujoco.mj_name2id(robot.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in selected]
# Inject selected files into the unchanged official TrackingState methods,
# without overwriting its default dance configuration or bundled policies.
state=TrackingState.__new__(TrackingState)
state.default_angles=joint_array(config,'default_angles')
state.stiffness=joint_array(config,'kps'); state.damping=joint_array(config,'kds')
state.effort_limit=joint_array(config,'effort_limit')
state.action_scale=float(config['action_scale'])*state.effort_limit/state.stiffness
state.motion_torso_index=all_names.index('torso_link')
state.policy=OnnxPolicy(model_path,observation_size=154)
state.motion=Motion.load(motion_path)
state.action=np.zeros(29,dtype=np.float32)
assert np.isclose(state.motion.fps,1/robot.control_dt)
robot.data.qpos[:3]=reference['body_pos_w'][0,0]
robot.data.qpos[3:7]=reference['body_quat_w'][0,0]
robot.data.qpos[robot.qpos_indices]=reference['joint_pos'][0]
robot.data.qvel[:3]=reference['body_lin_vel_w'][0,0]
rotation=Rotation.from_quat(reference['body_quat_w'][0,0],scalar_first=True).as_matrix()
robot.data.qvel[3:6]=rotation.T@reference['body_ang_vel_w'][0,0]
robot.data.qvel[robot.dof_indices]=reference['joint_vel'][0]
mujoco.mj_forward(robot.model,robot.data)
state.enter(robot.get_state())
renderer=None; writer=None
if a.video:
 import imageio.v2 as imageio
 renderer=mujoco.Renderer(robot.model,height=480,width=640)
 camera=mujoco.MjvCamera(); camera.type=mujoco.mjtCamera.mjCAMERA_FREE
 camera.distance=2.7; camera.azimuth=90; camera.elevation=-12
 writer=imageio.get_writer(a.video,fps=50)
joint_errors=[]; body_errors=[]; root_errors=[]; velocities=[]; speed_errors=[]
foot_counts=[]; foot_flags=[]; max_torque=0.; failed=False; trajectory=[]
steps=reference['joint_pos'].shape[0]-2
for i in range(steps):
 target=state.step(robot.get_state(),np.zeros(3,dtype=np.float32))
 assert not target.stabilize_base
 robot.send_command(target)
 t=min(i+1,reference['joint_pos'].shape[0]-1)
 pose=robot.get_state()
 if pose.base_height<.45 or pose.projected_gravity[2]>-.7071 or not np.isfinite(robot.data.qpos).all():
  failed=True; break
 qerr=np.mean((robot.data.qpos[robot.qpos_indices]-reference['joint_pos'][t])**2)
 actual=robot.data.xpos[body_ids]; ref=reference['body_pos_w'][t,ref_ids]
 bodyerr=np.mean(np.sum(((actual-actual[:1])-(ref-ref[:1]))**2,axis=-1))
 rooterr=np.sum((robot.data.qpos[:3]-reference['body_pos_w'][t,0])**2)
 r=Rotation.from_quat(robot.data.qpos[3:7],scalar_first=True).as_matrix()
 vx=float((r.T@robot.data.qvel[:3])[0])
 refspeed=float(np.linalg.norm(reference['body_lin_vel_w'][t,0,:2]))
 joint_errors.append(qerr); body_errors.append(bodyerr); root_errors.append(rooterr)
 velocities.append(vx); speed_errors.append(abs(vx-refspeed))
 contact=[False,False]
 for k in range(robot.data.ncon):
  c=robot.data.contact[k]
  names=[mujoco.mj_id2name(robot.model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in (c.geom1,c.geom2)]
  if not any('floor' in n or 'ground' in n for n in names): continue
  force=np.zeros(6); mujoco.mj_contactForce(robot.model,robot.data,k,force)
  if np.linalg.norm(force[:3])<=1.: continue
  for j,side in enumerate(['left','right']):
   if any(n.startswith(side+'_foot') for n in names): contact[j]=True
 foot_counts.append(sum(contact)); foot_flags.append(contact)
 max_torque=max(max_torque,float(np.max(np.abs(robot.data.qfrc_actuator[robot.dof_indices]))))
 trajectory.append(dict(t=(i+1)*robot.control_dt,reference_cursor=t,
    root_actual=robot.data.qpos[:3].tolist(),root_reference=reference['body_pos_w'][t,0].tolist(),
    joint_rmse=float(np.sqrt(qerr)),relative_body_rmse=float(np.sqrt(bodyerr)),
    forward_velocity=vx,reference_speed=refspeed,foot_contact=contact,alive=True,
    knee_actual=robot.data.qpos[np.array(robot.qpos_indices)[[3,9]]].tolist(),
    knee_reference=reference['joint_pos'][t,[3,9]].tolist()))
 if renderer is not None:
  camera.lookat[:]=robot.data.qpos[:3]+[0,0,-.1]
  renderer.update_scene(robot.data,camera=camera); writer.append_data(renderer.render())
contacts=np.array(foot_counts)
warm=contacts[25:]==0
segments=int((warm&~np.r_[False,warm[:-1]]).sum()) if len(warm) else 0
result=dict(policy=str(model_path),motion=str(motion_path),deployment='Official TrackingState + CPU MuJoCoRobot',
 completed=not failed and len(joint_errors)==steps,simulated_s=len(joint_errors)*robot.control_dt,
 reference_wraps=0,virtual_support=False,joint_rmse_rad=float(np.sqrt(np.mean(joint_errors))) if joint_errors else None,
 relative_body_position_rmse_m=float(np.sqrt(np.mean(body_errors))) if body_errors else None,
 root_position_rmse_m=float(np.sqrt(np.mean(root_errors))) if root_errors else None,
 forward_velocity_mean_mps=float(np.mean(velocities)) if velocities else None,
 speed_abs_error_mean_mps=float(np.mean(speed_errors)) if speed_errors else None,
 flight_fraction_after_half_second=float(warm.mean()) if len(warm) else 0.,
 flight_segments_after_half_second=segments,maximum_joint_actuator_torque_nm=max_torque)
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2)); out.with_suffix('.trajectory.json').write_text(json.dumps(trajectory,indent=2))
print('CPU_TRACKING_RESULT',json.dumps(result),flush=True)
if writer is not None: writer.close(); renderer.close()
robot.close()
