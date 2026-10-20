"""Independent CPU MuJoCo + official deployment observation/control path."""
import os,sys,argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import mujoco
from run_config import ROOT,REPO

sys.path.insert(0,str(REPO/'deploy'))
from common.config import load_config
from common.policy import OnnxPolicy,read_metadata
from inference.mujoco_robot import MuJoCoRobot
from fsm.loco.loco import LocomotionState


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--duration',type=float,default=600.);p.add_argument('--speed',type=float,default=2.4)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    os.chdir(REPO)
    robot=MuJoCoRobot(load_config(REPO/'deploy/config.toml'),launch_viewer=False)
    loco=LocomotionState();loco.policy_path=args.policy.resolve();loco.policy=OnnxPolicy(loco.policy_path,observation_size=960)
    loco._validate_metadata();metadata=read_metadata(loco.policy_path)
    for field,actual in [('joint_stiffness',loco.stiffness),('joint_damping',loco.damping),('default_joint_pos',loco.default_angles)]:
        assert np.allclose(np.fromstring(metadata[field],sep=','),actual,atol=1e-5),field
    assert abs(float(metadata['action_scale'])-loco.action_scale)<1e-8
    robot.data.qpos[robot.qpos_indices]=loco.default_angles
    mujoco.mj_forward(robot.model,robot.data);loco.enter(robot.get_state())
    command=np.zeros(3,dtype=np.float32);velocity=[];trace=[];energy=0.;distance=0.;failed=False;all_forward=0.;all_samples=0
    previous=robot.data.qpos[:2].copy();maximum_torque=0.;squared_torque=np.zeros(29);torque_samples=0
    start=time.perf_counter();dt=robot.control_dt;duration=0.
    for i in range(round(args.duration/dt)):
        t=i*dt;target_speed=0. if t<2. else args.speed
        command[0]+=np.clip(target_speed-command[0],-.8*dt,.8*dt)
        target=loco.step(robot.get_state(),command)
        assert not target.stabilize_base
        robot.send_command(target);duration=(i+1)*dt;state=robot.get_state()
        if state.base_height<.4 or state.projected_gravity[2]>-.5 or not np.isfinite(state.joint_pos).all():
            failed=True;break
        body_velocity=robot.data.xmat[robot.pelvis_body_id].reshape(3,3).T@robot.data.qvel[:3]
        all_forward+=float(body_velocity[0]);all_samples+=1
        if t>5. and command[0]>.95*args.speed:velocity.append(float(body_velocity[0]))
        torque=robot.data.qfrc_actuator[robot.dof_indices]
        energy+=float(np.abs(torque*robot.data.qvel[robot.dof_indices]).sum())*dt
        distance+=float(np.linalg.norm(robot.data.qpos[:2]-previous));previous=robot.data.qpos[:2].copy()
        maximum_torque=max(maximum_torque,float(np.abs(torque).max()));squared_torque+=torque**2;torque_samples+=1
        if i%5==0:trace.append(dict(t=duration,command_mps=float(command[0]),vx_mps=float(body_velocity[0]),root=robot.data.qpos[:3].tolist()))
        if i%5000==0:print('CPU_SIM_TIME',duration,flush=True)
    result=dict(backend='CPU MuJoCo + official LocomotionState / MuJoCoRobot deployment path',policy=str(loco.policy_path),
        policy_file_sha256=hashlib.sha256(loco.policy_path.read_bytes()).hexdigest(),
        requested_duration_s=args.duration,simulated_duration_s=duration,completion_fraction=float(not failed),failed=failed,
        steady_speed_mean_mps=float(np.mean(velocity)) if velocity else None,commanded_speed_mps=args.speed,
        whole_trial_forward_speed_mps=all_forward/max(all_samples,1),whole_trial_path_speed_mps=distance/max(duration,.02),
        absolute_mechanical_work_j=energy,distance_m=distance,mechanical_cost_per_meter_jpm=energy/max(distance,.01),
        maximum_joint_torque_nm=maximum_torque,joint_torque_rms_nm=np.sqrt(squared_torque/max(1,torque_samples)).tolist(),
        virtual_support=False,reference_state_teleport=False,stationary_initialization=True,
        energy_scope='Absolute mechanical joint power sampled at 50Hz; not battery consumption or thermal certification',wall_time_s=time.perf_counter()-start)
    result['passes_speed_and_duration']=bool(not failed and result['steady_speed_mean_mps']>=2. and duration>=args.duration-1e-6)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    args.output.with_suffix('.trajectory.json').write_text(json.dumps(trace,indent=2))
    print('CPU_RESULT',json.dumps(result),flush=True);robot.close()


if __name__=='__main__':main()
