"""Command-driven physics evaluations; failures remain failures after autoreset."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from run_config import ROOT,REPO,make_config,agent_config
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper,MjlabOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends


def target_at(t,scenario,speed):
    if scenario=='standing':return [0.,0.,0.]
    if t<2.:return [0.,0.,0.]
    if scenario=='transitions':
        if t<12:return [speed,0.,0.]
        if t<20:return [speed,0.,.3]
        if t<25:return [0.,0.,0.]
        if t<35:return [speed,0.,-.3]
        return [0.,0.,0.]
    return [speed,0.,0.]


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--onnx')
    p.add_argument('--stage',choices=['flat','robust','rough'],default='flat')
    p.add_argument('--scenario',choices=['standing','straight','transitions'],default='transitions')
    p.add_argument('--duration',type=float,default=40.);p.add_argument('--speed',type=float,default=2.2)
    p.add_argument('--num-envs',type=int,default=16);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--video');p.add_argument('--output',type=Path,required=True);p.add_argument('--export-onnx')
    p.add_argument('--noise',action='store_true',help='Evaluate with the training IMU/joint observation noise')
    p.add_argument('--video-env',type=int,default=0)
    args=p.parse_args();configure_torch_backends()
    if args.export_onnx:
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    cfg=make_config(args.stage,args.num_envs,False,args.seed);cfg.episode_length_s=args.duration+5
    if not 0<=args.video_env<args.num_envs:raise ValueError('video-env outside environment batch')
    cfg.viewer.env_idx=args.video_env
    cfg.observations['actor'].enable_corruption=args.noise
    cfg.commands['base_velocity'].resampling_time_range=(1e6,1e6)
    # Evaluation begins at nominal standing state with no random initial twist.
    cfg.events['reset_base'].params.update(pose_range={},velocity_range={})
    cfg.events['reset_robot_joints'].params.update(position_range=(1.,1.),velocity_range=(0.,0.))
    raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array' if args.video else None)
    env=RslRlVecEnvWrapper(raw)
    if args.checkpoint:
        runner=MjlabOnPolicyRunner(env,asdict(agent_config()),device='cuda:0')
        runner.load(args.checkpoint,load_cfg={'actor':True},map_location='cuda:0')
        policy=runner.get_inference_policy(device='cuda:0')
        if args.export_onnx:
            from mjlab.rl.exporter_utils import get_base_metadata,attach_metadata_to_onnx
            out=Path(args.export_onnx);out.parent.mkdir(parents=True,exist_ok=True)
            runner.export_policy_to_onnx(str(out.parent),out.name)
            attach_metadata_to_onnx(str(out),get_base_metadata(raw,'local'))
    else:
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(args.onnx or str(REPO/'deploy/checkpoints/loco.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        def policy(obs):
            return torch.tensor(np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs['actor'].cpu().numpy()]),device=raw.device)
    verifier=None;onnx_error=0.
    if args.export_onnx:
        import onnxruntime as ort
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        verifier=ort.InferenceSession(args.export_onnx,sess_options=opts,providers=['CPUExecutionProvider'])
    writer=None
    if args.video:
        import imageio.v2 as iio
        Path(args.video).parent.mkdir(parents=True,exist_ok=True);writer=iio.get_writer(args.video,fps=50)
    cmd=raw.command_manager.get_term('base_velocity');d=raw.scene['robot'].data
    physics_snapshot={}
    if args.stage!='flat':
        for field,column,key in [('body_mass',None,'mass_ratio'),('actuator_gainprm',0,'kp_ratio'),('actuator_forcerange',1,'effort_ratio')]:
            value=getattr(raw.sim.model,field)[:]
            default=raw.sim.get_default_field(field)
            if column is not None:value=value[...,column];default=default[...,column]
            positive=default>0
            ratio=value[...,positive]/default[positive]
            physics_snapshot[key]=dict(minimum=float(ratio.min()),maximum=float(ratio.max()),mean_by_env=ratio.mean(-1).tolist())
        physics_snapshot['total_model_mass_kg']=raw.sim.model.body_mass[:].sum(-1).tolist()
        terrain=raw.scene.terrain
        if hasattr(terrain,'terrain_types'):physics_snapshot['terrain_types']=terrain.terrain_types.tolist()
    alive=torch.ones(args.num_envs,dtype=torch.bool,device=raw.device)
    lifetime=torch.full((args.num_envs,),args.duration,device=raw.device)
    velocity_sum=torch.zeros(args.num_envs,device=raw.device);velocity_count=torch.zeros_like(velocity_sum)
    all_forward_velocity_sum=torch.zeros_like(velocity_sum)
    energy=torch.zeros_like(velocity_sum);distance=torch.zeros_like(velocity_sum)
    command_error=torch.zeros_like(velocity_sum);samples=torch.zeros_like(velocity_sum)
    flight=torch.zeros_like(velocity_sum);flight_count=torch.zeros_like(velocity_sum)
    yaw_error=torch.zeros_like(velocity_sum);stop_speed=torch.zeros_like(velocity_sum);stop_count=torch.zeros_like(velocity_sum)
    torque_squared=torch.zeros((args.num_envs,29),device=raw.device)
    torque_samples=torch.zeros_like(velocity_sum);max_joint_acc=0.;terrain_exits=torch.zeros_like(alive)
    tilt_squared=torch.zeros_like(velocity_sum);turn_error=torch.zeros_like(velocity_sum);turn_count=torch.zeros_like(velocity_sum)
    yaw_filtered=torch.zeros_like(velocity_sum);filtered_turn_error=torch.zeros_like(velocity_sum)
    directed_turn_error=torch.zeros((args.num_envs,2),device=raw.device);directed_turn_count=torch.zeros_like(directed_turn_error)
    action_rate_squared=torch.zeros_like(velocity_sum);last_action=torch.zeros((args.num_envs,29),device=raw.device)
    peak_contacts=torch.zeros((),device=raw.device,dtype=torch.long)
    peak_constraints=torch.zeros((),device=raw.device,dtype=torch.long)
    max_torque=0.;trace=[];failure_events=[];wall=time.perf_counter()
    with torch.inference_mode():
        obs,_=env.reset();previous=d.root_link_pos_w.clone();previous_jv=d.joint_vel.clone();cmd.external_target=torch.zeros_like(cmd.command)
        for i in range(round(args.duration/raw.step_dt)):
            if i%1000==0:print('EVAL_SIM_TIME_S',round(i*raw.step_dt,2),flush=True)
            t=i*raw.step_dt;cmd.external_target[:]=torch.tensor(target_at(t,args.scenario,args.speed),device=raw.device)
            before_height=d.root_link_pos_w[:,2].clone()
            before_gravity=d.projected_gravity_b[:,2].clone()
            action=policy(obs)
            if verifier is not None and i%50==0:
                expected=verifier.run(None,{'obs':obs['actor'][:1].cpu().numpy()})[0]
                onnx_error=max(onnx_error,float(np.abs(expected-action[:1].cpu().numpy()).max()))
            obs,reward,done,extra=env.step(action)
            peak_contacts=torch.maximum(peak_contacts,raw.sim.data.nacon[:].max().long())
            peak_constraints=torch.maximum(peak_constraints,raw.sim.data.nefc[:].max().long())
            failed=done.bool()|~torch.isfinite(d.joint_pos).all(-1)
            if args.stage=='rough':
                exited=(d.root_link_pos_w[:,:2]-raw.scene.env_origins[:,:2]).abs().amax(-1)>79.
                terrain_exits|=exited&alive;failed|=exited
            new=alive&failed
            for failed_id in new.nonzero(as_tuple=False).flatten().tolist():
                terms=[name for name in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(name)[failed_id])]
                failure_events.append(dict(env_id=failed_id,time_s=(i+1)*raw.step_dt,termination_terms=terms,
                    previous_root_height_m=float(before_height[failed_id]),previous_gravity_z=float(before_gravity[failed_id])))
            lifetime[new]=(i+1)*raw.step_dt;valid=alive&~failed;alive&=~failed
            v=d.root_link_lin_vel_b[:,0];steady=(cmd.command[:,0]>.95*args.speed)&(t>5.)
            mask=valid&steady
            velocity_sum[mask]+=v[mask];velocity_count[mask]+=1
            energy[valid]+=(d.qfrc_actuator*d.joint_vel).abs().sum(-1)[valid]*raw.step_dt
            displacement=(d.root_link_pos_w-previous)[:,:2].norm(dim=-1)
            distance[valid]+=displacement[valid];previous=d.root_link_pos_w.clone()
            command_error[valid]+=(cmd.command[:,0]-v).abs()[valid];samples[valid]+=1
            all_forward_velocity_sum[valid]+=v[valid]
            yaw_error[valid]+=(cmd.command[:,2]-d.root_link_ang_vel_b[:,2]).abs()[valid]
            turning=valid&(cmd.command[:,2].abs()>.15)
            turn_error[turning]+=(cmd.command[:,2]-d.root_link_ang_vel_b[:,2]).abs()[turning];turn_count[turning]+=1
            yaw_filtered+=(d.root_link_ang_vel_b[:,2]-yaw_filtered)*(raw.step_dt/(.2+raw.step_dt))
            filtered_turn_error[turning]+=(cmd.command[:,2]-yaw_filtered).abs()[turning]
            for direction,sign in [(0,1),(1,-1)]:
                selected=valid&(sign*cmd.command[:,2]>.15)
                directed_turn_error[selected,direction]+=(d.root_link_ang_vel_b[:,2]-cmd.command[:,2])[selected]
                directed_turn_count[selected,direction]+=1
            tilt_squared[valid]+=torch.acos((-d.projected_gravity_b[:,2]).clamp(-1.,1.)).square()[valid]
            action_rate_squared[valid]+=((action-last_action)/raw.step_dt).square().mean(-1)[valid];last_action=action.clone()
            stopped=valid&((24.<=t<25.)|(38.<=t)) if args.scenario=='transitions' else valid&(args.scenario=='standing')
            stop_speed[stopped]+=d.root_link_lin_vel_b[stopped,:2].norm(dim=-1);stop_count[stopped]+=1
            torque_squared[valid]+=d.qfrc_actuator[valid].square();torque_samples[valid]+=1
            if valid.any():max_joint_acc=max(max_joint_acc,float(((d.joint_vel-previous_jv)/raw.step_dt)[valid].abs().max()))
            previous_jv=d.joint_vel.clone()
            foot=raw.scene['feet_ground_contact'].data.force.norm(dim=-1)>1.
            flight[mask]+=(~foot[mask].any(-1)).float();flight_count[mask]+=1
            if valid.any():max_torque=max(max_torque,float(d.qfrc_actuator[valid].abs().max()))
            if i%5==0:trace.append(dict(t=round((i+1)*raw.step_dt,3),command=cmd.command[0].tolist(),
                actual_v=d.root_link_lin_vel_b[0].tolist(),actual_omega=d.root_link_ang_vel_b[0].tolist(),
                root=d.root_link_pos_w[0].tolist(),alive=bool(alive[0])))
            if writer is not None:
                if alive[args.video_env]:writer.append_data(raw.render())
                else:writer.close();writer=None
            if not alive.any():break
    steady_speed=velocity_sum/velocity_count.clamp_min(1)
    result=dict(checkpoint=args.checkpoint,onnx=args.onnx,stage=args.stage,scenario=args.scenario,
        policy_file_sha256=hashlib.sha256(Path(args.checkpoint or args.onnx or REPO/'deploy/checkpoints/loco.onnx').read_bytes()).hexdigest(),
        seed=args.seed,num_envs=args.num_envs,requested_duration_s=args.duration,
        observation_noise=args.noise,
        physics_variation_snapshot=physics_snapshot,
        numerical_capacity=dict(peak_total_contacts=int(peak_contacts),total_contact_capacity=int(raw.sim.wp_data.naconmax),
            peak_constraints_per_world=int(peak_constraints),constraint_capacity_per_world=cfg.sim.njmax),
        completion_fraction=float(alive.float().mean()),lifetimes_s=lifetime.tolist(),
        failure_events=failure_events,
        commanded_speed_mps=args.speed,steady_speed_each_env_mps=steady_speed.tolist(),
        steady_speed_mean_mps=float(steady_speed.mean()),steady_samples_each_env=velocity_count.tolist(),
        whole_trial_forward_speed_each_env_mps=(all_forward_velocity_sum/samples.clamp_min(1)).tolist(),
        whole_trial_path_speed_each_env_mps=(distance/lifetime.clamp_min(.02)).tolist(),
        command_speed_mae_mps=(command_error/samples.clamp_min(1)).tolist(),
        yaw_rate_mae_rads=(yaw_error/samples.clamp_min(1)).tolist(),
        turn_phase_yaw_rate_mae_rads=(turn_error/turn_count.clamp_min(1)).tolist(),turn_samples=turn_count.tolist(),
        turn_filtered_yaw_rate_mae_rads=(filtered_turn_error/turn_count.clamp_min(1)).tolist(),
        turn_mean_bias_by_direction_rads=(directed_turn_error/directed_turn_count.clamp_min(1)).tolist(),
        turn_direction_samples=directed_turn_count.tolist(),yaw_filter_time_constant_s=.2,
        base_tilt_rms_rad=(tilt_squared/samples.clamp_min(1)).sqrt().tolist(),
        action_rate_rms_per_s=(action_rate_squared/samples.clamp_min(1)).sqrt().tolist(),
        stopped_speed_mps=(stop_speed/stop_count.clamp_min(1)).tolist(),stop_measurement_samples=stop_count.tolist(),
        joint_torque_rms_nm=(torque_squared/torque_samples.clamp_min(1)[:,None]).sqrt().tolist(),
        peak_joint_acceleration_rads2=max_joint_acc,terrain_exit_count=int(terrain_exits.sum()),
        flight_fraction_steady_each_env=(flight/flight_count.clamp_min(1)).tolist(),
        absolute_mechanical_work_j=energy.tolist(),distance_m=distance.tolist(),
        mechanical_cost_per_meter_jpm=(energy/distance.clamp_min(.01)).tolist(),
        maximum_joint_torque_nm=max_torque,virtual_support=False,reference_state_teleport=False,
        reset_initialization='Nominal stationary pose, zero velocity',
        energy_scope='Sum of absolute joint mechanical power sampled at 50Hz; excludes electronics, efficiency, regeneration, battery and motor heat',
        onnx_max_abs_action_error=onnx_error if verifier is not None else None,
        randomization=('Mass 0.9-1.1, torso COM +/-15mm, PD 0.85-1.15, effort 0.85-1.0, friction 0.65-0.95, target delay 0-20ms, velocity kicks +/-0.25m/s every 7-12s' if args.stage!='flat' else 'Nominal physics; repeated nominal stationary starts')+('; training observation corruption enabled' if args.noise else '; observation corruption disabled'),
        terrain=('Synthetic height noise -15..25mm and waves 15..40mm amplitude range; not a certification of outdoor terrain' if args.stage=='rough' else 'Infinite plane; asphalt proxy only'),
        wall_time_s=time.perf_counter()-wall)
    result['passes_speed_and_completion']=bool(alive.float().mean()>=.95 and steady_speed.mean()>=2. and velocity_count.min()>50)
    result['passes_raw_transition_checks']=bool(result['passes_speed_and_completion'] and args.scenario=='transitions'
        and stop_count.min()>=50 and (stop_speed/stop_count.clamp_min(1)).max()<=.15
        and turn_count.min()>=50 and (turn_error/turn_count.clamp_min(1)).max()<=.15)
    result['passes_transition_checks']=bool(result['passes_speed_and_completion'] and args.scenario=='transitions'
        and stop_count.min()>=50 and (stop_speed/stop_count.clamp_min(1)).max()<=.15
        and directed_turn_count.min()>=50 and (directed_turn_error/directed_turn_count.clamp_min(1)).abs().max()<=.08
        and (filtered_turn_error/turn_count.clamp_min(1)).max()<=.15
        and (tilt_squared/samples.clamp_min(1)).sqrt().max()<=.25)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    args.output.with_suffix('.trajectory.json').write_text(json.dumps(trace,indent=2))
    if verifier is not None:assert onnx_error<1e-3,onnx_error
    print('RUNNING_EVALUATION',json.dumps(result),flush=True)
    if writer:writer.close()
    env.close()


if __name__=='__main__':main()
