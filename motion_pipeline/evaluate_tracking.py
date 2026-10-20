"""Finite-clip dynamics evaluation. Detects reference-wrap teleports explicitly."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import os
import time
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from tracking_common import configure_tracking,ROOT

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint')
p.add_argument('--onnx')
p.add_argument('--agent',choices=['zero','reference_pd'],default='zero')
p.add_argument('--num-envs',type=int,default=32)
p.add_argument('--jitter',action='store_true')
p.add_argument('--seed',type=int,default=123)
p.add_argument('--video')
p.add_argument('--export-onnx')
p.add_argument('--verify-onnx')
p.add_argument('--output',required=True)
a=p.parse_args()
configure_torch_backends()
if a.verify_onnx:
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
cfg=configure_tracking(load_env_cfg('VitaBoy-Tracking',play=True),jitter=a.jitter)
cfg.scene.num_envs=a.num_envs
cfg.seed=a.seed
raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array' if a.video else None)
vec=RslRlVecEnvWrapper(raw)
agent=load_rl_cfg('VitaBoy-Tracking')
cmd=raw.command_manager.get_term('motion')
if a.checkpoint:
    cls=load_runner_cls('VitaBoy-Tracking') or MjlabOnPolicyRunner
    runner=cls(vec,asdict(agent),device='cuda:0')
    runner.load(a.checkpoint,load_cfg={'actor':True},strict=True,map_location='cuda:0')
    if a.export_onnx:
        from mjlab.rl.exporter_utils import get_base_metadata,attach_metadata_to_onnx
        path=Path(a.export_onnx); path.parent.mkdir(parents=True,exist_ok=True)
        runner.export_policy_to_onnx(str(path.parent),path.name)
        meta=get_base_metadata(raw,'local')
        meta.update({'anchor_body_name':cmd.cfg.anchor_body_name,'body_names':list(cmd.cfg.body_names)})
        attach_metadata_to_onnx(str(path),meta)
    policy=runner.get_inference_policy(device='cuda:0')
elif a.onnx:
    import onnxruntime as ort
    opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
    session=ort.InferenceSession(a.onnx,sess_options=opts,providers=['CPUExecutionProvider'])
    obs_name=session.get_inputs()[0].name
    def policy(obs):
        return torch.as_tensor(np.concatenate([session.run(None,{obs_name:r[None]})[0] for r in obs['actor'].cpu().numpy()]),device=raw.device)
elif a.agent=='reference_pd':
    action_term=raw.action_manager.get_term('joint_pos')
    def policy(obs):
        return (cmd.joint_pos-raw.scene['robot'].data.default_joint_pos)/action_term._scale
else:
    def policy(obs): return torch.zeros((a.num_envs,29),device=raw.device)
verifier=None
if a.verify_onnx:
    import onnxruntime as ort
    opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
    verifier=ort.InferenceSession(a.verify_onnx,sess_options=opts,providers=['CPUExecutionProvider'])
writer=None
if a.video:
    import imageio.v2 as imageio
    Path(a.video).parent.mkdir(parents=True,exist_ok=True)
    writer=imageio.get_writer(a.video,fps=50)
with torch.inference_mode():
    obs,_=vec.reset()
    robot=raw.scene['robot']
    steps=int(cmd.motion.time_step_total)-2
    initial=robot.data.root_link_pos_w.clone()
    alive=torch.ones(a.num_envs,dtype=torch.bool,device=raw.device)
    lifetimes=torch.full((a.num_envs,),steps*raw.step_dt,device=raw.device)
    sums=torch.zeros(7,device=raw.device)
    count=0; reference_wraps=0; max_torque=0.
    flights=torch.zeros(3,device=raw.device)
    previous_flight=torch.zeros(a.num_envs,dtype=torch.bool,device=raw.device)
    flight_segments=torch.zeros(a.num_envs,dtype=torch.long,device=raw.device)
    warm_flight_samples=0; warm_valid_samples=0
    final_position=initial.clone()
    trajectory=[]
    onnx_max_error=0.
    print('EVAL_START',steps,raw.step_dt,'OBS',list(obs['actor'].shape),flush=True)
    for i in range(steps):
        cursor=cmd.time_steps.clone()
        action=policy(obs)
        if verifier is not None and i%20==0:
            expected=verifier.run(None,{'obs':obs['actor'][:1].cpu().numpy()})[0]
            onnx_max_error=max(onnx_max_error,float(np.abs(expected-action[:1].cpu().numpy()).max()))
        obs,_,done,_=vec.step(action)
        wrapped=(cmd.time_steps<cursor)&alive&~done.bool()
        reference_wraps+=int(wrapped.sum())
        if wrapped.any(): raise RuntimeError('Reference wrapped/teleported during finite-clip evaluation')
        finite=torch.isfinite(robot.data.joint_pos).all(-1)&torch.isfinite(robot.data.root_link_pos_w).all(-1)&torch.isfinite(robot.data.root_link_lin_vel_w).all(-1)
        posture=(robot.data.root_link_pos_w[:,2]-raw.scene.env_origins[:,2]<.45)|(robot.data.projected_gravity_b[:,2]>-.7071)|~finite
        failed=done.bool()|posture
        new=alive&failed
        lifetimes[new]=(i+1)*raw.step_dt
        valid=alive&~failed
        alive&=~failed
        actual=cmd.robot_body_pos_w
        reference=cmd.body_pos_w
        rel_actual=actual-actual[:,:1]
        rel_ref=reference-reference[:,:1]
        qerr=(robot.data.joint_pos-cmd.joint_pos).square().mean(-1)
        body_err=(rel_actual-rel_ref).square().sum(-1).mean(-1)
        root_err=(actual[:,0]-reference[:,0]).square().sum(-1)
        root_vel=robot.data.root_link_lin_vel_b[:,0]
        ref_vel=cmd.body_lin_vel_w[:,0,:2].norm(dim=-1)
        if valid.any():
            sums+=torch.stack([qerr[valid].sum(),body_err[valid].sum(),root_err[valid].sum(),root_vel[valid].sum(),
                (root_vel[valid]-ref_vel[valid]).abs().sum(),ref_vel[valid].sum(),robot.data.root_link_pos_w[valid,2].sum()])
            force=raw.scene['feet_ground_contact'].data.force[valid]
            foot_count=(force.norm(dim=-1)>1.).sum(-1)
            flights+=torch.stack([(foot_count==k).sum() for k in (0,1,2)])
            current_flight=torch.zeros(a.num_envs,dtype=torch.bool,device=raw.device)
            current_flight[valid]=foot_count==0
            if (i+1)*raw.step_dt>.5:
                flight_segments+=(current_flight&~previous_flight&valid).long()
                warm_flight_samples+=int(current_flight[valid].sum())
                warm_valid_samples+=int(valid.sum())
            previous_flight=current_flight
            count+=int(valid.sum())
            final_position[valid]=robot.data.root_link_pos_w[valid]
            max_torque=max(max_torque,float(robot.data.qfrc_actuator[valid].abs().max()))
        trajectory.append(dict(t=(i+1)*raw.step_dt,reference_cursor=int(cmd.time_steps[0]),
            root_actual=actual[0,0].tolist(),root_reference=reference[0,0].tolist(),
            joint_rmse=float(qerr[0].sqrt()),relative_body_rmse=float(body_err[0].sqrt()),alive=bool(alive[0]),
            forward_velocity=float(root_vel[0]),reference_speed=float(ref_vel[0]),
            foot_contact=(raw.scene['feet_ground_contact'].data.force[0].norm(dim=-1)>1.).tolist(),
            knee_actual=robot.data.joint_pos[0,[3,9]].tolist(),knee_reference=cmd.joint_pos[0,[3,9]].tolist()))
        if writer is not None: writer.append_data(raw.render())
    result=dict(agent=a.checkpoint or a.onnx or a.agent,motion=str(cfg.commands['motion'].motion_file),
        seed=a.seed,num_envs=a.num_envs,jitter=a.jitter,clip_steps=steps,clip_duration_s=steps*raw.step_dt,
        completed_fraction=float(alive.float().mean()),complete_count=int(alive.sum()),
        first_failure_or_completion_mean_s=float(lifetimes.mean()),reference_wraps=reference_wraps,
        joint_rmse_rad=float((sums[0]/max(1,count)).sqrt()),
        relative_body_position_rmse_m=float((sums[1]/max(1,count)).sqrt()),
        root_position_rmse_m=float((sums[2]/max(1,count)).sqrt()),
        forward_velocity_mean_mps=float(sums[3]/max(1,count)),
        speed_abs_error_mean_mps=float(sums[4]/max(1,count)),
        reference_speed_mean_mps=float(sums[5]/max(1,count)),
        foot_contact_fraction_none_single_double=(flights/max(1,count)).tolist(),
        flight_fraction_after_half_second=warm_flight_samples/max(1,warm_valid_samples),
        flight_segments_after_half_second_mean=float(flight_segments.float().mean()),
        flight_segments_after_half_second_min=int(flight_segments.min()),
        world_x_displacement_mean_m=float((final_position-initial)[:,0].mean()),
        maximum_joint_actuator_torque_nm=max_torque,valid_samples=count,
        onnx_max_abs_action_error=onnx_max_error if verifier is not None else None,
        virtual_support=False,score_note='Errors use valid pre-failure samples; assess completion together with errors.')
    criteria=dict(completion_at_least_95_percent=result['completed_fraction']>=.95,
        joint_rmse_at_most_0_15_rad=result['joint_rmse_rad']<=.15,
        relative_body_rmse_at_most_0_06_m=result['relative_body_position_rmse_m']<=.06,
        root_rmse_at_most_0_20_m=result['root_position_rmse_m']<=.20,
        speed_error_at_most_0_25_mps=result['speed_abs_error_mean_mps']<=.25,
        no_reference_wrap_or_teleport=reference_wraps==0,
        measured_flight_after_half_second_at_least_1_percent=result['flight_fraction_after_half_second']>=.01,
        at_least_two_flight_segments_after_half_second=result['flight_segments_after_half_second_min']>=2)
    result['engineering_acceptance_criteria']=criteria
    result['passes_engineering_acceptance']=all(criteria.values())
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2))
    Path(a.output).with_suffix('.trajectory.json').write_text(json.dumps(trajectory,indent=2))
    print('TRACKING_RESULT',json.dumps(result),flush=True)
if writer is not None: writer.close()
vec.close()
