"""Arm-only constrained projection AFTER filtering/resampling.

This is a kinematic reference repair, not a physics controller. The original
reference is retained in a separate folder. Every exported frame must pass a
nonlinear signed-distance check; failed optimization never silently exports.
"""
import argparse
import json
import shutil
import numpy as np
from scipy.optimize import minimize
from self_collision import ROOT, ASSET, ArmCollisionChecker, mujoco


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=__import__('pathlib').Path,default=ROOT/'outputs/vita_boy')
    p.add_argument('--out',type=__import__('pathlib').Path,default=ROOT/'outputs/vita_boy_collision_free')
    p.add_argument('--margin',type=float,default=.012)
    p.add_argument('--max-frames',type=int)
    args=p.parse_args()
    if args.input.resolve()==args.out.resolve():
        raise ValueError('Use a separate output folder to preserve the original motion.')
    args.out.mkdir(parents=True,exist_ok=True)
    motion=np.load(args.input/'retargeted_qpos.npz')
    original=motion['qpos'].copy()
    if args.max_frames: original=original[:args.max_frames]
    fps=float(motion['fps']);model=mujoco.MjModel.from_xml_path(str(ASSET))
    checker=ArmCollisionChecker(model);ids=checker.arm_qids
    jointids=[model.joint(n).id for n in motion['joint_names'].tolist()]
    limits=model.jnt_range[jointids][ids-7]
    low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
    weights=np.array([1. if 'shoulder' in model.joint(int(j)).name else .7
                      for j in np.array(jointids)[ids-7]])
    # Begin from a clear arm posture, then follow the closest feasible branch.
    seed=original[0,ids].copy()
    for k,qi in enumerate(ids):
        name=model.joint(int(qi-6)).name
        seed[k]=(.2 if 'shoulder_pitch' in name else
                 .5 if name=='left_shoulder_roll_joint' else
                 -.5 if name=='right_shoulder_roll_joint' else
                 .6 if 'elbow' in name else 0.)
    q=original[0].copy();q[ids]=seed
    print('SEED_CLEARANCE',checker.distances(q).min(),flush=True)
    result=[];iterations=[];max_step=2.5/fps
    for i,ref in enumerate(original):
        previous=seed if i==0 else result[-1][ids]
        # Keep the change in the correction smooth, rather than delaying motion.
        target=ref[ids].copy()
        predicted=target if i==0 else previous+ref[ids]-original[i-1,ids]
        def objective(x):
            a=x-target;b=x-predicted
            return .5*np.sum(weights*a*a)+.3*np.sum(b*b)
        def gradient(x):return weights*(x-target)+.6*(x-predicted)
        cache_x=None; cache=None
        def evaluate(x):
            nonlocal cache_x,cache
            if cache_x is None or not np.array_equal(x,cache_x):
                q=ref.copy();q[ids]=x
                cache=checker.distances(q,True);cache_x=x.copy()
            return cache
        constraint=dict(type='ineq',fun=lambda x:evaluate(x)[0]-args.margin,jac=lambda x:evaluate(x)[1])
        # Maximum per-frame angular step prevents IK branch jumps. The first
        # frame has no previous timestamp, so use only joint position bounds.
        lo=low if i==0 else np.maximum(low,previous-max_step)
        hi=high if i==0 else np.minimum(high,previous+max_step)
        opt=minimize(objective,np.clip(previous,lo,hi),jac=gradient,method='SLSQP',
                     bounds=list(zip(lo,hi)),constraints=[constraint],
                     options=dict(maxiter=100,ftol=1e-8,eps=1e-5))
        distance=evaluate(opt.x)[0].min()
        if distance<args.margin-2e-5:
            raise RuntimeError(f'Frame {i} at {i/fps:.3f}s failed: {distance:.6f}m; {opt.message}')
        q=ref.copy();q[ids]=opt.x;result.append(q);iterations.append(opt.nit)
        if i%100==0:print('PROJECT_FRAME',i,'clearance',float(distance),'iterations',opt.nit,flush=True)
    poses=np.array(result)
    np.savez(args.out/'retargeted_qpos.npz',qpos=poses,fps=fps,
             joint_names=motion['joint_names'],sole_clearance=motion['sole_clearance'][:len(poses)])
    np.savetxt(args.out/'video_motion.csv',np.column_stack([poses[:,:3],poses[:,[4,5,6,3]],poses[:,7:]]),
               delimiter=',',fmt='%.9f')
    for name in ['raw_gmr_qpos.npz','smplx_to_vita_boy.json']:
        if (args.input/name).exists():shutil.copy2(args.input/name,args.out/name)
    manifest=json.loads((args.input/'manifest.json').read_text())
    manifest['self_collision_repair']=dict(method='Sequential arm-only SLSQP with exact mesh signed-distance Jacobians',
        input_folder=args.input.name,margin_m=args.margin,frame_step_limit_rad=max_step,
        temporal_correction_weight=.6,iterations_max=max(iterations),
        arm_correction_max_rad=float(np.abs(poses[:,ids]-original[:,ids]).max()),
        arm_correction_rms_rad=float(np.sqrt(np.mean((poses[:,ids]-original[:,ids])**2))),
        frozen_non_arm_qpos=True,source_timing_preserved=True)
    velocity=np.diff(poses[:,7:],axis=0)*fps
    manifest['max_joint_velocity_rads']=float(np.abs(velocity).max())
    manifest['joint_velocity_p99_rads']=float(np.percentile(np.abs(velocity),99))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report=checker.audit(poses,fps,subdivisions=4)
    if report['minimum_distance_m']<.005:
        from repair_interpolation import repair
        pending=sorted({min(i//4,len(poses)-2) for i,d in enumerate(report['frame_minimum_m']) if d<.005})
        poses,fixes=repair(poses,checker,fps,pending=pending)
        np.savez(args.out/'retargeted_qpos.npz',qpos=poses,fps=fps,
                 joint_names=motion['joint_names'],sole_clearance=motion['sole_clearance'][:len(poses)])
        np.savetxt(args.out/'video_motion.csv',np.column_stack([poses[:,:3],poses[:,[4,5,6,3]],poses[:,7:]]),delimiter=',',fmt='%.9f')
        manifest['interpolation_repair']=dict(margin_m=.008,fixes=fixes,subdivisions=4)
        manifest['self_collision_repair']['arm_correction_max_rad']=float(np.abs(poses[:,ids]-original[:,ids]).max())
        manifest['self_collision_repair']['arm_correction_rms_rad']=float(np.sqrt(np.mean((poses[:,ids]-original[:,ids])**2)))
        velocity=np.diff(poses[:,7:],axis=0)*fps
        manifest['max_joint_velocity_rads']=float(np.abs(velocity).max())
        manifest['joint_velocity_p99_rads']=float(np.percentile(np.abs(velocity),99))
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        report=checker.audit(poses,fps,subdivisions=4)
    (args.out/'self_collision_report.json').write_text(json.dumps(report,indent=2))
    assert report['minimum_distance_m']>=.005, report['minimum_distance_m']
    print('REPAIR_PASSED',report['minimum_distance_m'],report['penetrating_samples'],flush=True)


if __name__=='__main__':main()
