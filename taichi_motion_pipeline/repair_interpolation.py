"""Repair interpolated arm paths by jointly adjusting both interval endpoints.

Used after per-frame projection. Includes neighboring intervals in each solve,
so correcting a midpoint cannot silently create a neighboring collision.
"""
import json
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from self_collision import ROOT,ASSET,ArmCollisionChecker,mujoco


def repair(poses,checker,fps,margin=.008,pending=None):
    poses=poses.copy();ids=checker.arm_qids;dims=len(ids);step=2.5/fps
    model=checker.model;limits=model.jnt_range[ids-6]
    fixes=[]
    def sample(qs,j,t):
        q=qs[j].copy();q[7:]=(1-t)*qs[j,7:]+t*qs[j+1,7:]
        return q
    def bad_intervals(qs):
        # Endpoint distances were checked by the preceding projection.
        bad=[]
        for j in range(len(qs)-1):
            if any(checker.distances(sample(qs,j,t)).min()<.005 for t in [.25,.5,.75]):bad.append(j)
        return bad
    # The caller can reuse its full projection audit for the first pass.
    if pending is None:pending=bad_intervals(poses)
    print('INTERPOLATION_BAD_INTERVALS',pending,flush=True)
    for sweep in range(4):
        if not pending:break
        for j in pending:
            assert 0<j<len(poses)-2
            baseline=poses[j:j+2,ids].copy();flat=baseline.ravel()
            bounds_low=np.maximum(limits[:,0]+1e-4,baseline-.12)
            bounds_high=np.minimum(limits[:,1]-1e-4,baseline+.12)
            bounds_low[0]=np.maximum(bounds_low[0],poses[j-1,ids]-step)
            bounds_high[0]=np.minimum(bounds_high[0],poses[j-1,ids]+step)
            bounds_low[1]=np.maximum(bounds_low[1],poses[j+2,ids]-step)
            bounds_high[1]=np.minimum(bounds_high[1],poses[j+2,ids]+step)
            active=set()
            for k in [j-1,j,j+1]:
                for t in [.25,.5,.75,1.]:
                    active.update(np.flatnonzero(checker.distances(sample(poses,k,t))<.08))
            active=np.array(sorted(active));cache_x=None;cache=None
            def evaluate(x):
                nonlocal cache_x,cache
                if cache_x is None or not np.array_equal(cache_x,x):
                    candidate=poses.copy();candidate[j:j+2,ids]=x.reshape(2,dims)
                    ds=[];rows=[]
                    for k in [j-1,j,j+1]:
                        for t in [.25,.5,.75,1.]:
                            d,g=checker.distances(sample(candidate,k,t),True)
                            row=np.zeros((len(active),2*dims))
                            for frame,weight in [(k,1-t),(k+1,t)]:
                                if frame in [j,j+1]:row[:,(frame-j)*dims:(frame-j+1)*dims]+=weight*g[active]
                            ds.append(d[active]);rows.append(row)
                    cache=(np.concatenate(ds),np.vstack(rows));cache_x=x.copy()
                return cache
            a=np.column_stack([-np.eye(dims),np.eye(dims)])
            opt=minimize(lambda x:.5*np.sum((x-flat)**2),flat,jac=lambda x:x-flat,
                         method='SLSQP',bounds=list(zip(bounds_low.ravel(),bounds_high.ravel())),
                         constraints=[dict(type='ineq',fun=lambda x:evaluate(x)[0]-margin,
                                           jac=lambda x:evaluate(x)[1]),LinearConstraint(a,-step,step)],
                         options=dict(maxiter=100,ftol=1e-9))
            if evaluate(opt.x)[0].min()<margin-2e-5:
                raise RuntimeError(f'Interpolation interval {j} infeasible: {opt.message}; clearance {evaluate(opt.x)[0].min()}')
            poses[j:j+2,ids]=opt.x.reshape(2,dims)
            fixes.append(dict(interval=j,time_s=j/fps,iterations=opt.nit,
                              maximum_extra_joint_change_rad=float(np.max(np.abs(opt.x-flat)))))
            print('INTERPOLATION_FIXED',j,fixes[-1],flush=True)
        pending=bad_intervals(poses)
    if pending:raise RuntimeError(f'Unresolved interpolated intervals: {pending}')
    return poses,fixes


if __name__=='__main__':
    folder=ROOT/'outputs/vita_boy_collision_free'
    motion=np.load(folder/'retargeted_qpos.npz');fps=float(motion['fps'])
    checker=ArmCollisionChecker(mujoco.MjModel.from_xml_path(str(ASSET)))
    original=motion['qpos'];poses,fixes=repair(original,checker,fps)
    np.savez(folder/'retargeted_qpos.npz',qpos=poses,fps=fps,joint_names=motion['joint_names'],sole_clearance=motion['sole_clearance'])
    np.savetxt(folder/'video_motion.csv',np.column_stack([poses[:,:3],poses[:,[4,5,6,3]],poses[:,7:]]),delimiter=',',fmt='%.9f')
    manifest=json.loads((folder/'manifest.json').read_text())
    manifest['interpolation_repair']=dict(margin_m=.008,fixes=fixes,subdivisions=4)
    vel=np.diff(poses[:,7:],axis=0)*fps
    manifest['max_joint_velocity_rads']=float(np.abs(vel).max())
    manifest['joint_velocity_p99_rads']=float(np.percentile(np.abs(vel),99))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report=checker.audit(poses,fps,4)
    (folder/'self_collision_report.json').write_text(json.dumps(report,indent=2))
    assert report['minimum_distance_m']>=.005
    print('INTERPOLATION_REPAIR_PASSED',report['minimum_distance_m'],flush=True)
