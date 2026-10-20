"""Explicit upper-body collision checks, independent of contact filters.

Use the supplied robot's collision meshes and FCL convex signed distances.
Shoulder pitch/roll mounts are deliberately excluded from torso checks:
their CAD geometry overlaps at the connected shoulder assembly.
"""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSET = ROOT.parent/'honor_rl_mjlab/src/whole_body_control/assets/robots/vita_boy/xml/main.xml'
sys.path.insert(0,str(ROOT/'collision_vendor'))
import fcl
from scipy.spatial import ConvexHull


class ArmCollisionChecker:
    def __init__(self, model, pipeline='fcl',compact_hulls=True):
        self.model = model
        self.pipeline = pipeline
        self.data = mujoco.MjData(model)
        links = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
        arms = {s: [model.geom(f'{s}_{n}_link_1_collision').id for n in links]
                for s in ['left', 'right']}
        trunk = [model.geom(f'torso_link_{i}_collision').id for i in [1, 2, 3]]
        trunk += [model.geom('pelvis_link_1_collision').id]
        self.pairs = ([(a, b) for s in arms for a in arms[s][2:] for b in trunk]
                      + [(a, b) for a in arms['left'] for b in arms['right']])
        self.names = [[model.geom(a).name, model.geom(b).name] for a, b in self.pairs]
        self.pair_ids=np.array(self.pairs)
        self.bounding_radii=model.geom_rbound[self.pair_ids].sum(1)
        self.objects={}
        if pipeline=='fcl':
            for gid in sorted({g for pair in self.pairs for g in pair}):
                mid=model.geom_dataid[gid]
                assert model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH
                vertices=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]].astype(float)
                hull=ConvexHull(vertices);faces=hull.simplices.copy()
                for i,face in enumerate(faces):
                    if np.dot(np.cross(vertices[face[1]]-vertices[face[0]],vertices[face[2]]-vertices[face[0]]),hull.equations[i,:3])<0:
                        faces[i,[1,2]]=faces[i,[2,1]]
                if compact_hulls:
                    remap=np.full(len(vertices),-1,dtype=int)
                    remap[hull.vertices]=np.arange(len(hull.vertices))
                    vertices=vertices[hull.vertices];faces=remap[faces]
                geometry=fcl.Convex(vertices,len(faces),np.column_stack([np.full(len(faces),3),faces]).flatten().astype(np.int32))
                self.objects[gid]=fcl.CollisionObject(geometry)
            self.request=fcl.DistanceRequest(enable_nearest_points=True,enable_signed_distance=True)
        self.arm_qids = np.array([model.jnt_qposadr[j] for j in range(1, model.njnt)
                                 if any(t in model.joint(j).name for t in ['shoulder', 'elbow', 'wrist'])])
        self.arm_dofs = self.arm_qids - 1  # free base: seven qpos, six tangent DOFs
        self.fromto = np.zeros(6)
        self.jac1 = np.zeros((3, model.nv))
        self.jac2 = np.zeros((3, model.nv))

    def distances(self, q, jacobian=False):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        for gid,obj in self.objects.items():
            obj.setTransform(fcl.Transform(self.data.geom_xmat[gid].reshape(3,3),self.data.geom_xpos[gid]))
        distances = np.empty(len(self.pairs))
        jac = np.zeros((len(self.pairs), len(self.arm_qids))) if jacobian else None
        sphere_lower_bound=np.linalg.norm(self.data.geom_xpos[self.pair_ids[:,0]]-
                                         self.data.geom_xpos[self.pair_ids[:,1]],axis=1)-self.bounding_radii
        for k, (a, b) in enumerate(self.pairs):
            # A strict bounding-sphere lower bound: distant pairs cannot violate
            # the constraint. Like distmax in mj_geomDistance, cap far results.
            if sphere_lower_bound[k]>.08:
                distances[k]=.08
                continue
            # Keep the distance pipeline fixed during optimization; switching
            # algorithms at a distance threshold makes the constraint discontinuous.
            if self.pipeline=='fcl':
                res=fcl.DistanceResult()
                d=fcl.distance(self.objects[a],self.objects[b],self.request,res)
                self.fromto[:3]=res.nearest_points[0];self.fromto[3:]=res.nearest_points[1]
            else:
                flags=self.model.opt.disableflags
                try:
                    if self.pipeline=='legacy':
                        self.model.opt.disableflags=flags | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
                    d=mujoco.mj_geomDistance(self.model,self.data,a,b,.5,self.fromto)
                finally:
                    self.model.opt.disableflags=flags
            distances[k] = d
            if jacobian and d < .499:
                normal = self.fromto[3:] - self.fromto[:3]
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    # For penetration the closest-point vector reverses direction.
                    normal *= (1 if d >= 0 else -1) / norm
                    mujoco.mj_jac(self.model, self.data, self.jac1, None,
                                  self.fromto[:3], self.model.geom_bodyid[a])
                    mujoco.mj_jac(self.model, self.data, self.jac2, None,
                                  self.fromto[3:], self.model.geom_bodyid[b])
                    jac[k] = (normal @ (self.jac2-self.jac1))[self.arm_dofs]
        return (distances, jac) if jacobian else distances

    def audit(self, poses, fps, subdivisions=1):
        minima = np.full(len(self.pairs), np.inf)
        worst = np.zeros(len(self.pairs), dtype=int)
        frame_min = []
        for i in range((len(poses)-1)*subdivisions+1):
            frame, sub = divmod(i, subdivisions)
            q = poses[frame].copy()
            if sub:
                # Global root rigid transform does not affect self distances.
                q[7:] += (poses[frame+1, 7:]-q[7:])*(sub/subdivisions)
            ds = self.distances(q)
            better = ds < minima
            minima[better] = ds[better]; worst[better] = i
            frame_min.append(float(ds.min()))
        records = [dict(pair=self.names[k], minimum_m=float(minima[k]),
                        time_s=float(worst[k]/fps/subdivisions))
                   for k in np.argsort(minima)]
        return dict(geometry='Convex hulls of actual MJCF compiled collision mesh vertices; compiled geom world transforms',ccd_pipeline=self.pipeline,
                    python_fcl_version=fcl.__version__,robot_xml_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
                    motion_qpos_sha256=hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest(),
                    far_distance_cap_m=.08,
                    scope='Distal arms versus torso/head/pelvis; all opposite-arm pairs; excludes shoulder attachment CAD overlaps',
                    pairs=len(self.pairs), samples=len(frame_min), sample_hz=fps*subdivisions,
                    minimum_distance_m=min(frame_min),
                    penetrating_samples=int(np.sum(np.array(frame_min)<-1e-5)),
                    below_5mm_samples=int(np.sum(np.array(frame_min)<.005-1e-5)),
                    worst_pairs=records, frame_minimum_m=frame_min)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--folder',type=Path,default=ROOT/'outputs/vita_boy')
    p.add_argument('--subdivisions',type=int,default=1); args=p.parse_args()
    motion=np.load(args.folder/'retargeted_qpos.npz')
    checker=ArmCollisionChecker(mujoco.MjModel.from_xml_path(str(ASSET)))
    report=checker.audit(motion['qpos'], float(motion['fps']),args.subdivisions)
    (args.folder/'self_collision_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['frame_minimum_m','worst_pairs']},indent=2))
    print(json.dumps(report['worst_pairs'][:15],indent=2))
