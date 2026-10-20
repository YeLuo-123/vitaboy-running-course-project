"""Verify exported motion contracts and decodability of the full videos."""
from pathlib import Path
import json
import numpy as np
import mujoco
import imageio.v2 as iio
import argparse
from self_collision import ArmCollisionChecker

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,default=ROOT/'outputs/vita_boy_collision_free')
folder=parser.parse_args().folder
q=np.load(folder/'retargeted_qpos.npz')
model=mujoco.MjModel.from_xml_path(str(ROOT.parent/'honor_rl_mjlab/src/whole_body_control/assets/robots/vita_boy/xml/main.xml'))
motion=np.load(folder/'taichi_motion.npz')
assert q['qpos'].shape==(2139,36)
assert np.isfinite(q['qpos']).all()
assert np.max(np.abs(np.linalg.norm(q['qpos'][:,3:7],axis=1)-1))<1e-6
assert np.all(q['qpos'][:,7:]>=model.jnt_range[1:,0]-1e-7)
assert np.all(q['qpos'][:,7:]<=model.jnt_range[1:,1]+1e-7)
assert q['sole_clearance'].min()>=.001-1e-8
assert motion['joint_pos'].shape==(2138,29)
assert np.max(np.abs(motion['joint_pos']-q['qpos'][:-1,7:]))<5e-5
assert all(np.isfinite(motion[k]).all() for k in motion.files)
assert np.isclose(motion['fps'],50.).all()
# Recompute using the delivered qpos rather than trusting cached report values.
collision=ArmCollisionChecker(model).audit(q['qpos'],50,subdivisions=4)
assert collision['minimum_distance_m']>=.005
assert collision['penetrating_samples']==0
(folder/'self_collision_report.json').write_text(json.dumps(collision,indent=2))
original=np.load(ROOT/'outputs/vita_boy/retargeted_qpos.npz')['qpos']
ids=ArmCollisionChecker(model).arm_qids
frozen=np.setdiff1d(np.arange(model.nq),ids)
assert np.array_equal(q['qpos'][:,frozen],original[:,frozen])
arm_speed=np.abs(np.diff(q['qpos'][:,ids],axis=0)*50).max()
assert arm_speed<=2.5+1e-5
media={}
for name in ['taichi_to_vita_comparison.mp4','taichi_vita_kinematic.mp4']:
    v=iio.get_reader(folder/name)
    count=v.count_frames();last=v.get_data(count-1);meta=v.get_meta_data()
    assert count==2139 and meta['fps']==50.
    media[name]=dict(frames=count,fps=meta['fps'],size=meta['size']);v.close()
report=dict(status='passed',source_frames=1284,reference_frames=2139,tracking_format_frames=2138,
            joints=29,finite_values=True,unit_root_quaternions=True,joint_limits_passed=True,
            sole_clearance_min_m=float(q['sole_clearance'].min()),media=media,
            upper_body_collision_pairs=collision['pairs'],collision_check_hz=collision['sample_hz'],
            collision_check_samples=collision['samples'],collision_minimum_m=collision['minimum_distance_m'],
            penetrating_samples=collision['penetrating_samples'],non_arm_qpos_unchanged=True,
            max_arm_frame_difference_rads=float(arm_speed),tracking_joint_positions_match_reference=True,
            dynamic_tracking_evaluated=False,hardware_execution=False)
(folder/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
