"""Audit physics, real ROS navigation and the delivered composite recording together."""
import argparse
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import imageio.v2 as iio
import imageio_ffmpeg

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
a=p.parse_args();r=a.results
physics=json.loads((r/'simulation.json').read_text());ros=json.loads((r/'integration.json').read_text())
trace=physics['trace'];times=np.array([t['sim_time'] for t in trace]);v=np.array([t['twist_body'][0] for t in trace])
means=np.convolve(v,np.ones(10)/10,mode='valid') if len(v)>=10 else v
video=iio.get_reader(r/'running_slam_rviz.mp4');meta=video.get_meta_data();count=video.count_frames()
for i in [int(count*f) for f in [.1,.3,.5,.7,.9]]:iio.imwrite(r/f'review_{i}.jpg',video.get_data(i))
video.close()
expected=hashlib.sha256(a.policy.read_bytes()).hexdigest()
checks=dict(trained_running_policy=physics['policy_sha256']==expected and physics['running_mode'],
            online_slam=ros['slam_executed'],all_goals=ros['navigation_passed'],
            no_fall=not physics['failed'],no_obstacle_contact=not physics['obstacle_collision'],
            no_virtual_support=not physics['virtual_support'] and not physics['reference_teleport'],
            occupied_goal_rejected=ros.get('occupied_or_unknown_goal_rejected',False),
            final_stop=ros.get('stopped_after_rejected_goal',False),
            running_speed_observed=bool(len(means) and means.max()>=2.0),
            actual_rviz_captured=bool(physics['rviz_capture_display']),
            combined_video=meta['size']==(1920,768) and abs(count/meta['fps']-physics['duration_s'])<.2,
            odometry_rate=45<=ros['odometry_hz_sim']<=55,scan_rate=14<=ros['scan_hz_sim']<=16)
report=dict(checks=checks,all_checks_passed=all(checks.values()),policy_sha256=expected,
            simulated_duration_s=physics['duration_s'],maximum_1s_mean_forward_mps=float(means.max()) if len(means) else None,
            sampled_time_above_2mps_s=float(np.count_nonzero(v>=2)*.1),
            video=dict(filename='running_slam_rviz.mp4',frames=count,fps=meta['fps'],size=meta['size']),
            ideal_odometry=True,synthetic_level_laser=True,laser_max_range_m=physics['synthetic_laser_max_range_m'],
            hardware_tested=False,scope='One nominal flat synthetic room, three sequential goals; actual running actor and live RViz; not outdoor or hardware validation.')
moving=[t['sim_time'] for t in trace if abs(t['command'][0])>.1 or abs(t['command'][2])>.1]
cut=round(max(0.,moving[0]-5.)*meta['fps'])/meta['fps'] if moving else 0.
compact=r/'running_slam_rviz_demo.mp4'
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-ss',str(cut),
                '-i',str(r/'running_slam_rviz.mp4'),'-an','-c:v','libx264','-preset','fast',
                '-crf','20','-pix_fmt','yuv420p',str(compact)],check=True)
vshort=iio.get_reader(compact);short_count=vshort.count_frames();short_meta=vshort.get_meta_data();vshort.get_data(short_count-1);vshort.close()
assert abs(short_count-(count-round(cut*meta['fps'])))<=1
report['presentation_video']=dict(filename=compact.name,frames=short_count,fps=short_meta['fps'],
                                  size=short_meta['size'],source_start_s=cut,
                                  edits='Leading stationary warm-up removed only. Original speed and synchronized panels retained. Full raw recording preserved.')
(r/'acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
root=Path(__file__).resolve().parents[1]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
sources={}
for name in ['vitaboy_navigation','launch','config','scripts','tests']:
    for path in (root/name).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:sources[str(path.relative_to(root))]=sha(path)
fingerprints={path.name:sha(path) for path in r.iterdir() if path.suffix in {'.json','.mp4','.png'} and path.name!='provenance.json'}
(r/'provenance.json').write_text(json.dumps(dict(source_sha256=sources,artifacts_sha256=fingerprints,
    policy_sha256=expected,rviz='Actual rviz2 process from ROS launch, captured from Xvfb in each MuJoCo video frame'),indent=2))
if not report['all_checks_passed']:raise SystemExit(1)
