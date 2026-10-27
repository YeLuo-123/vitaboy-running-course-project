"""Combine independent physics/ROS records and fingerprint the delivered source."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess

root=Path(__file__).resolve().parents[1]
results=root/'results'
integration=json.loads((results/'integration.json').read_text())
simulation=json.loads((results/'simulation.json').read_text())
control=json.loads((results/'control_smoke.json').read_text())
checks={
    'online_slam':integration['slam_executed'],
    'all_three_navigation_goals':len(integration['goals'])==3 and integration['navigation_passed'],
    'physical_no_fall':not simulation['failed'],
    'physical_no_obstacle_collision':not simulation['obstacle_collision'],
    'no_virtual_support':not simulation['virtual_support'] and not simulation['reference_teleport'],
    'invalid_goal_refused':integration['occupied_or_unknown_goal_rejected'],
    'final_stop':integration['stopped_after_rejected_goal'],
    'odometry_rate':45 <= integration['odometry_hz_sim'] <=55,
    'scan_rate':14 <= integration['scan_hz_sim'] <=16,
    'independent_control':control['passed'],
    'navigation_video_exists':(results/'navigation.mp4').stat().st_size>10000,
}
sources={}
for p in sorted(root.rglob('*')):
    if p.is_file() and not set(p.relative_to(root).parts)&{'results','build','install','log','__pycache__'}:
        sources[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(schema_version=1,checks=checks,all_checks_passed=all(checks.values()),
            source_sha256=sources,python=platform.python_version(),
            hostname=subprocess.check_output(['hostname'],text=True).strip(),
            oracle_odometry=True,synthetic_level_laser=True,hardware_tested=False,
            running_navigation_tested=False,
            scope='Three low-speed goals in one synthetic room, with ideal odometry; not hardware or long-term SLAM validation')
(results/'acceptance.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
if not report['all_checks_passed']:raise SystemExit(1)
