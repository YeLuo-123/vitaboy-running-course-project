"""Offline deployment contract and measured calibration/latency audit.

Contains no robot command sender. Encoder mappings are intentionally unknown
until measured; simulation's default pose is not an encoder zero calibration.
"""
import argparse,csv,json,re,hashlib
from pathlib import Path
import numpy as np
from run_config import ROOT,REPO


def template(policy):
    import onnx
    import mujoco
    from whole_body_control.assets.robots.vita_boy.vita_boy_constants import VITA_BOY_ARTICULATION,VITA_BOY_XML
    metadata={x.key:x.value for x in onnx.load(str(policy)).metadata_props}
    names=metadata['joint_names'].split(',');model=mujoco.MjModel.from_xml_path(str(VITA_BOY_XML))
    kp=list(map(float,metadata['joint_stiffness'].split(',')))
    kd=list(map(float,metadata['joint_damping'].split(',')))
    default=list(map(float,metadata['default_joint_pos'].split(',')))
    joints=[]
    for i,name in enumerate(names):
        actuator=next(a for a in VITA_BOY_ARTICULATION.actuators if any(re.fullmatch(p,name) for p in a.target_names_expr))
        joints.append(dict(name=name,simulation_index=i,sim_default_pose_rad=default[i],
            sim_kp=kp[i],sim_kd=kd[i],sim_peak_effort_nm=actuator.effort_limit,
            sim_joint_limits_rad=model.joint(name).range.tolist(),
            sdk_index=None,encoder_direction=None,encoder_zero_offset_rad=None,
            measured_kp=None,measured_kd=None,hardware_peak_effort_nm=None,hardware_continuous_effort_nm=None))
    return dict(schema_version=1,policy=str(policy),policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),policy_hz=50,simulation_hz=200,
        obs_dim=960,history_frames=10,action_scale=.25,quaternion_order='wxyz in simulation; adapt documented hardware order',
        pose_mapping='q_sim = encoder_direction * (q_hardware - encoder_zero_offset_rad)',
        measured_calibration_completed=False,trained_command_delay_range_ms=[0.,20.],
        note='Simulation reference only. All unknown hardware mappings must be measured. No hardware execution authorized by this file.',joints=joints)


def audit(measured,evaluation,latency,contract=None):
    issues=[];joints=measured['joints'];indices=[]
    if not measured.get('measured_calibration_completed',False):issues.append('Physical calibration is not marked completed')
    if contract is not None:
        if [j['name'] for j in joints]!=[j['name'] for j in contract['joints']]:issues.append('Joint ordering/names differ from policy contract')
        if measured.get('policy_sha256')!=contract['policy_sha256']:issues.append('Calibration policy hash differs from selected policy')
    required=['sdk_index','encoder_direction','encoder_zero_offset_rad','measured_kp','measured_kd',
              'hardware_peak_effort_nm','hardware_continuous_effort_nm']
    for j in joints:
        reference=next((x for x in contract['joints'] if x['name']==j['name']),None) if contract else j
        if reference is None:issues.append(j['name']+': unknown policy joint');continue
        for k in ['simulation_index','sim_kp','sim_kd','sim_peak_effort_nm','sim_default_pose_rad','sim_joint_limits_rad']:
            if j.get(k)!=reference[k]:issues.append(j['name']+': modified simulation contract '+k)
        missing=[k for k in required if j.get(k) is None]
        if missing:issues.append(f"{j['name']}: missing {','.join(missing)}");continue
        if not all(isinstance(j[k],(int,float)) and np.isfinite(j[k]) for k in required):
            issues.append(j['name']+': calibration fields must be finite numbers');continue
        if int(j['sdk_index'])!=j['sdk_index'] or j['sdk_index']<0:issues.append(j['name']+': invalid hardware index')
        if j['hardware_continuous_effort_nm']<=0 or j['hardware_peak_effort_nm']<=0:issues.append(j['name']+': nonpositive physical effort limit')
        if j['hardware_continuous_effort_nm']>j['hardware_peak_effort_nm']:issues.append(j['name']+': continuous effort exceeds peak effort')
        indices.append(j['sdk_index'])
        if j['encoder_direction'] not in [-1,1]:issues.append(j['name']+': invalid encoder direction')
        if not np.isfinite(j['encoder_zero_offset_rad']):issues.append(j['name']+': nonfinite zero offset')
        for label,ref in [('measured_kp','sim_kp'),('measured_kd','sim_kd')]:
            if not .85*reference[ref]<=j[label]<=1.15*reference[ref]:issues.append(j['name']+': '+label+' outside trained randomization')
        if j['hardware_peak_effort_nm']<reference['sim_peak_effort_nm']:
            issues.append(j['name']+': physical peak effort below simulation limit; retrain with corrected limit')
    if len(joints)!=29 or len(set(j['name'] for j in joints))!=29:issues.append('Joint list must contain 29 unique joints')
    if indices and len(set(indices))!=len(indices):issues.append('Duplicate hardware joint index')
    latency_report=None
    if latency is None:issues.append('No measured same-clock send/ack latency log')
    else:
        with open(latency,newline='') as f:rows=list(csv.DictReader(f))
        sends=np.array([float(r['send_monotonic_ns']) for r in rows])
        values=np.array([(float(r['ack_monotonic_ns'])-float(r['send_monotonic_ns']))/1e6 for r in rows])
        if len(values)<1000 or not np.isfinite(values).all() or (values<0).any() or (np.diff(sends)<=0).any():issues.append('Latency samples invalid, unordered/duplicated, or fewer than 1000')
        else:
            latency_report=dict(samples=len(values),median_ms=float(np.median(values)),p99_ms=float(np.percentile(values,99)),maximum_ms=float(values.max()),measurement='Same-host-clock send to acknowledgement roundtrip')
            if latency_report['p99_ms']>20.:issues.append('Measured roundtrip p99 exceeds trained 20ms delay envelope')
    if evaluation is None:issues.append('No completed 600s dynamics evaluation')
    else:
        values=[evaluation.get(k) for k in ['requested_duration_s','completion_fraction','steady_speed_mean_mps']]
        if not all(isinstance(v,(int,float)) and np.isfinite(v) for v in values) or values[0]<600 or values[1]<.95 or values[2]<2.:
            issues.append('Simulation endurance/speed criteria not met')
        if contract is not None and evaluation.get('policy_file_sha256')!=contract['policy_sha256']:
            issues.append('Endurance evaluation does not belong to selected ONNX policy')
        for k,j in enumerate(joints):
            if j.get('hardware_continuous_effort_nm') is not None:
                reference=next((x for x in contract['joints'] if x['name']==j['name']),None) if contract else j
                if reference is None:continue
                rms=float(np.atleast_2d(evaluation['joint_torque_rms_nm'])[:,reference['simulation_index']].max())
                if not np.isfinite(rms) or rms>j['hardware_continuous_effort_nm']:issues.append(j['name']+': RMS effort invalid or exceeds physical continuous effort')
    return dict(calibration_and_simulation_checks_passed=not issues,hardware_deployment_approved=False,
        issues=issues,latency=latency_report,
        remaining_real_validation='Physical torque-speed curve, thermal derating, approved deployment interface and battery-side V*I over a full 600s run')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,default=REPO/'deploy/checkpoints/loco.onnx')
    p.add_argument('--measurements',type=Path);p.add_argument('--evaluation',type=Path);p.add_argument('--latency',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    contract=template(a.policy)
    result=contract if a.measurements is None else audit(json.loads(a.measurements.read_text()),json.loads(a.evaluation.read_text()) if a.evaluation else None,a.latency,contract)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(a.output)
