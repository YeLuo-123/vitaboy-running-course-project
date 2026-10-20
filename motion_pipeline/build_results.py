"""Assemble inspectable evidence, source manifests, comparison video and plots."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; ART=ROOT/'artifacts'
motion=ROOT/'data/retargeted/lafan_run2_s4_geometry'
clean=json.loads((ART/'eval_clean.json').read_text())
trials=[json.loads((ART/f'eval_jitter_{seed}.json').read_text()) for seed in [123,456,789]]
cpu=json.loads((ART/'eval_cpu_deploy.json').read_text())
summary=dict(selected_checkpoint='model_500.pt',motion=json.loads((motion/'manifest.json').read_text()),
    clean=clean,perturbed_evaluations=trials,cpu_deployment=cpu,
    perturbed_trials=sum(x['num_envs'] for x in trials),
    perturbed_completed=sum(x['complete_count'] for x in trials),
    all_gpu_acceptance_checks_pass=clean['passes_engineering_acceptance'] and all(x['passes_engineering_acceptance'] for x in trials),
    scope='Finite approximately four-second running clip, initialized from its reference pose and velocity. Not a stand-to-run, long-duration, terrain, AMP, or hardware validation.',
    training_seed=42,evaluation_seeds=[123,456,789],
    experiment_note='Training was configured for 2000 iterations and stopped after iteration 821 once checkpoint 500 passed the milestone; checkpoints through 750 retained remotely.')
(ART/'validation_summary.json').write_text(json.dumps(summary,indent=2))
assert summary['all_gpu_acceptance_checks_pass']
assert cpu['completed'] and cpu['root_position_rmse_m']<.2 and cpu['relative_body_position_rmse_m']<.06
assert clean['onnx_max_abs_action_error']<1.e-5

destination=ART/'motion'; destination.mkdir(exist_ok=True)
for name in ['run.csv','run.npz','retargeted_qpos.npz','raw_gmr_qpos.npz','manifest.json','bvh_lafan1_to_vita_boy.json',
             'reference_kinematic.mp4','reference_frames.png','human_to_vita_retarget.mp4','human_to_vita_frames.png']:
    shutil.copy2(motion/name,destination/name)
provenance=ART/'provenance'; provenance.mkdir(exist_ok=True)
shutil.copy2(ROOT.parent/'GMR_source/source_manifest.json',provenance/'gmr_source_manifest.json')
shutil.copy2(ROOT.parent/'GMR_source/LICENSE',provenance/'GMR_LICENSE')
for name in ['LAFAN1_license.txt','LAFAN1_README.md','source_manifest.json']:
    shutil.copy2(ROOT/'data/raw'/name,provenance/name)
for name in ['eval_official_geometry.json','eval_zero_geometry.json','eval_reference_pd_geometry.json']:
    shutil.copy2(ROOT/name,ART/name)
provenance.joinpath('honor_training_commit.txt').write_text(subprocess.check_output(
    ['git','-C',str(ROOT.parent/'honor_rl_mjlab'),'rev-parse','HEAD'],text=True))

trajectory=json.loads((ART/'eval_clean.trajectory.json').read_text())
reference_video=imageio.get_reader(motion/'reference_kinematic.mp4')
policy_video=imageio.get_reader(ART/'tracking_policy.mp4')
sheet=Image.new('RGB',(1280,512*3),'white')
with imageio.get_writer(ART/'reference_vs_tracking.mp4',fps=50) as writer:
    for i,row in enumerate(trajectory):
        canvas=Image.new('RGB',(1280,512),'white')
        canvas.paste(Image.fromarray(reference_video.get_data(row['reference_cursor'])),(0,32))
        canvas.paste(Image.fromarray(policy_video.get_data(i)),(640,32))
        draw=ImageDraw.Draw(canvas)
        draw.text((8,8),f'GMR reference (kinematic) | t={row["t"]:.2f}s',fill='black')
        draw.text((648,8),f'Tracking policy (physics) | vx={row["forward_velocity"]:.2f} m/s | no virtual support',fill='black')
        writer.append_data(np.array(canvas))
        if i in [25,85,145]: sheet.paste(canvas,(0,[25,85,145].index(i)*512))
sheet.save(ART/'reference_vs_tracking_frames.png')
reference_video.close(); policy_video.close()

t=np.array([r['t'] for r in trajectory]); actual=np.array([r['root_actual'] for r in trajectory]); ref=np.array([r['root_reference'] for r in trajectory])
actual-=ref[0]; ref-=ref[0]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(14,7.6),layout='constrained')
ax=axes[0,0]; ax.plot(t,ref[:,0],label='Reference',color='#6b7280',ls='--'); ax.plot(t,actual[:,0],label='Policy',color='#1463bf'); ax.set(title='Forward displacement',xlabel='Time (s)',ylabel='Displacement (m)'); ax.legend()
ax=axes[0,1]; ax.plot(t,[r['reference_speed'] for r in trajectory],label='Reference planar speed',color='#6b7280',ls='--'); ax.plot(t,[r['forward_velocity'] for r in trajectory],label='Policy forward speed',color='#1463bf'); ax.set(title='Speed tracking',xlabel='Time (s)',ylabel='Speed (m/s)'); ax.legend(fontsize=8)
ax=axes[0,2]; ax.plot(t,[r['joint_rmse'] for r in trajectory],color='#1463bf'); ax.set(title=f'Joint error (aggregate RMSE {clean["joint_rmse_rad"]:.3f} rad)',xlabel='Time (s)',ylabel='Joint RMSE (rad)')
ax=axes[1,0]
for j,(side,color) in enumerate([('Left','#1463bf'),('Right','#d97706')]):
    ax.plot(t,[r['knee_reference'][j] for r in trajectory],color=color,ls='--',alpha=.6,label=side+' reference')
    ax.plot(t,[r['knee_actual'][j] for r in trajectory],color=color,label=side+' policy')
ax.set(title='Knee motion',xlabel='Time (s)',ylabel='Joint position (rad)'); ax.legend(ncol=2,fontsize=8)
ax=axes[1,1]; contact=np.array([r['foot_contact'] for r in trajectory],dtype=bool)
ax.fill_between(t,0,contact[:,1].astype(float)*.7,step='post',color='#d97706',label='Right contact')
ax.fill_between(t,1,1+contact[:,0].astype(float)*.7,step='post',color='#1463bf',label='Left contact')
ax.fill_between(t,0,1.8,where=~contact.any(1),color='#16a34a',alpha=.25,step='post',label='Both feet airborne')
ax.axvline(.5,color='#6b7280',ls=':',lw=1); ax.set(title='Measured foot contact (>1 N)',xlabel='Time (s)',yticks=[]); ax.legend(fontsize=8,loc='upper right')
ax=axes[1,2]; x=np.arange(3); vals=[c['root_position_rmse_m'] for c in trials]
ax.bar(x,vals,color='#1463bf',width=.55); ax.axhline(.2,color='#d97706',ls='--',label='Milestone mean-error limit')
ax.set_xticks(x,[str(c['seed']) for c in trials]); ax.set(title=f'Perturbed starts: {summary["perturbed_completed"]}/{summary["perturbed_trials"]} completed',xlabel='Evaluation seed (32 trials each)',ylabel='Root position RMSE (m)',ylim=(0,.23)); ax.legend(fontsize=8)
for ax in axes.flat: ax.grid(axis='y',alpha=.18)
fig.suptitle('VitaBoy retargeted running: finite-clip Tracking validation (3.92 s)',fontsize=15)
fig.savefig(ART/'tracking_validation.png',dpi=170); plt.close(fig)
hashes={str(p.relative_to(ART)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ART.rglob('*') if p.is_file() and p.suffix not in ['.log'] and p.name!='sha256_manifest.json'}
(ART/'sha256_manifest.json').write_text(json.dumps(hashes,indent=2))
print('VALIDATION_SUMMARY',json.dumps({k:summary[k] for k in ['perturbed_trials','perturbed_completed','all_gpu_acceptance_checks_pass']}),flush=True)
print('ARTIFACTS',ART,flush=True)
