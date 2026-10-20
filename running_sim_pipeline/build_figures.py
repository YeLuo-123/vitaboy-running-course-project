"""Plot actual evaluation trajectories and TensorBoard scalar evidence."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_config import ROOT

p=argparse.ArgumentParser();p.add_argument('--prefix',default='flat_799');args=p.parse_args()
out=ROOT/'artifacts';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
cpu=ROOT/'results'/(args.prefix+'_cpu_endurance_600s.json')
if cpu.exists():
    data=json.loads(cpu.with_suffix('.trajectory.json').read_text());report=json.loads(cpu.read_text())
    t=np.array([r['t'] for r in data]);v=np.array([r['vx_mps'] for r in data]);root=np.array([r['root'] for r in data])
    fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    axes[0].plot(t,v,color='#7795aa',alpha=.35,lw=.6,label='Measured body-forward speed')
    axes[0].plot(t[49:],np.convolve(v,np.ones(50)/50,'valid'),color='#176699',label='5-second trailing mean')
    axes[0].axhline(2.,color='#38804b',ls='--',label='2 m/s criterion')
    axes[0].set(xlabel='Simulated time (s)',ylabel='Actual speed (m/s)',title='600-second independent CPU deployment simulation')
    axes[0].legend(loc='lower right')
    axes[1].plot(root[:,0],root[:,1],color='#176699');axes[1].set(xlabel='World X (m)',ylabel='World Y (m)',title=f"Actual path | {report['distance_m']:.1f} m | no reference wrap or virtual support")
    fig.savefig(out/(args.prefix+'_endurance.png'),dpi=160);plt.close(fig)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
stages=[('flat_v2_seed42','Flat','#4287b0',1500),('robust_seed42','Randomized flat','#bd862c',1500),('rough_seed42','Randomized rough','#3b8c61',1000),('rough_focus_seed42','Sustained running','#9665b0',1500),('rough_stability_seed42','Foot margin + yaw','#ba574a',1500),('rough_precision_seed42','Clearance + mean yaw','#555555',1500)]
for folder,label,color,maxlen in stages:
    path=ROOT/'logs'/folder
    if not path.exists():continue
    ea=EventAccumulator(str(path),size_guidance={'scalars':0});ea.Reload()
    def curve(tag):
        if tag not in ea.Tags()['scalars']:return np.array([]),np.array([])
        items=ea.Scalars(tag);return np.array([s.step for s in items]),np.array([s.value for s in items])
    x,y=curve('Train/mean_episode_length');axes[0].plot(x,y/maxlen,color=color,label=label)
    x,y=curve('AMP/style_reward');axes[1].plot(x,y,color=color,label=label)
    x,y=curve('Loss/amp_expert_score');axes[2].plot(x,y,color=color,label=label+' expert')
    x,y=curve('Loss/amp_policy_score');axes[2].plot(x,y,color=color,ls='--',label=label+' policy')
axes[0].set(ylabel='Mean episode length / horizon',title='Training evidence; rollout statistics are not held-out acceptance')
axes[1].set(ylabel='AMP style reward (running mask)')
axes[2].set(ylabel='Discriminator score',xlabel='Checkpoint iteration label')
for ax in axes:ax.legend(ncol=2,fontsize=8)
fig.savefig(out/'training_evidence.png',dpi=160);plt.close(fig)
print('FIGURES_WRITTEN',out)
