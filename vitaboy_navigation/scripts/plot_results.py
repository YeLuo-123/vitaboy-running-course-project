"""Plot measured SLAM occupancy and map-frame trajectory, not ground-truth geometry."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
args=parser.parse_args()
mapping=json.loads((args.results/'integration.map.json').read_text())
report=json.loads((args.results/'integration.json').read_text())
data=np.array(mapping['data']).reshape(mapping['height'],mapping['width'])
image=np.where(data<0,.6,np.where(data>=50,0.,1.))
ox,oy=mapping['origin'];res=mapping['resolution']
fig,ax=plt.subplots(figsize=(8,8))
ax.imshow(image,origin='lower',cmap='gray',vmin=0,vmax=1,
          extent=[ox,ox+data.shape[1]*res,oy,oy+data.shape[0]*res])
trajectory=np.array(mapping['trajectory_map'])
if len(trajectory):
    ax.plot(trajectory[:,1],trajectory[:,2],color='#1545ba',linewidth=2,label='Measured map-frame trajectory')
for i,result in enumerate(report['goals']):
    x,y=result['goal'];ax.scatter(x,y,c='#ee7722',marker='x',s=70)
    ax.annotate(str(i+1),(x+.08,y+.08))
ax.set(xlabel='Map X (m)',ylabel='Map Y (m)',title='VitaBoy: online SLAM and physical navigation')
ax.set_aspect('equal');ax.legend(loc='upper left');fig.tight_layout()
fig.savefig(args.results/'slam_navigation.png',dpi=180)
