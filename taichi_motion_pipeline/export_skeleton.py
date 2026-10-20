"""Batch GVHMR SMPL-X evaluation for the complete 42-second Tai Chi clip."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'GMR_source'))
import numpy as np
import torch
import smplx
from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast

parser=argparse.ArgumentParser()
parser.add_argument('--prediction',type=Path,default=ROOT/'outputs/user_video/hmr4d_results.pt')
parser.add_argument('--batch-size',type=int,default=96)
args=parser.parse_args()
pred=torch.load(args.prediction,map_location='cpu')
params=pred['smpl_params_global']
num_betas=params['betas'].shape[-1]
model=smplx.create(ROOT.parent/'GVHMR/inputs/checkpoints/body_models','smplx',
                   gender='neutral',use_pca=False,num_betas=num_betas)
count=params['body_pose'].shape[0]
frames=[]
with torch.no_grad():
    for start in range(0,count,args.batch_size):
        end=min(start+args.batch_size,count);size=end-start
        chunk={k:(v[start:end] if v.shape[0]==count else v.expand(size,-1)) for k,v in params.items()}
        output=model(**chunk,left_hand_pose=torch.zeros(size,45),right_hand_pose=torch.zeros(size,45),
                     jaw_pose=torch.zeros(size,3),leye_pose=torch.zeros(size,3),reye_pose=torch.zeros(size,3),
                     expression=torch.zeros(size,10),return_full_pose=True)
        data=dict(pose_body=chunk['body_pose'].numpy(),mocap_frame_rate=torch.tensor(30.))
        batch,fps=get_gvhmr_data_offline_fast(data,model,output,tgt_fps=30)
        frames.extend(batch)
        print('EXPORTED_BATCH',start,end,flush=True)
names=list(frames[0])
positions=np.array([[frame[n][0] for n in names] for frame in frames])
rot=np.array([[frame[n][1] for n in names] for frame in frames])
assert len(frames)==count and np.isfinite(positions).all() and np.isfinite(rot).all()
np.savez(ROOT/'outputs/human_skeleton.npz',positions=positions,quaternions_wxyz=rot,
         joint_names=names,parents=model.parents.numpy(),fps=fps,
         human_height=float(1.66+.1*params['betas'][0,0]))
print('EXPORTED',positions.shape,'fps',fps,flush=True)
