"""Export official GVHMR result to portable world-Z-up SMPL-X skeleton."""
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
args=parser.parse_args()
model_path=ROOT.parent/'GVHMR/inputs/checkpoints/body_models'
pred=torch.load(args.prediction,map_location='cpu')
params=pred['smpl_params_global']
# Match the model's shape dimension to GVHMR's actual betas. Do not pad
# 10-dimensional GVHMR betas to 16 while constructing a 10-beta body model.
num_betas=params['betas'].shape[-1]
model=smplx.create(model_path,'smplx',gender='neutral',use_pca=False,num_betas=num_betas)
count=params['body_pose'].shape[0]
with torch.no_grad():
    output=model(**params,left_hand_pose=torch.zeros(count,45),right_hand_pose=torch.zeros(count,45),
                 jaw_pose=torch.zeros(count,3),leye_pose=torch.zeros(count,3),reye_pose=torch.zeros(count,3),
                 expression=torch.zeros(count,10),
                 return_full_pose=True)
height=float(1.66+.1*params['betas'][0,0])
data=dict(pose_body=params['body_pose'].numpy(),mocap_frame_rate=torch.tensor(30.))
frames,fps=get_gvhmr_data_offline_fast(data,model,output,tgt_fps=30)
names=list(frames[0])
pos=np.array([[frame[n][0] for n in names] for frame in frames])
rot=np.array([[frame[n][1] for n in names] for frame in frames])
assert np.isfinite(pos).all() and np.isfinite(rot).all()
np.savez(ROOT/'outputs/human_skeleton.npz',positions=pos,quaternions_wxyz=rot,
         joint_names=names,parents=model.parents.numpy(),fps=fps,human_height=height)
print('EXPORTED',pos.shape,'fps',fps,'height',height)
