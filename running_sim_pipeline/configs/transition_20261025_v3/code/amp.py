"""Transition discriminator and LS-AMP reward, trained alongside PPO.

Expert features come from the previously retargeted running reference. No
reference state is ever written to the simulated robot during training.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply_inverse
from run_config import WS
from run_config import ROOT


def state_features(raw,foot_ids):
    d=raw.scene['robot'].data
    feet=d.body_link_pos_w[:,foot_ids]-d.root_link_pos_w[:,None,:]
    quat=d.root_link_quat_w[:,None,:].expand(-1,2,-1)
    feet=quat_apply_inverse(quat.reshape(-1,4),feet.reshape(-1,3)).reshape(raw.num_envs,6)
    height=(d.root_link_pos_w[:,2]-raw.scene.env_origins[:,2])[:,None]
    return torch.cat([d.joint_pos-d.default_joint_pos,d.joint_vel*.1,d.projected_gravity_b,
                      height,d.root_link_lin_vel_b*.5,d.root_link_ang_vel_b*.2,feet],dim=-1).float()


def build_experts(default_q,foot_ids,device,source='kinematic'):
    physical=ROOT/'assets/dynamic_amp_expert.npz'
    if source=='dynamic':
        if not physical.exists():raise FileNotFoundError('Run collect_dynamic_expert.py first')
        transitions=np.load(physical)['transitions'].astype(np.float32)
        return torch.tensor(transitions,device=device),dict(source=str(physical),
            sha256=hashlib.sha256(physical.read_bytes()).hexdigest(),feature_dim=74,transition_dim=148,
            expert_transitions=len(transitions),time_warp_factors=[],
            source_type='Validated Tracking dynamics rollouts; no kinematic reconstruction gap')
    path=WS/'motion_pipeline/artifacts/motion/run.npz';motion=np.load(path)
    q=motion['joint_pos'];qd=motion['joint_vel']
    root=motion['body_pos_w'][:,0];quat=motion['body_quat_w'][:,0]
    inv=Rotation.from_quat(quat,scalar_first=True).inv()
    gravity=inv.apply(np.tile([0.,0.,-1.],(len(q),1)))
    vel=inv.apply(motion['body_lin_vel_w'][:,0])
    omega=inv.apply(motion['body_ang_vel_w'][:,0])
    feet=np.stack([inv.apply(motion['body_pos_w'][:,j]-root) for j in foot_ids],axis=1).reshape(len(q),6)
    state=np.column_stack([q-default_q,qd*.1,gravity,root[:,2],vel*.5,omega*.2,feet]).astype(np.float32)
    pairs=[]
    # Physically unvalidated time-warps are augmentation, not evaluation results.
    for speed_factor in [.9,1.,1.1,1.2,1.3]:
        for t in np.arange(0,len(state)-1-speed_factor,.25):
            pair=[]
            for cursor in [t,t+speed_factor]:
                i=int(cursor);f=cursor-i;x=(1-f)*state[i]+f*state[i+1]
                x=x.copy();x[29:58]*=speed_factor;x[62:68]*=speed_factor
                pair.append(x)
            pairs.append(np.concatenate(pair))
    return torch.tensor(np.asarray(pairs),device=device,dtype=torch.float32),dict(
        source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        feature_dim=74,transition_dim=148,expert_transitions=len(pairs),
        time_warp_factors=[.9,1.,1.1,1.2,1.3],
        caveat='Time-warped references are expert augmentation, not proven physically executable motions')


class AmpPrior:
    def __init__(self,env,source='kinematic'):
        self.raw=env;self.device=env.device
        self.legacy=source=='kinematic'
        robot=env.scene['robot'];self.foot_ids=robot.find_bodies(('left_ankle_roll_link','right_ankle_roll_link'))[0]
        self.expert,self.provenance=build_experts(robot.data.default_joint_pos[0].cpu().numpy(),self.foot_ids,self.device,source)
        self.mean=self.expert.mean(0);self.std=self.expert.std(0).clamp_min(.15)
        self.discriminator=nn.Sequential(nn.Linear(148,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1)).to(self.device)
        self.optimizer=torch.optim.Adam(self.discriminator.parameters(),lr=1e-4 if self.legacy else 2e-5,
                                        weight_decay=1e-4 if self.legacy else 1e-3)
        self.replay=torch.zeros((65536,148),device=self.device);self.used=0;self.cursor=0

    def features(self):
        return state_features(self.raw,self.foot_ids)

    def normalize(self,x):return ((x-self.mean)/self.std).clamp(-10.,10.)

    def observe(self,pair,valid):
        chosen=pair[valid]
        if not len(chosen):return
        if len(chosen)>256:chosen=chosen[torch.randperm(len(chosen),device=self.device)[:256]]
        n=len(chosen);indices=(torch.arange(n,device=self.device)+self.cursor)%len(self.replay)
        self.replay[indices]=chosen;self.cursor=(self.cursor+n)%len(self.replay);self.used=min(self.used+n,len(self.replay))

    def reward(self,pair):
        score=self.discriminator(self.normalize(pair)).squeeze(-1)
        return (1-.25*(score-1).square()).clamp(0.,1.)

    def update(self):
        if self.used<256:return dict(amp_discriminator=0.,amp_expert_score=0.,amp_policy_score=0.)
        values=[]
        for _ in range(4 if self.legacy else 2):
            expert=self.expert[torch.randint(len(self.expert),(1024,),device=self.device)].detach().clone()
            policy=self.replay[torch.randint(self.used,(1024,),device=self.device)].detach().clone()
            x=self.normalize(expert);y=self.normalize(policy)
            if self.legacy:
                x.requires_grad_();e=self.discriminator(x);p=self.discriminator(y)
                grad=torch.autograd.grad(e.sum(),x,create_graph=True)[0]
                loss=.5*((e-1).square().mean()+(p+1).square().mean())+5.*grad.square().sum(-1).mean()
            else:
                alpha=torch.rand((len(x),1),device=self.device)
                mixed=(alpha*x+(1-alpha)*y).requires_grad_()
                e=self.discriminator(x);p=self.discriminator(y)
                score=self.discriminator(mixed)
                grad=torch.autograd.grad(score.sum(),mixed,create_graph=True)[0]
                loss=.5*((e-.8).square().mean()+(p+.8).square().mean())+5.*grad.square().sum(-1).mean()
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.discriminator.parameters(),1.)
            self.optimizer.step();values.append([float(loss.detach()),float(e.mean().detach()),float(p.mean().detach())])
        result=np.mean(values,axis=0)
        return dict(zip(['amp_discriminator','amp_expert_score','amp_policy_score'],result.tolist()))

    def state_dict(self):
        return dict(discriminator=self.discriminator.state_dict(),optimizer=self.optimizer.state_dict(),
                    mean=self.mean,std=self.std,expert_provenance=self.provenance)

    def load_state_dict(self,state):
        if state['expert_provenance']['sha256']!=self.provenance['sha256']:raise ValueError('AMP expert changed')
        self.discriminator.load_state_dict(state['discriminator']);self.optimizer.load_state_dict(state['optimizer'])
        self.mean=state['mean'].to(self.device);self.std=state['std'].to(self.device)


class AmpWrapper(RslRlVecEnvWrapper):
    def __init__(self,env,weight=.4,source='kinematic'):
        super().__init__(env);self.amp=AmpPrior(env,source);self.amp_weight=weight

    def step(self,actions):
        before=self.amp.features()
        obs,reward,done,extra=super().step(actions)
        pair=torch.cat([before,self.amp.features()],-1)
        running=self.unwrapped.command_manager.get_command('base_velocity')[:,0]>1.3
        valid=~done.bool()&running
        self.amp.observe(pair,valid)
        style=self.amp.reward(pair)*valid
        reward=reward+self.amp_weight*self.unwrapped.step_dt*style
        extra.setdefault('log',{})['AMP/style_reward']=style.mean()
        return obs,reward,done,extra
