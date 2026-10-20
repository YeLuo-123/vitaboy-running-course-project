"""Isolated command-driven running task; official assets are never edited."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parent
WS=ROOT.parent
REPO=WS/'honor_rl_mjlab'
sys.path.insert(0,str(REPO))
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.managers import RewardTermCfg,TerminationTermCfg,EventTermCfg,SceneEntityCfg
from mjlab.envs.mdp import dr
from mjlab.terrains import TerrainEntityCfg,TerrainGeneratorCfg,HfRandomUniformTerrainCfg,HfWaveTerrainCfg
from whole_body_control.tasks.velocity.mdp.velocity_command import UniformVelocityCommand,UniformVelocityCommandCfg


class SmoothRunningCommand(UniformVelocityCommand):
    """Acceleration-limited commands; never changes robot pose or velocity."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.target=torch.zeros_like(self.vel_command_b)
        self.external_target=None
        self.accel=torch.tensor([.8,.4,.6],device=self.device)

    def _resample_command(self,env_ids):
        progress=min(float(self._env.common_step_counter)/(24*800),1.)
        maximum=self.cfg.ranges.lin_vel_x[1]
        upper=maximum if not self.cfg.speed_curriculum else min(maximum,1.2+progress*(maximum-1.2))
        self.target[env_ids,0]=torch.empty(len(env_ids),device=self.device).uniform_(.3,upper)
        self.target[env_ids,1]=torch.empty(len(env_ids),device=self.device).uniform_(-.12,.12)
        self.target[env_ids,2]=torch.empty(len(env_ids),device=self.device).uniform_(-.6,.6)
        if self.cfg.focus_running:
            fast=torch.rand(len(env_ids),device=self.device)<.75
            self.target[env_ids[fast],0]=torch.empty(int(fast.sum()),device=self.device).uniform_(min(2.,upper),upper)
            straight=torch.rand(len(env_ids),device=self.device)<.65
            self.target[env_ids[straight],1:]=0.
        standing=torch.rand(len(env_ids),device=self.device)<self.cfg.rel_standing_envs
        self.target[env_ids[standing]]=0.

    def reset(self,env_ids):
        self.vel_command_b[env_ids]=0.
        return super().reset(env_ids)

    def _update_command(self):
        target=self.target if self.external_target is None else self.external_target
        desired=target.clone()
        # Limit commanded lateral acceleration while turning at running speed.
        limit=.9/desired[:,0].abs().clamp_min(.6)
        desired[:,2]=torch.maximum(torch.minimum(desired[:,2],limit),-limit)
        delta=self.accel*self._env.step_dt
        self.vel_command_b+=torch.maximum(torch.minimum(desired-self.vel_command_b,delta),-delta)


@dataclass(kw_only=True)
class SmoothRunningCommandCfg(UniformVelocityCommandCfg):
    speed_curriculum:bool=True
    focus_running:bool=False
    def build(self,env):return SmoothRunningCommand(self,env)


def total_power(env):
    d=env.scene['robot'].data
    return (d.qfrc_actuator*d.joint_vel).abs().sum(-1)


def stand_still(env):
    d=env.scene['robot'].data
    standing=env.command_manager.get_command('base_velocity').abs().sum(-1)<.08
    return (d.joint_vel.square().mean(-1)+d.root_link_lin_vel_b[:,:2].square().sum(-1))*standing


def low_base(env):
    d=env.scene['robot'].data
    return (d.root_link_pos_w[:,2]-env.scene.env_origins[:,2]<.40)|(d.projected_gravity_b[:,2]>-.5)


def make_config(stage='flat',num_envs=2048,training=True,seed=42,focus_running=False):
    cfg=deepcopy(load_env_cfg('VitaBoy-Velocity-Flat',play=not training))
    cfg.seed=seed;cfg.scene.num_envs=num_envs
    cfg.scene.terrain=TerrainEntityCfg(terrain_type='plane',env_spacing=4.)
    cfg.curriculum={}
    if stage=='rough':
        cfg.scene.terrain=TerrainEntityCfg(terrain_type='generator',max_init_terrain_level=0,
            terrain_generator=TerrainGeneratorCfg(seed=seed,size=(160.,160.),num_rows=1,num_cols=2,
                border_width=5.,curriculum=False,add_lights=True,
                sub_terrains={
                    'rough':HfRandomUniformTerrainCfg(proportion=.7,noise_range=(-.015,.025),
                        noise_step=.005,horizontal_scale=.25,vertical_scale=.0025,border_width=1.),
                    'wave':HfWaveTerrainCfg(proportion=.3,amplitude_range=(.015,.04),num_waves=16,
                        horizontal_scale=.25,vertical_scale=.0025)}))
    command=SmoothRunningCommandCfg(entity_name='robot',resampling_time_range=(3.,7.),
        rel_standing_envs=.2,heading_command=False,debug_vis=False,
        speed_curriculum=training and stage=='flat',
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.,2.6),lin_vel_y=(-.12,.12),ang_vel_z=(-.6,.6)))
    if focus_running:
        command.focus_running=True;command.resampling_time_range=(7.,14.)
    cfg.commands={'base_velocity':command}
    cfg.episode_length_s=20. if stage=='rough' else 30.
    if focus_running:cfg.episode_length_s=30.
    cfg.sim.nconmax=128;cfg.sim.njmax=768
    cfg.observations['actor'].history_length=10
    cfg.observations['critic'].history_length=10
    cfg.observations['actor'].enable_corruption=training
    cfg.events['reset_base'].params.update(pose_range={'x':(-.05,.05),'y':(-.05,.05),'yaw':(-3.14,3.14)},
        velocity_range={})
    cfg.events['reset_robot_joints'].params.update(position_range=(.98,1.02),velocity_range=(0.,0.))
    cfg.events.pop('add_base_mass',None)
    cfg.events['physics_material'].params['ranges']=(.65,.95)
    cfg.events.pop('push_robot',None)
    if stage in ['robust','rough']:
        cfg.events['mass']=EventTermCfg(func=dr.body_mass,mode='startup',
            params={'asset_cfg':SceneEntityCfg('robot',body_names=('.*',)),'ranges':(.9,1.1),'operation':'scale'})
        cfg.events['com']=EventTermCfg(func=dr.body_com_offset,mode='startup',
            params={'asset_cfg':SceneEntityCfg('robot',body_names=('torso_link',)),'ranges':(-.015,.015),'operation':'add'})
        cfg.events['pd']=EventTermCfg(func=dr.pd_gains,mode='startup',params={'kp_range':(.85,1.15),'kd_range':(.85,1.15)})
        cfg.events['strength']=EventTermCfg(func=dr.effort_limits,mode='startup',params={'effort_limit_range':(.85,1.)})
        from whole_body_control.tasks.velocity import mdp
        cfg.events['push_robot']=EventTermCfg(func=mdp.push_by_setting_velocity,mode='interval',
            interval_range_s=(7.,12.),params={'velocity_range':{'x':(-.25,.25),'y':(-.25,.25)}})
        robot=deepcopy(cfg.scene.entities['robot'])
        for actuator in robot.articulation.actuators:
            actuator.delay_min_lag=0;actuator.delay_max_lag=4
            actuator.delay_update_period=1
        cfg.scene.entities['robot']=robot
    if not training and stage=='flat':
        cfg.events.pop('physics_material',None)
    r=cfg.rewards
    r.pop('fly',None)
    r['track_lin_vel_xy_exp'].weight=4.;r['track_lin_vel_xy_exp'].params['std']=.65
    r['track_ang_vel_z_exp'].weight=2.5 if stage=='rough' else 1.
    if focus_running:r['track_ang_vel_z_exp'].weight=5.
    r['energy']=RewardTermCfg(func=total_power,weight=-.0005)
    r['termination_penalty'].weight=-50.
    r['lin_vel_z_l2'].weight=-.3
    r['feet_air_time'].weight=.15
    r['feet_too_near'].params['threshold']=.10
    r['joint_deviation_hip'].weight=-.015
    r['joint_deviation_arms'].weight=-.03
    r['stand_still']=RewardTermCfg(func=stand_still,weight=-.15)
    cfg.terminations['fallen']=TerminationTermCfg(func=low_base)
    cfg.viewer.distance=2.8;cfg.viewer.elevation=-15;cfg.viewer.azimuth=45
    cfg.viewer.height=512;cfg.viewer.width=768;cfg.viewer.max_extra_envs=0
    return cfg


def agent_config():
    cfg=deepcopy(load_rl_cfg('VitaBoy-Velocity-Flat'))
    cfg.actor.distribution_cfg['init_std']=.25
    cfg.algorithm.learning_rate=1e-4;cfg.algorithm.entropy_coef=.002
    cfg.algorithm.num_learning_epochs=5
    cfg.experiment_name='vita_boy_running_sim';cfg.save_interval=250
    cfg.upload_model=False
    return cfg
