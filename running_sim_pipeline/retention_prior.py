"""Conditional actor distillation to retain validated straight-running behavior.

Only training gradients are affected. The released policy is a single actor;
no teacher, state correction or command compensation is used in evaluation.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from amp import AmpWrapper


class RetentionWrapper(AmpWrapper):
    def __init__(self, env, weight=.4, source='dynamic'):
        super().__init__(env, weight, source)
        self.retention_buffer = torch.zeros((8192,960),device=env.device)
        self.retention_used=0
        self.retention_cursor=0

    def step(self, actions):
        obs,reward,done,extra=super().step(actions)
        command=self.unwrapped.command_manager.get_command('base_velocity')
        valid=(~done.bool()) & (command[:,0]>1.5) & (command[:,1].abs()<.08) & (command[:,2].abs()<.05)
        chosen=obs['actor'][valid].detach()
        if len(chosen):
            if len(chosen)>128:chosen=chosen[torch.randperm(len(chosen),device=chosen.device)[:128]]
            count=len(chosen)
            indices=(torch.arange(count,device=chosen.device)+self.retention_cursor)%len(self.retention_buffer)
            self.retention_buffer[indices]=chosen
            self.retention_cursor=(self.retention_cursor+count)%len(self.retention_buffer)
            self.retention_used=min(self.retention_used+count,len(self.retention_buffer))
        return obs,reward,done,extra


def install_retention_update(runner, env, teacher_path, weight=1.):
    """Attach one clipped actor-only distillation step after every PPO update."""
    path=Path(teacher_path)
    teacher=deepcopy(runner.alg.actor).to(runner.device)
    checkpoint=torch.load(path,weights_only=False,map_location=runner.device)
    teacher.load_state_dict(checkpoint['actor_state_dict'])
    teacher.eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    update=runner.alg.update

    def retained_update():
        losses=update()
        if env.retention_used<256:
            losses['retention_actor_mse']=0.
            return losses
        index=torch.randint(env.retention_used,(1024,),device=runner.device)
        obs={'actor':env.retention_buffer[index].detach()}
        with torch.no_grad():target=teacher(obs)
        prediction=runner.alg.actor(obs)
        mse=(prediction-target).square().mean()
        runner.alg.optimizer.zero_grad()
        (weight*mse).backward()
        torch.nn.utils.clip_grad_norm_(runner.alg.actor.parameters(),1.)
        runner.alg.optimizer.step()
        losses['retention_actor_mse']=float(mse.detach())
        return losses

    runner.alg.update=retained_update
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                weight=weight,mask='vx>1.5m/s, |vy|<0.08m/s, |yaw_rate|<0.05rad/s, not done',
                batch_size=1024,steps_per_ppo_update=1,inference_teacher=False)
