"""Contract checks for curriculum commands without a physics simulation."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from transition_optimization import TransitionRunningCommand
from retention_prior import install_retention_update


class TransitionCommandContract(unittest.TestCase):
    def make_command(self):
        c = TransitionRunningCommand.__new__(TransitionRunningCommand)
        c._env = SimpleNamespace(step_dt=.02, device='cpu', episode_length_buf=torch.zeros(4))
        c.target = torch.tensor([[1., .1, .1]] * 4)
        c.external_target = None
        c.vel_command_b = torch.zeros(4, 3)
        c.accel = torch.tensor([.8, .4, .6])
        c.sequence_mask = torch.tensor([True, False, True, False])
        c.sequence_speed = torch.full((4,), 2.2)
        c.sequence_turn = torch.full((4,), .3)
        return c

    def test_sequence_covers_both_turns_and_braking_preserves_rehearsal(self):
        c = self.make_command()
        expected = [(0., 0., 0.), (6., 2.2, 0.), (14., 2.2, .3),
                    (23., 0., 0.), (29., 2.2, -.3), (38., 0., 0.)]
        for seconds, vx, yaw in expected:
            c._env.episode_length_buf[:] = seconds / .02
            before = c.vel_command_b.clone()
            c._update_command()
            self.assertAlmostEqual(float(c.target[0, 0]), vx, places=5)
            self.assertAlmostEqual(float(c.target[0, 2]), yaw, places=5)
            torch.testing.assert_close(c.target[1], torch.tensor([1., .1, .1]))
            self.assertTrue(bool(((c.vel_command_b-before).abs() <= c.accel*.02+1e-6).all()))

    def test_external_commands_bypass_training_sequence(self):
        c = self.make_command()
        c._env.episode_length_buf[:] = 29. / .02
        c.external_target = torch.zeros(4, 3)
        original = c.target.clone()
        c._update_command()
        torch.testing.assert_close(c.vel_command_b, torch.zeros(4, 3))
        torch.testing.assert_close(c.target, original)

    def test_high_speed_turns_obey_existing_lateral_acceleration_limit(self):
        c = self.make_command()
        c.external_target = torch.tensor([[2.6, 0., .6]] * 4)
        for _ in range(1000):
            c._update_command()
        self.assertLessEqual(float((c.vel_command_b[:, 0]*c.vel_command_b[:, 2]).abs().max()), .90001)


class ToyActor(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.layer=torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():self.layer.weight.fill_(value)

    def forward(self,obs):
        return self.layer(obs['actor'])


class RetentionGradientContract(unittest.TestCase):
    def test_teacher_is_training_only_and_student_moves_toward_teacher(self):
        actor=ToyActor(0.)
        called=[]
        alg=SimpleNamespace(actor=actor,optimizer=torch.optim.Adam(actor.parameters(),lr=.01),
                            update=lambda:called.append(True) or {})
        runner=SimpleNamespace(alg=alg,device='cpu')
        env=SimpleNamespace(retention_used=256,retention_buffer=torch.ones(256,2))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'teacher.pt'
            torch.save({'actor_state_dict':ToyActor(1.).state_dict()},path)
            original=path.read_bytes()
            metadata=install_retention_update(runner,env,path,1.)
            losses=runner.alg.update()
            self.assertEqual(len(called),1)
            self.assertGreater(float(actor.layer.weight.detach().min()),0.)
            self.assertGreater(losses['retention_actor_mse'],0.)
            self.assertFalse(metadata['inference_teacher'])
            self.assertEqual(path.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
