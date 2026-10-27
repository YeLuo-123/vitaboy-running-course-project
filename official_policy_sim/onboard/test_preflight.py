"""Offline regression tests for the hardware start gate. Never initialize ROS."""
import copy
import unittest
from preflight_readonly import EXPECTED, evaluate


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.last = {'low_state': {'mode_pr': 0, 'mode_machine': 0,
            'motor_state': [{'joint_id': i, 'error_code': 0, 'pos_fb': 0., 'vel_fb': 0.,
                             'tau_fb': 0., 'temp_fb': [25., 25.]} for i in EXPECTED]},
            'robot_state': {'is_sim': False}, 'imu': {'orientation': dict(x=0., y=0., z=0., w=1.),
            'orientation_covariance': [0.]*9, 'angular_velocity': dict(x=0., y=0., z=0.),
            'linear_acceleration': dict(x=0., y=0., z=9.81)}}
        self.counts = {k: 100 for k in ('low_state', 'imu', 'monitor', 'robot_state')}
        self.ages = {k: .01 for k in self.counts}
        self.first = {i: (100, 100) for i in EXPECTED}
        self.final = {i: (2100, 2100) for i in EXPECTED}
        self.pubs = []

    def failures(self, exclusive=True):
        return evaluate(self.last, self.counts, self.ages, self.first, self.final,
                        self.pubs, [{'node': 'controller_node'}], exclusive)[0]

    def test_healthy_and_exclusive(self):
        self.assertEqual(self.failures(), [])

    def test_factory_publisher_blocks_start_but_not_inspection(self):
        self.pubs = [{'node': 'motion_intelligence'}]
        self.assertTrue(self.failures())
        self.assertEqual(self.failures(False), [])

    def test_live_ros_with_frozen_motor_response_is_rejected(self):
        self.final[20006] = (2100, 100)
        self.assertTrue(any('20006' in f for f in self.failures()))

    def test_wrong_joint_order_is_rejected(self):
        self.last['low_state']['motor_state'].reverse()
        self.assertTrue(self.failures())

    def test_stale_state_is_rejected(self):
        self.ages['low_state'] = 1.
        self.assertTrue(self.failures())

    def test_simulator_feedback_is_rejected(self):
        self.last['robot_state']['is_sim'] = True
        self.assertTrue(self.failures())

    def test_invalid_imu_is_rejected(self):
        self.last['imu']['orientation']['w'] = float('nan')
        self.assertTrue(self.failures())

    def test_wrapped_counters_are_allowed(self):
        self.first[10006] = (2**32-100, 2**32-100)
        self.final[10006] = (1900, 1900)
        self.assertEqual(self.failures(), [])


if __name__ == '__main__':
    unittest.main()
