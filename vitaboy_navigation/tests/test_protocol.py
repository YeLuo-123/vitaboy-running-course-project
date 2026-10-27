import json
import math
import unittest
from vitaboy_navigation.protocol import (CommandGate, transform_planar, approach_command,
                                        brake_forward, RUNNING_LIMITS,running_approach_command,brake_running)
from vitaboy_navigation.planning import astar_grid


class ProtocolTest(unittest.TestCase):
    def message(self, **changes):
        data = dict(schema=1, robot='vitaboy', seq=1, session='test', command=[.2, 0., 0.])
        data.update(changes)
        return json.dumps(data).encode()

    def test_no_command_stops(self):
        self.assertEqual(CommandGate().target(100), (0., 0., 0.))

    def test_watchdog_and_bad_packet_does_not_extend_it(self):
        gate = CommandGate(); self.assertTrue(gate.receive(self.message(), 1.))
        self.assertFalse(gate.receive(b'{broken', 1.24))
        self.assertEqual(gate.target(1.26), (0., 0., 0.))

    def test_nonfinite_and_wrong_robot(self):
        gate = CommandGate()
        self.assertFalse(gate.receive(self.message(command=[float('nan'), 0., 0.]), 1))
        self.assertFalse(gate.receive(self.message(robot='g1'), 1))

    def test_duplicate_cannot_keep_motion_alive(self):
        gate = CommandGate(); gate.receive(self.message(), 1)
        self.assertFalse(gate.receive(self.message(), 1.24))
        self.assertEqual(gate.target(1.26), (0., 0., 0.))

    def test_limits(self):
        gate = CommandGate(); gate.receive(self.message(command=[99, -99, 99]), 1)
        self.assertEqual(gate.target(1), (.35, -.15, .35))

    def test_slam_map_to_odom_rotation_and_translation(self):
        x, y, heading = transform_planar(1, 0, .2, 4, -3, math.pi/2)
        self.assertAlmostEqual(x, 4); self.assertAlmostEqual(y, -2)
        self.assertAlmostEqual(heading, math.pi/2+.2)

    def test_walking_deadzone_and_rotate_first(self):
        vx, vy, wz = approach_command(.25,0,0,1.)
        self.assertGreaterEqual(vx,.22)
        self.assertEqual(vy,0.); self.assertEqual(wz,0.)
        vx, vy, wz = approach_command(0,1.,0,1.)
        self.assertEqual(vx,0.);self.assertGreaterEqual(wz,.30)
        self.assertEqual(approach_command(1,0,0,.1),(0.,0.,0.))

    def test_braking_stops_for_front_obstacle(self):
        self.assertEqual(brake_forward(.35,[(.4,0.)]),0.)

    def test_scan_density_and_parallel_wall_do_not_create_repulsion(self):
        points=[(.7,math.pi/2),(.8,0.)]
        self.assertAlmostEqual(brake_forward(.35,points),.24)
        self.assertAlmostEqual(brake_forward(.35,points*100),.24)

    def test_running_envelope_and_separate_gate(self):
        gate=CommandGate(RUNNING_LIMITS);gate.receive(self.message(command=[2.4,0,.6]),1.)
        self.assertEqual(gate.target(1.),(2.4,0.,.6))
        self.assertEqual(gate.target(1.3),(0.,0.,0.))
        self.assertGreater(running_approach_command(8.,0.,0.,8.)[0],2.)

    def test_running_slowdown_turn_and_wall_stopping_distance(self):
        self.assertEqual(running_approach_command(0.,5.,0.,5.)[0],0.)
        self.assertEqual(running_approach_command(.2,0.,0.,.2),(0.,0.,0.))
        self.assertLess(running_approach_command(1.,0.,0.,1.)[0],1.)
        self.assertAlmostEqual(brake_running(2.4,[(1.8,0.)]),1.)
        self.assertEqual(brake_running(2.4,[(.7,0.)]),0.)

    def test_astar_detours_and_does_not_cut_blocked_corners(self):
        blocked=[False]*49
        for y in range(6):blocked[y*7+3]=True
        path=astar_grid(blocked,7,7,(1,2,0.),(5,2))
        self.assertIsNotNone(path)
        self.assertIn((3,6),path)
        for (x,y),(nx,ny) in zip(path,path[1:]):
            self.assertFalse(blocked[ny*7+nx])
            if nx!=x and ny!=y:
                self.assertFalse(blocked[y*7+nx]);self.assertFalse(blocked[ny*7+x])
        self.assertIsNone(astar_grid([False,True,True,False],2,2,(0,0,0.),(1,1)))


if __name__ == '__main__': unittest.main()
