import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from exploration import Exploration
from core import frontier_plan


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.now=100.
        self.clock=patch('exploration.time.monotonic',side_effect=lambda:self.now)
        self.clock.start();self.addCleanup(self.clock.stop)
        self.c=SimpleNamespace(mode='mapping',ready=Mock(return_value=True),search=Mock(),
            manual=False,goal_handle=None,pending=False,armed=False,exploring=False,
            speaker=Mock(),lease=100.,map_version=1,map_time=100.,odom_linear=0.,odom_angular=0.,
            grid=np.zeros(10000,dtype=np.int8),pose=[2.,2.,0.],goal_time=100.,
            info={'width':100,'height':100,'resolution':.05,'origin':[0.,0.,0.]},
            send_goal=Mock(),save_map=Mock(return_value={'name':'auto_test'}))
        self.c.search.active.return_value=False
        self.c.speaker.available.return_value=True;self.c.speaker.busy.return_value=False
        self.c.speaker.status.return_value={'busy':False,'error':None}
        self.e=Exploration(self.c,Path(self.directory.name))
        def stop():
            self.e.cancel();self.c.armed=False;self.c.exploring=False;self.c.speaker.cancel()
        self.c.stop=Mock(side_effect=stop)

    def start(self,**data):self.e.start({'name':'auto_test','_client':'page',**data})
    def advance(self,seconds=2,new_map=True):
        self.now+=seconds;self.c.lease=self.c.map_time=self.now
        if new_map:self.c.map_version+=1
        self.e.tick()

    def test_speech_must_finish_before_first_goal(self):
        self.start();self.c.speaker.say.assert_called_once_with('自主探索建图开始')
        self.assertFalse(self.c.armed)
        with patch('exploration.frontier_plan',return_value={'goal':(3.,2.,0.),'frontiers':1,'reason':'goal'}):
            self.c.speaker.status.return_value={'busy':True,'error':None}
            self.advance();self.c.send_goal.assert_not_called()
            self.c.speaker.status.return_value={'busy':False,'error':None}
            self.advance();self.c.send_goal.assert_called_once_with(3.,2.,0.)

    def test_playback_failure_never_drives(self):
        self.start();self.c.speaker.status.return_value={'busy':False,'error':'USB unavailable'}
        self.advance();self.assertEqual(self.e.phase,'failed');self.assertFalse(self.c.armed)
        self.c.send_goal.assert_not_called()

    def test_completion_requires_new_stable_maps_and_standstill(self):
        self.start();self.e.tick()
        self.advance(20,new_map=False);self.assertEqual(self.e.phase,'checking')
        self.c.save_map.assert_not_called()
        for _ in range(3):self.advance()
        self.assertEqual(self.e.phase,'finishing');self.assertFalse(self.c.armed)
        self.c.odom_linear=.1;self.advance();self.c.save_map.assert_not_called()
        self.c.odom_linear=0.;self.advance()
        self.assertEqual(self.e.phase,'complete');self.assertEqual(self.e.saved_map,'auto_test')
        self.assertFalse(self.c.exploring);self.c.save_map.assert_called_once_with('auto_test')

    def test_map_growth_restarts_completion_window(self):
        self.start();self.e.tick();self.advance(10)
        self.c.grid[:20]=-1
        with patch('exploration.frontier_plan',return_value={'goal':None,'frontiers':0,'reason':'no_frontiers'}):
            self.advance();self.assertEqual(self.e.phase,'checking')
            self.assertEqual(self.e.empty_since,self.now)

    def test_blockage_is_not_completion(self):
        self.start()
        with patch('exploration.frontier_plan',return_value={'goal':None,'frontiers':1,'reason':'blocked'}):self.e.tick()
        self.assertEqual(self.e.phase,'blocked');self.c.save_map.assert_not_called()

    def test_failed_goal_tries_another_frontier_but_never_false_completes(self):
        self.start();self.e.phase='running';self.e.finished(6,[3.,2.,0.])
        with patch('exploration.frontier_plan',return_value={'goal':(4.,3.,0.),'frontiers':1,'reason':'goal'}):self.advance(4)
        self.c.send_goal.assert_called_once_with(4.,3.,0.)
        with patch('exploration.frontier_plan',return_value={'goal':None,'frontiers':0,'reason':'no_frontiers'}):self.advance()
        self.assertEqual(self.e.phase,'blocked');self.c.save_map.assert_not_called()

    def test_cancel_during_speech_prevents_late_movement(self):
        self.start();self.c.stop();self.advance()
        self.c.send_goal.assert_not_called();self.assertEqual(self.e.phase,'stopped')

    def test_new_frontier_during_parking_prevents_premature_save(self):
        self.start();self.e.phase='finishing';self.e.finish_due=self.now
        with patch('exploration.frontier_plan',return_value={'goal':(3.,2.,0.),'frontiers':1,'reason':'goal'}):self.advance()
        self.assertEqual(self.e.phase,'running');self.c.save_map.assert_not_called()

    def test_geometry_change_restarts_completion_window(self):
        self.start();self.e.tick();self.advance(10)
        self.c.info['origin'][0]=.05
        self.advance();self.assertEqual(self.e.phase,'checking')
        self.assertEqual(self.e.empty_since,self.now)

    def test_save_failure_keeps_map_for_manual_save(self):
        self.start();self.e.phase='finishing';self.e.finish_due=self.now
        self.c.save_map.side_effect=OSError('disk full');self.advance()
        self.assertEqual(self.e.phase,'save_failed');self.assertFalse(self.c.armed)

    def test_user_can_disable_auto_save(self):
        self.start(auto_save=False);self.e.phase='finishing';self.e.finish_due=self.now;self.advance()
        self.assertEqual(self.e.phase,'complete');self.c.save_map.assert_not_called()

    def test_heartbeat_loss_stops_instead_of_finishing(self):
        self.start();self.now+=4;self.e.tick()
        self.assertEqual(self.e.phase,'failed');self.c.send_goal.assert_not_called()

    def test_duplicate_and_busy_start_rejected(self):
        (Path(self.directory.name)/'auto_test').mkdir()
        with self.assertRaisesRegex(ValueError,'已存在'):self.start()
        self.c.speaker.say.assert_not_called()


class FrontierTests(unittest.TestCase):
    def setUp(self):
        self.info={'width':100,'height':100,'resolution':.05,'origin':[0.,0.,0.]}

    def test_pose_near_wall_does_not_mean_completed(self):
        grid=np.full((100,100),100,dtype=np.int8)
        grid[20:80,20:80]=0;grid[20:80,80:]=-1
        result=frontier_plan(grid,self.info,[1.075,2.,0.])
        self.assertEqual(result['reason'],'goal');self.assertGreater(result['goal'][0],3.)

    def test_all_candidates_excluded_is_blocked(self):
        grid=np.full((100,100),100,dtype=np.int8)
        grid[20:80,20:80]=0;grid[20:80,80:]=-1
        excluded=[(x/10,y/10) for x in range(10,45,3) for y in range(10,45,3)]
        result=frontier_plan(grid,self.info,[2.,2.,0.],excluded)
        self.assertEqual(result['reason'],'blocked');self.assertGreater(result['frontiers'],0)

    def test_wall_separates_unknown_from_reachable_frontier(self):
        grid=np.full((100,100),100,dtype=np.int8);grid[20:80,20:80]=0;grid[:,90:]=-1
        self.assertEqual(frontier_plan(grid,self.info,[2.,2.,0.])['reason'],'no_frontiers')


if __name__=='__main__':unittest.main()
