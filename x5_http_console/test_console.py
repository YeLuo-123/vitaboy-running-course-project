import concurrent.futures
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from core import valid_name, cell_to_world, world_to_cell, frontier_goal
from areas import save_area, set_target, load_areas, validate_polygon, validate_target


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.info={'width':100,'height':100,'resolution':.05,'origin':[-2.,-3.,.7]}
    def test_rotated_origin_roundtrip(self):
        for col,row in [(0,0),(9,67),(99,99)]:
            self.assertEqual((col,row),world_to_cell(*cell_to_world(col,row,self.info),self.info))
    def test_path_traversal_rejected(self):
        for s in ['../map','/tmp/map','a;rm','',None,'a'*65]:
            with self.assertRaises(ValueError):valid_name(s)
        self.assertEqual(valid_name('room_1-2'),'room_1-2')
    def test_frontier_reachable_free(self):
        a=np.full((100,100),-1,dtype=np.int8);a[10:90,10:65]=0;a[10:90,10:12]=100
        pose=cell_to_world(30,50,self.info)+(0.,)
        goal=frontier_goal(a,self.info,pose)
        self.assertIsNotNone(goal); c,r=world_to_cell(*goal[:2],self.info)
        self.assertEqual(a[r,c],0);self.assertGreaterEqual(c,54)
    def test_fully_known_map(self):
        self.assertIsNone(frontier_goal(np.zeros((100,100),dtype=np.int8),self.info,cell_to_world(30,50,self.info)+(0.,)))
    def test_unreachable_unknown_not_selected(self):
        a=np.full((100,100),100,dtype=np.int8);a[10:35,10:35]=0;a[60:90,60:80]=0;a[60:90,80:95]=-1
        self.assertIsNone(frontier_goal(a,self.info,cell_to_world(20,20,self.info)+(0.,)))

    def test_area_storage_geometry_and_free_target(self):
        info={'width':100,'height':100,'resolution':.05,'origin':[0.,0.,0.]}
        grid=np.full(10000,31,dtype=np.int8)
        polygon=[[1.,1.],[3.,1.],[3.,3.],[1.,3.]]
        with tempfile.TemporaryDirectory() as directory:
            first=save_area(directory,{'area_name':'客厅','polygon':polygon},info,grid)
            self.assertEqual(load_areas(directory)[0]['name'],'客厅')
            target=set_target(directory,first['id'],{'x':2.,'y':2.,'yaw':.5},info,grid)
            self.assertEqual(target['target']['yaw'],.5)
            with self.assertRaisesRegex(ValueError,'所属区域'):
                validate_target({'x':4.,'y':2.},polygon,info,grid)
            grid[40*100+41]=100
            with self.assertRaisesRegex(ValueError,'过近'):
                validate_target({'x':2.,'y':2.},polygon,info,grid)
            with self.assertRaisesRegex(ValueError,'交叉'):
                validate_polygon([[1.,1.],[3.,3.],[1.,3.],[2.5,1.]],info)
            with self.assertRaisesRegex(ValueError,'已存在'):
                save_area(directory,{'area_name':'客厅','polygon':polygon},info,grid)


class RosTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rclpy
        rclpy.init()
    @classmethod
    def tearDownClass(cls):
        import rclpy
        rclpy.shutdown()
    def setUp(self):
        import server
        self.server=server
        self.node=server.Console(dry_run=True)
        self.node.vel=Mock();self.node.action=Mock();self.node.action.server_is_ready.return_value=True
    def tearDown(self):self.node.destroy_node()
    def configure(self):
        n=self.node;n.mode='mapping';n.process=Mock();n.process.poll.return_value=None
        n.scan_time=n.odom_time=n.map_time=n.lease=time.monotonic();n.pose=[1.,1.,0.]
        n.info={'width':100,'height':100,'resolution':.05,'origin':[0.,0.,0.]}
        n.grid=np.zeros(10000,dtype=np.int8)
        n.scan_match=.9;n.scan_match_time=time.monotonic()
        return n
    def test_idle_cannot_move(self):
        with self.assertRaises(ValueError):self.node.send_goal(1.,1.,0.)
        self.assertFalse(self.node.armed)
        with self.assertRaises(ValueError):self.node.manual_start('tab')
    def test_unknown_and_outside_goal_rejected(self):
        n=self.configure();n.grid[:]=-1
        for x,y in [(1.,1.),(-1.,0.),(100.,100.),(math.nan,1.)]:
            with self.assertRaises(ValueError):n.send_goal(x,y,0.)
        self.assertFalse(n.armed)
    def test_navigation_requires_initialpose(self):
        n=self.configure();n.mode='navigation'
        self.assertFalse(n.ready());n.initialized=True;self.assertTrue(n.ready())
    def test_navigation_start_waits_for_clock_sync(self):
        with patch.object(self.server.time,'time',return_value=946684800):
            with self.assertRaisesRegex(ValueError,'时钟尚未同步'):
                self.node.start_stack('navigation','rooms')
    def test_clock_jump_stops_navigation(self):
        n=self.configure();n.mode='navigation'
        n.last_wall_clock=time.time()-60
        n.last_monotonic_clock=time.monotonic()
        with patch.object(n,'stop_stack') as stop_stack:
            n.tick()
        stop_stack.assert_called_once_with()
        self.assertIn('时钟刚发生跳变',n.message)
    def test_navigation_rejects_uncertain_localization(self):
        n=self.configure();n.mode='navigation';n.initialized=True
        with self.assertRaisesRegex(ValueError,'定位不可靠'):
            n.send_goal(1.,1.,0.)
        n.amcl_covariance=[.04,.04,.03];n.amcl_time=time.monotonic()
        self.assertFalse(n.localization_uncertainty()['uncertain'])
        n.send_goal(1.,1.,0.)
        self.assertTrue(n.armed)
        n.amcl_covariance=[.36,.04,.03]
        self.assertTrue(n.localization_uncertainty()['uncertain'])
        n.amcl_covariance=[.04,.04,.03];n.amcl_time-=30;n.goal_time-=10
        self.assertTrue(n.localization_uncertainty()['uncertain'])
    def test_navigation_requires_recent_scan_match(self):
        n=self.configure();n.mode='navigation';n.initialized=True
        n.amcl_covariance=[.04,.04,.03];n.amcl_time=time.monotonic()
        n.scan_match=None
        self.assertTrue(n.localization_uncertainty()['uncertain'])
        with self.assertRaises(ValueError):n.send_goal(1.,1.,0.)
        n.scan_match=.9;n.scan_match_time=time.monotonic()-3
        self.assertTrue(n.localization_uncertainty()['uncertain'])
        with self.assertRaises(ValueError):n.send_goal(1.,1.,0.)
        n.scan_match_time=time.monotonic()
        self.assertFalse(n.localization_uncertainty()['uncertain'])
    def test_live_scan_mismatch_blocks_navigation(self):
        from sensor_msgs.msg import LaserScan
        n=self.configure();n.mode='navigation';n.initialized=True;n.pose=[.5,.5,0.]
        n.amcl_covariance=[.04,.04,.03];n.amcl_time=time.monotonic()
        n.obstacle_distance=np.ones((100,100));n.obstacle_distance[:,20]=0.
        n.scan_match=None
        transform=Mock()
        transform.translation.x=transform.translation.y=0.
        transform.rotation.x=transform.rotation.y=transform.rotation.z=0.
        transform.rotation.w=1.
        n.tf.lookup_transform=Mock(return_value=Mock(transform=transform))
        scan=LaserScan();scan.header.frame_id='laser';scan.range_min=.2;scan.range_max=6.
        scan.angle_min=scan.angle_increment=0.;scan.ranges=[.5]*90
        n.on_scan(scan)
        self.assertAlmostEqual(n.scan_match,1.)
        self.assertFalse(n.localization_uncertainty()['uncertain'])
        scan.ranges=[.2]*90
        for _ in range(4):n.on_scan(scan)
        self.assertLess(n.scan_match,.70)
        self.assertTrue(n.localization_uncertainty()['uncertain'])
    def test_stationary_amcl_estimate_and_nomotion_sampling(self):
        n=self.configure();n.mode='navigation';n.initialized=True
        n.nomotion_client=Mock();n.nomotion_client.service_is_ready.return_value=True
        done=concurrent.futures.Future();done.set_result(Mock())
        n.nomotion_client.call_async.return_value=done
        n.start_sampling();n.sampling_tick()
        self.assertEqual(n.sampling_count,1)
        msg=Mock();msg.header.frame_id='map';msg.pose.covariance=[0.]*36
        msg.pose.covariance[0]=msg.pose.covariance[7]=.04;msg.pose.covariance[35]=.03
        msg.pose.pose.position.x=1.;msg.pose.pose.position.y=1.
        msg.pose.pose.orientation.x=msg.pose.pose.orientation.y=msg.pose.pose.orientation.z=0.
        msg.pose.pose.orientation.w=1.
        n.on_amcl_pose(msg)
        self.assertEqual(n.dispatch('localization',{})['history'][0][1:3],[1.,1.])
        cloud=Mock();cloud.header.frame_id='map'
        cloud.particles=[Mock(pose=Mock(position=Mock(x=float(i),y=1.))) for i in range(500)]
        n.on_particle_cloud(cloud)
        self.assertLessEqual(len(n.dispatch('localization',{})['particles']),240)
        self.assertTrue(n.localization_uncertainty()['uncertain'])  # Sampling is still running.
        n.sampling_until=0.;n.amcl_time-=30
        self.assertFalse(n.localization_uncertainty()['uncertain'])  # No new AMCL sample while stationary.
    def test_stalled_localization_is_activated(self):
        from lifecycle_msgs.msg import State, Transition
        n=self.configure();n.mode='navigation';n.stack_started=time.monotonic()-20
        lifecycle={}
        for name in ('map_server','amcl'):
            get_state=Mock();change_state=Mock()
            get_state.service_is_ready.return_value=True
            change_state.service_is_ready.return_value=True
            state=concurrent.futures.Future()
            state.set_result(Mock(current_state=Mock(id=State.PRIMARY_STATE_INACTIVE)))
            changed=concurrent.futures.Future();changed.set_result(Mock(success=True))
            get_state.call_async.return_value=state
            change_state.call_async.return_value=changed
            lifecycle[name]=(get_state,change_state)
        n.lifecycle=lifecycle
        n.recover_localization()
        self.assertFalse(n.recovery_busy)
        for _,change_state in lifecycle.values():
            self.assertEqual(change_state.call_async.call_args.args[0].transition.id,
                             Transition.TRANSITION_ACTIVATE)
    def test_initialpose_requires_map_and_localization_subscriber(self):
        n=self.configure();n.mode='navigation';n.initial_pub=Mock()
        n.initial_pub.get_subscription_count.return_value=0
        with self.assertRaisesRegex(ValueError,'定位节点'):
            n.dispatch('initialpose',{'x':1.,'y':1.,'yaw':0.})
        n.initial_pub.get_subscription_count.return_value=1
        n.grid[:]=-1
        with self.assertRaisesRegex(ValueError,'空闲区域'):
            n.dispatch('initialpose',{'x':1.,'y':1.,'yaw':0.})
        n.grid[:]=0
        n.dispatch('initialpose',{'x':1.,'y':1.,'yaw':0.})
        n.initial_pub.publish.assert_called_once()
        self.assertTrue(n.initialized)
        msg=n.initial_pub.publish.call_args.args[0]
        self.assertAlmostEqual(msg.pose.covariance[0],(.75)**2)
        self.assertAlmostEqual(msg.pose.covariance[35],math.radians(45)**2)
        self.assertEqual(n.initial_range,{'radius':1.5,'heading_range_deg':90.})
        with self.assertRaisesRegex(ValueError,'位置范围'):
            n.dispatch('initialpose',{'x':1.,'y':1.,'yaw':0.,'radius':4.})
        with self.assertRaisesRegex(ValueError,'朝向范围'):
            n.dispatch('initialpose',{'x':1.,'y':1.,'yaw':0.,'heading_range_deg':190.})
    def test_broad_initial_pose_covariance(self):
        n=self.configure();n.mode='navigation';n.initial_pub=Mock()
        n.dispatch('initialpose',{'x':1.,'y':1.,'yaw':0.,'radius':2.,'heading_range_deg':180.})
        msg=n.initial_pub.publish.call_args.args[0]
        self.assertAlmostEqual(msg.pose.covariance[0],1.)
        self.assertAlmostEqual(msg.pose.covariance[7],1.)
        self.assertAlmostEqual(msg.pose.covariance[35],(math.pi/2)**2)
        self.assertEqual(n.dispatch('localization',{})['initial_range']['radius'],2.)
    def test_sensor_stale(self):
        n=self.configure();self.assertTrue(n.ready());n.scan_time-=5;self.assertFalse(n.ready())
    def test_stop_cancels_pending_acceptance(self):
        n=self.configure();n.send_goal(1.,1.,0.);g=n.generation
        n.stop();h=Mock();h.accepted=True;f=concurrent.futures.Future();f.set_result(h)
        n.accepted(f,g);h.cancel_goal_async.assert_called_once();self.assertFalse(n.armed)
    def test_stop_cancels_active_goal(self):
        n=self.configure();handle=Mock();n.goal_handle=handle;n.armed=True;n.exploring=True
        n.stop();handle.cancel_goal_async.assert_called_once();self.assertFalse(n.armed);self.assertFalse(n.exploring)
        self.assertIsNone(n.owner)
    def test_watchdog_drops_velocity(self):
        n=self.configure();n.armed=True;n.lease-=4
        n.tf.lookup_transform=Mock(side_effect=RuntimeError())
        n.tick();self.assertFalse(n.armed)
        self.assertTrue(all(call.args[0].linear.x==0 for call in n.vel.publish.call_args_list))
    def test_manual_drive_owner_speed_and_release(self):
        n=self.configure();n.manual_start('tab')
        with self.assertRaises(ValueError):
            n.manual_drive({'_client':'other','forward':1,'turn':0})
        with self.assertRaises(ValueError):
            n.manual_drive({'_client':'tab','forward':2,'turn':0})
        n.manual_drive({'_client':'tab','forward':1,'turn':-1})
        with self.assertRaisesRegex(ValueError,'先停止小车'):
            n.save_map('while_moving')
        n.manual_ready=Mock(return_value=True)
        n.tf.lookup_transform=Mock(side_effect=RuntimeError())
        n.tick();command=n.vel.publish.call_args.args[0]
        self.assertAlmostEqual(command.linear.x,.12)
        self.assertAlmostEqual(command.angular.z,-.35)
        n.manual_time-=1
        n.tick();self.assertEqual(n.vel.publish.call_args.args[0].linear.x,0)
        n.dispatch('manual/stop',{'_client':'tab'})
        self.assertFalse(n.manual);self.assertFalse(n.armed)
    def test_manual_drive_stops_on_sensor_loss(self):
        n=self.configure();n.manual_start('tab')
        n.manual_drive({'_client':'tab','forward':1,'turn':0})
        n.scan_time-=5
        n.tf.lookup_transform=Mock(side_effect=RuntimeError())
        n.tick();self.assertFalse(n.manual);self.assertFalse(n.armed)
        self.assertEqual(n.vel.publish.call_args.args[0].linear.x,0)
    def test_map_orientation_and_no_overwrite(self):
        n=self.configure();n.info={'width':2,'height':2,'resolution':.05,'origin':[1.,2.,.5]};n.grid=np.array([0,100,-1,31],dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory,patch.object(self.server,'MAPS',Path(directory)):
            n.save_map('test');data=(Path(directory)/'test/map.pgm').read_bytes()
            self.assertTrue(data.endswith(bytes([255,31,0,100])))
            self.assertIn('mode: raw',(Path(directory)/'test/map.yaml').read_text())
            with self.assertRaises(ValueError):n.save_map('test')
    def test_goal_result_propagates(self):
        n=self.configure();n.armed=True;f=concurrent.futures.Future();f.set_result(Mock(status=4))
        n.finished(f,n.generation);self.assertEqual(n.goal_status,'succeeded');self.assertFalse(n.armed)
    def test_route_speaks_waits_then_advances(self):
        n=self.configure();n.mode='navigation';n.active_map='rooms';n.initialized=True
        n.amcl_covariance=[.04,.04,.03];n.amcl_time=time.monotonic()
        n.speaker=Mock();n.speaker.available.return_value=True
        n.speaker.status.return_value={'busy':False,'error':None}
        polygon1=[[.5,.5],[2.,.5],[2.,2.],[.5,2.]]
        polygon2=[[2.5,.5],[4.,.5],[4.,2.],[2.5,2.]]
        with tempfile.TemporaryDirectory() as directory,patch.object(self.server,'MAPS',Path(directory)):
            mapdir=Path(directory)/'rooms';mapdir.mkdir()
            first=save_area(mapdir,{'area_name':'客厅','polygon':polygon1},n.info,n.grid)
            second=save_area(mapdir,{'area_name':'卧室','polygon':polygon2},n.info,n.grid)
            set_target(mapdir,first['id'],{'x':1.,'y':1.},n.info,n.grid)
            set_target(mapdir,second['id'],{'x':3.,'y':1.},n.info,n.grid)
            n.start_route([first['id'],second['id']],'tab')
            n.speaker.say.assert_called_with('自主导航开始，目前前往客厅')
            self.assertFalse(n.armed)
            n.route_tick();self.assertEqual(n.route_phase,'running');self.assertTrue(n.armed)
            n.pending=False
            completed=concurrent.futures.Future();completed.set_result(Mock(status=4))
            n.finished(completed,n.generation)
            self.assertEqual(n.route_phase,'dwell');self.assertFalse(n.armed)
            n.speaker.say.assert_called_with('已到达客厅，现在前往卧室')
            n.route_tick();self.assertGreater(n.route_due,time.monotonic()+2)
            self.assertEqual(n.route_index,0)
            n.route_due=time.monotonic()-1;n.route_tick()
            self.assertEqual(n.route_index,1);self.assertEqual(n.route_phase,'running')
            n.pending=False;n.finished(completed,n.generation)
            self.assertEqual(n.route_phase,'complete');self.assertFalse(n.armed)
            n.speaker.say.assert_called_with('已到达卧室，任务完成')
    def test_route_failed_goal_never_advances(self):
        n=self.configure();n.route_steps=[{'name':'客厅','target':{'x':1.,'y':1.,'yaw':0.}}]
        n.route_index=0;n.route_phase='running';n.armed=True;n.speaker=Mock()
        failed=concurrent.futures.Future();failed.set_result(Mock(status=6))
        n.finished(failed,n.generation)
        self.assertEqual(n.route_phase,'failed');self.assertFalse(n.armed)

if __name__=='__main__':unittest.main(verbosity=2)
