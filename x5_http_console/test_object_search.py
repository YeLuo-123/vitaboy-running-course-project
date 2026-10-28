import concurrent.futures
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import object_search
from object_search import RoomSearch, room_viewpoints
from object_prompts import resolve_prompts
from core import world_to_cell


class FakeVision:
    def __init__(self, detections=()):
        self.health = {'ready': True, 'camera_ready': True}
        self.results = list(detections); self.sequence = 0; self.job_id = ''
        self.exception = None; self.reuse_sequence = False

    def submit(self, path, data):
        future = concurrent.futures.Future()
        if path == '/configure':
            self.job_id = data['job_id']; future.set_result({'job_id': self.job_id})
        elif self.exception:
            future.set_exception(self.exception)
        else:
            if not self.reuse_sequence: self.sequence += 1
            future.set_result({'job_id': self.job_id, 'sequence': self.sequence,
                               'capture_age_sec': .01, 'inference_sec': .4,
                               'detections': self.results.pop(0) if self.results else []})
        return future


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.info = {'width': 80, 'height': 80, 'resolution': .05, 'origin': [0., 0., 0.]}
        self.area = {'id': 'living', 'name': '客厅', 'polygon': [[.5,.5],[3.5,.5],[3.5,3.5],[.5,3.5]],
                     'target': {'x': 2., 'y': 2., 'yaw': 0.}}
        directory = Path(self.directory.name)/'rooms'; directory.mkdir()
        (directory/'areas.json').write_text(json.dumps({'version': 1, 'areas': [self.area]}))
        c = self.c = Mock()
        c.search_maps = Path(self.directory.name); c.active_map = 'rooms'
        c.info = self.info; c.grid = np.zeros(6400, np.int8); c.mode = 'navigation'
        c.manual = c.exploring = c.pending = c.armed = False; c.goal_handle = None
        c.route_phase = 'idle'; c.ready.return_value = True
        c.localization_uncertainty.return_value = {'uncertain': False}
        c.odom_linear = c.odom_angular = 0.; c.pose = [2.,2.,0.]
        c.speaker.available.return_value = True; c.goal_status = 'idle'
        self.now = 100.
        self.clock = patch.object(object_search.time, 'monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.vision = FakeVision(); self.search = RoomSearch(c, self.vision)
        def stop():
            c.armed = False; c.goal_status = 'stopped'; self.search.cancel()
        c.stop.side_effect = stop
        c.send_goal.side_effect = lambda *args: setattr(c, 'goal_status', 'executing')

    def tearDown(self):
        self.clock.stop(); self.directory.cleanup()

    def tick(self, count=1):
        for _ in range(count):
            self.c.lease = self.now; self.search.tick(); self.now += .5

    def begin(self, check_only=True):
        self.search.start({'item':'纸杯','area_id':'living'}, 'page1', check_only)

    def test_chinese_color_and_arbitrary_english_prompt(self):
        self.assertEqual(resolve_prompts('纸杯')['texts'][0], 'paper cup')
        self.assertEqual(resolve_prompts('红色纸杯')['texts'][0], 'red paper cup')
        self.assertEqual(resolve_prompts('我的玩具', 'small toy robot')['texts'][0], 'small toy robot')
        with self.assertRaisesRegex(ValueError, '英文识别提示'):
            resolve_prompts('未知奇怪物品')

    def test_viewpoints_keep_clearance_and_connected_room(self):
        grid = self.c.grid.reshape(80,80); grid[:,50] = 100
        points, steps, inside = room_viewpoints(self.area, self.info, grid)
        self.assertLessEqual(len(points), 3); self.assertEqual(len(steps), 6*len(points))
        for x, y in points:
            self.assertLess(x, 2.5-.20)
            col, row = world_to_cell(x, y, self.info)
            self.assertTrue(inside[row,col]); self.assertEqual(grid[row,col], 0)

    def test_two_consistent_fresh_frames_confirm_and_speak(self):
        hit = [{'score':.7,'box':[10.,10.,40.,50.]}]
        self.vision.results = [hit, hit]; self.begin(); self.tick(12)
        self.assertEqual(self.search.phase, 'found')
        self.c.speaker.say.assert_called_once_with('已找到纸杯')
        self.c.send_goal.assert_not_called()

    def test_single_false_positive_is_not_confirmation(self):
        self.vision.results = [[{'score':.7,'box':[10.,10.,40.,50.]}], [], []]
        self.begin(); self.tick(16)
        self.assertEqual(self.search.phase, 'not_found')
        self.c.speaker.say.assert_called_once_with('未找到')

    def test_camera_failure_is_error_not_not_found(self):
        self.vision.exception = RuntimeError('camera unplugged')
        self.begin(); self.tick(12)
        self.assertEqual(self.search.phase, 'failed')
        self.assertIn('camera unplugged', self.search.note)
        self.c.speaker.say.assert_not_called()

    def test_repeated_frame_does_not_confirm(self):
        self.vision.reuse_sequence = True
        self.vision.results = [[{'score':.7,'box':[10.,10.,40.,50.]}]]*3
        self.begin(); self.tick(16)
        self.assertEqual(self.search.phase, 'failed')
        self.c.speaker.say.assert_not_called()

    def test_page_disconnect_cancels_navigation(self):
        self.begin(False); self.tick()
        self.assertEqual(self.search.phase, 'navigating')
        self.now += 4.; self.search.tick()
        self.assertEqual(self.search.phase, 'failed'); self.c.stop.assert_called_once()

    def test_plan_leaving_room_is_stopped(self):
        self.begin(False); self.tick()
        self.c.pose = self.search.steps[0]; self.c.goal_status = 'succeeded'
        self.tick(); self.search.phase = 'navigating'
        self.search.accept_plan([[2.,2.],[3.9,3.9]])
        self.assertEqual(self.search.phase, 'failed')
        self.assertIn('离开目标房间', self.search.note)

    def test_cancellation_ignores_late_detection(self):
        self.begin(); self.tick(3)
        pending = concurrent.futures.Future(); pending.set_running_or_notify_cancel()
        self.search.future = pending; self.search.cancel()
        pending.set_result({'job_id':self.search.job_id,'detections':[{'score':.9}]})
        self.tick(15)
        self.assertEqual(self.search.phase, 'stopped')
        self.c.speaker.say.assert_not_called()

    def test_room_navigation_failure_never_reports_not_found(self):
        self.begin(False); self.tick(); self.c.goal_status = 'failed'; self.tick()
        self.assertEqual(self.search.phase, 'failed'); self.c.speaker.say.assert_not_called()

    def test_no_camera_never_starts_robot(self):
        self.vision.health['camera_ready'] = False
        with self.assertRaisesRegex(ValueError, '相机尚未就绪'): self.begin(False)
        self.c.send_goal.assert_not_called()

    def test_camera_disconnect_during_navigation_stops_task(self):
        self.begin(False); self.tick()
        self.vision.health['camera_ready'] = False; self.tick()
        self.assertEqual(self.search.phase, 'failed')
        self.c.stop.assert_called_once(); self.c.speaker.say.assert_not_called()

    def test_complete_room_search_observes_all_planned_directions(self):
        self.begin(False)
        for _ in range(200):
            if self.search.phase == 'navigating':
                self.c.pose = self.search.steps[self.search.index]
                self.c.goal_status = 'succeeded'
            self.tick()
            if not self.search.active(): break
        self.assertEqual(self.search.phase, 'not_found')
        self.assertEqual(self.search.observed, len(self.search.steps))
        self.assertEqual(self.c.send_goal.call_count, len(self.search.steps))
        self.c.speaker.say.assert_called_once_with('未找到')

    def test_timeout_without_observation_is_failure(self):
        self.begin(False); self.tick()
        self.search.search_started = self.now-self.search.budget-1
        self.tick()
        self.assertEqual(self.search.phase, 'failed')
        self.c.speaker.say.assert_not_called()


if __name__ == '__main__': unittest.main()
