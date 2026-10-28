"""Bounded, room-constrained visual search. Called only by the ROS executor."""
import concurrent.futures
import json
import math
import secrets
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from scipy import ndimage as ndi
from areas import point_in_polygon
from core import FREE_LIMIT, cell_to_world, world_to_cell
from object_prompts import resolve_prompts

ACTIVE_PHASES = ('preparing', 'navigating', 'settling', 'scanning')


def room_viewpoints(area, info, grid, max_positions=3):
    """Known free cells with room/footprint clearance; one connected component."""
    if not area.get('target'):
        raise ValueError('请先为该房间设置区域目标点')
    cells = np.asarray(grid, dtype=np.int8).reshape(info['height'], info['width'])
    inside = np.zeros(cells.shape, bool)
    for row in range(info['height']):
        for col in range(info['width']):
            inside[row, col] = point_in_polygon(cell_to_world(col, row, info), area['polygon'])
    free = (cells >= 0) & (cells <= FREE_LIMIT)
    clearance = ndi.distance_transform_edt(free)*info['resolution']
    boundary = ndi.distance_transform_edt(inside)*info['resolution']
    # The padded chassis diagonal needs ~0.21 m for a stationary turn.
    safe = free & inside & (clearance >= .23) & (boundary >= .18)
    rows, cols = np.where(safe)
    if not len(rows):
        raise ValueError('该房间没有可安全转向的观察位置，请检查区域轮廓和地图')
    target = area['target']; tx, ty = target['x'], target['y']
    coordinates = np.asarray([cell_to_world(c, r, info) for r, c in zip(rows, cols)])
    first = int(np.argmin(np.linalg.norm(coordinates-[tx, ty], axis=1)))
    if np.linalg.norm(coordinates[first]-[tx, ty]) > .65:
        raise ValueError('房间目标点附近没有安全观察位置，请重新设置区域目标点')
    seed = np.zeros(cells.shape, bool); seed[rows[first], cols[first]] = True
    component = ndi.binary_propagation(seed, mask=safe)
    rows, cols = np.where(component)
    coordinates = np.asarray([cell_to_world(c, r, info) for r, c in zip(rows, cols)])
    first = int(np.argmin(np.linalg.norm(coordinates-[tx, ty], axis=1)))
    selected = [first]
    while len(selected) < max_positions:
        distance = np.min(np.linalg.norm(coordinates[:, None, :]-coordinates[selected][None, :, :], axis=2), axis=1)
        candidate = int(np.argmax(distance*np.minimum(clearance[rows, cols], .5)))
        if distance[candidate] < .60:
            break
        selected.append(candidate)
    centre = np.mean(coordinates, axis=0)
    points, steps = [], []
    for index in selected:
        x, y = coordinates[index]
        heading = math.atan2(centre[1]-y, centre[0]-x) if np.linalg.norm(centre-[x, y]) > .2 else target['yaw']
        points.append([float(x), float(y)])
        for direction in range(6):
            angle = heading+direction*math.pi/3
            angle = math.atan2(math.sin(angle), math.cos(angle))
            steps.append([float(x), float(y), angle])
    return points, steps, inside


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
    total = max(0., a[2]-a[0])*max(0., a[3]-a[1])+max(0., b[2]-b[0])*max(0., b[3]-b[1])-intersection
    return intersection/total if total > 0 else 0.


class VisionClient:
    def __init__(self, enabled=True, base='http://127.0.0.1:8091'):
        self.base = base; self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.health = {'ready': False, 'camera_ready': False, 'error': '视觉服务未连接'}
        self.running = enabled
        if enabled:
            threading.Thread(target=self._poll_health, daemon=True).start()

    def _request(self, path, data=None, timeout=30):
        request = Request(self.base+path, data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            try: detail = json.load(exc).get('error', '视觉服务异常')
            except Exception: detail = '视觉服务异常'
            raise RuntimeError(detail) from exc

    def _poll_health(self):
        while self.running:
            try: self.health = self._request('/health', timeout=2)
            except Exception as exc:
                self.health = {'ready': False, 'camera_ready': False, 'error': str(exc)[:180]}
            time.sleep(2)

    def submit(self, path, data):
        return self.pool.submit(self._request, path, data)

    def close(self):
        self.running = False; self.pool.shutdown(wait=False, cancel_futures=True)


class RoomSearch:
    def __init__(self, console, vision):
        self.console = console; self.vision = vision
        self.phase = 'idle'; self.note = ''; self.item = ''; self.room = ''
        self.check_only = False; self.future = None; self.points = []; self.steps = []
        self.index = 0; self.observed = 0; self.started = 0.; self.search_started = 0.
        self.budget = 180.; self.due = 0.; self.goal_started = 0.
        self.last_result = None; self.last_sequence = -1; self.last_box = None; self.hits = 0; self.frames = 0
        self.job_id = ''; self.inside = None

    def active(self): return self.phase in ACTIVE_PHASES

    def status(self):
        return {'phase': self.phase, 'active': self.active(), 'note': self.note, 'item': self.item,
                'room': self.room, 'check_only': self.check_only, 'observed_views': self.observed,
                'total_views': len(self.steps) if not self.check_only else 1,
                'view_index': self.index, 'points': self.points, 'last_result': self.last_result,
                'elapsed_sec': round(time.monotonic()-self.started, 1) if self.active() else None,
                'budget_sec': self.budget}

    def start(self, data, client, check_only=False):
        c = self.console
        if self.active() or c.manual or c.exploring or c.pending or c.goal_handle or c.armed or c.route_phase in ('announcing', 'running', 'dwell'):
            raise ValueError('请先停止当前运动或寻物任务')
        prompt = resolve_prompts(data.get('item'), data.get('english_hint', ''))
        health = self.vision.health
        if not health.get('ready') or not health.get('camera_ready'):
            raise ValueError('视觉模型或相机尚未就绪：'+str(health.get('error') or health.get('camera_error') or '等待服务'))
        if not c.speaker.available(): raise ValueError('小车语音播报不可用')
        budget = data.get('timeout_sec', 180)
        if type(budget) not in (int, float) or not math.isfinite(budget) or not 30 <= budget <= 600:
            raise ValueError('搜索时限须为 30–600 秒')
        room = ''
        if not check_only:
            if c.mode != 'navigation' or not c.ready() or c.localization_uncertainty()['uncertain']:
                raise ValueError('请先加载地图并确认定位和导航就绪')
            from areas import load_areas
            from pathlib import Path
            area = next((a for a in load_areas(Path(c.search_maps)/c.active_map) if a['id'] == data.get('area_id')), None)
            if not area: raise ValueError('请选择当前地图的房间')
            points, steps, inside = room_viewpoints(area, c.info, c.grid)
            room = area['name']
        else:
            points, steps, inside = [], [], None
        self.cancel(); self.item = prompt['name']; self.room = room; self.check_only = check_only
        self.points = points; self.steps = steps; self.inside = inside
        self.index = 0; self.observed = 0; self.started = time.monotonic(); self.search_started = 0.
        self.last_result = None; self.last_sequence = -1; self.budget = float(budget)
        self.job_id = secrets.token_urlsafe(16)
        c.owner = client; c.lease = self.started
        self.future = self.vision.submit('/configure', {'job_id': self.job_id, 'item': self.item,
                                                       'english_hint': data.get('english_hint', '')})
        self.phase = 'preparing'; self.note = '正在准备识别 '+self.item
        c.message = self.note

    def cancel(self):
        if self.future: self.future.cancel()
        self.future = None
        if self.active(): self.phase = 'stopped'; self.note = '寻物任务已取消'

    def accept_plan(self, points):
        if self.phase != 'navigating' or not self.search_started or self.check_only or not points:
            return
        # Modest tolerance for grid/pose noise, not a corridor outside the room.
        allowed = ndi.binary_dilation(self.inside, iterations=max(1, math.ceil(.10/self.console.info['resolution'])))
        for x, y in points:
            col, row = world_to_cell(x, y, self.console.info)
            if not (0 <= row < allowed.shape[0] and 0 <= col < allowed.shape[1] and allowed[row, col]):
                self.fail('当前路径会离开目标房间，已停止搜索'); return

    def fail(self, reason):
        self.console.stop(); self.phase = 'failed'; self.note = reason
        self.console.message = '寻物已停止：'+reason

    def finish(self, found, reason=''):
        c = self.console; c.stop()
        self.phase = 'found' if found else 'not_found'
        speech = '已找到'+self.item if found else '未找到'
        self.note = speech+('；'+reason if reason else '')
        c.message = self.note
        try: c.speaker.say(speech)
        except Exception as exc:
            self.note += '；播报失败：'+str(exc); c.message = self.note

    def _navigate(self):
        self.console.send_goal(*self.steps[self.index])
        self.goal_started = time.monotonic(); self.phase = 'navigating'
        self.note = ('正在前往'+self.room if not self.search_started else
                     f'房间内自主观察 {self.index+1}/{len(self.steps)}')
        self.console.message = self.note

    def _settle(self):
        self.phase = 'settling'; self.due = time.monotonic()+.8
        self.hits = 0; self.frames = 0; self.last_box = None

    def tick(self):
        if not self.active(): return
        c = self.console; now = time.monotonic()
        if now-c.lease > 3: return self.fail('控制页面心跳超时')
        if not self.vision.health.get('ready') or not self.vision.health.get('camera_ready'):
            return self.fail('视觉模型或相机中断')
        if not self.check_only and (not c.ready() or c.localization_uncertainty()['uncertain']):
            return self.fail('导航传感器或定位未就绪')
        if self.search_started and now-self.search_started > self.budget and self.phase != 'preparing':
            if not self.observed:
                return self.fail('搜索超时，尚未完成有效观察')
            return self.finish(False, f'本轮搜索超时，已观察 {self.observed}/{len(self.steps) or 1} 个方向')
        if self.phase == 'preparing':
            if now-self.started > 35: return self.fail('视觉模型准备超时')
            if not self.future or not self.future.done(): return
            try:
                result = self.future.result(); self.future = None
                if result.get('job_id') != self.job_id: raise RuntimeError('视觉任务编号不一致')
                if self.check_only:
                    self.search_started = now; self._settle()
                else: self._navigate()
            except Exception as exc: self.fail('模型准备失败：'+str(exc))
            return
        if self.phase == 'navigating':
            if now-self.goal_started > (150 if not self.search_started else 40):
                return self.fail('导航至观察位置超时')
            if c.goal_status in ('pending', 'executing'): return
            if c.goal_status != 'succeeded': return self.fail('无法到达房间或观察位置：'+c.goal_status)
            col, row = world_to_cell(c.pose[0], c.pose[1], c.info)
            if not (0 <= row < self.inside.shape[0] and 0 <= col < self.inside.shape[1] and self.inside[row, col]):
                return self.fail('机器人未到达所选房间内')
            if not self.search_started: self.search_started = now
            self._settle(); return
        if self.phase == 'settling':
            if not self.check_only and (abs(c.odom_linear) > .015 or abs(c.odom_angular) > .03):
                self.due = now+.5; return
            if now < self.due: return
            self.phase = 'scanning'; self.note = '正在观察 '+self.item
        if self.phase == 'scanning':
            if self.future is None:
                if now < self.due: return
                self.future = self.vision.submit('/detect', {'job_id': self.job_id,
                                                            'after_sequence': self.last_sequence})
                self.due = now+30; return
            if not self.future.done():
                if now > self.due: self.fail('视觉推理超时')
                return
            try:
                result = self.future.result(); self.future = None
                if result.get('job_id') != self.job_id: raise RuntimeError('检测结果属于旧任务')
                if result['sequence'] <= self.last_sequence or result['capture_age_sec'] > 2:
                    raise RuntimeError('检测图像重复或已过期')
                self.last_sequence = result['sequence']; self.last_result = result; self.frames += 1
                detections = result.get('detections', [])
                if detections:
                    best = max(detections, key=lambda x: x['score'])
                    self.hits = self.hits+1 if self.last_box is not None and box_iou(best['box'], self.last_box) >= .20 else 1
                    self.last_box = best['box']
                    if self.hits >= 2:
                        self.observed += 1; return self.finish(True)
                else: self.hits = 0; self.last_box = None
                if self.frames < 3 and detections:
                    self.due = now+.25; return
                self.observed += 1
                if self.check_only: return self.finish(False, '当前视角未确认目标')
                self.index += 1
                if self.index >= len(self.steps):
                    return self.finish(False, '已完成本轮房间观察，遮挡位置可能仍无法看到')
                self._navigate()
            except Exception as exc: self.fail('视觉检测失败：'+str(exc))
