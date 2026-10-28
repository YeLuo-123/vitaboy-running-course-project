"""ROS-independent geometry and frontier selection."""
import math
import re
from collections import deque
import numpy as np
from scipy import ndimage as ndi

FREE_LIMIT = 40
OCCUPIED_LIMIT = 65


def amcl_map_pixels(grid):
    """Encode a raw PGM whose OccupancyGrid contains only -1, 0 and 100."""
    a = np.asarray(grid)
    pixels = np.full(a.shape, 255, dtype=np.uint8)
    pixels[(a >= 0) & (a <= FREE_LIMIT)] = 0
    pixels[(a >= OCCUPIED_LIMIT) & (a <= 100)] = 100
    return pixels


def valid_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
        raise ValueError('地图名称只允许 1–64 位字母、数字、下划线或短横线')
    return name


def cell_to_world(col, row, info):
    x, y = (col + .5) * info['resolution'], (row + .5) * info['resolution']
    ox, oy, yaw = info['origin']
    return ox + math.cos(yaw)*x - math.sin(yaw)*y, oy + math.sin(yaw)*x + math.cos(yaw)*y


def world_to_cell(x, y, info):
    ox, oy, yaw = info['origin']; x -= ox; y -= oy
    return (math.floor((math.cos(yaw)*x + math.sin(yaw)*y)/info['resolution']),
            math.floor((-math.sin(yaw)*x + math.cos(yaw)*y)/info['resolution']))


def frontier_goal(grid, info, pose, excluded=()):
    """Reachable free-space goals near unknown edges, with 23cm obstacle clearance."""
    a = np.asarray(grid, dtype=np.int8).reshape(info['height'], info['width'])
    free = (a >= 0) & (a <= FREE_LIMIT)
    clearance = ndi.distance_transform_edt(a < 50) * info['resolution']
    safe = free & (clearance >= .23)
    col, row = world_to_cell(pose[0], pose[1], info)
    if not (0 <= row < a.shape[0] and 0 <= col < a.shape[1]) or not safe[row, col]:
        return None
    seed = np.zeros(a.shape, bool); seed[row, col] = True
    reachable = ndi.binary_propagation(seed, mask=safe)
    if not np.any(a < 0):
        return None
    unknown_distance = ndi.distance_transform_edt(a >= 0) * info['resolution']
    candidates = reachable & (unknown_distance >= .20) & (unknown_distance <= .55)
    labels, n = ndi.label(candidates)
    best = None
    for label, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None: continue
        ys, xs = np.where(labels[sl] == label)
        if len(xs) < 3: continue
        xs += sl[1].start; ys += sl[0].start
        order = np.argsort((xs-xs.mean())**2 + (ys-ys.mean())**2)
        for i in order[::max(1, len(order)//30)]:
            x, y = cell_to_world(int(xs[i]), int(ys[i]), info)
            distance = math.hypot(x-pose[0], y-pose[1])
            if distance < .35 or any(math.hypot(x-ex, y-ey) < .65 for ex, ey in excluded): continue
            score = min(len(xs)*info['resolution'], 5)*.25 - distance
            goal = (x, y, math.atan2(y-pose[1], x-pose[0]))
            if best is None or score > best[0]: best = (score, goal)
            break
    return best[1] if best else None
