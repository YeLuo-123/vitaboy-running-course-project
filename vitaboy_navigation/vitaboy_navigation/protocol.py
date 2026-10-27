"""Loopback protocol and command watchdog, independent of ROS/MuJoCo."""
import json
import math

VERSION = 1
LIMITS = (0.35, 0.15, 0.35)
RUNNING_LIMITS = (2.4, 0.12, 0.6)


def finite_vector(value, count):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError('Wrong vector length')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('Expected finite numeric values')
    return tuple(float(x) for x in value)


def yaw(q):
    w, x, y, z = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def transform_planar(x, y, heading, tx, ty, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (tx+c*x-s*y, ty+s*x+c*y,
            math.atan2(math.sin(heading+angle), math.cos(heading+angle)))


def approach_command(dx, dy, heading, goal_distance, limits=LIMITS):
    """Low-speed envelope measured against the official walking policy.

    Small (~0.1 m/s, ~0.06 rad/s) commands can settle into a standing attractor.
    Use rotate-first steering and a minimum walking command outside the goal zone.
    Local obstacle avoidance still has final authority to reduce this command.
    """
    if goal_distance <= .18:
        return 0., 0., 0.
    error = math.atan2(math.sin(math.atan2(dy, dx)-heading),
                       math.cos(math.atan2(dy, dx)-heading))
    turn = math.copysign(min(limits[2], max(.30, 1.5*abs(error))), error) if abs(error)>.15 else 0.
    forward = 0. if abs(error)>.65 else min(limits[0], max(.22, .8*math.hypot(dx,dy)))
    return forward, 0., turn


def brake_forward(speed, scan_points, clearance=.5):
    result = max(0., speed)
    for distance, angle in scan_points:
        projection = math.cos(angle)
        if projection > .05:
            result = min(result, .8*max(0.,distance-clearance)/projection)
    return result


def running_approach_command(dx, dy, heading, goal_distance, limits=RUNNING_LIMITS):
    """Speed-dependent steering and stopping for the trained velocity running actor."""
    if goal_distance <= .30:
        return 0., 0., 0.
    error = math.atan2(math.sin(math.atan2(dy, dx)-heading),
                       math.cos(math.atan2(dy, dx)-heading))
    turn = max(-limits[2], min(limits[2], 2.0*error))
    # Conservative 0.5 m/s^2 stopping envelope, below the 0.8 command ramp.
    speed = min(limits[0], .85*goal_distance, math.sqrt(max(0.,goal_distance-.30)))
    speed *= max(0., math.cos(error))**2
    if abs(error) > .55:
        speed = 0.
    turn = max(-.9/max(speed,.6), min(.9/max(speed,.6), turn))
    return speed, 0., turn


def brake_running(speed, scan_points, clearance=.8):
    """Allow stopping distance before walls; independent of scan density."""
    result=max(0., speed)
    for distance,angle in scan_points:
        projection=math.cos(angle)
        if projection > .15:
            available=max(0.,distance-clearance)/projection
            result=min(result,math.sqrt(available))
    return result


class CommandGate:
    def __init__(self, limits=LIMITS, timeout=0.25):
        self.limits = finite_vector(limits, 3)
        if any(x <= 0 for x in self.limits) or not 0 < timeout <= 1:
            raise ValueError('Invalid limits or watchdog')
        self.timeout = timeout
        self.last_time = -math.inf
        self.command = (0., 0., 0.)
        self.session = None
        self.seq = -1

    def receive(self, packet, now):
        try:
            data = json.loads(packet)
            if data.get('schema') != VERSION or data.get('robot') != 'vitaboy':
                return False
            values = finite_vector(data['command'], 3)
            session, seq = data['session'], data['seq']
            if not isinstance(session, str) or not 1 <= len(session) <= 64:
                return False
            if type(seq) is not int or seq < 0:
                return False
            if session == self.session and seq <= self.seq:
                return False
            self.command = tuple(max(-limit, min(limit, x)) for x, limit in zip(values, self.limits))
            self.session, self.seq, self.last_time = session, seq, now
            return True
        except (ValueError, KeyError, TypeError, AttributeError, UnicodeDecodeError):
            return False

    def target(self, now):
        return self.command if now-self.last_time <= self.timeout else (0., 0., 0.)


def decode_state(packet):
    data = json.loads(packet)
    if data.get('schema') != VERSION or data.get('robot') != 'vitaboy':
        raise ValueError('Not VitaBoy telemetry')
    for key, length in [('position', 3), ('quaternion_wxyz', 4), ('twist_body', 6)]:
        finite_vector(data[key], length)
    q = data['quaternion_wxyz']
    if abs(sum(x*x for x in q)-1) > .02:
        raise ValueError('Invalid orientation')
    names = data['joint_names']
    if len(names) != 29 or len(set(names)) != 29 or not all(isinstance(x, str) for x in names):
        raise ValueError('Invalid joint names')
    finite_vector(data['joint_position'], 29)
    finite_vector([data['sim_time']], 1)
    if type(data['seq']) is not int or data['seq'] < 0:
        raise ValueError('Invalid telemetry sequence')
    scan = data['scan']
    maximum_range=scan.get('range_max',8.)
    if isinstance(maximum_range,bool) or not isinstance(maximum_range,(int,float)) or not math.isfinite(maximum_range) or not .1<maximum_range<=32:
        raise ValueError('Invalid configured laser range')
    if len(scan['ranges']) != 360:
        raise ValueError('Invalid scan')
    if any(x is not None and (isinstance(x, bool) or not isinstance(x, (float, int)) or
                              not math.isfinite(x) or not .1 <= x <= maximum_range) for x in scan['ranges']):
        raise ValueError('Invalid laser range')
    if not math.isfinite(scan['stamp']) or scan['stamp'] > data['sim_time']+.001:
        raise ValueError('Invalid scan stamp')
    if not math.isfinite(scan['height']) or scan['height'] < 0:
        raise ValueError('Invalid laser frame')
    return data
