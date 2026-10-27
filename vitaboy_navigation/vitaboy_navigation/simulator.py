"""Official VitaBoy ONNX + contact dynamics, never a kinematic base integrator."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import signal
import sys
import time

import numpy as np
import mujoco
from vitaboy_navigation.protocol import CommandGate, LIMITS, RUNNING_LIMITS, VERSION, yaw


def create_robot(repo, world, viewer=False):
    sys.path.insert(0, str(repo/'deploy'))
    from common.config import load_config
    from common.robot_interface import JOINT_NAMES
    from inference.mujoco_robot import MuJoCoRobot, _JOINT_ARMATURE

    class NavigationRobot(MuJoCoRobot):
        @staticmethod
        def _build_model(physics_dt):
            # Mirrors pinned official deployment actuator setup; only adds static obstacles.
            spec = mujoco.MjSpec.from_file(str(repo/'deploy/inference/scene.xml'))
            for act in list(spec.actuators):
                spec.delete(act)
            for geom in spec.geoms:
                if geom.name.endswith('_collision'):
                    foot = geom.name.startswith(('left_foot', 'right_foot'))
                    geom.condim, geom.priority = (3, 1) if foot else (1, 0)
                    if foot:
                        geom.friction[0] = .6
            for name, armature in zip(JOINT_NAMES, _JOINT_ARMATURE, strict=True):
                spec.joint(name).armature = armature
                act = spec.add_actuator(name=f'{name}_position', target=name)
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.dyntype = mujoco.mjtDyn.mjDYN_NONE
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                act.gainprm[0] = act.biasprm[1] = act.biasprm[2] = 0.
                act.inheritrange = 0.; act.ctrllimited = False
                act.forcelimited = True; act.forcerange[:] = (-1., 1.)
            for index, (x, y, hx, hy) in enumerate(world['rectangles']):
                if not all(math.isfinite(v) for v in (x, y, hx, hy)) or min(hx, hy) <= 0:
                    raise ValueError('Invalid world geometry')
                height = world['wall_height']
                spec.worldbody.add_geom(name=f'nav_wall_{index}', type=mujoco.mjtGeom.mjGEOM_BOX,
                                        pos=[x, y, height/2], size=[hx, hy, height/2],
                                        rgba=[.35, .45, .65, 1.], group=3,
                                        contype=1, conaffinity=1)
            model = spec.compile()
            model.opt.timestep = physics_dt
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            model.opt.iterations = 100; model.opt.tolerance = 1e-8
            model.opt.ls_iterations = 50; model.opt.ls_tolerance = .01
            model.opt.ccd_iterations = 128
            return model

    robot = NavigationRobot(load_config(repo/'deploy/config.toml'), launch_viewer=viewer)
    if robot.viewer is not None:
        robot.viewer.opt.geomgroup[3] = 1
    return robot, JOINT_NAMES


class Simulation:
    def __init__(self, repo, world, policy=None, viewer=False, running=False):
        self.repo = Path(repo).resolve()
        self.world = world
        self.limits = RUNNING_LIMITS if running else LIMITS
        self.running = running
        self.robot, self.joints = create_robot(self.repo, world, viewer)
        from fsm.loco.loco import LocomotionState
        from common.policy import OnnxPolicy, read_metadata
        self.loco = LocomotionState()
        if policy:
            self.loco.policy_path = Path(policy).resolve()
            self.loco.policy = OnnxPolicy(self.loco.policy_path)
            self.loco._validate_metadata()
        if running and not policy:
            raise ValueError('Running navigation requires an explicit trained policy file')
        metadata = read_metadata(self.loco.policy_path)
        for field, actual in [('joint_stiffness', self.loco.stiffness),
                              ('joint_damping', self.loco.damping),
                              ('default_joint_pos', self.loco.default_angles)]:
            if field in metadata and not np.allclose(np.fromstring(metadata[field], sep=','), actual, atol=1e-5):
                raise ValueError('Policy/deployment contract mismatch: '+field)
        if 'action_scale' in metadata and abs(float(metadata['action_scale'])-self.loco.action_scale) > 1e-8:
            raise ValueError('Action scale mismatch')
        # Set the initial standing state once. No base pose/velocity writes after initialization.
        self.robot.data.qpos[:2] = world['spawn']
        self.robot.data.qpos[self.robot.qpos_indices] = self.loco.default_angles
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.loco.enter(self.robot.get_state())
        self.command = np.zeros(3, dtype=np.float32)
        self.sequence = 0
        self.scan = {'stamp': 0., 'height': world['scan_height'], 'ranges': [None]*360}
        self.scan_next = 0.
        self.robot_geoms = set(np.where(self.robot.model.geom_bodyid != 0)[0])

    def step(self, desired):
        dt = self.robot.control_dt
        delta = np.asarray([.8, .4, .6])*dt
        desired = np.asarray(desired, dtype=np.float32)
        if not np.isfinite(desired).all() or desired.shape != (3,):
            raise ValueError('Invalid command')
        desired = np.clip(desired, -np.asarray(self.limits), self.limits)
        if self.running:
            desired[2]=np.clip(desired[2],-.9/max(abs(desired[0]),.6),.9/max(abs(desired[0]),.6))
        self.command += np.clip(desired-self.command, -delta, delta)
        target = self.loco.step(self.robot.get_state(), self.command)
        if target.stabilize_base:
            raise RuntimeError('Virtual support is forbidden in navigation')
        self.robot.send_command(target)
        self.sequence += 1
        state = self.robot.get_state()
        failed = (state.base_height < .4 or state.projected_gravity[2] > -.5 or
                  not np.isfinite(self.robot.data.qpos).all())
        return failed

    def collision(self):
        for contact in self.robot.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if (self.robot.model.geom_group[a] == 3 and b in self.robot_geoms or
                    self.robot.model.geom_group[b] == 3 and a in self.robot_geoms):
                return True
        return False

    def telemetry(self, scan=True):
        r = self.robot; d = r.data
        position = d.xpos[r.pelvis_body_id].copy()
        quat = d.xquat[r.pelvis_body_id].copy()
        heading = yaw(quat)
        if scan and d.time >= self.scan_next:
            point = np.asarray([position[0], position[1], self.world['scan_height']])
            group = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
            ranges = [];maximum_range=float(self.world.get('scan_range',8.))
            for angle in heading-np.pi+np.arange(360)*2*np.pi/360:
                direction = np.asarray([np.cos(angle), np.sin(angle), 0.])
                hit = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(r.model, d, point, direction, group, 1, -1, hit)
                ranges.append(float(distance) if .1 <= distance <= maximum_range else None)
            self.scan = {'stamp': float(d.time), 'height': self.world['scan_height'], 'ranges': ranges,'range_max':maximum_range}
            while self.scan_next <= d.time:
                self.scan_next += 1/15
        twist = np.zeros(6)
        mujoco.mj_objectVelocity(r.model, d, mujoco.mjtObj.mjOBJ_BODY, r.pelvis_body_id, twist, 0)
        c, s = math.cos(heading), math.sin(heading)
        planar_twist = [c*twist[3]+s*twist[4], -s*twist[3]+c*twist[4],
                        twist[5], twist[0], twist[1], twist[2]]
        return dict(schema=VERSION, robot='vitaboy', seq=self.sequence, sim_time=float(d.time),
                    position=position.tolist(), quaternion_wxyz=quat.tolist(),
                    twist_body=planar_twist,
                    joint_names=list(self.joints), joint_position=d.qpos[r.qpos_indices].tolist(),
                    scan=self.scan, command=self.command.tolist(),
                    oracle_odometry=True, virtual_support=False,
                    collision=self.collision())


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--world', type=Path, default=Path(__file__).resolve().parents[1]/'config/world.json')
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--running', action='store_true')
    parser.add_argument('--rviz-display', help='Capture an actual RViz X11 display next to MuJoCo')
    parser.add_argument('--status-file', type=Path)
    parser.add_argument('--duration', type=float, default=0.)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path, default=Path('navigation_session.json'))
    args = parser.parse_args()
    args.repo = args.repo.resolve(); args.world = args.world.resolve()
    args.output = args.output.resolve()
    if args.policy: args.policy = args.policy.resolve()
    if args.video: args.video = args.video.resolve()
    world = json.loads(args.world.read_text())
    os.chdir(args.repo)
    sim = Simulation(args.repo, world, args.policy, args.viewer, args.running)
    gate = CommandGate(sim.limits)
    receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receive.bind(('127.0.0.1', 15100)); receive.setblocking(False)
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.monotonic(); trace = []; failed = False; collision = False
    renderer = writer = None; next_frame = 0.
    if args.video:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        height,width,fps=(720,960,20) if args.rviz_display else (480,640,10)
        sim.robot.model.vis.global_.offheight=max(height,sim.robot.model.vis.global_.offheight)
        sim.robot.model.vis.global_.offwidth=max(width,sim.robot.model.vis.global_.offwidth)
        renderer = mujoco.Renderer(sim.robot.model, height=height, width=width)
        writer = imageio.get_writer(str(args.video), fps=fps,codec='libx264',quality=8)
        camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = sim.robot.pelvis_body_id; camera.distance = 4.
        camera.azimuth = -130.; camera.elevation = -25.
        scene_option = mujoco.MjvOption(); scene_option.geomgroup[3] = 1
        if args.rviz_display:
            from PIL import Image, ImageGrab, ImageDraw, ImageFont
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
            camera.distance=4.8;camera.elevation=-24.
    try:
        while sim.robot.is_running() and (args.duration <= 0 or sim.robot.data.time < args.duration):
            now = time.monotonic()
            while True:
                try:
                    packet, source = receive.recvfrom(4096)
                except BlockingIOError:
                    break
                if source[0] == '127.0.0.1':
                    gate.receive(packet, now)
            desired = gate.target(now) if sim.robot.data.time > 2. else (0., 0., 0.)
            failed = sim.step(desired)
            state = sim.telemetry()
            collision |= state['collision']
            send.sendto(json.dumps(state, allow_nan=False).encode(), ('127.0.0.1', 15101))
            if writer and sim.robot.data.time >= next_frame:
                renderer.update_scene(sim.robot.data, camera=camera, scene_option=scene_option)
                frame=renderer.render()
                if args.rviz_display:
                    rviz=ImageGrab.grab(xdisplay=args.rviz_display)
                    if rviz.size != (960,720):raise ValueError('RViz capture must be 960x720')
                    canvas=Image.new('RGB',(1920,768),(24,29,36))
                    canvas.paste(Image.fromarray(frame),(0,48));canvas.paste(rviz,(960,48))
                    draw=ImageDraw.Draw(canvas)
                    speed=math.hypot(*state['twist_body'][:2])
                    draw.text((16,12),f'MuJoCo | trained running policy | t={state["sim_time"]:.2f}s | v={speed:.2f}m/s',font=font,fill='white')
                    status={}
                    if args.status_file and args.status_file.exists():
                        try:status=json.loads(args.status_file.read_text())
                        except (ValueError,OSError):pass
                    label=status.get('label','Online SLAM initialization')
                    draw.text((976,12),f'RViz2 | {label}',font=font,fill='white')
                    frame=np.asarray(canvas)
                writer.append_data(frame)
                while next_frame <= sim.robot.data.time: next_frame += 1/fps
            if sim.sequence % 5 == 0:
                trace.append({key: state[key] for key in ('sim_time', 'position', 'command','twist_body','quaternion_wxyz')})
            if failed or collision:
                break
            delay = start+sim.robot.data.time-time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(backend='MuJoCo official VitaBoy deployment', failed=failed,
                      obstacle_collision=collision, duration_s=float(sim.robot.data.time),
                      virtual_support=False, reference_teleport=False, oracle_odometry=True,
                      policy=str(sim.loco.policy_path),
                      policy_sha256=hashlib.sha256(sim.loco.policy_path.read_bytes()).hexdigest(),
                      running_mode=args.running,limits=list(sim.limits),
                      synthetic_laser_max_range_m=world.get('scan_range',8.),
                      rviz_capture_display=args.rviz_display,video_fps=(fps if writer else None),
                      trace=trace)
        args.output.write_text(json.dumps(report, indent=2))
        sim.robot.close(); receive.close(); send.close()
        print(json.dumps({k:v for k,v in report.items() if k != 'trace'}), flush=True)


if __name__ == '__main__':
    main()
