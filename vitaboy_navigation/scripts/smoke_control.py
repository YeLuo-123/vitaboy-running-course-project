"""Validate actual standing/walking/turning contact dynamics before ROS tests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from vitaboy_navigation.simulator import Simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--running',action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    world='running_world.json' if args.running else 'world.json'
    sim = Simulation(args.repo.resolve(), json.loads((root/'config'/world).read_text()),args.policy,running=args.running)
    dt = sim.robot.control_dt; trace = []; failed = False; collided = False
    renderer = writer = None; next_frame = 0.
    if args.video:
        import mujoco
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(sim.robot.model, height=480, width=640)
        writer = imageio.get_writer(str(args.video), fps=15)
        camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = sim.robot.pelvis_body_id; camera.distance = 4.
        camera.azimuth = -130.; camera.elevation = -20.
    try:
        for i in range(round(22/dt)):
            t = i*dt
            command = [0., 0., 0.]
            if 3 <= t < 9: command[0] = 2.4 if args.running else .25
            if 13 <= t < 18: command[2] = .6 if args.running else .35
            failed = sim.step(command); collided |= sim.collision()
            state = sim.telemetry(scan=False)
            trace.append(dict(t=state['sim_time'], position=state['position'],
                              command=state['command'], joints=state['joint_position'],
                              quaternion=state['quaternion_wxyz'],
                              velocity=state['twist_body']))
            if writer and sim.robot.data.time >= next_frame:
                renderer.update_scene(sim.robot.data, camera=camera)
                writer.append_data(renderer.render())
                while next_frame <= sim.robot.data.time: next_frame += 1/15
            if failed or collided: break
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
        sim.robot.close()
    forward = [x for x in trace if 5 < x['t'] < 9]
    stopped = [x for x in trace if 11 < x['t'] < 13]
    turning = [x for x in trace if 15 < x['t'] < 18]
    positions = np.array([x['position'] for x in trace])
    joint_range = float(np.ptp(np.array([x['joints'] for x in trace]),axis=0).max())
    report = dict(backend='Official VitaBoy ONNX / MuJoCo with physical obstacles',
                  failed=failed, obstacle_collision=collided,
                  simulated_duration_s=trace[-1]['t'],
                  forward_mean_mps=float(np.mean([x['velocity'][0] for x in forward])) if forward else None,
                  stopped_planar_mean_mps=float(np.mean([np.linalg.norm(x['velocity'][:2]) for x in stopped])) if stopped else None,
                  turning_mean_radps=float(np.mean([x['velocity'][5] for x in turning])) if turning else None,
                  actual_displacement_m=float(np.linalg.norm(positions[-1,:2]-positions[0,:2])),
                  joint_excursion_max_rad=joint_range,
                  minimum_base_height_m=float(positions[:,2].min()),
                  virtual_support=False, base_pose_integration=False,
                  policy_sha256=hashlib.sha256(sim.loco.policy_path.read_bytes()).hexdigest())
    report['passed'] = bool(not failed and not collided and report['simulated_duration_s'] >=21.99
                            and report['forward_mean_mps'] > .10 and joint_range>.15
                            and abs(report['turning_mean_radps']) > .10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    args.output.with_suffix('.trajectory.json').write_text(json.dumps(trace))
    print(json.dumps(report), flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
