"""Run the real official C++ keyboard controller against isolated MuJoCo.

No physical-robot launch mode exists. The frozen initial feedback phase is
explicitly logged; free dynamics begin only after entering locomotion.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import pty
import select
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path.home() / "honor_deploy_ws"
HERE = ROOT / "official_policy_sim"
sys.path.insert(0, str(ROOT / "src/honor_mujoco/scripts"))


def check_environment():
    assert os.environ.get("ROS_DOMAIN_ID") == "117"
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    assert os.environ.get("RMW_IMPLEMENTATION") == "rmw_fastrtps_cpp"
    assert Path(os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"]).resolve() == HERE / "localhost_dds.xml"
    r = ET.parse(HERE / "localhost_dds.xml").getroot()
    values = [(x.tag.split("}")[-1], x.text) for x in r.iter()]
    assert ("address", "127.0.0.1") in values and ("useBuiltinTransports", "false") in values


def make_model(output, profile):
    src = ROOT / "src/honor_mujoco/robots_assets/vita_boy/vita_boy/xml/vita_boy_29dof_v1.xml"
    tree = ET.parse(src)
    tree.getroot().find("compiler").set("meshdir", str((src.parent / "../meshes").resolve()))
    floors = tree.getroot().findall(".//worldbody/geom[@type='plane']")
    assert len(floors) == 1, "Expected exactly one ground plane in the official asset"
    floors[0].set("friction", ".8 .02 .001")
    tree.write(output / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    model.opt.timestep = .002
    assert model.nu == 29 and model.nq == 36 and model.nv == 35
    data = mujoco.MjData(model)
    cfg = yaml.safe_load((profile / "rl_locomotion/rl_locomotion.yaml").read_text())["vita_boy"]
    joints = model.actuator_trnid[:, 0]
    data.qpos[model.jnt_qposadr[joints]] = cfg["default_dof_pos"]
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    # Place the lowest foot collision primitive 2 mm above the floor once.
    foot_ids = {model.body("left_ankle_roll_link").id, model.body("right_ankle_roll_link").id}
    bottoms = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] in foot_ids and model.geom_contype[i]:
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                radius = np.abs(data.geom_xmat[i].reshape(3, 3)[2]) @ model.geom_size[i]
            elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = model.geom_size[i, 0]
            else:
                continue
            bottoms.append(data.geom_xpos[i, 2] - radius)
    assert bottoms
    data.qpos[2] += .002 - min(bottoms)
    mujoco.mj_forward(model, data)
    # Match the project's nominal per-joint peak-effort simulation caps.
    limits = np.array([150,139,88,139,60,60]*2 + [88,35,35] + [35,35,18,35,7,7,7]*2, dtype=float)
    contract = HERE / "deployment_contract_template.json"
    if contract.exists():
        entries = json.loads(contract.read_text())["joints"]
        by_name = {j["name"]: j["sim_peak_effort_nm"] for j in entries}
        limits = np.array([by_name[model.joint(int(j)).name] for j in joints])
    model.actuator_ctrllimited[:] = True
    model.actuator_ctrlrange[:, 0] = -limits
    model.actuator_ctrlrange[:, 1] = limits
    return model, data, limits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--manual", action="store_true", help="Open a viewer and forward this terminal's keyboard to the C++ controller")
    args = p.parse_args()
    check_environment()
    session_lock = (ROOT / ".keyboard_sim.lock").open("w")
    fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profile = ROOT / "src/honor_rl_deploy/source/config/vita_boy/SIM_ONLY"
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    model, data, limits = make_model(output, profile)
    initial_qpos = data.qpos.copy()
    from ros2_bridge import Ros2Bridge
    import rclpy

    lock = threading.RLock()
    # Both the controller and simulator use renamed topics in a separate domain.
    sys.argv = [sys.argv[0], "--ros-args", "-r", "/xlab/hr/low_cmd:=/honor_sim_only/low_cmd",
                "-r", "/xlab/hr/low_state:=/honor_sim_only/low_state", "-r", "/xlab/hr/body_upper_imu:=/honor_sim_only/imu"]

    class LockedBridge(Ros2Bridge):
        commands = 0
        last_cmd_wall = 0.

        def __init__(self, mj_model, mj_data):
            # Reuse the official serialization and PD handlers. Own the executor
            # here so Ctrl+C and worker shutdown finish cleanly in this harness.
            from interaction_msgs.msg import LowCommand, LowState
            from sensor_msgs.msg import Imu
            from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException, ShutdownException
            from rclpy.signals import SignalHandlerOptions
            self.mj_model, self.mj_data = mj_model, mj_data
            self.num_motor = mj_model.nu
            self.dim_motor_sensor = 3 * self.num_motor
            self._has_imu = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat") >= 0
            self.tau, self.kp, self.kd, self.q, self.dq = (np.zeros(self.num_motor) for _ in range(5))
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            self._node = rclpy.create_node("ros2_bridge_node")
            self._low_state, self._imu = LowState(), Imu()
            self._low_state_pub = self._node.create_publisher(LowState, "/honor_sim_only/low_state", 10)
            self._imu_pub = self._node.create_publisher(Imu, "/honor_sim_only/imu", 10)
            self._low_cmd_sub = self._node.create_subscription(LowCommand, "/honor_sim_only/low_cmd", self.low_cmd_handler, 10)
            self._timer = self._node.create_timer(.003125, self._on_timer)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self.closing = False
            self.callback_error = None
            def spin():
                try:
                    self._executor.spin()
                except (ExternalShutdownException, ShutdownException):
                    pass
                except Exception as exc:
                    if not self.closing:
                        self.callback_error = repr(exc)
            self._spin_thread = threading.Thread(target=spin)
            self._spin_thread.start()

        def _on_timer(self):
            with lock:
                super()._on_timer()

        def low_cmd_handler(self, msg):
            assert len(msg.motor_cmd) == model.nu
            with lock:
                super().low_cmd_handler(msg)
                self.commands += 1
                self.last_cmd_wall = time.monotonic()

    bridge = LockedBridge(model, data)
    stop = threading.Event()
    free = threading.Event()
    samples, poses, errors, keys, graphs = [], [], [], [], []
    master = slave = process = reader = viewer = saved_terminal = None
    controller_text = []
    started = time.monotonic()
    release_at = None

    def physics(viewer=None):
        next_tick = time.monotonic()
        counter = 0
        try:
            while not stop.is_set() and (viewer is None or viewer.is_running()):
                if bridge.callback_error:
                    raise RuntimeError("ROS feedback thread failed: " + bridge.callback_error)
                with lock:
                    if free.is_set():
                        bridge.update_torque()
                        mujoco.mj_step(model, data)
                        mujoco.mj_rnePostConstraint(model, data)
                        if not np.isfinite(data.qpos).all():
                            raise RuntimeError("nonfinite simulation state")
                    if counter % 10 == 0:
                        rot = np.empty(9)
                        mujoco.mju_quat2Mat(rot, data.qpos[3:7])
                        rot = rot.reshape(3, 3)
                        tilt = math.acos(float(np.clip(rot[2, 2], -1, 1)))
                        samples.append([time.monotonic()-started, data.time, *data.qpos[:3], *data.qvel[:3],
                                        *(rot.T @ data.qvel[:3]), tilt, float(np.max(np.abs(data.qvel[6:]))), bridge.commands, data.qvel[5]])
                        poses.append(data.qpos.copy())
                counter += 1
                if viewer is not None and counter % 10 == 0:
                    with lock:
                        viewer.cam.lookat[:] = data.qpos[:3]
                        viewer.sync()
                next_tick += model.opt.timestep
                time.sleep(max(0., next_tick-time.monotonic()))
                if time.monotonic()-next_tick > .05:
                    next_tick = time.monotonic()
        except Exception as e:
            errors.append(repr(e))
            stop.set()

    def read_controller():
        with (output / "controller.log").open("wb") as f:
            while not stop.is_set():
                try:
                    chunk = os.read(master, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                f.write(chunk)
                f.flush()
                controller_text.append(chunk.decode(errors="replace"))
                if args.manual:
                    sys.stdout.write(chunk.decode(errors="replace"))
                    sys.stdout.flush()

    def wait(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            if stop.is_set() or (process is not None and process.poll() is not None):
                raise RuntimeError("simulation/controller stopped unexpectedly")
            time.sleep(min(.05, max(0., until-time.monotonic())))

    def key(value):
        os.write(master, value.encode())
        keys.append({"wall_s": time.monotonic()-started, "key": value})

    def capture_graph():
        topics = bridge._node.get_topic_names_and_types()
        names = [n for n, _ in topics]
        assert not any(n.startswith('/xlab/') or n.startswith('/xrobot/') for n in names), names
        graphs.append({"wall_s": time.monotonic()-started, "nodes": bridge._node.get_node_names_and_namespaces(), "topics": topics})

    worker = None
    try:
        if args.manual:
            import importlib
            viewer = importlib.import_module("mujoco.viewer").launch_passive(model, data)
            viewer.cam.distance = 3.
            viewer.cam.azimuth = -140.
            viewer.cam.elevation = -15.
        worker = threading.Thread(target=physics, args=(viewer,), daemon=True)
        worker.start()
        master, slave = pty.openpty()
        process = subprocess.Popen(["bash", str(HERE / "run_controller_sim.sh")], stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        slave = None
        reader = threading.Thread(target=read_controller, daemon=True)
        reader.start()
        for _ in range(200):
            if "FSM is running" in "".join(controller_text):
                break
            wait(.05)
        else:
            raise RuntimeError("controller did not receive simulated feedback within 10 s")
        if args.manual:
            import termios
            import tty
            assert sys.stdin.isatty(), "Manual mode needs a real terminal"
            saved_terminal = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            print("\nSIM ONLY: focus THIS terminal. R, wait >=4 s, L; then W/S/A/D/Q/E. Release stops after a 400 ms input lease plus filtering. Ctrl+C exits simulation.\nThe initial state is frozen until L has entered for 0.7 s. This is NOT a physical recovery test.\n", flush=True)
            entered_at = None
            try:
                while viewer.is_running() and not stop.is_set():
                    ready, _, _ = select.select([sys.stdin.fileno()], [], [], .02)
                    if ready:
                        raw_keys = os.read(sys.stdin.fileno(), 32).decode(errors="ignore")
                        for c in raw_keys:
                            key(c)
                    if process.poll() is not None:
                        raise RuntimeError("controller exited")
                    if entered_at is None and "FSMStateRLLocomotion::onEnter()" in "".join(controller_text):
                        entered_at = time.monotonic()
                    if entered_at is not None and not free.is_set() and time.monotonic()-entered_at > .7:
                        capture_graph()
                        release_at = time.monotonic()-started
                        free.set()
                        print("\nFree dynamics started. No virtual support.\n", flush=True)
            except KeyboardInterrupt:
                pass
            (output / "manual_session.json").write_text(json.dumps({"hardware_execution": False, "initial_state_frozen_until_locomotion": True,
                "release_wall_s": release_at, "keyboard_events": keys, "graph_snapshots": graphs, "errors": errors}, indent=2))
            np.savez_compressed(output / "trajectory.npz", samples=np.asarray(samples), qpos=np.asarray(poses))
            return
        key("r")
        wait(4.4)
        key("l")
        wait(.7)
        assert "FSMStateRLLocomotion::onEnter()" in "".join(controller_text), "locomotion not entered"
        assert bridge.commands > 100
        capture_graph()
        release_at = time.monotonic()-started
        free.set()
        phases = [("standing", 4., None), ("forward", 6., "w"), ("stop", 3., None), ("left_turn", 4., "q"), ("stop_after_turn", 3., None)]
        phase_records = []
        for name, seconds, k in phases:
            begin = time.monotonic()-started
            end = time.monotonic()+seconds
            while time.monotonic() < end:
                if k:
                    key(k)
                wait(min(.1, max(0., end-time.monotonic())))
            phase_records.append({"name": name, "start_wall_s": begin, "end_wall_s": time.monotonic()-started})
            capture_graph()
        with lock:
            final = data.qpos.copy()
        raw = np.asarray(samples)
        phase_metrics = []
        for phase in phase_records:
            selected = raw[(raw[:, 0] >= phase['start_wall_s']+.8) & (raw[:, 0] <= phase['end_wall_s'])]
            full = raw[(raw[:, 0] >= phase['start_wall_s']) & (raw[:, 0] <= phase['end_wall_s'])]
            phase_metrics.append({**phase, "samples": len(selected), "mean_body_vx": float(np.mean(selected[:, 8])),
                "mean_body_vy": float(np.mean(selected[:, 9])), "mean_yaw_rate_rad_s": float(np.mean(selected[:, 14])), "min_root_z": float(np.min(full[:, 4])),
                "max_tilt_rad": float(np.max(full[:, 11])), "distance_xy_m": float(np.linalg.norm(full[-1,2:4]-full[0,2:4]))})
        free_samples = raw[raw[:,0] >= release_at]
        failed = bool(np.min(free_samples[:,4]) < .45 or np.max(free_samples[:,11]) > 1.0)
        report = {"status": "failed" if failed or errors else "completed", "hardware_execution": False,
            "isolation": {"ros_domain_id": 117, "localhost_only": True, "dds_interface": "127.0.0.1", "topic_prefix": "/honor_sim_only/"},
            "policy": str(profile / "rl_locomotion/vita_boy_loco.onnx"),
            "policy_sha256": hashlib.sha256((profile / "rl_locomotion/vita_boy_loco.onnx").read_bytes()).hexdigest(),
            "mujoco_version": mujoco.__version__, "model_mass_kg": float(model.body_mass.sum()), "ground_plane_count": int(np.sum(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)),
            "initial_qpos": initial_qpos.tolist(), "final_qpos": final.tolist(), "release_wall_s": release_at,
            "initialization": "frozen nominal standing state through R/4.4s/L/0.7s; then free dynamics, no band, root reset, external force or state teleport",
            "simulation_dt_s": model.opt.timestep, "effort_caps_nm": limits.tolist(), "physics_real_time_factor": float((free_samples[-1,1]-free_samples[0,1])/(free_samples[-1,0]-free_samples[0,0])),
            "motor_command_messages": bridge.commands, "fallen": failed, "phase_metrics": phase_metrics,
            "keyboard_events": keys, "graph_snapshots": graphs, "errors": errors}
        (output / "result.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(output / "trajectory.npz", samples=raw, qpos=np.asarray(poses), columns=np.array(["wall_s","sim_s","x","y","z","world_vx","world_vy","world_vz","body_vx","body_vy","body_vz","tilt","max_joint_speed","cmd_messages","yaw_rate"]))
        print(json.dumps({k: report[k] for k in ['status','fallen','physics_real_time_factor','motor_command_messages','phase_metrics']}, indent=2), flush=True)
    except Exception as e:
        (output / "failure.json").write_text(json.dumps({"hardware_execution": False, "exception": repr(e), "errors": errors, "keyboard_events": keys}, indent=2))
        raise
    finally:
        if saved_terminal is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved_terminal)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        stop.set()
        if reader:
            reader.join(timeout=1)
        if master is not None:
            os.close(master)
        if slave is not None:
            os.close(slave)
        if worker:
            worker.join(timeout=2)
        bridge.closing = True
        bridge._timer.cancel()
        bridge._executor.shutdown(timeout_sec=2)
        bridge._spin_thread.join(timeout=2)
        bridge._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if viewer:
            viewer.close()
        session_lock.close()


if __name__ == "__main__":
    main()
