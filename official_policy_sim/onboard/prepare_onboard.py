"""Stage a separate official controller copy. Never edit factory paths."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_SHA = "97cd2a7fa35414e4633053438ec684fe7533c141297bddefc757733832ec1ed4"
MODEL_SHA = "c8978c07a7ce2cd85370ff1010cc23a1b2ec0a0686b024fc7533e947852d19a7"


def main():
    assert ROOT.parent == Path("/data/hihonor_onnx_keyboard"), ROOT
    archive = ROOT / "honor_rl_deploy.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == DEPLOY_SHA
    source = ROOT / "honor_rl_deploy/source"
    if not source.exists():
        with tarfile.open(archive) as tar:
            for m in tar:
                assert m.name == "honor_rl_deploy" or m.name.startswith("honor_rl_deploy/")
                assert not m.issym() and not m.islnk()
                assert ROOT in (ROOT / m.name).resolve().parents
            tar.extractall(ROOT)
    assert not list((ROOT / "honor_rl_deploy").rglob("AGENTS.md")), "Review instructions first"
    cmake = source / "CMakeLists.txt"
    original = ROOT / "logs/CMakeLists.original.txt"
    if not original.exists():
        shutil.copy2(cmake, original)
    cmake.write_text(original.read_text().replace("find_package(nav2_util REQUIRED)\n", "").replace("sensor_msgs nav2_util cv_bridge", "sensor_msgs cv_bridge"))
    profile = source / "config/vita_boy/ONBOARD_WALK"
    if not profile.exists():
        shutil.copytree(source / "config/vita_boy/V1", profile)
    base_file = profile / "base.yaml"
    base = yaml.safe_load(base_file.read_text())
    base["vita_boy"].update(robot_version="ONBOARD_WALK", is_sim=False, joy_type="keyboard", bind_main_thread_to_core=False,
        supported_states=["FSMStatePassive", "FSMStateDamper", "FSMStateRecoveryStand", "FSMStateRLLocomotion"])
    base_file.write_text(yaml.safe_dump(base, sort_keys=False))
    loco_file = profile / "rl_locomotion/rl_locomotion.yaml"
    loco = yaml.safe_load(loco_file.read_text())
    loco["vita_boy"]["bind_inference_thread_to_core"] = False
    loco_file.write_text(yaml.safe_dump(loco, sort_keys=False))
    cmd_file = profile / "command.yaml"
    command = yaml.safe_load(cmd_file.read_text())
    command["vita_boy"]["joystick_command_max"].update(joystick_vel_x_max=.3, joystick_vel_y_max=.25, joystick_vel_yaw_max=.3)
    command["vita_boy"].update(vel_des_min=[-.3,-.25,-1.5], vel_des_max=[.3,.25,1.5], ome_des_min=[-1.,-1.,-.3], ome_des_max=[1.,1.,.3])
    cmd_file.write_text(yaml.safe_dump(command, sort_keys=False))
    policy = profile / "rl_locomotion/vita_boy_loco.onnx"
    assert hashlib.sha256(policy.read_bytes()).hexdigest() == MODEL_SHA
    (ROOT / "deployment_manifest.json").write_text(json.dumps({"stage_only": True, "hardware_controller_started": False,
        "repository": "https://github.com/HonorRobotics/honor_rl_deploy", "commit": "709c809d98ada109f8d5587ea655f1d4ca89ce47",
        "source_archive_sha256": DEPLOY_SHA, "model_sha256": MODEL_SHA, "root": str(ROOT), "native_sdk": "/opt/build_dist/common",
        "system_installations": False, "factory_services_modified": False, "profile": "ONBOARD_WALK", "keyboard_vx_limit_mps": .3,
        "tracking_disabled_in_profile": True, "configuration_note": "Staged for future supervised hardware testing; not proof of actuator readiness or calibrated limits"}, indent=2))
    print("Prepared independent copy:", ROOT, flush=True)


if __name__ == "__main__":
    main()
