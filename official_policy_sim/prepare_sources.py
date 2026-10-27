"""Prepare pinned official sources and a separate simulation-only profile."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import yaml

ROOT = Path.home() / "honor_deploy_ws"
SOURCES = {
    "honor_rl_deploy": ("709c809d98ada109f8d5587ea655f1d4ca89ce47", "97cd2a7fa35414e4633053438ec684fe7533c141297bddefc757733832ec1ed4"),
    "honor_mujoco": ("7c76b0cc6d7b706e5726d4acfbe4e5fa75267a89", "c5ff07fb9529e790f929ffb5846ad994e78e5b27d36712cd8b9814615083488f"),
    "honor_robot_sdk": ("43f1dba12e2e0d25aaaec0fe4978da179a8c26f2", "50bb7f62bb20f7dde8096fc2628373287235c24e67eafea59472fbede02f5f2e"),
}


def main():
    manifest = {"source": "git archive HEAD from existing official checkouts on training host", "repositories": {}, "hardware_execution": False}
    for name, (commit, digest) in SOURCES.items():
        archive = ROOT / "archives" / (name + ".tar.gz")
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest, name
        target = ROOT / "src" / name
        if not target.exists():
            with tarfile.open(archive) as tar:
                for member in tar:
                    assert member.name == name or member.name.startswith(name + "/")
                    assert not member.issym() and not member.islnk()
                    resolved = (ROOT / "src" / member.name).resolve()
                    assert (ROOT / "src").resolve() in resolved.parents
                tar.extractall(ROOT / "src")
        assert not list(target.rglob("AGENTS.md")), "Review repository instructions before proceeding"
        manifest["repositories"][name] = {"url": "https://github.com/HonorRobotics/" + name, "commit": commit, "archive_sha256": digest}

    source = ROOT / "src/honor_rl_deploy/source"
    # nav2_util is never referenced by the C++ source. Avoid installing the
    # navigation stack just to satisfy this redundant upstream CMake dependency.
    cmake = source / "CMakeLists.txt"
    original = ROOT / "archives/deploy_CMakeLists.original.txt"
    if not original.exists():
        shutil.copy2(cmake, original)
    text = original.read_text().replace("find_package(nav2_util REQUIRED)\n", "").replace("sensor_msgs nav2_util cv_bridge", "sensor_msgs cv_bridge")
    cmake.write_text(text)

    profile = source / "config/vita_boy/SIM_ONLY"
    if not profile.exists():
        shutil.copytree(source / "config/vita_boy/V1", profile)
    base_path = profile / "base.yaml"
    base = yaml.safe_load(base_path.read_text())
    base["vita_boy"].update(is_sim=True, robot_version="SIM_ONLY", joy_type="keyboard", bind_main_thread_to_core=False,
        lowcmd_topic="/honor_sim_only/low_cmd", lowstate_topic="/honor_sim_only/low_state", imu_topic="/honor_sim_only/imu", joy_topic="/honor_sim_only/joy")
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    loco = profile / "rl_locomotion/rl_locomotion.yaml"
    data = yaml.safe_load(loco.read_text())
    data["vita_boy"]["bind_inference_thread_to_core"] = False
    loco.write_text(yaml.safe_dump(data, sort_keys=False))
    cmd = profile / "command.yaml"
    data = yaml.safe_load(cmd.read_text())
    data["vita_boy"]["joystick_command_max"].update(joystick_vel_x_max=0.5, joystick_vel_y_max=0.3, joystick_vel_yaw_max=0.35)
    cmd.write_text(yaml.safe_dump(data, sort_keys=False))
    manifest["simulation_profile"] = "SIM_ONLY; V1 preserved; keyboard vx=0.5 m/s, vy=0.3 m/s, yaw=0.35 rad/s"
    manifest["build_patch"] = "Only remove unused nav2_util CMake dependency; C++ source unchanged"
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
