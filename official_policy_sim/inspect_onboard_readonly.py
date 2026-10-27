"""Inspect the onboard build environment and factory service; no ROS publishers."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def run(*args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=15)
    return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def main():
    service = run("systemctl", "show", "motion_intelligence.service", "-p", "LoadState", "-p", "FragmentPath", "-p", "DropInPaths", "-p", "ExecStart", "-p", "WorkingDirectory", "-p", "ActiveState", "-p", "UnitFileState", "-p", "MainPID")
    processes = []
    files = {}
    for p in Path("/proc").glob("[0-9]*"):
        try:
            comm = (p / "comm").read_text().strip()
            if any(comm.startswith(x) for x in ["motion_intelli", "controller_node", "joystick_node"]):
                exe, cwd = (p / "exe").resolve(), (p / "cwd").resolve()
                processes.append({"pid": int(p.name), "comm": comm, "exe": str(exe), "cwd": str(cwd)})
                if exe.is_file():
                    files[str(exe)] = hashlib.sha256(exe.read_bytes()).hexdigest()
        except (OSError, PermissionError):
            pass
    for line in service["stdout"].splitlines():
        if line.startswith("FragmentPath="):
            p = Path(line.split("=", 1)[1])
            if p.is_file():
                files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    message_root = Path("/opt/build_dist/common/share/interaction_msgs")
    messages = {}
    for name in ["LowCommand", "LowState", "MotorCommand", "MotorState", "RobotState", "MotorMonitor", "ImuState"]:
        p = message_root / "msg" / (name + ".msg")
        if p.exists():
            messages[name] = p.read_text()
            files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    # Record factory deployment configs and start scripts without printing contents.
    candidates = set()
    for item in processes:
        cwd = Path(item["cwd"])
        if cwd != Path("/") and cwd.exists():
            for root in [cwd / "config", cwd.parent / "config", cwd / "scripts", cwd.parent / "scripts"]:
                if root.is_dir():
                    candidates.update(p for p in root.rglob("*") if p.is_file() and p.suffix in [".yaml", ".yml", ".toml", ".sh", ".json", ".onnx"])
    for p in sorted(candidates):
        try:
            files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            pass
    protected_roots = [Path('/opt/build_dist'), Path('/etc/systemd/system')]
    for root in protected_roots:
        for p in root.rglob('*'):
            if not p.is_file() or p.is_symlink():
                continue
            try:
                if p.suffix in ['.service', '.yaml', '.yml', '.toml', '.sh', '.onnx'] or (os.access(p, os.X_OK) and p.open('rb').read(4) == b'\x7fELF'):
                    files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                pass
    report = {"hostname": platform.node(), "architecture": platform.machine(), "kernel": platform.release(), "user": os.environ.get("USER"),
        "factory_service": service, "factory_processes": processes, "factory_file_sha256": files, "native_messages": messages,
        "compilers": {x: shutil.which(x) for x in ["cmake", "g++", "make", "python3", "readelf"]},
        "disk": run("df", "-h", str(Path.home()), "/data"), "memory": run("free", "-h"),
        "dependency_packages": run("dpkg-query", "-W", "-f=${Package} ${Status} ${Version}\n", "libeigen3-dev", "libyaml-cpp-dev", "libgoogle-glog-dev", "libopencv-dev", "libcurl4-openssl-dev", "zlib1g-dev"),
        "native_cmake_configs": [str(p) for p in message_root.glob("cmake/*Config.cmake")],
        "safety": "read-only process/file/service inspection; no ROS initialization, mode changes, control publishing or service mutations"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
