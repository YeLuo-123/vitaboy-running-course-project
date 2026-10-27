#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$ROOT" in /data/hihonor_onnx_keyboard/official_walk_*) ;; *) echo 'Unexpected staging directory'; exit 2;; esac
mkdir -p "$ROOT/logs" "$ROOT/tools"
/usr/bin/python3 "$ROOT/onboard/prepare_onboard.py"
source /opt/ros/humble/setup.bash
source /opt/iceoryx_cyclone_dds/install/setup.bash
source /opt/build_dist/common/share/interaction_msgs/local_setup.bash
SRC="$ROOT/honor_rl_deploy/source"
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build "$SRC/build" -j 2
ORT="$SRC/thirdparty/onnxruntime-linux-aarch64-1.22.0"
g++ -O2 -std=c++17 "$ROOT/onboard/onnx_cpu_probe.cpp" -I "$ORT/include" \
  -L "$ORT/lib" -Wl,-rpath,"$ORT/lib" -l:libonnxruntime.so.1.22.0 -o "$ROOT/tools/onnx_cpu_probe"
# This standalone program does not link ROS or access network/motor interfaces.
"$ROOT/tools/onnx_cpu_probe" "$SRC/config/vita_boy/ONBOARD_WALK/rl_locomotion/vita_boy_loco.onnx" > "$ROOT/logs/offline_inference.json"
readelf -h "$SRC/build/motion_intelligence"
readelf -h "$ROOT/tools/onnx_cpu_probe"
cat "$ROOT/logs/offline_inference.json"
echo 'STAGED AND BUILT ONLY. The hardware controller has NOT been launched.'
