#!/usr/bin/env bash
set -eo pipefail
ROOT="$HOME/honor_deploy_ws"
export PATH="/usr/bin:/bin:$ROOT/.venv/bin"
unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
"$ROOT/.venv/bin/python" "$ROOT/official_policy_sim/prepare_sources.py"
cd "$ROOT"
"$ROOT/.venv/bin/colcon" --log-base "$ROOT/logs/sdk_colcon" build \
  --base-paths "$ROOT/src/honor_robot_sdk/common/interaction_msgs" \
  --build-base "$ROOT/sdk_build" --install-base "$ROOT/sdk_install" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DPython3_EXECUTABLE="$ROOT/.venv/bin/python" -DPYTHON_EXECUTABLE="$ROOT/.venv/bin/python" \
  --parallel-workers 2
source "$ROOT/sdk_install/setup.bash"
cmake -S "$ROOT/src/honor_rl_deploy/source" -B "$ROOT/src/honor_rl_deploy/source/build" \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$ROOT/.venv/bin/python" -DPYTHON_EXECUTABLE="$ROOT/.venv/bin/python"
cmake --build "$ROOT/src/honor_rl_deploy/source/build" -j 4
