#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
WS=$(dirname "$ROOT")
cd "$ROOT"
mkdir -p logs outputs bin
echo "Checking isolated reconstruction environment"
PY="$ROOT/.venv/bin/python"
"$PY" -c 'import torch,pytorch3d,smplx,wis3d; print(torch.__version__,torch.cuda.is_available())'
if ! "$PY" -c 'import chumpy' >/dev/null 2>&1; then
  "$HOME/.local/bin/uv" pip install --python "$PY" --no-build-isolation chumpy==0.70
fi
ffmpeg_bin=$("$PY" -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
ln -sfn "$ffmpeg_bin" "$ROOT/bin/ffmpeg"
export PATH="$ROOT/bin:$PATH"
export PYTHONPATH="$WS/GVHMR"
export OMP_NUM_THREADS=4
export MUJOCO_GL=egl
echo "STAGE GVHMR inference and human rendering"
cd "$WS/GVHMR"
"$PY" tools/demo/demo.py --video "$ROOT/input/inference/user_video.mp4" \
  --output_root "$ROOT/outputs" --static_cam --verbose
cd "$ROOT"
echo "STAGE export SMPL-X skeleton"
"$PY" export_skeleton.py
echo "STAGE GMR VitaBoy retarget"
"$WS/honor_rl_mjlab/.venv/bin/python" retarget_video.py
echo "STAGE render synchronized comparison"
"$WS/honor_rl_mjlab/.venv/bin/python" render_comparison.py
"$HOME/.local/bin/uv" pip freeze --python "$PY" > assets/environment_freeze.txt
git -C "$WS/GVHMR" rev-parse HEAD > assets/GVHMR_commit.txt
echo "RECONSTRUCTION_COMPLETE"
