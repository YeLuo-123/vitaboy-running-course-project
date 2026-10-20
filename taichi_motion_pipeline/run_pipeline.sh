#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
WS=$(dirname "$ROOT")
PY="$WS/video_motion_pipeline/.venv/bin/python"
SIMPY="$WS/honor_rl_mjlab/.venv/bin/python"
export OMP_NUM_THREADS=4
export MUJOCO_GL=egl
export PYTHONPATH="$WS/GVHMR"
export PATH="$ROOT/bin:$PATH"
cd "$WS/GVHMR"
"$PY" tools/demo/demo.py --video "$ROOT/input/inference/user_video.mp4" \
  --output_root "$ROOT/outputs" --static_cam --verbose
cd "$ROOT"
"$PY" - <<'PY'
from pathlib import Path
import json,torch,numpy as np
root=Path.cwd();info=json.loads((root/'inspection/video_info.json').read_text())
b=torch.load(root/'outputs/user_video/preprocess/bbx.pt',map_location='cpu')['bbx_xyxy'].numpy()
w,h=info['size'];left=max(0,int(np.floor(b[:,0].min()))-50);right=min(w,int(np.ceil(b[:,2].max()))+50)
(root/'inspection/display_roi.json').write_text(json.dumps(dict(xyxy=[left,0,right,h],
    purpose='Display-only fixed crop; all inference uses the complete original frame.'),indent=2))
PY
"$PY" export_skeleton.py
"$SIMPY" retarget_taichi.py
"$SIMPY" repair_self_collision.py
FINAL="$ROOT/outputs/vita_boy_collision_free"
"$SIMPY" check_motion_quality.py --folder "$FINAL"
"$SIMPY" render_comparison.py --folder "$FINAL"
"$SIMPY" render_collision_review.py
cd "$WS/honor_rl_mjlab"
"$SIMPY" datasets/csv_to_npz.py --input-file "$FINAL/video_motion.csv" \
  --output-name "$FINAL/taichi_motion.npz" --input-fps 50 --output-fps 50
cd "$ROOT"
"$SIMPY" validate_artifacts.py --folder "$FINAL"
echo "TAICHI_RETARGET_COMPLETE"
