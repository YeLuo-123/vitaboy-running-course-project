#!/usr/bin/env bash
set -eo pipefail
source "$HOME/honor_deploy_ws/official_policy_sim/sim_env.sh"
"$ROOT/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml
assert os.environ['ROS_DOMAIN_ID']=='117' and os.environ['ROS_LOCALHOST_ONLY']=='1'
base=Path.home()/'honor_deploy_ws/src/honor_rl_deploy/source/config/vita_boy/SIM_ONLY/base.yaml'
d=yaml.safe_load(base.read_text())['vita_boy']
assert d['is_sim'] is True and d['joy_type']=='keyboard'
for key in ['lowcmd_topic','lowstate_topic','imu_topic','joy_topic']:
    assert d[key].startswith('/honor_sim_only/'), (key,d[key])
print('SIMULATION ONLY: domain=117, transport=loopback, namespace=/honor_sim_only/',flush=True)
PY
cd "$ROOT/src/honor_rl_deploy/source/build"
exec ./motion_intelligence -v SIM_ONLY
