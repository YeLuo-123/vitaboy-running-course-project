#!/usr/bin/env bash
set -eo pipefail
source "$HOME/honor_deploy_ws/official_policy_sim/sim_env.sh"
exec "$ROOT/.venv/bin/python" "$ROOT/official_policy_sim/smoke_keyboard.py" "$@"
