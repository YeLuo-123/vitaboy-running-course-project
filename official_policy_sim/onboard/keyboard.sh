#!/usr/bin/env bash
# Default/read-only mode NEVER launches the controller. --start is manual only.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$ROOT" in /data/hihonor_onnx_keyboard/official_walk_*) ;; *) echo 'Unexpected deployment directory'; exit 2;; esac
MODE="${1:---check}"
case "$MODE" in --check|--check-exclusive|--offline|--start) ;; *) echo 'Usage: bash onboard/keyboard.sh [--check|--check-exclusive|--offline|--start]'; exit 2;; esac
source "$ROOT/onboard/native_env.sh"
SRC="$ROOT/honor_rl_deploy/source"
if [[ "$MODE" == --offline ]]; then
    exec "$ROOT/tools/onnx_cpu_probe" "$SRC/config/vita_boy/ONBOARD_WALK/rl_locomotion/vita_boy_loco.onnx"
fi
if [[ "$MODE" == --start && ( ! -t 0 || ! -t 1 ) ]]; then
    echo 'A foreground interactive SSH terminal is required (ssh -tt).'; exit 2
fi
exec 9>"$ROOT/logs/manual_keyboard.lock"
flock -n 9 || { echo 'Another check/controller owns this deployment lock.'; exit 2; }
EXTRA=()
[[ "$MODE" == --check ]] || EXTRA+=(--exclusive)
/usr/bin/python3 "$ROOT/onboard/preflight_readonly.py" "${EXTRA[@]}" --output "$ROOT/logs/preflight_latest.json"
[[ "$MODE" == --start ]] || exit 0
# Only this branch can actuate. No services, modes, recovery or power commands are sent here.
/usr/bin/python3 "$ROOT/onboard/verify_deployment.py"
echo 'Starting LIVE controller in Passive. It publishes joint commands immediately.'
echo 'R=recovery (4s, not balance); L=policy balance; W/S A/D Q/E=motion.'
echo 'P/Space/Ctrl-C are NOT balance-preserving stops. Keep physical support and independent emergency stop ready.'
cd "$SRC/build"
exec ./motion_intelligence -v ONBOARD_WALK
