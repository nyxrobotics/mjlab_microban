#!/usr/bin/env bash
# One-command checker/launcher for the audited legacy PICO simulation fallback.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly WALK_CHECKPOINT="${PROJECT_ROOT}/checkpoints/xc330_velocity/model_14999.pt"
readonly EXPECTED_SHA256="b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141"
readonly USER_HOME_DIR="$(getent passwd "$(id -u)" | cut -d: -f6)"
readonly SIM_NATIVE_CONFIG="${USER_HOME_DIR}/.config/microban-teleop/native_transport_sim.json"
readonly SIM_PICO_CONFIG="${USER_HOME_DIR}/.config/microban-teleop/pico-usb-sim.json"
readonly PICO_CLIENT_ROOT="${PROJECT_ROOT}/../microban_pico_client"

mode="${1:-launch}"
if [[ "${mode}" == "launch" || "${mode}" == "launch-webxr" || "${mode}" == "check" ]]; then
    if (($# > 0)); then
        shift
    fi
else
    mode="launch"
fi

cd -- "${PROJECT_ROOT}"
actual_sha256="$(sha256sum -- "${WALK_CHECKPOINT}" | cut -d ' ' -f 1)"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    echo "[FAIL] audited legacy checkpoint SHA-256 mismatch" >&2
    exit 1
fi

if [[ "${mode}" == "check" ]]; then
    uv lock --check
    uv run --locked --with pytest python -m pytest -q \
        tests/test_live_pico_teleop_sim.py \
        tests/test_simulation_camera.py
    MUJOCO_GL="${MUJOCO_GL:-egl}" uv run --locked python -m \
        mjlab_microban.scripts.legacy_pico_fallback_smoke \
        --walk-checkpoint "${WALK_CHECKPOINT}" "$@"
    exit 0
fi

input="pico-app"
control_port="63903"
runtime_args=(--native-config "${SIM_NATIVE_CONFIG}")
if [[ "${mode}" == "launch-webxr" ]]; then
    input="webxr"
    control_port="8443"
    runtime_args=()
fi

for port in "${control_port}" 8081; do
    if ss -H -ltnp | rg -q ":${port}[[:space:]]"; then
        echo "[FAIL] localhost TCP port ${port} is already in use:" >&2
        ss -H -ltnp | rg ":${port}[[:space:]]" >&2 || true
        echo "Stop/reconfigure the listed owner, or use launch-webxr on free ports." >&2
        exit 1
    fi
done

if [[ "${input}" == "pico-app" ]]; then
    if [[ ! -f "${SIM_NATIVE_CONFIG}" || -L "${SIM_NATIVE_CONFIG}" ||
          ! -f "${SIM_PICO_CONFIG}" || -L "${SIM_PICO_CONFIG}" ]]; then
        echo "[FAIL] isolated 63903 simulation pairing is missing." >&2
        exit 1
    fi
    if [[ ! -x "${PICO_CLIENT_ROOT}/scripts/provision_pico_usb.sh" ]]; then
        echo "[FAIL] PICO provisioning script is missing." >&2
        exit 1
    fi
    "${PICO_CLIENT_ROOT}/scripts/provision_pico_usb.sh" \
        --config "${SIM_PICO_CONFIG}" \
        --camera-mode simulation
elif command -v adb >/dev/null 2>&1 && adb get-state >/dev/null 2>&1; then
    adb reverse "tcp:${control_port}" "tcp:${control_port}" >/dev/null
    adb reverse tcp:8081 tcp:8081 >/dev/null
else
    echo "[WARN] PICO is not visible to adb; control/camera reverse was not installed." >&2
fi

exec uv run --locked python -m mjlab_microban.scripts.live_pico_teleop_sim \
    --input "${input}" \
    "${runtime_args[@]}" \
    --walk-checkpoint "${WALK_CHECKPOINT}" \
    "$@"
