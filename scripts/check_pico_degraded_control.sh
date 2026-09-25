#!/usr/bin/env bash
# Reproducible, motor-free fault-matrix check for the PICO simulation fallback.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
teleop_root="${workspace_root}/microban_teleop"
unity_project="${workspace_root}/microban_pico_client"
unity_editor="/home/kanade/Unity/Hub/Editor/2022.3.16f1/Editor/Unity"
run_unity=true

while (($#)); do
    case "$1" in
        --skip-unity)
            run_unity=false
            shift
            ;;
        --unity)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --unity requires an executable path" >&2
                exit 2
            }
            unity_editor="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

[[ -d "${teleop_root}" ]] || {
    echo "ERROR: sibling microban_teleop repository is missing" >&2
    exit 2
}

cd "${repo_root}"
uv run --locked --with pytest python -m pytest -q \
    tests/test_pico_degraded_control_matrix.py \
    tests/test_live_pico_teleop_sim.py::WatchdogTests::test_legacy_only_ignores_x_before_native_mapper_body_gate \
    tests/test_live_pico_teleop_sim.py::DualActorDispatchTests::test_camera_failure_degrades_video_without_disarming_control \
    tests/test_live_pico_teleop_sim.py::CliSafetyTests::test_default_legacy_checkpoint_is_the_audited_model_14999

"${teleop_root}/.venv/bin/python" -m pytest -q \
    "${teleop_root}/tests/test_native_server.py::test_disconnect_clears_mailbox_and_held_trigger_must_release_after_reconnect" \
    "${teleop_root}/tests/test_native_tracking.py::test_controller_and_body_freshness_are_independent" \
    "${teleop_root}/tests/test_native_tracking.py::test_malformed_body_does_not_poison_controls_and_recovery_needs_progress"

if [[ "${run_unity}" == true ]]; then
    [[ -x "${unity_editor}" ]] || {
        echo "ERROR: Unity editor not executable: ${unity_editor}" >&2
        exit 2
    }
    [[ -d "${unity_project}" ]] || {
        echo "ERROR: sibling microban_pico_client repository is missing" >&2
        exit 2
    }
    audit_logs="$(mktemp -d -t microban-pico-fault-matrix.XXXXXX)"
    methods=(
        NyxRobotics.Microban.StereoCamera.Editor.SimulationStereoCameraSelfTest.Run
        NyxRobotics.Microban.Transport.Editor.MicrobanNativeTransportSelfTest.Run
        NyxRobotics.Microban.Transport.Editor.PicoRawTrackingSamplerSelfTest.Run
        NyxRobotics.Microban.Transport.Editor.PicoDegradedControlSelfTest.Run
    )
    markers=(
        MICROBAN_SIM_STEREO_CAMERA_SELF_TEST=PASS
        MICROBAN_NATIVE_TRANSPORT_SELF_TEST=PASS
        PICO_RAW_TRACKING_SAMPLER_SELF_TEST=PASS
        PICO_DEGRADED_CONTROL_SELF_TEST=PASS
    )
    for index in "${!methods[@]}"; do
        log_path="${audit_logs}/unity-${index}.log"
        "${unity_editor}" \
            -batchmode \
            -nographics \
            -quit \
            -projectPath "${unity_project}" \
            -executeMethod "${methods[index]}" \
            -logFile "${log_path}"
        rg -q "${markers[index]}" "${log_path}" || {
            echo "ERROR: Unity PASS marker missing; inspect ${log_path}" >&2
            exit 1
        }
    done
    echo "Unity audit logs: ${audit_logs}"
fi

echo "PICO_DEGRADED_CONTROL_MATRIX=PASS (simulation/control only; no robot socket)"
