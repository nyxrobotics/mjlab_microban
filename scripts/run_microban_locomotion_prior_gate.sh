#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

prior=""
expected_prior_sha256=""
robot_xml=""
expected_robot_xml_sha256=""
static_output="artifacts/microban_locomotion_prior_suitability.json"
dynamic_output="artifacts/microban_locomotion_prior_dynamic_gate.json"
force=0

usage() {
  echo "Usage: $0 [--prior PATH] [--expected-prior-sha256 SHA256]" >&2
  echo "          [--robot-xml PATH] [--expected-robot-xml-sha256 SHA256]" >&2
  echo "          [--static-output PATH] [--dynamic-output PATH] [--force]" >&2
}

while (($#)); do
  case "$1" in
    --prior) prior="$2"; shift 2 ;;
    --expected-prior-sha256) expected_prior_sha256="$2"; shift 2 ;;
    --robot-xml) robot_xml="$2"; shift 2 ;;
    --expected-robot-xml-sha256) expected_robot_xml_sha256="$2"; shift 2 ;;
    --static-output) static_output="$2"; shift 2 ;;
    --dynamic-output) dynamic_output="$2"; shift 2 ;;
    --force) force=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 64 ;;
  esac
done

force_args=()
if ((force)); then
  force_args+=(--force)
fi

input_args=()
if [[ -n "${prior}" ]]; then
  input_args+=(--prior "${prior}")
fi
if [[ -n "${expected_prior_sha256}" ]]; then
  input_args+=(--expected-prior-sha256 "${expected_prior_sha256}")
fi
if [[ -n "${robot_xml}" ]]; then
  input_args+=(--robot-xml "${robot_xml}")
fi
if [[ -n "${expected_robot_xml_sha256}" ]]; then
  input_args+=(--expected-robot-xml-sha256 "${expected_robot_xml_sha256}")
fi

# `set -e` is intentional: the CUDA evaluator is never imported or started
# unless the exact candidate first produces a passing static receipt.
uv run --locked python -m mjlab_microban.scripts.audit_locomotion_prior_suitability \
  "${input_args[@]}" \
  --output "${static_output}" \
  "${force_args[@]}"

uv run --locked python -m mjlab_microban.scripts.evaluate_locomotion_prior_dynamics \
  "${input_args[@]}" \
  --static-receipt "${static_output}" \
  --output "${dynamic_output}" \
  "${force_args[@]}"
