#!/usr/bin/env python3
"""Retrain, export, install and commit every Microban policy at the HOME of config/home_pose.yaml.

One command after a HOME edit (docs/home_pose_workflow.md):

    python3 scripts/retrain_all_for_home.py \\
        --robot-repo ../microban_home-<label> --robot-branch home-<label> [--training-branch home-<label>]

(run it in the home-config training checkout; --robot-repo must be a robot
worktree whose code reads config/home_pose.yaml, i.e. made from the robot
branch home-config -- the deploy checkout is refused at preflight)

Steps (each one is skipped when its outputs already exist and validate, so the
same command resumes after a crash, a stall or a fixed failure):

1. HOME: balance check of config/home_pose.yaml (COM vs. sole contact area,
   scripts/home_pipeline/home_check.py and config/balance_home_pose.py
   --check; the YAML is never modified), the training-line check
   (config/home_pose_tool.py show) and the robot's generated
   config/home_pose.yaml (home_pose_tool.py write-robot).
2. Walking: Mjlab-Velocity-Microban 15000 iterations from scratch, then a
   continuation to 30000; every 1000-iteration checkpoint is probed in the
   PICO teleop env (9 x 300 steps, the v12 source gate) while training runs;
   the best candidates are re-probed and the one with the best worst-case
   margin over all repeats is selected (if none passes every repeat, the best
   one is used and reported as a fallback).  It is copied to
   checkpoints/<prefix>_walk/ (the v12 provenance re-hashes it there).
3. Get-up stages 1-5 (Mjlab-Getup-Microban 2500 -> ImuDelay +1500 -> action
   std reset 0.5 -> CalmRoll-ImuDelay +8000 -> CalmEffortStrong-ImuDelay +6500
   -> CalmPush-ImuDelay +3000, entropy 0.001 from stage 3), each stage
   evaluated (scripts/home_pipeline/getup_eval.py); the final stage must pass
   the get-up acceptance gate.  Runs in parallel with step 2 (GPU memory
   permitting).
4. PICO v12: a fresh pose-release chain from the selected walker
   (scripts/train_microban_teleop_v12.sh start), stage
   route 0 -> 3000 -> 3100 -> 7000 -> 7100 -> 10000 -> 10100 -> 15000 with
   the stage gate at every boundary (the three evaluators of
   scripts/evaluate_microban_teleop_v12_stage.sh, each run to the end so a
   failing gate is a verdict with every report; then teleop_v12_stage create).
   Retrains use a new training seed (train_microban_teleop_v12.sh --seed; 42,
   then 43, ...): same-seed retrains from the same gated parent failed
   identically on the 2026-10 forward-lean chain.  Automatic
   rescues: an accuracy-only canary failure is retrained once with seed 43 (a
   retry interrupted before it saved its checkpoint is rerun on resume; a gate
   whose evaluator died without a report is not a verdict and stops the run
   for a re-evaluation on rerun).  The 10000
   boundary escalates on its own (the route the 2026-10 forward-lean chain
   took by hand): a failed 9999 gate tries the pose-release model_9900 corner
   rescues, one run per mix of --pr-corner-rescue-mixes (default lf60, lf90,
   lf72, lf65; scripts/train_microban_teleop_v12_corner_rescue.sh
   --hand-pose-release --mix M, whose validator accepts only a fresh
   pose-release parent failing hand accuracy only), each gated at 9999; if
   they all fail, 7100->10000 is retrained from the gated model_7099 as a new
   attempt (run <prefix>_v12_7100_to10000_a2, gated, then its own rescues), up
   to --v12-9999-attempts (default 2, attempt k trains with seed 41+k); then
   the run stops.  The 15000 boundary escalates the same way: a failed 14999
   gate whose locomotion and ONNX checks passed and whose tracking failures
   are all rescuable (hand/foot accuracy, actual_soft_limits,
   twist_directional_response) tries the pose-release final-scenario rescues
   (scripts/train_microban_teleop_v12_hand_pose_release_final_rescue.sh
   MODEL_14900 FAILED_TRACKING_REPORT --mix M --seed S, one run per mix of
   --pr-final-rescue-mixes, default pr_v1,...,pr_v6; a mix whose
   scenarios do not cover every failed scenario is refused by the validator
   and skipped), each gated at 14999; if they all fail, 10100->15000 is
   retrained from the gated model_10099 with the next seed (run
   <prefix>_v12_10100_to15000_a<k>), up to --v12-15000-attempts (default 2);
   then the run stops.  The first rescue
   commits config/home_pose.yaml on the training branch (the rescue launchers
   train only from a clean committed tree).  A passing rescue's model_9999
   (lineage fresh_chain_model9900_corner_rescue) or attempt is resumed as an
   ordinary pose-release checkpoint.  Every gate is judged under the one
   profile of its clock (evaluate_teleop_v12_tracking; hand RMS 0.040 m at
   every HOME), and the package must record the 10000 / 10100 gates (step 5
   stops if either is missing).
5. Export walk.onnx / getup.onnx, install them, the robot HOME yaml and the
   run pins (walking source / probe / walk.onnx SHA-256s, HOME literals of the
   robot tests) into --robot-repo, package PICO against that robot tree
   (export_teleop_v12_deployment --microban-repo, real robot validator),
   install pico_teleop.onnx, then run tools/validate_pico_policy.py and the
   robot test suite.
6. Only if everything passed: commit the robot branch, then the training
   branch (config/home_pose.yaml + config/releases/<tag>.json), and push both.
   Otherwise the command stops with a report and a non-zero exit code.

Status: <state dir>/STATUS.log (one line per event; subprocess output in
<state dir>/logs/), machine state in <state dir>/state.json.  The default
state dir is artifacts/home_pipeline/<tag>/ (a new HOME gets a new one).
``--status`` prints the state of a run.

GPU: before each GPU job the command waits until the GPU has the free memory
the job needs; a job whose log is not written for the stall limit is killed
(only its own process group: this command never signals any other process)
and the command exits 2 (rerun to resume).  --serial-gpu (the dry-run
default) runs at most one of its GPU jobs at a time, for a GPU shared with
another training.  Ctrl-C / SIGTERM stops its own children and exits.

--dry-run exercises the whole plumbing in minutes at any HOME: 2-3 iterations
per stage at 64 envs, PICO stage boundaries reached by clock lifts (every
segment, canaries included, lifted to 3 updates before its real end and the
lift's parent recorded in its params/agent.yaml), every gate
evaluated and recorded but not enforced (plumbing mode; only
--dry-run-simulate-failures decides the rescue route), the final package
built by scripts/home_pipeline/dry_run_tools.py (artifacts labelled DRYRUN,
the package marked dry_run_not_deployable, which the robot refuses outside
the dry run's own checks),
the robot repo a scratch clone (origin push URL a local path), the robot
commit local, and nothing pushed or committed in the training repo.  It needs
one of:

  --dry-run-walk-init WALKER  walking continues from a checkpoint stamped with
                              this HOME (e.g. the forward-lean cont2
                              model_29000 for the forward-lean yaml), so the
                              v12 source probe can pass and the robot
                              validator and tests are enforced;
  --dry-run-plumbing          from scratch at any edited HOME: the v12 source
                              probe, the robot validator and the robot tests
                              run and are recorded but do not stop the run (a
                              failed source probe is copied to
                              DRYRUN_FORCED_PASS_<receipt> and the chain
                              bootstraps from it).

--dry-run-simulate-failures fails the first gate of each canary and the 9999
gate; --dry-run-simulate-9999 retrain|rescue|stop picks the 9999 route (all
rescues fail and attempt 2 passes / the second rescue mix passes / everything
fails).  A dry corner rescue records the real rescue validator's verdict on
the dry model_9900 and stands in for the 2048-env rescue run with
dry_run_tools.py stamp-corner-rescue (the dry model_9999 with the exact
pose-release corner-rescue marker), so the downstream lineage validators run.
--dry-run-simulate-15000 pass|rescue|retrain|stop does the same for the 14999
gate (default pass: no simulated failure); a dry final rescue stands in with
dry_run_tools.py stamp-final-rescue (the dry model_14999 with the exact
pose-release final-rescue marker and the launcher's seed/params layout, so the
packager's lineage and resume-ancestry checks run on it).

Exit codes: 0 done, 1 a check/gate failed, 2 a job stalled, 3 bad input or
preflight refusal, 4 another instance holds the state dir, 130 interrupted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "home_pipeline"))
import robot_pins  # noqa: E402

HOME_YAML = REPO / "config" / "home_pose.yaml"
LOG_ROOT = REPO / "logs" / "rsl_rl"
WALK_EXP = LOG_ROOT / "mjlab_microban_velocity"
GETUP_EXP = LOG_ROOT / "mjlab_microban_getup"
V12_EXP = LOG_ROOT / "mjlab_microban_teleop_v12"
GATE_ROOT = REPO / "artifacts" / "teleop_v12_gates"
PROBE_ROOT = REPO / "artifacts" / "legacy_teleop_probe"
UV = ["uv", "run", "--locked"]
UV_ONNX = ["uv", "run", "--locked", "--with", "onnxruntime", "--with", "protobuf<7"]
TRAILER_DEFAULT = ""

# v12 source gate minimum signed responses (teleop_v12_bootstrap / stage gates).
PROBE_THRESHOLDS = {
    "forward_0p1": 0.04, "forward_0p2": 0.08, "backward_0p1": 0.02, "backward_0p2": 0.04,
    "lateral_left_0p1": 0.02, "lateral_right_0p1": 0.02, "yaw_left_0p5": 0.2,
    "yaw_right_0p5": 0.2,
}
PROBE_MAX_SOFT_LIMIT_OVERSHOOT_RAD = 0.0873  # 5 deg, as the bootstrap gate

# (label suffix, task, iterations, entropy coef, reset std before)
GETUP_STAGES = [
    ("getup_s1", "Mjlab-Getup-Microban", 2500, None, False),
    ("getup_s2_delay", "Mjlab-Getup-Microban-ImuDelay", 1500, None, False),
    ("getup_s3_calmroll", "Mjlab-Getup-Microban-CalmRoll-ImuDelay", 8000, "0.001", True),
    ("getup_s4_effort", "Mjlab-Getup-Microban-CalmEffortStrong-ImuDelay", 6500, "0.001", False),
    ("getup_s5_push", "Mjlab-Getup-Microban-CalmPush-ImuDelay", 3000, "0.001", False),
]
# (name, extra args) of the evaluations run after every get-up stage.
GETUP_EVALS = [
    ("stand_s11", ["stand", "Mjlab-Getup-Microban-Redesign", "--seed", "11", "--imu-delay", "3", "--noise"]),
    ("stand_s5", ["stand", "Mjlab-Getup-Microban-Redesign", "--seed", "5", "--imu-delay", "3", "--noise"]),
    ("push_s11", ["stand", "Mjlab-Getup-Microban-Redesign", "--seed", "11", "--imu-delay", "3", "--noise",
                  "--push", "x:0.3"]),
    ("posture_s11", ["posture", "Mjlab-Getup-Microban-ImuDelay", "--seed", "11"]),
]
# Get-up acceptance of the final stage (2026-10 lean/centered finals: 0.98
# fallen-start standing, 0/63 push falls, 0.07 rad/s, 60/64 posture).
GETUP_GATE = {
    "min_fallen_standing_fraction": 0.85,
    "max_push_fall_fraction": 0.10,
    "max_standing_joint_abs_vel_rad_s": 0.30,
    "min_posture_standing_fraction": 0.80,
}

# (segment suffix, end model index, canary)
V12_SEGMENTS = [
    ("0_to3000", 2999, False),
    ("3000_to3100", 3099, True),
    ("3100_to7000", 6999, False),
    ("7000_to7100", 7099, True),
    ("7100_to10000", 9999, False),
    ("10000_to10100", 10099, True),
    ("10100_to15000", 14999, False),
]
# export_teleop_v12_deployment.DRY_RUN_POLICY_ALLOW_ENV: the robot runtime
# accepts a dry_run_not_deployable PICO package only with this set to "1".
DRY_RUN_POLICY_ALLOW_ENV = "MICROBAN_ALLOW_DRYRUN_POLICY"
V12_ACCURACY_CHECKS = {"hand_tracking_rms", "hand_tracking_p95", "foot_tracking_rms",
                       "foot_tracking_p95"}
# microban_teleop_v12_hand_pose_release_final_rescue
# MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_RESCUABLE_CHECKS.
V12_FINAL_RESCUABLE_CHECKS = V12_ACCURACY_CHECKS | {"actual_soft_limits", "twist_directional_response"}
# All mixes of MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES; pr_v5 /
# pr_v6 (forward-lean-v2 005f55c) also replay the evaluator's fixed push on the
# replayed episodes.
V12_FINAL_RESCUE_MIXES = ("pr_v1", "pr_v2", "pr_v3", "pr_v4", "pr_v5", "pr_v6")
# The stage trainer's default training seed (train_microban_teleop_v12.sh
# --seed); retry k of a segment from the same parent trains with seed + k.
V12_TRAIN_SEED = 42
# evaluate_teleop_v12_tracking.HMD_HAND_PROFILE / FINAL_PROFILE: the profile of
# the model_9900 clock that the pose-release corner-rescue validator
# (validate_hand_pose_release_corner_rescue_parent_report) requires of the
# parent report, and the final profile.
V12_RESCUE_PARENT_PROFILE = "hmd_hand_reachable_performance_foot_exposure_v2_deployed_accuracy_v1"
V12_FINAL_PROFILES = {
    "full_body_reachable_performance_perturbation_v2_deployed_accuracy_v1",
}

STALL_S = {"train": 1200, "gate": 3600, "probe": 1800, "eval": 1800, "cpu": 1800}
GPU_RESERVATION_S = 120  # a started job has this long to allocate its memory
# A job has imported config/home_pose.yaml within this long after it started.
YAML_LOAD_WINDOW_S = 600
GPU_POLL_S = 60
# Dry run: v12 start attempts before stopping.  Its walkers are 3-iteration
# continuations whose probe margin is within the probe's repeat noise, so a
# failed start probe re-probes the candidates instead of stopping the dry run.
DRY_START_ATTEMPTS = 4
EXIT_FAILED, EXIT_STALL, EXIT_INPUT, EXIT_BUSY = 1, 2, 3, 4


class PipelineError(Exception):
    def __init__(self, message: str, code: int = EXIT_FAILED, *, secondary: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.secondary = secondary  # stopped only because another parallel step failed


class HomeYamlChanged(PipelineError):
    """config/home_pose.yaml changed during the run (at ``edit_time``, its mtime)."""

    def __init__(self, message: str, edit_time: float) -> None:
        super().__init__(message, EXIT_INPUT)
        self.edit_time = edit_time


class SourceProbeFailed(PipelineError):
    """The fresh v12 start probe of the selected walker failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_index(name: str) -> int | None:
    m = re.fullmatch(r"model_(\d+)\.pt", name)
    return int(m.group(1)) if m else None


def checkpoints(run_dir: Path) -> list[int]:
    if not run_dir.is_dir():
        return []
    return sorted(i for i in (model_index(p.name) for p in run_dir.iterdir()) if i is not None)


def runs_with_suffix(exp: Path, label: str) -> list[Path]:
    if not exp.is_dir():
        return []
    return sorted(p for p in exp.iterdir() if p.is_dir() and p.name.endswith("_" + label))


def resume_parent(run_dir: Path) -> Path | None:
    """The checkpoint ``run_dir`` resumed from (its params/agent.yaml), if any."""

    params = run_dir / "params" / "agent.yaml"
    try:
        text = params.read_text()
    except OSError:
        return None
    fields = dict(re.findall(r"^(resume|load_run|load_checkpoint): *(.*?) *$", text, re.M))
    run = re.fullmatch(r"\^?([A-Za-z0-9][A-Za-z0-9_-]*)\$?", fields.get("load_run", "").strip("'\""))
    ckpt = re.fullmatch(r"\^?(model_[0-9]+)(?:\[\.\]|\\\.|\.)pt\$?", fields.get("load_checkpoint", "").strip("'\""))
    if fields.get("resume") != "true" or run is None or ckpt is None:
        return None
    parent = run_dir.parent / run[1] / f"{ckpt[1]}.pt"
    return parent if parent.is_file() else None


KNOWN_TEST_FAILURES = REPO / "scripts" / "home_pipeline" / "known_test_failures.txt"


def known_test_failures() -> list[str]:
    """The training suite's pre-existing failures (pytest node ids)."""

    lines = KNOWN_TEST_FAILURES.read_text().splitlines() if KNOWN_TEST_FAILURES.is_file() else []
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def report_profile(path: Path) -> str | None:
    """The ``profile`` of a v12 tracking report, or None if it is unreadable."""

    try:
        profile = json.loads(path.read_text()).get("profile")
    except (OSError, ValueError, AttributeError):
        return None
    return profile if isinstance(profile, str) else None


def probe_verdict(path: Path) -> dict:
    """Pass/fail and worst margin of one 9x300 teleop-env probe receipt."""

    d = json.loads(path.read_text())
    s = d["summary"]
    resp, below = {}, []
    for r in d["results"]:
        dr = r.get("directional_response")
        name = r.get("scenario", r.get("name"))
        if dr is not None and name in PROBE_THRESHOLDS:
            resp[name] = float(dr["signed_response"])
            if resp[name] < PROBE_THRESHOLDS[name]:
                below.append(name)
    margins = {k: resp.get(k, float("-inf")) - t for k, t in PROBE_THRESHOLDS.items()}
    worst_key = min(margins, key=margins.get)
    ok = (
        not below and len(resp) == len(PROBE_THRESHOLDS)
        and s.get("completed_scenario_count") == 9 and s.get("fall_scenario_count") == 0
        and s.get("nonfinite_scenario_count", 0) == 0
        and s.get("directionally_correct_scenario_count") == 8
        and s.get("raw_action_recurrence_all_steps") is True
        and float(s.get("maximum_actual_soft_limit_violation_rad", 0.0)) <= PROBE_MAX_SOFT_LIMIT_OVERSHOOT_RAD
    )
    return {"ok": ok, "worst_margin": margins[worst_key], "worst": worst_key, "below": below,
            "falls": s.get("fall_scenario_count"), "completed": s.get("completed_scenario_count"),
            "responses": resp}


def select_walker(results: dict[str, list[dict]]) -> tuple[str, dict, bool]:
    """Pick the checkpoint with the best worst-case margin over its repeats.

    ``results`` maps a checkpoint key to its probe verdicts.  Returns (key,
    summary, fallback): fallback is True when no checkpoint passed every repeat.
    """

    rows = {}
    for key, verdicts in results.items():
        if not verdicts:
            continue
        rows[key] = {
            "repeats": len(verdicts),
            "passes": sum(v["ok"] for v in verdicts),
            "all_pass": all(v["ok"] for v in verdicts),
            "worst_case_margin": min(v["worst_margin"] for v in verdicts),
        }
    if not rows:
        raise PipelineError("no walking checkpoint was probed")
    passing = [k for k, r in rows.items() if r["all_pass"]]
    pool = passing or list(rows)
    best = max(pool, key=lambda k: (rows[k]["worst_case_margin"], rows[k]["repeats"]))
    return best, rows[best], not passing


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dry = args.dry_run
        self.plumbing = bool(args.dry_run and args.dry_run_plumbing)
        self.lock = threading.RLock()
        self.gpu_lock = threading.Lock()
        # --serial-gpu: at most one of this command's GPU jobs at a time.
        self.gpu_serial = threading.Lock() if args.serial_gpu else None
        self.gpu_reservations: dict[int, tuple[int, float]] = {}
        self.gpu_reservation_seq = 0
        self.gpu_last_free = 0
        self.owns_state = False
        self.yaml_sha256: str | None = None
        self.abort = threading.Event()
        self.children: dict[int, str] = {}
        # Every job of this command: {"name", "started", "ended" (None while running)}.
        self.job_log: list[dict] = []
        self.robot =Path(args.robot_repo).resolve() if args.robot_repo else None
        self.state_dir = Path(args.state_dir).resolve() if args.state_dir else None
        self.state: dict = {}
        self.home: dict = {}

    # ------------------------------------------------------------ infra
    def log(self, message: str) -> None:
        line = f"{datetime.now():%F %T} {message}"
        with self.lock:
            print(line, flush=True)
            if self.state_dir is not None and self.owns_state:
                with open(self.state_dir / "STATUS.log", "a") as f:
                    f.write(line + "\n")

    def save(self) -> None:
        if not self.owns_state:
            return  # never touch the state of a run another instance holds
        with self.lock:
            tmp = self.state_dir / "state.json.tmp"
            tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True, default=str))
            os.replace(tmp, self.state_dir / "state.json")

    def put(self, *keys_and_value) -> None:
        *keys, value = keys_and_value
        with self.lock:
            node = self.state
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value
            self.save()

    def get(self, *keys, default=None):
        node = self.state
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def check_abort(self) -> None:
        if self.abort.is_set():
            raise PipelineError("stopped because another step failed", secondary=True)

    def run(self, name: str, cmd: list[str], kind: str, *, cwd: Path = REPO, env: dict | None = None,
            allow_fail: bool = False, poll=None, stdout_path: Path | None = None) -> int:
        """Run one job in its own process group with stall detection."""

        self.check_abort()
        self.verify_home_yaml()
        log_dir = self.state_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        path = log_dir / f"{datetime.now():%m%d-%H%M%S}_{name}.log"
        self.log(f"start {name}: {' '.join(cmd)}  (log {path.name})")
        started = time.time()
        merged_env = dict(os.environ, **(env or {}))
        out_file = open(stdout_path, "w") if stdout_path else None
        with open(path, "w") as f:
            job = {"name": name, "started": started, "ended": None}
            with self.lock:
                self.job_log.append(job)
            proc = subprocess.Popen(cmd, cwd=cwd, env=merged_env, stdout=out_file or f,
                                    stderr=f if out_file else subprocess.STDOUT, start_new_session=True)
            with self.lock:
                self.children[proc.pid] = name
                self.put("children", str(proc.pid), {"name": name, "cmd": cmd, "started": started})
            try:
                while proc.poll() is None:
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    if proc.poll() is not None:
                        break
                    if self.abort.is_set():
                        self.kill(proc, name)
                        raise PipelineError(f"{name} stopped because another step failed", secondary=True)
                    newest = max(path.stat().st_mtime, stdout_path.stat().st_mtime if stdout_path else 0)
                    if time.time() - newest > STALL_S[kind]:
                        self.kill(proc, name)
                        raise PipelineError(
                            f"STALL {name}: no output for > {STALL_S[kind]} s, killed (log {path})",
                            EXIT_STALL)
                    if poll is not None:
                        try:
                            poll()
                        except BaseException:
                            self.kill(proc, name)
                            raise
            finally:
                if out_file:
                    out_file.close()
                with self.lock:
                    if proc.poll() is not None:
                        job["ended"] = time.time()
                    self.children.pop(proc.pid, None)
                    self.state.get("children", {}).pop(str(proc.pid), None)
                    self.save()
        # The job imports the YAML some seconds after it starts (uv / torch
        # start-up), so the check before it started is not enough: an edit in
        # that window means it ran at the edited HOME.  Stop before anything
        # of its output is used; main() quarantines that output.
        self.verify_home_yaml()
        minutes = (time.time() - started) / 60
        if proc.returncode != 0 and not allow_fail:
            tail = path.read_text(errors="replace")[-1500:].strip()
            raise PipelineError(f"{name} failed rc={proc.returncode} after {minutes:.1f} min "
                                f"(log {path}):\n{tail}")
        self.log(f"done {name} rc={proc.returncode} in {minutes:.1f} min")
        return proc.returncode

    def verify_home_yaml(self) -> None:
        """Stop at once if config/home_pose.yaml changed since step 1 of this run.

        Every job imports the YAML when it starts, so an edit during the run
        (for example trying the next HOME with balance_home_pose.py --write in
        this worktree) would mix HOMEs across stages.
        """

        if self.yaml_sha256 is None:
            return
        try:
            current = sha256(HOME_YAML)
        except OSError:
            current = None
        if current != self.yaml_sha256:
            try:
                edit_time = HOME_YAML.stat().st_mtime
            except OSError:
                edit_time = time.time()
            raise HomeYamlChanged(
                f"{HOME_YAML} changed during the run (sha256 {self.yaml_sha256[:12]} at step 1, now "
                f"{(current or 'missing')[:12]}); restore it (git -C {REPO} diff config/home_pose.yaml) and "
                "rerun, or use another worktree for the next HOME (the output of every job that ran "
                "after the edit is moved to <state-dir>/quarantine/ and is not reused)", edit_time)

    def quarantine_after_yaml_change(self, edit_time: float) -> None:
        """Move aside the output of every job that may have loaded the edited YAML.

        A job still running at ``edit_time`` that started less than
        YAML_LOAD_WINDOW_S before it may have imported the edited HOME (jobs
        load it during start-up), so nothing it wrote may be reused by a
        rerun: run directories created, and gates, probe receipts and this
        state's evaluation reports written, since the earliest such job
        started are moved to ``<state-dir>/quarantine/<time>/`` (kept for
        inspection).  Older jobs had loaded the HOME before the edit.
        """

        with self.lock:
            affected = [j for j in self.job_log if (j["ended"] is None or j["ended"] >= edit_time)
                        and j["started"] >= edit_time - YAML_LOAD_WINDOW_S]
        if not affected or self.state_dir is None:
            return
        cutoff = min(j["started"] for j in affected) - 1.0
        candidates: list[Path] = []
        if LOG_ROOT.is_dir():
            candidates += [run for exp in LOG_ROOT.iterdir() if exp.is_dir() for run in exp.iterdir()]
        for root in (GATE_ROOT, PROBE_ROOT, *(self.state_dir / d for d in
                                              ("walk_probe", "getup_eval", "pico", "export"))):
            if root.is_dir():
                candidates += list(root.iterdir())

        def written(path: Path) -> float:
            """A file's mtime; a directory's creation, approximated by its oldest file."""

            try:
                if not path.is_dir() or path.is_symlink():
                    return path.lstat().st_mtime
                times = [os.lstat(os.path.join(base, f)).st_mtime
                         for base, _dirs, files in os.walk(path) for f in files]
                return min(times) if times else path.lstat().st_mtime
            except OSError:
                return 0.0

        target = self.state_dir / "quarantine" / f"{datetime.now():%Y%m%d-%H%M%S}"
        moved = []
        for path in candidates:
            if written(path) < cutoff:
                continue
            for base in (LOG_ROOT, GATE_ROOT.parent, PROBE_ROOT.parent, self.state_dir):
                if path.is_relative_to(base):
                    rel = path.relative_to(base)
                    break
            else:
                base, rel = path.parent, Path(path.name)
            dest = target / base.name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            moved.append(str(path))
            self.log(f"quarantined {path} -> {dest} (written after the HOME YAML edit window opened)")
        self.put("quarantined", target.name, {"edit_time": edit_time, "cutoff": cutoff,
                                              "jobs": [j["name"] for j in affected], "moved": moved})

    def kill(self, proc: subprocess.Popen, name: str) -> None:
        """Stop one of this command's own jobs (its process group only)."""

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=60)
                return
            except subprocess.TimeoutExpired:
                continue
        self.log(f"killed {name} (pid {proc.pid})")

    def kill_all_children(self) -> None:
        for pid, name in list(self.children.items()):
            try:
                os.killpg(pid, signal.SIGTERM)
                self.log(f"stopping own job {name} (pid {pid})")
            except ProcessLookupError:
                pass

    def capture(self, cmd: list[str], *, cwd: Path = REPO, env: dict | None = None,
                timeout: float = 1800, check: bool = False) -> subprocess.CompletedProcess:
        self.verify_home_yaml()
        result = subprocess.run(cmd, cwd=cwd, env=dict(os.environ, **(env or {})),
                                capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise PipelineError(f"{' '.join(cmd)} failed rc={result.returncode}: "
                                f"{(result.stderr or result.stdout).strip()[-1500:]}")
        return result

    def git(self, repo: Path, *args: str, check: bool = True) -> str:
        return self.capture(["git", "-C", str(repo), *args], check=check).stdout.rstrip("\n")

    def gpu_free_mib(self) -> int:
        out = self.capture(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"], timeout=60).stdout.split()
        return int(out[0]) if out else 0

    def gpu_reserve(self, need_mib: int) -> int | None:
        """Reserve ``need_mib`` of free GPU memory for a job about to start.

        Returns a reservation id, or None when the memory is not free now.
        The lock is held only for this check (never while waiting), so a job
        waiting for a lot of memory never blocks a small one that fits, nor
        the stall checks of running jobs.  A reservation counts against the
        free memory until the job has had GPU_RESERVATION_S to allocate it
        (or ends), so two of this command's jobs never start on the same
        free memory.
        """

        with self.gpu_lock:
            now = time.time()
            self.gpu_reservations = {k: v for k, v in self.gpu_reservations.items() if v[1] > now}
            free = self.gpu_free_mib() - sum(v[0] for v in self.gpu_reservations.values())
            self.gpu_last_free = free
            if free < need_mib:
                return None
            self.gpu_reservation_seq += 1
            self.gpu_reservations[self.gpu_reservation_seq] = (need_mib, now + GPU_RESERVATION_S)
            return self.gpu_reservation_seq

    def gpu_release(self, reservation: int) -> None:
        with self.gpu_lock:
            self.gpu_reservations.pop(reservation, None)

    def gpu_job(self, need_mib: int, name: str, cmd: list[str], kind: str, *, wait: bool = True,
                **kwargs) -> int | None:
        """Start a GPU job once the GPU has ``need_mib`` free.

        With ``wait=False`` returns None at once when the memory is not free
        (the walking probes use this from inside the walking job's poll, so
        the walking stall check keeps running).
        """

        waited, noted = 0, -1
        serial = self.gpu_serial
        while True:
            self.check_abort()
            if serial is not None and not serial.acquire(timeout=0 if not wait else GPU_POLL_S):
                if not wait:
                    return None
                waited += GPU_POLL_S
                continue
            reservation = self.gpu_reserve(need_mib)
            if reservation is not None:
                break
            if serial is not None:
                serial.release()
            if not wait:
                return None
            if noted < 0 or waited - noted >= 1800:
                self.log(f"waiting for GPU memory for {name}: free {self.gpu_last_free} MiB "
                         f"(after this command's reservations) < {need_mib} MiB")
                noted = waited
            self.abort.wait(GPU_POLL_S)
            waited += GPU_POLL_S
        try:
            return self.run(name, cmd, kind, **kwargs)
        finally:
            self.gpu_release(reservation)
            if serial is not None:
                serial.release()

    # --------------------------------------------------------- preflight
    def preflight(self) -> None:
        a = self.args
        if self.robot is None or not (self.robot / ".git").exists():
            raise PipelineError(f"--robot-repo is not a git checkout: {a.robot_repo}", EXIT_INPUT)
        if not (self.robot / "tools" / "validate_pico_policy.py").is_file():
            raise PipelineError(f"{self.robot} is not a microban robot checkout", EXIT_INPUT)
        # The robot code must read config/home_pose.yaml (robot branch home-config or
        # a branch made from it).  An older tree hard-codes the centered HOME, and a
        # retrained policy would be refused only at step 5, a day of GPU later.
        constants = self.robot / "src" / "constants.py"
        if not (self.robot / "src" / "home_pose.py").is_file() or "from home_pose import" not in (
                constants.read_text() if constants.is_file() else ""):
            raise PipelineError(
                f"{self.robot} does not read config/home_pose.yaml (no src/home_pose.py imported by "
                "src/constants.py, e.g. the deploy branch feature/neck-roll-pitch-camera). Use a worktree "
                "of the robot branch home-config: git -C ../microban worktree add -b home-<label> "
                "../microban_home-<label> origin/home-config", EXIT_INPUT)
        if not (REPO / "config" / "home_pose.yaml").is_file() or not (
                REPO / "src" / "mjlab_microban" / "robot" / "home_pose.py").is_file():
            raise PipelineError(f"{REPO} is not a home-config training checkout", EXIT_INPUT)
        if not a.robot_branch:
            raise PipelineError("--robot-branch is required", EXIT_INPUT)
        if self.dry and a.dry_run_walk_init:
            seed = Path(a.dry_run_walk_init)
            if not seed.is_file() or model_index(seed.name) is None:
                raise PipelineError(f"--dry-run-walk-init must be a walking model_<N>.pt file: {seed}", EXIT_INPUT)
        elif self.dry and not self.plumbing:
            raise PipelineError(
                "a dry run needs --dry-run-walk-init WALKER (a walking checkpoint trained at this HOME, so the "
                "v12 source probe can pass) or --dry-run-plumbing (from scratch at any HOME: the source probe and "
                "the robot checks are run and recorded, not enforced)", EXIT_INPUT)
        if self.dry:
            push_url = self.git(self.robot, "remote", "get-url", "--push", "origin", check=False)
            if re.match(r"^(git@|ssh://|https?://)", push_url):
                raise PipelineError(
                    "--dry-run installs DRYRUN policies and commits them locally: --robot-repo must be "
                    f"a scratch clone whose origin push URL is a local path (got {push_url!r})", EXIT_INPUT)

    def open_state(self) -> None:
        """Load the HOME identity, pick and lock the state dir."""

        yaml_sha256 = sha256(HOME_YAML) if HOME_YAML.is_file() else None
        home = json.loads(self.capture(
            [*UV, "python", "scripts/home_pipeline/home_check.py"], timeout=600).stdout or "{}")
        self.yaml_sha256 = yaml_sha256
        self.verify_home_yaml()  # not edited while it was being checked
        if not home.get("tag"):
            raise PipelineError("cannot load config/home_pose.yaml: "
                                + "; ".join(home.get("reasons") or ["no output"]), EXIT_INPUT)
        self.home = home
        prefix = self.args.run_prefix or (("dryrun_" if self.dry else "home_") + home["joint_hash"])
        if self.state_dir is None:
            self.state_dir = REPO / "artifacts" / "home_pipeline" / (
                f"{'dryrun_' if self.dry else ''}{home['tag']}_{home['joint_hash']}")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(self.state_dir / ".lock", "w")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PipelineError(f"another retrain_all_for_home.py holds {self.state_dir}", EXIT_BUSY)
        self.owns_state = True
        state_path = self.state_dir / "state.json"
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}
        for pid, child in list(self.state.get("children", {}).items()):
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except OSError:
                cmdline = ""
            if cmdline and child["cmd"][-1] in cmdline:
                raise PipelineError(f"job {child['name']} (pid {pid}) of an earlier run is still running; "
                                    "wait for it or stop it, then rerun", EXIT_BUSY)
        self.state["children"] = {}
        # A resumed run is no longer stopped: keep the old stop as history
        # so --status does not show it as the current state.
        if self.state.get("stopped"):
            self.state.setdefault("previous_stops", []).append(self.state.pop("stopped"))
        identity ={"joint_hash": home["joint_hash"], "label": home["label"], "tag": home["tag"]}
        if self.state.get("home_identity") and self.state["home_identity"] != identity:
            raise PipelineError(
                f"{self.state_dir} belongs to HOME {self.state['home_identity']}, but "
                f"config/home_pose.yaml is now {identity}; use another --state-dir", EXIT_INPUT)
        if self.state.get("dry_run") is not None and self.state["dry_run"] != self.dry:
            raise PipelineError(f"{self.state_dir} is a {'dry' if self.state['dry_run'] else 'real'} "
                                "run state; use another --state-dir", EXIT_INPUT)
        self.state.update(home_identity=identity, dry_run=self.dry, prefix=self.state.get("prefix", prefix))
        self.prefix = self.state["prefix"]
        self.state.setdefault("created", f"{datetime.now():%F %T}")
        self.save()
        self.log(f"==== retrain_all_for_home {'DRY RUN ' if self.dry else ''}HOME tag {home['tag']} "
                 f"(hash {home['joint_hash']}), prefix {self.prefix}, state {self.state_dir}")

    def prepare_branches(self) -> None:
        a = self.args
        # Training repository: code must be committed (only the HOME yaml may be edited).
        current = self.git(REPO, "rev-parse", "--abbrev-ref", "HEAD")
        branch = a.training_branch or current
        if branch != current:
            exists = self.git(REPO, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
            if exists:
                raise PipelineError(f"the training repo is on {current}; switch it to {branch} first "
                                    "(the pipeline does not switch the code under training)", EXIT_INPUT)
            self.git(REPO, "switch", "-c", branch)
            self.log(f"training repo: created branch {branch} from {current}")
        # Untracked (non-ignored) files count too: the 9999 corner rescue
        # (ensure_committed_tree, train_microban_teleop_v12_corner_rescue.sh)
        # trains only from a tree whose sole change is the HOME yaml, so a real
        # run refuses anything else here instead of a day of GPU later.
        dirty = sorted({line[3:] for line in self.git(REPO, "status", "--porcelain", "--untracked-files=all")
                        .splitlines() if line.strip()})
        others = [p for p in dirty if p not in ("config/home_pose.yaml",)]
        if others and not (a.allow_dirty and self.dry):
            raise PipelineError("the training repo has uncommitted or untracked files besides "
                                f"config/home_pose.yaml: {others} (commit, remove or .gitignore them; "
                                "--allow-dirty waives this only for --dry-run)", EXIT_INPUT)
        self.put("training", {"branch": branch, "head": self.git(REPO, "rev-parse", "HEAD"),
                              "dirty": dirty})
        # Robot repository.
        current = self.git(self.robot, "rev-parse", "--abbrev-ref", "HEAD")
        if current != a.robot_branch:
            exists = self.git(self.robot, "rev-parse", "--verify", "--quiet",
                              f"refs/heads/{a.robot_branch}", check=False)
            if exists:
                if self.git(self.robot, "status", "--porcelain", "--untracked-files=no"):
                    raise PipelineError(f"robot repo is on {current} with local changes; cannot switch "
                                        f"to {a.robot_branch}", EXIT_INPUT)
                self.git(self.robot, "switch", a.robot_branch)
            else:
                self.git(self.robot, "switch", "-c", a.robot_branch)
            self.log(f"robot repo: now on branch {a.robot_branch} (was {current})")
        self.put("robot", "branch", a.robot_branch)

    # ----------------------------------------------------------- step 1
    def step_training_suite(self) -> None:
        """Run the training test suite at this HOME (CPU) before any GPU time.

        A HOME branch is this code plus its config/home_pose.yaml, and its
        suite must pass at that HOME: a failure that is not one of the code's
        pre-existing ones (KNOWN_TEST_FAILURES, the same at every HOME) stops
        the run before days of training.  The verdict is cached per code
        commit and YAML, and recorded in the release record.
        """

        if self.args.skip_training_suite:
            self.log("training suite: skipped (--skip-training-suite, dry run only)")
            return
        key = {"head": self.git(REPO, "rev-parse", "HEAD"), "yaml_sha256": self.yaml_sha256}
        done = self.get("training_suite")
        if done and done.get("key") == key and done.get("passed"):
            self.log(f"training suite: passed earlier at this commit and HOME ({done.get('summary')})")
            return
        self.log("training suite: running tests/ at this HOME on the CPU (a few minutes)")
        output = self.state_dir / "training_suite.out"
        rc = self.run("training_suite", [*UV, "--with", "pytest", "python", "-m", "pytest", "-q",
                                         "-p", "no:cacheprovider", "tests"], "cpu",
                      env={"CUDA_VISIBLE_DEVICES": ""}, allow_fail=True, stdout_path=output)
        text = output.read_text(errors="replace") if output.is_file() else ""
        failed = sorted({m[1] for m in re.finditer(r"^(?:FAILED|ERROR) (\S+)", text, re.M)})
        summary = next((line.strip("= ") for line in reversed(text.splitlines())
                        if re.search(r"\d+ (passed|failed)", line)), None)
        known = set(known_test_failures())
        new = [node for node in failed if node not in known]
        record = {"key": key, "rc": rc, "summary": summary, "failed": failed, "new_failures": new,
                  "passed": bool(summary) and not new and rc in (0, 1)}
        self.put("training_suite", record)
        if summary is None or rc not in (0, 1):
            raise PipelineError(f"the training suite did not complete (rc={rc}, see {output})")
        if new:
            raise PipelineError(
                f"the training suite fails at this HOME beyond the known failures ({summary}): "
                + ", ".join(new[:20]) + (" ..." if len(new) > 20 else "")
                + f" (see {output}); fix them before retraining (the branch must be green at its HOME)",
                EXIT_INPUT)
        fixed = sorted(known - set(failed))
        self.log(f"training suite: {summary}; no failure outside the {len(known)} known ones"
                 + (f" ({len(fixed)} known failures now pass)" if fixed else ""))

    def step_home(self) -> None:
        self.log("[1/6] HOME check")
        home = self.home
        for warning in home.get("warnings", []):
            self.log(f"WARNING HOME: {warning}")
        loader, ground = home.get("loader_contact", {}), home.get("ground_contact", {})
        self.log(
            f"HOME COM - sole centre {loader.get('com_offset_x_m', float('nan')) * 1e3:+.4f} mm, heel/toe "
            f"margin {loader.get('heel_margin_m', 0) * 1e3:.2f}/{loader.get('toe_margin_m', 0) * 1e3:.2f} mm "
            f"({loader.get('contact_corner_count')} corners); floor contact "
            f"{ground.get('heel_margin_m', 0) * 1e3:.2f}/{ground.get('toe_margin_m', 0) * 1e3:.2f} mm "
            f"({ground.get('contact_corner_count')} corners); root z {home.get('root_pos_m', [0, 0, 0])[2]}")
        if home.get("status") == "refuse":
            raise PipelineError("HOME refused: " + "; ".join(home["reasons"]))
        balance = self.capture([*UV, "python", "config/balance_home_pose.py", "--check",
                                "--no-training-check"], timeout=900)
        canonical = balance.returncode == 0
        if not canonical:
            self.log("WARNING HOME: config/balance_home_pose.py --check: the yaml is not the canonical "
                     "balanced solution (run it with --write to re-centre the COM); continuing")
        show = self.capture([*UV, "python", "config/home_pose_tool.py", "show"], timeout=1200)
        if show.returncode != 0:
            raise PipelineError("config/home_pose_tool.py show refused the HOME: "
                                + (show.stderr or show.stdout).strip()[-800:])
        training_line = json.loads(show.stdout).get("training_line")
        self.log(f"training line: {training_line}")
        before = self.state_dir / "robot_home_pose.before.yaml"
        robot_yaml = self.robot / "config" / "home_pose.yaml"
        if not before.exists():
            before.write_text(robot_yaml.read_text() if robot_yaml.exists() else "")
        check = self.capture([*UV, "python", "config/home_pose_tool.py", "write-robot",
                              "--microban-repo", str(self.robot), "--check"], timeout=1200)
        if check.returncode != 0:
            self.capture([*UV, "python", "config/home_pose_tool.py", "write-robot",
                          "--microban-repo", str(self.robot)], timeout=1200, check=True)
            self.log(f"wrote {robot_yaml}")
        else:
            self.log(f"{robot_yaml} is up to date")
        self.put("home", {k: home.get(k) for k in (
            "tag", "joint_hash", "label", "name", "trunk_pitch_deg", "root_pos_m", "status", "warnings",
            "loader_contact", "ground_contact", "head_standing_height_m", "feet_lateral_m")}
            | {"balance_tool_canonical": canonical, "training_line": training_line,
               "yaml_sha256": self.yaml_sha256})

    def check_walk_init(self) -> None:
        """--dry-run-walk-init must be a walking checkpoint trained at this HOME (its stamp)."""

        seed = Path(self.args.dry_run_walk_init).resolve()
        out = self.capture([*UV, "python", "-m", "mjlab_microban.tasks.microban_velocity_runner", str(seed)],
                           timeout=900)
        if out.returncode != 0:
            raise PipelineError(
                f"--dry-run-walk-init {seed} was not trained at the HOME of config/home_pose.yaml under the "
                "walking contract (its run's params/env.yaml or its microban_walk_home_pose stamp differs, as "
                "train_microban_teleop_v12.sh start checks): use a walker of this HOME, or run from scratch "
                "with --dry-run-plumbing\n" + (out.stderr or out.stdout).strip()[-600:], EXIT_INPUT)
        self.log(f"dry run: warm-start walker {seed} was trained at this HOME")

    # -------------------------------------------------- generic segments
    def train_segment(self, exp: Path, label: str, task: str, iterations: int,
                      load: tuple[str, int] | None, extra: list[str], need_mib: int,
                      envs: int, poll=None) -> tuple[Path, int]:
        """Train ``label`` to its end checkpoint (resuming a partial run); return (run dir, end)."""

        start = load[1] if load else 0
        end = start + iterations - 1
        for run in reversed(runs_with_suffix(exp, label)):
            if end in checkpoints(run):
                self.log(f"skip {label}: {run.name}/model_{end}.pt exists")
                return run, end
        partial = [(i, run) for run in runs_with_suffix(exp, label) for i in checkpoints(run)
                   if start < i < end]
        if partial:
            k, run = max(partial)
            load, remaining = (run.name, k), end - k + 1
            self.log(f"{label}: resuming the interrupted run {run.name} at model_{k} ({remaining} left)")
        else:
            remaining = iterations
        cmd = [*UV, "train", task, "--env.scene.num-envs", str(envs), "--agent.logger", "tensorboard",
               "--agent.max-iterations", str(remaining), "--agent.run-name", label, *extra]
        if load:
            cmd += ["--agent.resume", "True", "--agent.load-run", f"^{re.escape(load[0])}$",
                    "--agent.load-checkpoint", f"^model_{load[1]}[.]pt$"]
        self.gpu_job(need_mib, label, cmd, "train", poll=poll)
        for run in reversed(runs_with_suffix(exp, label)):
            if end in checkpoints(run):
                return run, end
        raise PipelineError(f"{label}: model_{end}.pt missing after training")

    # ----------------------------------------------------------- step 2
    def walk_probe(self, ckpt: Path, rep: int, *, wait: bool = True) -> dict | None:
        """Probe one walking checkpoint (9 x 300); None if ``wait`` is False and the GPU is full."""

        out = self.state_dir / "walk_probe" / f"{ckpt.parent.name}__{ckpt.stem}_r{rep}.json"
        out.parent.mkdir(exist_ok=True)
        digest = sha256(ckpt)
        meta = out.with_suffix(".sha256")
        if not (out.exists() and meta.exists() and meta.read_text().strip() == digest):
            out.unlink(missing_ok=True)
            rc = self.gpu_job(self.args.probe_gpu_mib, f"probe_{ckpt.parent.name[-24:]}_{ckpt.stem}_r{rep}",
                              [*UV, "python", "-m", "mjlab_microban.scripts.probe_legacy_actor_in_teleop_env",
                               "--checkpoint", str(ckpt), "--expected-sha256", digest, "--output", str(out),
                               "--force"], "probe", allow_fail=True, wait=wait)
            if rc is None:
                return None
            if not out.exists():
                raise PipelineError(f"walking probe of {ckpt} wrote no receipt")
            meta.write_text(digest + "\n")
        try:
            verdict = probe_verdict(out)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise PipelineError(f"walking probe receipt {out} is malformed: {error!r}") from error
        verdict["checkpoint"] = str(ckpt)
        return verdict

    def walk_probe_due(self, label_runs: list[Path], finals: set[int]) -> list[Path]:
        every, minimum = self.args.probe_every, self.args.probe_min
        due = []
        for run in label_runs:
            for i in checkpoints(run):
                if (i % every == 0 and i >= minimum) or i in finals:
                    due.append(run / f"model_{i}.pt")
        return due

    def step_walk(self) -> None:
        self.log("[2/6] walking")
        a = self.args
        selected = self.get("walk", "selected")
        if selected and Path(selected["path"]).is_file() and sha256(Path(selected["path"])) == selected["sha256"]:
            self.log(f"skip walking: selected {selected['path']} ({selected['sha256'][:12]})")
            return
        base_label, cont_label = f"{self.prefix}_walk", f"{self.prefix}_walk_cont"
        load = None
        if a.dry_run_walk_init:
            seed = Path(a.dry_run_walk_init).resolve()
            k = model_index(seed.name)
            seed_dir = WALK_EXP / f"{self.prefix}_walk_seed"
            seed_dir.mkdir(parents=True, exist_ok=True)
            if not (seed_dir / seed.name).exists():
                shutil.copy2(seed, seed_dir / seed.name)
            load = (seed_dir.name, k)
            self.log(f"dry run: walking starts from {seed} (iteration {k})")
        probed: dict[str, list[dict]] = {}
        failures: dict[str, int] = {}
        finals: set[int] = set()
        training = {"running": True}
        noted = {"gpu_full": False}

        def poll() -> None:
            runs = runs_with_suffix(WALK_EXP, base_label) + runs_with_suffix(WALK_EXP, cont_label)
            for ckpt in self.walk_probe_due(runs, finals):
                newest = max(checkpoints(ckpt.parent))
                if str(ckpt) in probed or (ckpt.stat().st_mtime > time.time() - 20 and
                                           model_index(ckpt.name) == newest):
                    continue  # skip the file being written right now
                if failures.get(str(ckpt), 0) >= 3:
                    continue
                try:
                    # Never wait for GPU memory here: this runs inside the walking
                    # job's loop, whose stall check must keep running.
                    v = self.walk_probe(ckpt, 0, wait=not training["running"])
                except Exception as error:  # noqa: BLE001 - a bad probe must not kill training
                    if isinstance(error, PipelineError) and (error.secondary or error.code != EXIT_FAILED):
                        raise
                    failures[str(ckpt)] = failures.get(str(ckpt), 0) + 1
                    self.log(f"WARNING probe of {ckpt} failed ({failures[str(ckpt)]}/3): "
                             f"{str(error).splitlines()[0] if str(error) else repr(error)}")
                    continue
                if v is None:
                    if not noted["gpu_full"]:
                        self.log("walking probes deferred: not enough free GPU memory now (retried every poll)")
                        noted["gpu_full"] = True
                    return
                noted["gpu_full"] = False
                probed[str(ckpt)] = [v]
                self.log(f"probe {ckpt.parent.name}/{ckpt.name}: {'PASS' if v['ok'] else 'FAIL'} "
                         f"worst margin {v['worst_margin']:+.4f} ({v['worst']}) falls={v['falls']}")
                self.put("walk", "probes", str(ckpt), v)

        envs = a.dry_envs if self.dry else 4096
        base_iters = a.walk_iterations if not self.dry else 3
        cont_iters = a.walk_cont_iterations if not self.dry else 3
        base_run, base_end = self.train_segment(WALK_EXP, base_label, "Mjlab-Velocity-Microban", base_iters,
                                                load, [], a.train_gpu_mib, envs, poll=poll)
        finals.add(base_end)
        cont_run, cont_end = self.train_segment(WALK_EXP, cont_label, "Mjlab-Velocity-Microban", cont_iters,
                                                (base_run.name, base_end), [], a.train_gpu_mib, envs, poll=poll)
        finals.add(cont_end)
        self.put("walk", "runs", {"base": str(base_run), "cont": str(cont_run), "base_end": base_end,
                                  "cont_end": cont_end})
        training["running"] = False
        poll()
        for path, verdicts in (self.get("walk", "probes") or {}).items():
            probed.setdefault(path, [verdicts])
        # Repeat the best candidates.
        ranked = sorted(probed, key=lambda k: -probed[k][0]["worst_margin"])
        candidates = [k for k in ranked if probed[k][0]["ok"]][: a.select_top] or ranked[: a.select_top]
        for key in candidates:
            for rep in range(1, a.select_repeats):
                probed[key].append(self.walk_probe(Path(key), rep))
        self.put("walk", "candidates", {k: probed[k] for k in candidates})
        best, row, fallback = select_walker({k: probed[k] for k in candidates})
        table = ["checkpoint                                   repeats pass worst-case margin"]
        for key in candidates:
            vs = probed[key]
            table.append(f"{Path(key).parent.name}/{Path(key).name}  {len(vs)}  {sum(v['ok'] for v in vs)}"
                         f"/{len(vs)}  {min(v['worst_margin'] for v in vs):+.4f}")
        (self.state_dir / "walk_selection.txt").write_text("\n".join(table) + "\n")
        for line in table:
            self.log("  " + line)
        if fallback:
            self.log(f"WARNING walking: no candidate passed every repeated probe; FALLBACK to the best "
                     f"worst-case margin {row['worst_case_margin']:+.4f}: {best}")
        else:
            self.log(f"walking selected {best}: {row['passes']}/{row['repeats']} probes pass, worst-case "
                     f"margin {row['worst_case_margin']:+.4f}")
        self.install_walker(best, row, fallback)

    def install_walker(self, best: str, row: dict, fallback: bool) -> None:
        """Copy the selected walking checkpoint to checkpoints/<prefix>_walk[_<run>]/ and record it."""

        src = Path(best)
        dest_dir = REPO / "checkpoints" / f"{self.prefix}_walk"
        dest = dest_dir / src.name
        if dest.exists() and sha256(dest) != sha256(src):
            # Same iteration number from the other walking run: keep both apart.
            dest_dir = REPO / "checkpoints" / f"{self.prefix}_walk_{src.parent.name[-24:]}"
            dest = dest_dir / src.name
            if dest.exists() and sha256(dest) != sha256(src):
                raise PipelineError(f"{dest} exists with other bytes; remove it or use another --run-prefix")
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        if not (dest_dir / "params").exists() and (src.parent / "params").is_dir():
            shutil.copytree(src.parent / "params", dest_dir / "params")
        self.put("walk", "selected", {"path": str(dest), "relative": str(dest.relative_to(REPO)),
                                      "sha256": sha256(dest), "from": str(src), "fallback": fallback,
                                      **row})

    def reselect_walker(self, reason: str) -> bool:
        """Drop the selected walker after a failed v12 start probe; select the next candidate.

        Returns False when no candidate is left.  A dry run cycles through the
        candidates again (DRY_START_ATTEMPTS starts in all) instead.
        """

        selected = self.get("walk", "selected")
        rejected = list(self.get("walk", "rejected") or [])
        rejected.append({"from": selected["from"], "sha256": selected["sha256"], "reason": reason})
        self.put("walk", "rejected", rejected)
        excluded = {r["from"] for r in rejected}
        every = self.get("walk", "candidates") or {}
        candidates = {k: v for k, v in every.items() if k not in excluded}
        if not candidates and self.dry and len(rejected) < DRY_START_ATTEMPTS:
            candidates = {k: v for k, v in every.items() if k != selected["from"]} or dict(every)
        if not candidates:
            return False
        best, row, fallback = select_walker(candidates)
        self.log(f"walking: {Path(selected['from']).name} failed the v12 start probe ({reason}); "
                 f"trying the next candidate {best} (worst-case margin {row['worst_case_margin']:+.4f}"
                 f"{', FALLBACK' if fallback else ''})")
        self.install_walker(best, row, fallback)
        return True

    # ----------------------------------------------------------- step 3
    def getup_evaluate(self, label: str, ckpt: Path) -> dict:
        a = self.args
        out_dir = self.state_dir / "getup_eval"
        out_dir.mkdir(exist_ok=True)
        results = {}
        digest = sha256(ckpt)
        for name, extra in GETUP_EVALS:
            out = out_dir / f"{label}_{name}.json"
            if out.exists():
                data = json.loads(out.read_text())
                if data.get("checkpoint_sha256") == digest:
                    results[name] = data
                    continue
            stdout = out_dir / f"{label}_{name}.stdout"
            cmd = [*UV, "python", "scripts/home_pipeline/getup_eval.py", extra[0], extra[1], str(ckpt),
                   *extra[2:]]
            if self.dry:
                cmd += ["--envs", "16", "--steps", "150"]
            self.gpu_job(a.probe_gpu_mib, f"eval_{label}_{name}", cmd, "eval", stdout_path=stdout)
            lines = [l for l in stdout.read_text().splitlines() if l.startswith("RESULT ")]
            if not lines:
                raise PipelineError(f"get-up evaluation {name} of {ckpt} printed no RESULT")
            data = json.loads(lines[-1][7:])
            data["checkpoint_sha256"] = digest
            out.write_text(json.dumps(data, indent=1, sort_keys=True))
            results[name] = data
        stand = [results[n] for n in ("stand_s11", "stand_s5", "push_s11")]
        push = results["push_s11"]
        summary = {
            "checkpoint": str(ckpt),
            "min_fallen_standing_fraction": min(r["fallen_standing_fraction"] for r in stand),
            "push_fall_fraction": push.get("push_fell_within_3s", 0) / max(1, push.get("push_standing_before", 0)),
            "standing_joint_abs_vel_rad_s": max(r["standing_joint_abs_vel_rad_s"] for r in stand),
            "standing_targets_beyond_1p57": max(r["standing_targets_beyond_1p57"] for r in stand),
            "posture_standing_fraction": results["posture_s11"]["standing_fraction"],
            "final_tilt_deg": max(r["final_tilt_deg"] for r in stand),
        }
        self.log(f"get-up {label}: fallen-start standing >= {summary['min_fallen_standing_fraction']:.2f}, "
                 f"push falls {summary['push_fall_fraction']:.2f}, |vel| "
                 f"{summary['standing_joint_abs_vel_rad_s']:.2f} rad/s, |target|>1.57 "
                 f"{summary['standing_targets_beyond_1p57']:.2f}, posture "
                 f"{summary['posture_standing_fraction']:.2f}, tilt {summary['final_tilt_deg']:.1f} deg")
        return summary

    def step_getup(self) -> None:
        self.log("[3/6] get-up")
        a = self.args
        envs = a.dry_envs if self.dry else 4096
        prev: tuple[Path, int] | None = None
        for suffix, task, iterations, entropy, reset_std in GETUP_STAGES:
            label = f"{self.prefix}_{suffix}"
            iterations = 3 if self.dry else iterations
            load = None
            if prev is not None:
                prev_run, prev_end = prev
                load = (prev_run.name, prev_end)
                if reset_std:
                    std_run = GETUP_EXP / f"{prev_run.name}_std05"
                    if not (std_run / f"model_{prev_end}.pt").exists():
                        self.run(f"reset_std_{suffix}", [
                            *UV, "python", "-m", "mjlab_microban.scripts.reset_getup_action_std",
                            "--checkpoint", str(prev_run / f"model_{prev_end}.pt"), "--out-run", std_run.name],
                            "cpu")
                    load = (std_run.name, prev_end)
            extra = ["--agent.algorithm.entropy-coef", entropy] if entropy else []
            run, end = self.train_segment(GETUP_EXP, label, task, iterations, load, extra,
                                          a.train_gpu_mib, envs)
            prev = (run, end)
            summary = self.getup_evaluate(label, run / f"model_{end}.pt")
            self.put("getup", "stages", suffix, {"run": str(run), "end": end, **summary})
        run, end = prev
        final = self.get("getup", "stages", GETUP_STAGES[-1][0])
        g = GETUP_GATE
        failures = []
        if final["min_fallen_standing_fraction"] < g["min_fallen_standing_fraction"]:
            failures.append(f"fallen-start standing {final['min_fallen_standing_fraction']:.2f} < "
                            f"{g['min_fallen_standing_fraction']}")
        if final["push_fall_fraction"] > g["max_push_fall_fraction"]:
            failures.append(f"push falls {final['push_fall_fraction']:.2f} > {g['max_push_fall_fraction']}")
        if final["standing_joint_abs_vel_rad_s"] > g["max_standing_joint_abs_vel_rad_s"]:
            failures.append(f"standing tremble {final['standing_joint_abs_vel_rad_s']:.2f} rad/s > "
                            f"{g['max_standing_joint_abs_vel_rad_s']}")
        if final["posture_standing_fraction"] < g["min_posture_standing_fraction"]:
            failures.append(f"posture standing {final['posture_standing_fraction']:.2f} < "
                            f"{g['min_posture_standing_fraction']}")
        self.put("getup", "final", {"run": str(run), "end": end, "checkpoint": str(run / f"model_{end}.pt"),
                                    "gate_failures": failures})
        if failures:
            if self.dry:
                self.log(f"get-up gate (plumbing mode, not enforced): {failures}")
            else:
                raise PipelineError(f"get-up final gate failed for {run.name}/model_{end}.pt: {failures}")
        else:
            self.log(f"get-up gate PASS: {run.name}/model_{end}.pt")

    # ----------------------------------------------------------- step 4
    def stage_tool(self, *args: str) -> subprocess.CompletedProcess:
        return self.capture([*UV, "python", "-m", "mjlab_microban.scripts.teleop_v12_stage", *args],
                            timeout=1800)

    @staticmethod
    def last_line(out: subprocess.CompletedProcess) -> str:
        lines = out.stdout.strip().splitlines()
        return lines[-1] if out.returncode == 0 and lines else "invalid"

    @staticmethod
    def failed_checks(path: Path) -> list[str]:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return ["<missing>"]
        return sorted(k for k, v in data.get("checks", {}).items() if v is not True)

    def gate_ok(self, run: str, end: int) -> bool:
        gate = GATE_ROOT / f"{run}_model_{end}_gate.json"
        ckpt = V12_EXP / run / f"model_{end}.pt"
        return gate.is_file() and self.stage_tool("validate", str(gate), str(ckpt)).returncode == 0

    def latest_v12(self, suffix: str) -> str | None:
        runs = runs_with_suffix(V12_EXP, suffix)
        return runs[-1].name if runs else None

    def v12_train(self, seg: str, prev: str | None, *, lift_to: int | None = None,
                  seed: int | None = None) -> None:
        a = self.args
        envs = ["--num-envs", str(a.dry_envs)] if self.dry else []
        if prev is None:
            source = self.get("walk", "selected")
            cmd = ["scripts/train_microban_teleop_v12.sh", "start", "--source", source["relative"],
                   "--source-sha256", source["sha256"], "--agent.run-name", seg, *envs]
            if self.dry:
                cmd += ["--max-updates", "2"]
            probe = PROBE_ROOT / f"velocity_{source['sha256'][:16]}_teleop83_raw_9x300.json"
            forced = self.get("pico", "forced_probe")
            if forced and forced.get("source_sha256") == source["sha256"] and Path(forced["path"]).is_file():
                # Plumbing mode: the walker's measured probe failed (kept in
                # forced["from"]); the chain bootstraps from the forced copy.
                cmd += ["--dry-run-probe-receipt", forced["path"]]
                self.gpu_job(a.pico_gpu_mib, seg, cmd, "train")
                return
            started = time.time()
            seen = {"done": False}

            def watch() -> None:
                if seen["done"] or not probe.exists() or probe.stat().st_mtime < started:
                    return
                time.sleep(5)
                seen["done"] = True
                try:
                    v = probe_verdict(probe)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    raise SourceProbeFailed(f"malformed v12 source probe {probe}: {error!r}") from error
                self.log(f"v12 source probe {probe.name}: {'PASS' if v['ok'] else 'FAIL'} worst margin "
                         f"{v['worst_margin']:+.4f} ({v['worst']}) below={v['below']}")
                if not v["ok"]:
                    raise SourceProbeFailed(
                        f"fresh v12 source probe below the locomotion gate minimums (worst margin "
                        f"{v['worst_margin']:+.4f} {v['worst']})")

            try:
                self.gpu_job(a.pico_gpu_mib, seg, cmd, "train", poll=watch)
            except PipelineError as raised:
                error = raised
                if not isinstance(raised, SourceProbeFailed):
                    # The trainer may exit on its own probe failure before watch() saw it.
                    try:
                        fresh = probe.exists() and probe.stat().st_mtime >= started
                        bad = fresh and not probe_verdict(probe)["ok"]
                    except (OSError, ValueError, KeyError, TypeError):
                        bad = False
                    if not bad or raised.secondary:
                        raise
                    error = SourceProbeFailed(f"v12 start refused the fresh source probe {probe.name}")
                if self.plumbing:
                    self.force_source_probe(source, probe, str(error))
                    return self.v12_train(seg, None)
                # Repeat probes vary by about +-0.01; the next-best candidate
                # (already probed during selection) gets its own fresh probe.
                if not self.reselect_walker(str(error)):
                    raise PipelineError(f"{error}; no other walking candidate is left "
                                        f"(see {self.state_dir / 'walk_selection.txt'})") from error
                return self.v12_train(seg, None)
            return
        if lift_to is not None:
            parent_dir = V12_EXP / prev
            last = max(checkpoints(parent_dir))
            # Named by the lifted checkpoint's bytes, so a different parent (a
            # canary retry, another 9999 route) never reuses a stale lift.
            lifted = f"{self.prefix}_v12_lift_{lift_to}_{sha256(parent_dir / f'model_{last}.pt')[:12]}"
            if not (V12_EXP / lifted / f"model_{lift_to}.pt").exists():
                shutil.rmtree(V12_EXP / lifted, ignore_errors=True)
                self.run(f"lift_{lift_to}", [*UV, "python", "scripts/home_pipeline/dry_run_tools.py",
                                             "lift-clock", str(parent_dir / f"model_{last}.pt"),
                                             str(V12_EXP / lifted), str(lift_to)], "cpu")
            prev = lifted
        cmd = ["scripts/train_microban_teleop_v12.sh", "resume", prev,
               "--agent.run-name", seg, *envs]
        if seed is not None and seed != V12_TRAIN_SEED:
            cmd += ["--seed", str(seed)]
        if self.dry:
            cmd += ["--dry-run-skip-gate", "--max-updates", "3"]
        self.gpu_job(a.pico_gpu_mib, seg, cmd, "train")

    def force_source_probe(self, source: dict, probe: Path, reason: str) -> None:
        """Plumbing mode: bootstrap the chain from a forced-pass copy of the failed probe."""

        if not probe.is_file():
            raise PipelineError(f"plumbing mode: {reason}, and no probe receipt {probe} to force")
        forced = PROBE_ROOT / f"DRYRUN_FORCED_PASS_{probe.name}"
        self.run("force_probe_DRYRUN", [*UV, "python", "scripts/home_pipeline/dry_run_tools.py", "force-probe",
                                        str(probe), str(forced)], "cpu")
        self.put("pico", "forced_probe", {"path": str(forced), "from": str(probe), "reason": reason,
                                          "source_sha256": source["sha256"], "measured": probe_verdict(probe)})
        self.log(f"plumbing mode: the v12 source probe failed ({reason}); NOT ENFORCED, the chain "
                 f"bootstraps from the forced-pass receipt {forced.name} (not deployable)")

    def simulated_failure(self, key: str) -> bool:
        """--dry-run-simulate-failures: does this gate count as failed?

        Keys: ``canary:<segment>`` (its first gate), ``9999:a<attempt>``,
        ``rescue:a<attempt>r<index>_<mix>``, ``15000:a<attempt>`` and
        ``frescue:a<attempt>r<index>_<mix>``.  --dry-run-simulate-9999 picks the
        9999 route: ``retrain`` fails the 9999 gate and every corner rescue of
        attempt 1 and passes the retrained attempt 2 (the route of the 2026-10
        forward-lean chain); ``rescue`` passes the second rescue mix; ``stop``
        fails everything so the boundary stops.  --dry-run-simulate-15000 picks
        the 15000 route the same way (``pass``, the default, fails nothing).
        """

        if not (self.dry and self.args.dry_run_simulate_failures):
            return False
        mode = self.args.dry_run_simulate_9999
        if key.startswith("canary:"):
            done = set(self.get("pico", "simulated_failures") or [])
            if key in done:
                return False
            self.put("pico", "simulated_failures", sorted(done | {key}))
            return True
        if key == "9999:a1":
            return True
        if key.startswith("9999:a"):
            return mode == "stop"
        if key.startswith("rescue:"):
            index = int(re.match(r"rescue:a\d+r(\d+)_", key).group(1))
            return mode != "rescue" or index == 1
        final = self.args.dry_run_simulate_15000
        if final == "pass":
            return False
        if key == "15000:a1":
            return True
        if key.startswith("15000:a"):
            return final == "stop"
        if key.startswith("frescue:"):
            index = int(re.match(r"frescue:a\d+r(\d+)_", key).group(1))
            return final != "rescue" or index == 1
        return False

    def v12_gate(self, run: str, end: int) -> tuple[int, list[str], list[str]]:
        """Evaluate ``run``/model_<end> with the three stage evaluators; return (rc, tracking, other).

        The evaluators of scripts/evaluate_microban_teleop_v12_stage.sh run
        one by one, each to the end: that script (set -e) stops at the first
        evaluator that returns 1 for a "fail" verdict, so a gate failing on
        tracking never wrote its ONNX report, which is indistinguishable from
        an evaluator that died.  Here a report that is written is a verdict
        (pass or fail) and only a missing one is a crash.  Reports of an
        earlier evaluation of this checkpoint are removed first, so none of
        them stands in for this one.  A real run that passes all three
        publishes the hash-bound resume gate exactly as the stage script does
        (teleop_v12_stage create); a dry run records the reports only
        (plumbing mode: nothing is enforced).
        """

        prefix = GATE_ROOT / f"{run}_model_{end}"
        ckpt = V12_EXP / run / f"model_{end}.pt"
        digest = sha256(ckpt)
        GATE_ROOT.mkdir(parents=True, exist_ok=True)
        reports = {name: Path(f"{prefix}_{name}.json") for name in ("9x300", "tracking", "onnx")}
        for path in (*reports.values(), Path(f"{prefix}_gate.json")):
            path.unlink(missing_ok=True)
        steps = [
            ("loco", [*UV, "python", "-m", "mjlab_microban.scripts.evaluate_teleop_v12_checkpoint",
                      str(ckpt), "--expected-sha256", digest, "--output", str(reports["9x300"]), "--force"]),
            ("tracking", [*UV, "python", "-m", "mjlab_microban.scripts.evaluate_teleop_v12_tracking",
                          str(ckpt), "--expected-sha256", digest, "--output", str(reports["tracking"]),
                          "--force"]),
            ("onnx", [*UV_ONNX, "python", "-m", "mjlab_microban.scripts.teleop_v12_onnx_gate", str(ckpt),
                      "--expected-sha256", digest, "--onnx", f"{prefix}.onnx", "--output",
                      str(reports["onnx"]), "--force"]),
        ]
        rcs = {}
        for name, cmd in steps:
            rcs[name] = self.gpu_job(self.args.probe_gpu_mib, f"gate_{end}_{name}", cmd, "gate",
                                     allow_fail=True)
        rc = 0 if all(v == 0 for v in rcs.values()) else 1
        if rc == 0 and not self.dry:
            rc = self.run(f"gate_{end}_create", [
                *UV, "python", "-m", "mjlab_microban.scripts.teleop_v12_stage", "create", str(ckpt),
                str(reports["9x300"]), str(reports["tracking"]), str(reports["onnx"]),
                f"{prefix}_gate.json", "--force"], "cpu", allow_fail=True)
        trk = self.failed_checks(reports["tracking"])
        other = self.failed_checks(reports["9x300"]) + self.failed_checks(reports["onnx"])
        if rc != 0 and not trk and not other:
            # Every check passed but the stage tool refused to publish the gate
            # (or an evaluator returned non-zero with an all-pass report).
            other = ["stage_gate_create"]
        try:
            profile = json.loads(Path(f"{prefix}_tracking.json").read_text()).get("profile")
        except (OSError, ValueError):
            profile = None
        self.log(f"gate {end} {run}: rc={rc} profile={profile} tracking failed={trk} other failed={other}"
                 + ("  (plumbing mode: not enforced)" if self.dry else ""))
        self.put("pico", "gates", f"{run}_model_{end}", {"rc": rc, "tracking_failed": trk,
                                                         "other_failed": other, "profile": profile})
        return rc, trk, other

    def judged_gate(self, run: str, end: int, key: str) -> tuple[bool, list[str], list[str]]:
        """Gate ``run``/model_<end>; return (passed, tracking failed, other failed).

        A real run is judged by the gate itself (an existing validating gate
        is reused; a recorded failure of the same checkpoint bytes is not
        re-evaluated).  A dry run evaluates and records every gate it can, but
        only --dry-run-simulate-failures decides the route (plumbing mode).
        """

        ckpt = V12_EXP / run / f"model_{end}.pt"
        digest = sha256(ckpt)
        record = self.get("pico", "gates", f"{run}_model_{end}")
        if not self.dry and self.gate_ok(run, end):
            self.log(f"gate {end}: the existing gate validates for {run}")
            rc, trk, other = 0, [], []
        elif record and record.get("checkpoint_sha256") == digest and not self.gate_crashed(record):
            rc, trk, other = record["rc"], record["tracking_failed"], record["other_failed"]
            self.log(f"gate {end} {run}: recorded rc={rc} tracking failed={trk} other failed={other}")
        elif self.dry and self.args.dry_run_gates == "final" and end != 14999:
            rc, trk, other = 0, [], []
        else:
            rc, trk, other = self.v12_gate(run, end)
            crashed = self.gate_crashed({"tracking_failed": trk, "other_failed": other})
            if crashed and not self.dry:
                # An evaluator that died (e.g. OOM on the shared GPU) wrote no
                # report: that is not a verdict on the checkpoint, so it is not
                # cached and does not count as a failed gate; a rerun
                # re-evaluates it.
                raise PipelineError(f"the v12 gate evaluation of {run}/model_{end}.pt wrote no report (rc={rc}); "
                                    "not a gate verdict: rerun to re-evaluate it")
            if not crashed:
                # Only a real verdict (every report written) is cached.
                self.put("pico", "gates", f"{run}_model_{end}", "checkpoint_sha256", digest)
        if self.simulated_failure(key):
            self.log(f"dry run: simulating an accuracy-only failure of gate {end} ({key}, {run})")
            return False, ["hand_tracking_rms"], []
        if self.dry:
            return True, trk, other
        return rc == 0, trk, other

    @staticmethod
    def gate_crashed(record: dict) -> bool:
        """A gate record without a verdict: some report was never written."""

        return "<missing>" in (record.get("tracking_failed") or []) + (record.get("other_failed") or [])

    def check_recipe(self, run: str, end: int) -> None:
        ckpt = V12_EXP / run / f"model_{end}.pt"
        kind = self.last_line(self.stage_tool("checkpoint-recipe", str(ckpt), "--shell"))
        if kind != "hand_pose_release":
            raise PipelineError(f"{ckpt} recipe kind {kind} != hand_pose_release")

    def ensure_committed_tree(self, boundary: int = 9999) -> None:
        """The corner-rescue launcher trains only from a clean, committed tree.

        The HOME edit is the only change a run allows, so commit it now (the
        release commit of step 6 follows on the same branch); anything else
        stops the run.
        """

        status = self.git(REPO, "status", "--porcelain", "--untracked-files=all")
        dirty = sorted({line[3:] for line in status.splitlines() if line.strip()})
        if not dirty:
            return
        if dirty != ["config/home_pose.yaml"] or sha256(HOME_YAML) != self.yaml_sha256:
            raise PipelineError(f"{boundary} gate failed; the rescue needs a clean, committed training tree "
                                f"but these files are modified or untracked: {dirty}")
        h = self.home
        commit = self.commit(REPO, ["config/home_pose.yaml"], (
            f"Set HOME {h['tag']} for retraining every policy\n\n"
            f"HOME {h['tag']} (hash {h['joint_hash']}, trunk {h.get('trunk_pitch_deg')} deg).  Committed by "
            f"scripts/retrain_all_for_home.py before a pose-release rescue of the v12 {boundary}\n"
            "boundary, whose launcher trains only from a clean committed tree.  The release\n"
            "commit with the retrained policies follows on this branch."))
        self.log(f"committed config/home_pose.yaml ({(commit or '')[:12]}) for the {boundary} rescue")
        self.put("training", "home_yaml_commit", commit)

    def rescue_parent(self, run: str, attempt: int) -> Path | None:
        """model_9900 of a 7100->10000 attempt (a dry run lifts its model_9999 to 9900)."""

        parent = V12_EXP / run / "model_9900.pt"
        if parent.is_file() or not self.dry:
            return parent if parent.is_file() else None
        lifted = V12_EXP / f"{self.prefix}_v12_dry_parent9900_a{attempt}"
        if not (lifted / "model_9900.pt").is_file():
            shutil.rmtree(lifted, ignore_errors=True)
            self.run(f"lift_parent9900_a{attempt}", [*UV, "python", "scripts/home_pipeline/dry_run_tools.py",
                                                     "lift-clock", str(V12_EXP / run / "model_9999.pt"),
                                                     str(lifted), "9900"], "cpu")
        return lifted / "model_9900.pt"

    def corner_rescues(self, run: str, attempt: int) -> str | None:
        """Pose-release model_9900 corner rescues of a failed 9999 gate; return the passing run.

        The mixes of --pr-corner-rescue-mixes are tried in order (a repeated
        mix is a new run that differs only by GPU nondeterminism: the corner
        rescue launcher takes no seed; the default has no repeat), each gated at 9999 like any
        stage checkpoint.  A parent the rescue validator refuses (the 9999
        gate did not fail on hand accuracy only) skips the rescues.  A dry run
        records the validator's verdict and stands in for the 2048-env rescue
        with dry_run_tools.py stamp-corner-rescue.
        """

        parent = self.rescue_parent(run, attempt)
        if parent is None:
            self.log(f"corner rescue: {run} has no model_9900; no rescue for attempt {attempt}")
            self.put("pico", "b9999", "attempts", str(attempt), "rescue_parent", None)
            return None
        report = self.state_dir / "pico" / f"{parent.parent.name}_model_9900_strict_tracking.json"
        report.parent.mkdir(exist_ok=True)
        if report.exists() and report_profile(report) != V12_RESCUE_PARENT_PROFILE:
            # A report cached by an older pipeline under another profile:
            # evaluate it again under the one HMD/hand profile.
            self.log(f"corner rescue: re-evaluating {report.name} under {V12_RESCUE_PARENT_PROFILE} "
                     f"(cached profile {report_profile(report)})")
            report.unlink()
        if not report.exists():
            rc = self.gpu_job(self.args.probe_gpu_mib, f"rescue_parent_report_a{attempt}", [
                *UV, "python", "-m", "mjlab_microban.scripts.evaluate_teleop_v12_tracking", str(parent),
                "--expected-sha256", sha256(parent), "--profile", V12_RESCUE_PARENT_PROFILE,
                "--output", str(report), "--force"], "gate",
                allow_fail=True)
            if not report.exists() and not self.dry:
                # The evaluator died without a report (not a verdict on the
                # parent); a rerun evaluates it again instead of skipping rescues.
                raise PipelineError(f"the strict tracking evaluation of the rescue parent {parent} wrote no "
                                    f"report (rc={rc}); rerun to re-evaluate it")
        for index, mix in enumerate(self.args.pr_corner_rescue_mixes, 1):
            key = f"a{attempt}r{index}_{mix}"
            done = self.get("pico", "b9999", "rescues", key)
            if done and done.get("passed") is False:
                self.log(f"corner rescue {key}: failed earlier ({done.get('run')}), next")
                continue
            check = self.capture([*UV, "python", "-m", "mjlab_microban.scripts.teleop_v12_corner_rescue",
                                  "validate-parent", str(parent), str(report), "--hand-pose-release"],
                                 timeout=1800, env={"MICROBAN_V12_PR_CORNER_RESCUE_MIX": mix})
            verdict = "accepted" if check.returncode == 0 else (
                (check.stderr or check.stdout).strip().splitlines() or ["refused"])[-1][-400:]
            if check.returncode != 0:
                if not self.dry:
                    self.log(f"corner rescue: the validator refuses {parent.parent.name}/model_9900: {verdict}")
                    self.put("pico", "b9999", "attempts", str(attempt), "rescue_parent",
                             {"path": str(parent), "refused": verdict})
                    return None
                self.log(f"dry run: the rescue validator refuses the dry parent ({verdict}); plumbing mode, "
                         "not enforced")
            seg = f"{self.prefix}_v12_pr_rescue_{key}_9901_to10000"
            rescue = self.latest_v12(seg)
            if not (rescue and (V12_EXP / rescue / "model_9999.pt").is_file()):
                self.log(f"corner rescue {key}: mix {mix} from {parent.parent.name}/model_9900")
                if self.dry:
                    rescue = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{seg}"
                    self.run(f"rescue_{key}_DRYRUN", [
                        *UV, "python", "scripts/home_pipeline/dry_run_tools.py", "stamp-corner-rescue",
                        str(parent), str(report), str(V12_EXP / run / "model_9999.pt"), str(V12_EXP / rescue),
                        mix], "cpu")
                else:
                    self.ensure_committed_tree()
                    self.gpu_job(self.args.pico_gpu_mib, f"rescue_{key}", [
                        "scripts/train_microban_teleop_v12_corner_rescue.sh", str(parent), str(report),
                        "--hand-pose-release", "--mix", mix, "--agent.run-name", seg], "train")
                rescue = self.latest_v12(seg)
                if not (rescue and (V12_EXP / rescue / "model_9999.pt").is_file()):
                    raise PipelineError(f"corner rescue {key} wrote no model_9999.pt")
            self.check_recipe(rescue, 9999)
            passed, trk, other = self.judged_gate(rescue, 9999, f"rescue:{key}")
            self.put("pico", "b9999", "rescues", key, {"run": rescue, "mix": mix, "attempt": attempt,
                                                       "parent": str(parent), "validator": verdict,
                                                       "passed": passed, "tracking_failed": trk,
                                                       "other_failed": other})
            self.log(f"corner rescue {key} {rescue}: gate 9999 {'PASS' if passed else 'FAIL'} "
                     f"(tracking {trk}, other {other})")
            if passed:
                return rescue
        return None

    def attempt_9999(self, parent_7099: str, attempt: int) -> str:
        """Retrain 7100->10000 from the gated model_7099 as a new attempt; return its run."""

        seg = f"{self.prefix}_v12_7100_to10000_a{attempt}"
        current = self.latest_v12(seg)
        if current and (V12_EXP / current / "model_9999.pt").is_file():
            self.log(f"skip training {seg}: {current}/model_9999.pt exists")
            return current
        seed = V12_TRAIN_SEED + attempt - 1
        self.log(f"9999 boundary: retrain 7100->10000 from the gated {parent_7099}/model_7099 "
                 f"(attempt {attempt}, training seed {seed})")
        self.put("pico", "b9999", "attempts", str(attempt), "seed", seed)
        self.v12_train(seg, parent_7099, lift_to=9996 if self.dry else None, seed=seed)
        current = self.latest_v12(seg)
        if not (current and (V12_EXP / current / "model_9999.pt").is_file()):
            raise PipelineError(f"{seg}: model_9999.pt missing after training")
        return current

    def boundary_9999(self, parent_7099: str, first: str) -> str:
        """Pass the 10000 boundary automatically; return the run the chain continues from.

        The escalation of the 2026-10 forward-lean chain: the 9999 gate of
        the 7100->10000 segment; if it fails, the pose-release model_9900
        corner rescues (every mix of --pr-corner-rescue-mixes); if they all
        fail, retrain 7100->10000 from the gated model_7099 as a new attempt
        (gate, then rescues again), up to --v12-9999-attempts; then stop.
        """

        passed = self.get("pico", "b9999", "passed")
        if passed and (V12_EXP / passed["run"] / "model_9999.pt").is_file() and \
                sha256(V12_EXP / passed["run"] / "model_9999.pt") == passed["sha256"]:
            self.log(f"skip the 9999 boundary: {passed['run']} ({passed['kind']}, attempt {passed['attempt']})")
            return passed["run"]
        failures = []
        for attempt in range(1, self.args.v12_9999_attempts + 1):
            run = first if attempt == 1 else self.attempt_9999(parent_7099, attempt)
            self.put("pico", "b9999", "attempts", str(attempt), "run", run)
            self.check_recipe(run, 9999)
            ok, trk, other = self.judged_gate(run, 9999, f"9999:a{attempt}")
            kind, chosen = "segment", run
            if not ok:
                self.log(f"gate 9999 failed for attempt {attempt} {run} (tracking {trk}, other {other}); "
                         "trying the pose-release corner rescues")
                failures.append(f"attempt {attempt} {run}: tracking {trk}, other {other}")
                chosen = self.corner_rescues(run, attempt)
                kind = "corner_rescue"
            if chosen:
                record = {"run": chosen, "sha256": sha256(V12_EXP / chosen / "model_9999.pt"),
                          "attempt": attempt, "kind": kind}
                self.put("pico", "b9999", "passed", record)
                self.log(f"9999 boundary passed: {chosen} ({kind}, attempt {attempt})")
                return chosen
            self.log(f"9999 boundary: attempt {attempt} and its corner rescues failed")
        raise PipelineError(f"v12 9999 boundary failed after {self.args.v12_9999_attempts} attempt(s), each with "
                            f"the corner rescues {','.join(self.args.pr_corner_rescue_mixes)}: "
                            + "; ".join(failures) + f" (see {self.state_dir / 'STATUS.log'})")

    def final_rescue_parent(self, run: str, attempt: int) -> Path | None:
        """model_14900 of a 10100->15000 run (a dry run lifts its model_14999 to 14900)."""

        parent = V12_EXP / run / "model_14900.pt"
        if parent.is_file() or not self.dry:
            return parent if parent.is_file() else None
        source = V12_EXP / run / "model_14999.pt"
        lifted = V12_EXP / f"{self.prefix}_v12_dry_parent14900_{sha256(source)[:12]}"
        if not (lifted / "model_14900.pt").is_file():
            shutil.rmtree(lifted, ignore_errors=True)
            self.run(f"lift_parent14900_a{attempt}", [*UV, "python", "scripts/home_pipeline/dry_run_tools.py",
                                                      "lift-clock", str(source), str(lifted), "14900"], "cpu")
        return lifted / "model_14900.pt"

    def final_rescues(self, run: str, attempt: int, seed: int, trk: list[str], other: list[str]) -> str | None:
        """Pose-release final-scenario rescues of a failed 14999 gate; return the passing run.

        forward-lean-v2 fc1c313..cd0ea78: a 99-update replay (model_14900 ->
        model_14999) of the failed gate's scenarios.  Only a gate whose
        locomotion and ONNX checks passed and whose tracking failures are all
        rescuable is rescued.  Each mix of --pr-final-rescue-mixes is
        validated against the failed gate (a mix that does not replay every
        failed scenario is refused and skipped), trained with this attempt's
        seed and gated at 14999.  A dry run records the validator's verdict
        and stands in for the 2048-env rescue with dry_run_tools.py
        stamp-final-rescue.
        """

        if not self.dry and (other or not trk or not set(trk) <= V12_FINAL_RESCUABLE_CHECKS):
            self.log(f"final rescue: gate 14999 of {run} is not rescuable (tracking {trk}, other {other}; "
                     f"rescuable: tracking only, within {sorted(V12_FINAL_RESCUABLE_CHECKS)})")
            self.put("pico", "b15000", "attempts", str(attempt), "rescue_parent", None)
            return None
        parent = self.final_rescue_parent(run, attempt)
        if parent is None:
            self.log(f"final rescue: {run} has no model_14900; no rescue for attempt {attempt}")
            self.put("pico", "b15000", "attempts", str(attempt), "rescue_parent", None)
            return None
        report = GATE_ROOT / f"{run}_model_14999_tracking.json"
        for index, mix in enumerate(self.args.pr_final_rescue_mixes, 1):
            key = f"a{attempt}r{index}_{mix}"
            done = self.get("pico", "b15000", "rescues", key)
            if done and done.get("passed") is False:
                self.log(f"final rescue {key}: failed or refused earlier ({done.get('run')}), next")
                continue
            check = self.capture([*UV, "python", "-m", "mjlab_microban.scripts.teleop_v12_hand_pose_release_final_rescue",
                                  "validate-parent", str(parent), str(report), "--mix", mix, "--seed", str(seed)],
                                 timeout=1800)
            verdict = "accepted" if check.returncode == 0 else (
                (check.stderr or check.stdout).strip().splitlines() or ["refused"])[-1][-400:]
            if check.returncode != 0:
                if not self.dry:
                    self.log(f"final rescue {key}: the validator refuses {parent.parent.name}/model_14900 with "
                             f"mix {mix}: {verdict}")
                    self.put("pico", "b15000", "rescues", key, {"run": None, "mix": mix, "attempt": attempt,
                                                                "parent": str(parent), "validator": verdict,
                                                                "passed": False})
                    continue
                self.log(f"dry run: the final rescue validator refuses the dry parent ({verdict}); plumbing "
                         "mode, not enforced")
            seg = f"{self.prefix}_v12_pr_final_rescue_{key}_14901_to15000"
            rescue = self.latest_v12(seg)
            if not (rescue and (V12_EXP / rescue / "model_14999.pt").is_file()):
                self.log(f"final rescue {key}: mix {mix} seed {seed} from {parent.parent.name}/model_14900")
                if self.dry:
                    rescue = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{seg}"
                    self.run(f"final_rescue_{key}_DRYRUN", [
                        *UV, "python", "scripts/home_pipeline/dry_run_tools.py", "stamp-final-rescue",
                        str(parent), str(report), str(V12_EXP / run / "model_14999.pt"), str(V12_EXP / rescue),
                        mix, str(seed)], "cpu")
                else:
                    self.ensure_committed_tree(15000)
                    self.gpu_job(self.args.pico_gpu_mib, f"final_rescue_{key}", [
                        "scripts/train_microban_teleop_v12_hand_pose_release_final_rescue.sh", str(parent),
                        str(report), "--mix", mix, "--seed", str(seed), "--agent.run-name", seg], "train")
                rescue = self.latest_v12(seg)
                if not (rescue and (V12_EXP / rescue / "model_14999.pt").is_file()):
                    raise PipelineError(f"final rescue {key} wrote no model_14999.pt")
            self.check_recipe(rescue, 14999)
            passed, rtrk, rother = self.judged_gate(rescue, 14999, f"frescue:{key}")
            self.put("pico", "b15000", "rescues", key, {"run": rescue, "mix": mix, "attempt": attempt,
                                                        "seed": seed, "parent": str(parent), "validator": verdict,
                                                        "passed": passed, "tracking_failed": rtrk,
                                                        "other_failed": rother})
            self.log(f"final rescue {key} {rescue}: gate 14999 {'PASS' if passed else 'FAIL'} "
                     f"(tracking {rtrk}, other {rother})")
            if passed:
                return rescue
        return None

    def attempt_15000(self, parent_10099: str, attempt: int) -> str:
        """Retrain 10100->15000 from the gated model_10099 with the next seed; return its run."""

        seg = f"{self.prefix}_v12_10100_to15000_a{attempt}"
        current = self.latest_v12(seg)
        if current and (V12_EXP / current / "model_14999.pt").is_file():
            self.log(f"skip training {seg}: {current}/model_14999.pt exists")
            return current
        seed = V12_TRAIN_SEED + attempt - 1
        self.log(f"15000 boundary: retrain 10100->15000 from the gated {parent_10099}/model_10099 "
                 f"(attempt {attempt}, training seed {seed})")
        self.v12_train(seg, parent_10099, lift_to=14996 if self.dry else None, seed=seed)
        current = self.latest_v12(seg)
        if not (current and (V12_EXP / current / "model_14999.pt").is_file()):
            raise PipelineError(f"{seg}: model_14999.pt missing after training")
        return current

    def boundary_15000(self, parent_10099: str, first: str) -> str:
        """Pass the 15000 boundary automatically; return the final run.

        The 14999 gate of the 10100->15000 segment; if it fails, the
        pose-release final-scenario rescues (every mix of
        --pr-final-rescue-mixes, with the attempt's training seed); if they
        all fail or the gate is not rescuable, retrain 10100->15000 from the
        gated model_10099 with the next training seed as a new attempt (gate,
        then rescues again), up to --v12-15000-attempts; then stop.
        """

        passed = self.get("pico", "b15000", "passed")
        if passed and (V12_EXP / passed["run"] / "model_14999.pt").is_file() and \
                sha256(V12_EXP / passed["run"] / "model_14999.pt") == passed["sha256"]:
            self.log(f"skip the 15000 boundary: {passed['run']} ({passed['kind']}, attempt {passed['attempt']})")
            return passed["run"]
        failures = []
        for attempt in range(1, self.args.v12_15000_attempts + 1):
            seed = V12_TRAIN_SEED + attempt - 1
            run = first if attempt == 1 else self.attempt_15000(parent_10099, attempt)
            self.put("pico", "b15000", "attempts", str(attempt), "run", run)
            self.put("pico", "b15000", "attempts", str(attempt), "seed", seed)
            self.check_recipe(run, 14999)
            ok, trk, other = self.judged_gate(run, 14999, f"15000:a{attempt}")
            kind, chosen = "segment", run
            if not ok:
                self.log(f"gate 14999 failed for attempt {attempt} {run} (tracking {trk}, other {other}); "
                         "trying the pose-release final-scenario rescues")
                failures.append(f"attempt {attempt} {run}: tracking {trk}, other {other}")
                chosen = self.final_rescues(run, attempt, seed, trk, other)
                kind = "final_rescue"
            if chosen:
                record = {"run": chosen, "sha256": sha256(V12_EXP / chosen / "model_14999.pt"),
                          "attempt": attempt, "kind": kind}
                self.put("pico", "b15000", "passed", record)
                self.log(f"15000 boundary passed: {chosen} ({kind}, attempt {attempt})")
                return chosen
            self.log(f"15000 boundary: attempt {attempt} and its final rescues failed")
        raise PipelineError(f"v12 15000 boundary failed after {self.args.v12_15000_attempts} attempt(s), each "
                            f"with the final rescues {','.join(self.args.pr_final_rescue_mixes)}: "
                            + "; ".join(failures) + f" (see {self.state_dir / 'STATUS.log'})")

    def step_pico(self) -> None:
        self.log("[4/6] PICO v12 pose-release chain")
        final = self.get("pico", "final")
        if final and Path(final["checkpoint"]).is_file() and sha256(Path(final["checkpoint"])) == final["sha256"]:
            self.log(f"skip PICO: final {final['checkpoint']}")
            return
        prev: str | None = None
        for index, (suffix, real_end, canary) in enumerate(V12_SEGMENTS):
            seg = f"{self.prefix}_v12_{suffix}"
            # A dry run lifts every segment (canaries included) to 3 updates
            # before its real end, so every gate and the packager's 10000 /
            # 10100 boundary record see the real clocks.
            end = real_end
            current = self.latest_v12(seg)
            if current and (V12_EXP / current / f"model_{end}.pt").is_file():
                self.log(f"skip training {seg}: {current}/model_{end}.pt exists")
            else:
                # A recorded canary retry means the failed first run is
                # complete and this segment's newest run is the retry, which
                # was interrupted after it created its run directory: train
                # the retry again with the retry's seed, not the default one.
                retry_of = self.get("pico", "canary_retry", suffix) if canary else None
                seed = V12_TRAIN_SEED + 1 if retry_of else None
                if retry_of:
                    self.log(f"canary {end}: resuming the interrupted retry of {retry_of} with training seed "
                             f"{seed} (partial run {current} kept)")
                if self.dry and index == 0:
                    start_seg = f"{self.prefix}_v12_start"
                    if not self.latest_v12(start_seg):
                        self.v12_train(start_seg, None)
                    self.v12_train(seg, self.latest_v12(start_seg), lift_to=end - 3, seed=seed)
                elif self.dry:
                    self.v12_train(seg, prev, lift_to=end - 3, seed=seed)
                else:
                    self.v12_train(seg, prev, seed=seed)
                current = self.latest_v12(seg)
            ckpt = V12_EXP / current / f"model_{end}.pt"
            if not ckpt.is_file():
                raise PipelineError(f"{ckpt} missing after training")
            self.check_recipe(current, end)
            if real_end == 9999:
                prev = self.boundary_9999(prev, current)
                self.put("pico", "segments", suffix, prev)
                continue
            if real_end == 14999:
                prev = self.boundary_15000(prev, current)
                self.put("pico", "segments", suffix, prev)
                continue
            ok, trk, other = self.judged_gate(current, end, f"canary:{suffix}" if canary else f"stage:{suffix}")
            retried = self.get("pico", "canary_retry", suffix)
            if not ok and canary and (
                    (not retried and trk and set(trk) <= V12_ACCURACY_CHECKS and not other)
                    or retried == current):
                if retried == current:
                    # Resumed after the retry was interrupted before it saved
                    # model_<end>: the newest run is still the failed first one.
                    self.log(f"canary {end}: resuming the interrupted retry of {current}")
                else:
                    self.put("pico", "canary_retry", suffix, current)
                    self.log(f"canary {end} failed on accuracy only {trk}; retraining it once with "
                             f"training seed {V12_TRAIN_SEED + 1} (failed run {current} kept)")
                failed_run = current
                self.v12_train(seg, prev, lift_to=end - 3 if self.dry else None, seed=V12_TRAIN_SEED + 1)
                current = self.latest_v12(seg)
                if current == failed_run or not (V12_EXP / current / f"model_{end}.pt").is_file():
                    raise PipelineError(f"canary {end} retry of {failed_run} wrote no model_{end}.pt")
                self.check_recipe(current, end)
                ok, trk, other = self.judged_gate(current, end, f"canary:{suffix}")
            if not ok:
                raise PipelineError(f"v12 gate {end} failed for {current}: tracking {trk}, other {other}")
            self.put("pico", "segments", suffix, current)
            prev = current
        ckpt = V12_EXP / prev / "model_14999.pt"
        gate = GATE_ROOT / f"{prev}_model_14999_gate.json"
        profile = None
        if not self.dry:
            out = self.stage_tool("validate", str(gate), str(ckpt))
            if out.returncode != 0:
                raise PipelineError(f"final v12 gate does not validate: {out.stderr.strip()[-800:]}")
            try:
                profile = json.loads(out.stdout.strip().splitlines()[-1]).get("tracking_profile")
            except (IndexError, ValueError):
                profile = None
            if profile not in V12_FINAL_PROFILES:
                raise PipelineError(f"final v12 gate profile {profile} is not a final profile")
        self.put("pico", "final", {"run": prev, "checkpoint": str(ckpt), "sha256": sha256(ckpt),
                                   "gate": str(gate), "profile": profile})
        self.log(f"PICO chain done: {prev}/model_14999.pt profile={profile}")

    # ----------------------------------------------------------- step 5
    def provenance(self, ckpt: Path) -> dict:
        code = ("import json,sys,torch;i=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['infos'];"
                "p=i['legacy_velocity_actor_bootstrap_v12'];print(json.dumps({'source':p['source'],"
                "'probe':p['probe']}))")
        out = self.capture([*UV, "python", "-c", code, str(ckpt)], timeout=600, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def robot_env(self) -> dict[str, str]:
        """Environment of the robot validator / tests (a dry run's own package is accepted there only)."""

        env = {"PYTHONPATH": str(self.robot / "src"), "CUDA_VISIBLE_DEVICES": ""}
        if self.dry:
            env[DRY_RUN_POLICY_ALLOW_ENV] = "1"
        return env

    def robot_python(self, args: list[str], *, timeout: float = 900) -> subprocess.CompletedProcess:
        return self.capture(["uv", "run", "--project", str(self.robot), "--locked", "python", *args],
                            cwd=self.robot, env=self.robot_env(), timeout=timeout)

    def install(self, src: Path, rel: str) -> bool:
        dest = self.robot / rel
        if dest.exists() and sha256(dest) == sha256(src):
            return False
        tmp = dest.with_name(f".{dest.name}.new")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        self.log(f"installed {rel} ({sha256(dest)[:12]})")
        return True

    def edit_robot(self, rel: str, editor) -> bool:
        path = self.robot / rel
        text = path.read_text()
        new, changed = editor(text)
        if changed and new != text:
            path.write_text(new)
            return True
        return False

    def step_export_install(self) -> None:
        self.log("[5/6] export, install, package, validate, test")
        out = self.state_dir / "export"
        out.mkdir(exist_ok=True)
        walk_src = Path(self.get("walk", "selected", "path"))
        getup_src = Path(self.get("getup", "final", "checkpoint"))
        pico_ckpt = Path(self.get("pico", "final", "checkpoint"))
        exports = {}
        for name, module, ckpt in (("walk", "export_walk_onnx", walk_src),
                                   ("getup", "export_getup_onnx", getup_src)):
            onnx = out / f"{name}.onnx"
            digest = sha256(ckpt)
            if self.get("export", name, "checkpoint_sha256") != digest or not onnx.exists() or \
                    sha256(onnx) != self.get("export", name, "onnx_sha256"):
                self.gpu_job(self.args.probe_gpu_mib, f"export_{name}",
                             [*UV, "python", "-m", f"mjlab_microban.scripts.{module}", "--checkpoint",
                              str(ckpt), "--output", str(onnx), "--replace"], "cpu")
                self.put("export", name, {"checkpoint": str(ckpt), "checkpoint_sha256": digest,
                                          "onnx": str(onnx), "onnx_sha256": sha256(onnx)})
            exports[name] = onnx
        # Robot HOME yaml (step 1 wrote it; re-check it is still current).
        check = self.capture([*UV, "python", "config/home_pose_tool.py", "write-robot",
                              "--microban-repo", str(self.robot), "--check"], timeout=1200)
        if check.returncode != 0:
            self.capture([*UV, "python", "config/home_pose_tool.py", "write-robot", "--microban-repo",
                          str(self.robot)], timeout=1200, check=True)
            self.log("rewrote the robot config/home_pose.yaml")
        self.install(exports["walk"], "src/agents/walk.onnx")
        self.install(exports["getup"], "src/agents/getup.onnx")
        # The synthetic walk fixture carries the HOME stamp; regenerate it (deterministic).
        self.capture([*UV, "--project", str(REPO), "python", "tests/fixtures/make_walk_policy_fixture.py"],
                     cwd=self.robot, env={"PYTHONPATH": str(self.robot / "src")}, timeout=900, check=True)
        # Run pins: frozen walking source + probe of PICO v12, and the walk fallback.
        prov = self.provenance(pico_ckpt)
        src, probe = prov["source"], prov["probe"]
        self.put("export", "probe_receipt", probe["path"])
        if src["sha256"] != self.get("walk", "selected", "sha256"):
            raise PipelineError("the PICO checkpoint's walking source is not the selected walker")
        changed = self.edit_robot(robot_pins.PICO_HYBRID, lambda t: robot_pins.set_hex_pin(
            t, "EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256", src["sha256"]))
        changed |= self.edit_robot(robot_pins.PICO_HYBRID, lambda t: robot_pins.set_int_pin(
            t, "EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_ITERATION", src["iteration"]))
        changed |= self.edit_robot(robot_pins.PICO_HYBRID, lambda t: robot_pins.set_hex_pin(
            t, "EXPECTED_V12_LEGACY_PROBE_SHA256", probe["sha256"]))
        changed |= self.edit_robot(robot_pins.PICO_HYBRID, lambda t: robot_pins.set_source_comment(
            t, tag=self.home["tag"], source_path=src["path"], iteration=src["iteration"],
            probe_path=probe["path"]))
        changed |= self.edit_robot("tests/test_pico_hybrid.py", lambda t: robot_pins.set_quoted_values(t, {
            "v12_legacy_source_checkpoint_sha256": src["sha256"],
            "v12_legacy_source_checkpoint_iteration": str(src["iteration"]),
            "v12_legacy_probe_sha256": probe["sha256"]}))
        changed |= self.edit_robot(robot_pins.VALIDATOR, lambda t: robot_pins.set_hex_pin(
            t, "EXPECTED_WALK_FALLBACK_SHA256", sha256(self.robot / "src/agents/walk.onnx")))
        if changed:
            self.log(f"robot pins: source {src['sha256'][:12]} iter {src['iteration']}, probe "
                     f"{probe['sha256'][:12]}, walk.onnx {sha256(self.robot / 'src/agents/walk.onnx')[:12]}")
        # The robot tests derive every HOME-bound value from config/home_pose.yaml
        # (robot home-config); only the reviewed degree table TRAINING_HOME_DEG of
        # tests/test_shared_home.py pins the HOME, so rewrite it (the run pins and
        # PACKAGER_V12_HOME_POSE_JSON of tests/test_pico_hybrid.py follow below).
        new = robot_pins.parse_robot_home_yaml((self.robot / "config/home_pose.yaml").read_text())
        if self.edit_robot("tests/test_shared_home.py",
                           lambda t: robot_pins.set_training_home_deg(t, new["joint_pos_deg"])):
            self.log("updated TRAINING_HOME_DEG in tests/test_shared_home.py")
        # Package PICO against this robot tree, unless the installed package still validates.
        pico_out = out / ("DRYRUN_pico_teleop.onnx" if self.dry else "pico_teleop.onnx")
        receipt = out / "pico_receipt.json"
        installed = self.robot / "src/agents/pico_teleop.onnx"
        valid = self.robot_python(["tools/validate_pico_policy.py", str(installed)], timeout=600)
        packaged_sha = self.get("export", "pico", "onnx_sha256")
        if not (valid.returncode == 0 and packaged_sha and installed.exists() and sha256(installed) == packaged_sha
                and self.get("export", "pico", "checkpoint_sha256") == sha256(pico_ckpt)):
            if self.dry:
                prefix = GATE_ROOT / f"{self.get('pico', 'final', 'run')}_model_14999"
                if not all(Path(f"{prefix}{s}").is_file() for s in ("_9x300.json", "_tracking.json")):
                    # e.g. an evaluator that crashed on a fallen plumbing policy: re-evaluate.
                    self.v12_gate(self.get("pico", "final", "run"), 14999)
                self.run("package_pico_DRYRUN", [*UV_ONNX, "python", "scripts/home_pipeline/dry_run_tools.py",
                                                 "package", str(pico_ckpt), str(prefix),
                                                 str(out / "DRYRUN_gate_model_14999.json"), str(pico_out),
                                                 str(self.robot), *self.dry_boundary_args()], "cpu",
                         env={"CUDA_VISIBLE_DEVICES": ""}, stdout_path=receipt)
            else:
                self.run("package_pico", [*UV_ONNX, "python", "-m",
                                          "mjlab_microban.scripts.export_teleop_v12_deployment",
                                          "--checkpoint", str(pico_ckpt), "--stage-gate",
                                          self.get("pico", "final", "gate"), "--microban-repo", str(self.robot),
                                          *self.boundary_gate_args(), "--output", str(pico_out), "--force"],
                         "cpu", env={"CUDA_VISIBLE_DEVICES": ""}, stdout_path=receipt)
            self.check_receipt(receipt, pico_ckpt, pico_out)
            self.put("export", "pico", {"checkpoint": str(pico_ckpt), "checkpoint_sha256": sha256(pico_ckpt),
                                        "onnx": str(pico_out), "onnx_sha256": sha256(pico_out),
                                        "receipt": str(receipt)})
            self.install(pico_out, "src/agents/pico_teleop.onnx")
        else:
            self.log("skip PICO packaging: the installed pico_teleop.onnx is this checkpoint's package "
                     "and validates")
        # Test pin of the exact HOME JSON the packager writes.
        meta = self.capture([*UV_ONNX, "python", "-c",
                             "import sys,onnxruntime as o;print(o.InferenceSession(sys.argv[1],providers="
                             "['CPUExecutionProvider']).get_modelmeta().custom_metadata_map"
                             "['v12_training_home_pose_json'])", str(installed)],
                            env={"CUDA_VISIBLE_DEVICES": ""}, timeout=600, check=True)
        home_json = meta.stdout.strip().splitlines()[-1]
        if self.edit_robot("tests/test_pico_hybrid.py",
                           lambda t: robot_pins.set_packager_home_json(t, home_json)):
            self.log("updated PACKAGER_V12_HOME_POSE_JSON in tests/test_pico_hybrid.py")
        # Robot validator and test suite.
        valid = self.robot_python(["tools/validate_pico_policy.py", "src/agents/pico_teleop.onnx"], timeout=900)
        try:
            report = json.loads(valid.stdout)
        except ValueError:
            report = {}
        (out / "validate_pico_policy.json").write_text(valid.stdout or valid.stderr)
        self.put("export", "robot_validator", {"rc": valid.returncode, "status": report.get("status")})
        if (valid.returncode != 0 or report.get("status") != "pass") and self.plumbing:
            self.log("plumbing mode: tools/validate_pico_policy.py rejected the installed pico_teleop.onnx "
                     f"(recorded in {out / 'validate_pico_policy.json'}, NOT ENFORCED): "
                     + ((valid.stderr or valid.stdout).strip().splitlines() or [""])[-1][-300:])
        elif valid.returncode != 0 or report.get("status") != "pass":
            raise PipelineError("tools/validate_pico_policy.py rejected the installed pico_teleop.onnx: "
                                + (valid.stderr or valid.stdout).strip()[-1200:])
        expected = {"v12_legacy_source_checkpoint_sha256": src["sha256"],
                    "v12_legacy_probe_sha256": probe["sha256"]}
        for key, value in expected.items():
            if report.get(key) not in (None, value):
                raise PipelineError(f"validator report {key}={report.get(key)} != {value}")
        if report.get("status") == "pass":
            self.log(f"robot validator: pass (walk fallback {report.get('walk_fallback', {}).get('status')})")
        tests = self.capture(["uv", "run", "--project", str(self.robot), "--locked", "--with", "pytest", "python",
                              "-m", "pytest", "-q", "tests"], cwd=self.robot, env=self.robot_env(),
                             timeout=3600)
        (out / "robot_tests.log").write_text(tests.stdout + tests.stderr)
        summary = (tests.stdout.strip().splitlines() or ["(no output)"])[-1]
        if tests.returncode != 0 and self.plumbing:
            failed = [l for l in tests.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))][:30]
            self.log(f"plumbing mode: robot test suite failed ({summary}; NOT ENFORCED, log "
                     f"{out / 'robot_tests.log'}): " + "; ".join(failed[:8]))
            self.put("export", "robot_tests", f"NOT ENFORCED (plumbing): {summary}")
            return
        if tests.returncode != 0:
            failed = [l for l in tests.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))][:30]
            raise PipelineError("robot test suite failed: " + summary + "\n" + "\n".join(failed)
                                + f"\n(full log {out / 'robot_tests.log'}; HOME literals the pipeline could not "
                                "update must be edited by hand, then rerun: completed steps are skipped)")
        self.log(f"robot tests: {summary}")
        self.put("export", "robot_tests", summary)

    def boundary_runs(self) -> list[tuple[str | None, int]]:
        """(run, end) of the 10000 boundary and the 10100 canary the final continued from."""

        canary = self.get("pico", "segments", "10000_to10100") or self.latest_v12(f"{self.prefix}_v12_10000_to10100")
        return [(self.get("pico", "b9999", "passed", "run"), 9999), (canary, 10099)]

    def required_boundary_clocks(self) -> set[int]:
        """Clocks a pose-release package must record (the 10000 boundary and 10100 canary)."""

        return set(self.home.get("v12_required_boundary_gate_clocks", [10_000, 10_100]))

    def boundary_gate_args(self) -> list[str]:
        """--boundary-gate for the 10000 boundary and the 10100 canary of the final lineage.

        The package records which tracking profile judged them, so the robot
        sees it.  The packager refuses a pose-release final without both, so a
        missing or non-validating gate stops the run here.
        """

        required = self.required_boundary_clocks()
        args, missing = [], []
        for run, end in self.boundary_runs():
            gate = GATE_ROOT / f"{run}_model_{end}_gate.json"
            if run and gate.is_file() and self.gate_ok(run, end):
                args += ["--boundary-gate", str(gate)]
            elif end + 1 in required:
                missing.append(f"{end + 1} ({run}/model_{end}: {gate.name} missing or not validating)")
        if missing:
            raise PipelineError("the pose-release package must record its 10000 boundary and 10100 canary gates "
                                "at this HOME, but " + "; ".join(missing))
        return args

    def dry_boundary_args(self) -> list[str]:
        """Dry run: the model_9999 / model_10099 the final continued from, for the dry packager.

        dry_run_tools.py package builds each a forced gate and passes it as
        --boundary-gate, so the packager's real boundary checks run on the
        escalation's result: the checkpoint must be on the final's resume
        chain (lifted dry segments record their parent in params/agent.yaml),
        a stamped corner rescue's marker must be carried by the final, and
        where the HOME requires both clocks a missing one refuses packaging.
        """

        args = []
        for run, end in self.boundary_runs():
            prefix = GATE_ROOT / f"{run}_model_{end}"
            if not run or not all(Path(f"{prefix}{s}").is_file() for s in ("_9x300.json", "_tracking.json",
                                                                           "_onnx.json")):
                self.log(f"dry run: no evaluated {end + 1} gate ({run}) to pass to the packager")
                continue
            self.log(f"dry run: packaging with the {end + 1} gate of {run}/model_{end}"
                     + (f" ({self.get('pico', 'b9999', 'passed', 'kind')})" if end == 9999 else ""))
            args += [str(V12_EXP / run / f"model_{end}.pt"), str(prefix)]
        return args

    def check_receipt(self, receipt: Path, ckpt: Path, onnx: Path) -> None:
        """Bind the packager report to this checkpoint, this ONNX and the robot validator."""

        text = receipt.read_text()
        try:
            report = json.loads(text[text.index("{"):])
        except ValueError as error:
            raise PipelineError(f"PICO packager receipt is not JSON: {receipt}") from error
        runtime = report.get("microban_runtime_validator") or {}
        smoke = runtime.get("onnxruntime_compatibility_smoke") or {}
        checks = {
            "status": report.get("status") == "pass",
            "completed_updates": report.get("completed_updates") == 15000,
            "output_sha256": report.get("output_sha256") == sha256(onnx),
            "checkpoint_sha256": report.get("checkpoint_sha256") == sha256(ckpt),
            "runtime_validator": runtime.get("status") == "pass",
            "ort_smoke": smoke.get("status") == "pass",
        }
        if not all(checks.values()):
            raise PipelineError(f"PICO deployment receipt incomplete: {checks}")
        self.log(f"PICO packaged {onnx.name} sha256 {sha256(onnx)[:12]} (runtime validator pass)")
        for entry in report.get("boundary_stage_gates") or []:
            self.log(f"  boundary gate {entry.get('iteration')} ({entry.get('checkpoint_kind')}, "
                     f"{str(entry.get('checkpoint_sha256'))[:12]}): {entry.get('tracking_profile')}")

    # ----------------------------------------------------------- step 6
    def commit(self, repo: Path, paths: list[str], message: str, *, force_add: list[str] = ()) -> str | None:
        existing = [p for p in paths if (repo / p).exists() or self.git(repo, "ls-files", p)]
        self.git(repo, "add", "--", *existing)
        if force_add:  # gitignored release files, archived like the centered release (5b5a9d0)
            self.git(repo, "add", "-f", "--", *force_add)
        staged = self.capture(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode
        if staged == 0:
            return None
        if self.args.commit_trailer:
            message += "\n\n" + self.args.commit_trailer
        result = subprocess.run(["git", "-C", str(repo), "commit", "-q", "-F", "-"], input=message, text=True,
                                capture_output=True)
        if result.returncode != 0:
            raise PipelineError(f"git commit failed in {repo}: {result.stderr.strip()}")
        return self.git(repo, "rev-parse", "HEAD")

    def push(self, repo: Path, branch: str) -> None:
        if self.dry or self.args.no_push:
            self.log(f"not pushing {repo.name} {branch} ({'dry run' if self.dry else '--no-push'})")
            return
        result = self.capture(["git", "-C", str(repo), "push", "-u", "origin", branch], timeout=600)
        if result.returncode != 0:
            raise PipelineError(f"git push failed for {repo} {branch}: {result.stderr.strip()[-800:]}")
        self.log(f"pushed {repo.name} {branch}")

    def step_commit(self) -> None:
        self.log("[6/6] commit and push")
        h = self.home
        sel, gf, pf = self.get("walk", "selected"), self.get("getup", "final"), self.get("pico", "final")
        exp = self.get("export")
        body = (f"HOME {h['tag']} (hash {h['joint_hash']}, trunk {h['trunk_pitch_deg']} deg, root z "
                f"{h['root_pos_m'][2]}).\n\n"
                f"walk.onnx {exp['walk']['onnx_sha256']} from {sel['relative']} (sha256 {sel['sha256'][:16]}, "
                f"{sel['passes']}/{sel['repeats']} probe passes, worst-case margin {sel['worst_case_margin']:+.4f}"
                f"{', FALLBACK' if sel['fallback'] else ''}).\n"
                f"getup.onnx {exp['getup']['onnx_sha256']} from {Path(gf['run']).name}/model_{gf['end']}.pt.\n"
                f"pico_teleop.onnx {exp['pico']['onnx_sha256']} from {pf['run']}/model_14999.pt "
                f"(gate profile {pf['profile']}).\n"
                f"tools/validate_pico_policy.py passes; robot tests: {exp.get('robot_tests')}.\n"
                f"Generated by mjlab_microban scripts/retrain_all_for_home.py.")
        robot_paths = ["config/home_pose.yaml", "src/agents/walk.onnx", "src/agents/getup.onnx",
                       "src/agents/pico_teleop.onnx", "src/moves/pico_hybrid.py", "tools/validate_pico_policy.py",
                       "tests"]
        title = ("DRY RUN (not deployable): " if self.dry else "") + \
            f"Install walking, get-up and PICO policies retrained at HOME {h['tag']}"
        robot_commit = self.commit(self.robot, robot_paths, f"{title}\n\n{body}")
        robot_commit = robot_commit or self.git(self.robot, "rev-parse", "HEAD")
        self.log(f"robot commit {robot_commit[:12]} on {self.args.robot_branch}")
        self.push(self.robot, self.args.robot_branch)
        record = {
            "home": self.get("home"), "prefix": self.prefix, "walk": sel,
            "walk_runs": self.get("walk", "runs"), "getup": self.get("getup"), "pico": pf,
            "pico_gates": self.get("pico", "gates"), "pico_9999_boundary": self.get("pico", "b9999"),
            "pico_15000_boundary": self.get("pico", "b15000"),
            "pico_forced_probe": self.get("pico", "forced_probe"), "exports": exp,
            "robot": {"repo": str(self.robot), "branch": self.args.robot_branch, "commit": robot_commit},
            "training": self.get("training"), "training_suite": self.get("training_suite"),
            "finished": f"{datetime.now():%F %T}",
        }
        if self.dry:
            archive = self.release_archive()
            record["archived_files"] = {rel_path: sha256(REPO / rel_path) for rel_path in archive}
            self.log(f"dry run: {len(archive)} release files would be archived (not committed)")
            path = self.state_dir / "release_record.json"
            path.write_text(json.dumps(record, indent=1, sort_keys=True, default=str))
            self.log(f"dry run: training repo not committed; record written to {path}")
            self.put("commit", {"robot": robot_commit, "training": None})
            return
        archive = self.release_archive()
        record["archived_files"] = {rel_path: sha256(REPO / rel_path) for rel_path in archive}
        rel = f"config/releases/{h['tag']}.json"
        (REPO / "config" / "releases").mkdir(exist_ok=True)
        (REPO / rel).write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
        training_commit = self.commit(REPO, ["config/home_pose.yaml", rel],
                                      f"Retrain every policy at HOME {h['tag']}\n\n{body}\n\nRobot: "
                                      f"{self.args.robot_branch} {robot_commit}.\nArchived "
                                      f"{len(archive)} release files (checkpoints, gate reports, probe "
                                      "receipt, packaged ONNX).", force_add=archive)
        training_commit = training_commit or self.git(REPO, "rev-parse", "HEAD")
        self.log(f"training commit {training_commit[:12]} on {self.get('training', 'branch')}")
        self.push(REPO, self.get("training", "branch"))
        self.put("commit", {"robot": robot_commit, "training": training_commit})

    def release_archive(self) -> list[str]:
        """Repo-relative release files committed with the training release.

        The same set the centered release archived (5b5a9d0): the final PICO
        checkpoint with its params/git records, its stage-gate reports and
        ONNX, the packaged deployment ONNX and receipt; plus what the robot
        pins by SHA-256 (the selected walking checkpoint and its 9x300 probe
        receipt) and the final get-up checkpoint, so every pin can be
        re-validated from git on another machine.
        """

        files: list[Path] = []
        pf, sel, gf = self.get("pico", "final"), self.get("walk", "selected"), self.get("getup", "final")
        run = pf["run"]
        run_dir = V12_EXP / run
        files.append(run_dir / "model_14999.pt")
        for sub in ("params", "git"):
            if (run_dir / sub).is_dir():
                files += sorted(p for p in (run_dir / sub).iterdir() if p.is_file())
        gate_prefix = f"{run}_model_14999"
        files += sorted(p for p in GATE_ROOT.glob(f"{gate_prefix}*") if p.is_file())
        # A final-scenario rescue resumed from a staged seed (its model_14900,
        # the failed gate report and the parent run's resume record) that its
        # marker binds by SHA-256: archive the seed too.
        resumed = resume_parent(run_dir)
        if resumed is not None and resumed.name == "model_14900.pt":
            files += sorted(p for p in resumed.parent.rglob("*") if p.is_file())
        releases = REPO / "artifacts" / "teleop_v12_releases"
        releases.mkdir(parents=True, exist_ok=True)
        packaged = Path(self.get("export", "pico", "onnx"))
        receipt = Path(self.get("export", "pico", "receipt"))
        for src, name in ((packaged, f"{gate_prefix}.onnx"), (receipt, f"{gate_prefix}_deployment_receipt.json")):
            dest = releases / name
            if src.is_file() and (not dest.exists() or sha256(dest) != sha256(src)):
                shutil.copyfile(src, dest)
            files.append(dest)
        walker = Path(sel["path"])
        files.append(walker)
        if (walker.parent / "params").is_dir():
            files += sorted(p for p in (walker.parent / "params").iterdir() if p.is_file())
        probe = self.get("export", "probe_receipt")  # repo:// path of the PICO chain's provenance
        files.append(REPO / probe[len("repo://"):] if probe and probe.startswith("repo://") else
                     PROBE_ROOT / f"velocity_{sel['sha256'][:16]}_teleop83_raw_9x300.json")
        getup_ckpt = Path(gf["checkpoint"])
        files.append(getup_ckpt)
        if (getup_ckpt.parent / "params").is_dir():
            files += sorted(p for p in (getup_ckpt.parent / "params").iterdir() if p.is_file())
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise PipelineError(f"release files missing, cannot archive the release: {missing}")
        return sorted({str(p.resolve().relative_to(REPO)) for p in files})

    # ------------------------------------------------------------- flow
    def threaded(self, name: str, fn, errors: list) -> threading.Thread:
        def body() -> None:
            try:
                fn()
            except BaseException as error:  # noqa: BLE001 - reported by the main thread
                errors.append((name, error))
                self.abort.set()

        thread = threading.Thread(target=body, name=name, daemon=True)
        thread.start()
        return thread

    def execute(self) -> int:
        self.preflight()
        self.open_state()
        self.prepare_branches()
        self.step_home()
        self.step_training_suite()
        if self.dry and self.args.dry_run_walk_init:
            self.check_walk_init()
        errors: list = []
        walk = self.threaded("walk", self.step_walk, errors)
        getup = None
        if not self.args.sequential:
            time.sleep(5)
            getup = self.threaded("getup", self.step_getup, errors)
        walk.join()
        if not errors:
            pico = self.threaded("pico", self.step_pico, errors)
            pico.join()
        if getup is None and not errors:
            getup = self.threaded("getup", self.step_getup, errors)
        if getup is not None:
            getup.join()
        if errors:
            first = next((e for e in errors if not getattr(e[1], "secondary", False)), errors[0])
            for name, error in errors:
                if error is not first[1]:
                    self.log(f"({name}: {error})")
            raise first[1]
        try:
            self.step_export_install()
        except PipelineError as error:
            if error.secondary:
                raise
            raise PipelineError(
                f"{error}\nThe robot tree {self.robot} is left half-installed (new walk/getup ONNX, run pins "
                "and HOME literals, maybe the old pico_teleop.onnx, so its PICO validator can fail): rerun the "
                "same command to resume step 5, or revert it with git -C "
                f"{self.robot} checkout -- . (nothing there is committed yet).", error.code) from error
        self.step_commit()
        self.log("==== ALL STEPS DONE")
        return 0


def print_status(state_dir: Path) -> int:
    state_path = state_dir / "state.json"
    if not state_path.exists():
        print(f"no state in {state_dir}")
        return 1
    state = json.loads(state_path.read_text())
    print(json.dumps({k: state.get(k) for k in ("home_identity", "prefix", "dry_run", "created", "commit",
                                                "stopped", "quarantined")}, indent=1))
    if state.get("previous_stops"):
        print(f"earlier stops (resumed since): {len(state['previous_stops'])}, last: "
              f"{state['previous_stops'][-1].get('at')} code {state['previous_stops'][-1].get('code')}")
    walk = state.get("walk", {})
    if walk.get("selected"):
        print("walk selected:", walk["selected"]["path"], "fallback" if walk["selected"]["fallback"] else "")
    for stage, s in (state.get("getup", {}).get("stages") or {}).items():
        print(f"getup {stage}: {Path(s['run']).name}/model_{s['end']} standing>={s['min_fallen_standing_fraction']:.2f}")
    for gate, g in (state.get("pico", {}).get("gates") or {}).items():
        print(f"gate {gate}: rc={g['rc']} {g['profile']} failed={g['tracking_failed'] + g['other_failed']}")
    print("--- STATUS.log (last 25 lines)")
    lines = (state_dir / "STATUS.log").read_text().splitlines() if (state_dir / "STATUS.log").exists() else []
    print("\n".join(lines[-25:]))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--robot-repo", help="microban checkout to install into (e.g. ../microban)")
    p.add_argument("--robot-branch", help="robot branch to commit on (created from HEAD if missing)")
    p.add_argument("--training-branch", help="training branch to commit on (default: the current one; "
                   "created from HEAD if missing)")
    p.add_argument("--state-dir", help="default: artifacts/home_pipeline/<tag>_<hash>/")
    p.add_argument("--run-prefix", help="run-name prefix (default: home_<hash>, dryrun_<hash>)")
    p.add_argument("--status", action="store_true", help="print the state of --state-dir and exit")
    p.add_argument("--no-push", action="store_true", help="commit but do not push")
    p.add_argument("--allow-dirty", action="store_true",
                   help="--dry-run only: allow uncommitted or untracked training-repo files besides "
                   "config/home_pose.yaml (a real run refuses them: the 9999 corner rescue trains only "
                   "from a clean tree)")
    p.add_argument("--sequential", action="store_true", help="train get-up after walking instead of in parallel")
    p.add_argument("--serial-gpu", action="store_true",
                   help="run at most one GPU job of this command at a time (implies --sequential; walking "
                   "probes then wait for the end of each training segment); default for --dry-run")
    p.add_argument("--commit-trailer", default=TRAILER_DEFAULT, help="text appended to commit messages")
    p.add_argument("--walk-iterations", type=int, default=15000)
    p.add_argument("--walk-cont-iterations", type=int, default=15000)
    p.add_argument("--probe-every", type=int, default=1000)
    p.add_argument("--probe-min", type=int, default=1000)
    p.add_argument("--select-top", type=int, default=6, help="candidates re-probed for selection")
    p.add_argument("--select-repeats", type=int, default=3, help="probes per candidate")
    p.add_argument("--train-gpu-mib", type=int, default=11000, help="free GPU memory before a 4096-env job")
    p.add_argument("--pico-gpu-mib", type=int, default=17000,
                   help="free GPU memory before a v12 training job (2048 envs use about 16.1 GB)")
    p.add_argument("--probe-gpu-mib", type=int, default=3000, help="free GPU memory before a probe/eval/gate")
    p.add_argument("--dry-run", action="store_true", help="plumbing run with 2-3 iterations per stage")
    p.add_argument("--dry-run-walk-init", help="dry run: walking continues from this checkpoint, which must "
                   "be stamped with this HOME (so the v12 source probe can pass)")
    p.add_argument("--dry-run-plumbing", action="store_true",
                   help="dry run from scratch at any HOME: the v12 source probe, the robot validator and the "
                   "robot tests are run and recorded but do not stop the run (a failed source probe is forced "
                   "to pass in a DRYRUN_FORCED_PASS_ copy)")
    p.add_argument("--dry-run-gates", choices=("all", "final"), default="all")
    p.add_argument("--skip-training-suite", action="store_true",
                   help="--dry-run only: do not run the training test suite at this HOME first (a real run "
                   "always runs it and stops on a failure outside scripts/home_pipeline/known_test_failures.txt)")
    p.add_argument("--dry-run-simulate-failures", action="store_true",
                   help="dry run: treat the first gate of each canary and the 9999 gate as failed to exercise "
                   "the canary retry and the 9999 escalation")
    p.add_argument("--dry-run-simulate-9999", choices=("retrain", "rescue", "stop"), default="retrain",
                   help="with --dry-run-simulate-failures: retrain = every corner rescue of attempt 1 fails, the "
                   "retrained attempt 2 passes; rescue = the second rescue mix passes; stop = everything fails")
    p.add_argument("--dry-envs", type=int, default=64)
    p.add_argument("--pr-corner-rescue-mixes", default="lf60,lf90,lf72,lf65",
                   help="sampler mixes of the pose-release corner rescues tried, in order, after a failed 9999 "
                   "gate (lf60, lf65, lf72, lf90; a repeated mix is a new run); the 2026-10 forward-lean chain "
                   "tried lf60, lf90 and lf72, and registered lf65 for a more balanced parent")
    p.add_argument("--v12-9999-attempts", type=int, default=2,
                   help="7100->10000 attempts (the first plus retrains from the gated model_7099), each with "
                   "its corner rescues, before the 9999 boundary stops the run")
    p.add_argument("--pr-final-rescue-mixes", default=",".join(V12_FINAL_RESCUE_MIXES),
                   help="sampler mixes of the pose-release final-scenario rescues tried, in order, after a "
                   "failed 14999 gate (pr_v1-pr_v6; pr_v5/pr_v6 also replay the evaluator's push; the validator "
                   "refuses a mix that does not replay every failed scenario)")
    p.add_argument("--v12-15000-attempts", type=int, default=2,
                   help="10100->15000 attempts (the first plus retrains from the gated model_10099, attempt k "
                   "with training seed 41+k), each with its final rescues, before the 15000 boundary stops "
                   "the run")
    p.add_argument("--dry-run-simulate-15000", choices=("pass", "rescue", "retrain", "stop"), default="pass",
                   help="with --dry-run-simulate-failures: the 15000 route (pass = no simulated failure; "
                   "rescue = the second final-rescue mix passes; retrain = attempt 1 and its rescues fail, "
                   "attempt 2 passes; stop = everything fails)")
    args = p.parse_args(argv)
    mixes = [m.strip() for m in args.pr_corner_rescue_mixes.split(",") if m.strip()]
    bad = [m for m in mixes if m not in ("lf60", "lf65", "lf72", "lf90")]
    if bad:
        p.error(f"--pr-corner-rescue-mixes: unknown mix(es) {bad} (lf60, lf65, lf72, lf90)")
    args.pr_corner_rescue_mixes = mixes
    if args.v12_9999_attempts < 1:
        p.error("--v12-9999-attempts must be at least 1")
    final_mixes = [m.strip() for m in args.pr_final_rescue_mixes.split(",") if m.strip()]
    bad = [m for m in final_mixes if m not in V12_FINAL_RESCUE_MIXES]
    if bad:
        p.error(f"--pr-final-rescue-mixes: unknown mix(es) {bad} ({', '.join(V12_FINAL_RESCUE_MIXES)})")
    args.pr_final_rescue_mixes = final_mixes
    if args.v12_15000_attempts < 1:
        p.error("--v12-15000-attempts must be at least 1")
    if args.dry_run:
        args.probe_every, args.probe_min = 1, 0
        args.select_top, args.select_repeats = min(args.select_top, 2), min(args.select_repeats, 2)
        args.train_gpu_mib = args.pico_gpu_mib = args.probe_gpu_mib = 2500
        args.serial_gpu = True
    if args.serial_gpu:
        args.sequential = True
    if not args.dry_run and (args.dry_run_walk_init or args.dry_run_simulate_failures or args.dry_run_plumbing
                             or args.dry_run_simulate_15000 != "pass" or args.skip_training_suite):
        p.error("--dry-run-* options need --dry-run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.status:
        if not args.state_dir:
            print("--status needs --state-dir", file=sys.stderr)
            return EXIT_INPUT
        return print_status(Path(args.state_dir))
    if os.environ.get("MJLAB_MICROBAN_HOME_POSE_YAML"):
        # Every job must train at the committed config/home_pose.yaml.
        print("error: unset MJLAB_MICROBAN_HOME_POSE_YAML (the pipeline trains at "
              "config/home_pose.yaml)", file=sys.stderr)
        return EXIT_INPUT
    pipeline = Pipeline(args)

    def on_signal(signum, _frame) -> None:
        pipeline.log(f"signal {signum}: stopping own jobs")
        pipeline.abort.set()
        pipeline.kill_all_children()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        return pipeline.execute()
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001 - every stop is reported and cleans up
        pipeline.abort.set()
        pipeline.kill_all_children()
        if isinstance(error, PipelineError):
            code, reason = error.code, str(error)
        else:
            import traceback

            code = EXIT_FAILED
            reason = f"unexpected {type(error).__name__}: {error}\n" + "".join(
                traceback.format_exception(error)).rstrip()
        pipeline.log(f"STOPPED: {reason}")
        if isinstance(error, HomeYamlChanged) and pipeline.owns_state:
            # Let the stopped jobs exit before their output is moved aside.
            deadline = time.time() + 120
            while time.time() < deadline and any(Path(f"/proc/{pid}").exists() for pid in list(pipeline.children)):
                time.sleep(2)
            try:
                pipeline.quarantine_after_yaml_change(error.edit_time)
            except Exception as quarantine_error:  # noqa: BLE001 - the stop is still reported
                pipeline.log(f"quarantine failed: {quarantine_error!r}; remove the newest runs by hand")
        if pipeline.owns_state:  # a refused second instance never touches the live run's state
            pipeline.put("stopped", {"at": f"{datetime.now():%F %T}", "reason": reason, "code": code})
        return code


if __name__ == "__main__":
    sys.exit(main())
