"""The steps of one release run: home -> walk -> pico -> getup -> export -> install -> commit.

Each policy is one training process from scratch that runs its fixed number
of updates (mjlab_microban/schedules.py) and is judged once, on its last
checkpoint.  A failed judgment stops the run with exit code 1 and a report.
The test is questioned first: a test found wrong is fixed by a committed
change of the evaluation code or the judgment config, which never retrains
-- the rerun re-judges the trained policy (``begin``: a failed step runs
again only when its judgment inputs changed).  A valid test that fails is
the model's problem and stops the release; the reward is not changed to pass
a test.  Nothing is rescued or retrained with another seed.  A training that
stopped without a verdict (its process crashed or stalled, Ctrl-C:
``core.JobStopped``) leaves the step running and trains it again from update
0 on the rerun (a run continued from a checkpoint would not replay its
curriculum as one run does); an evaluator that wrote no report is run again on
the rerun; a training of the step's run name that is
already running (started by hand with the same command) is watched, not
restarted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from mjlab_microban import schedules
from mjlab_microban.pipeline.monitor import Monitor
from mjlab_microban.pipeline.core import (
    EXIT_INPUT,
    HOME_YAML,
    LOG_ROOT,
    REPO,
    UV,
    Jobs,
    JobStopped,
    PipelineError,
    State,
    capture,
    find_checkpoint,
    git,
    hash_inputs,
    last_json_line,
    load_config,
    record_files,
    resumed_run,
    resumes,
    running_training,
    sha256,
)

STEPS = ("home", "walk", "pico", "getup", "export", "install", "commit")
WALK_EXP, PICO_EXP, GETUP_EXP = (
    "mjlab_microban_velocity",
    "mjlab_microban_teleop_v12",
    "mjlab_microban_getup",
)
PROBE_ROOT = REPO / "artifacts" / "legacy_teleop_probe"
BOOTSTRAP_ROOT = REPO / "artifacts" / "teleop_v12_bootstrap"
DRY_RUN_POLICY_ALLOW_ENV = "MICROBAN_ALLOW_DRYRUN_POLICY"

# The keys of a step's config/pipeline.yaml section the training reads (the
# rest configures its judgment and is not a training input).
TRAINING_KEYS = ("task", "envs", "save_interval")
# The schedule entries (Pipeline._schedules) a step's training reads, besides
# its stage table.
STEP_SCHEDULE_KEYS = {
    "walk": ("walk_total",),
    "pico": ("pico", "pico_total"),
    "getup": ("getup", "getup_total"),
    "export": ("contract", "recipes"),
}
# Files whose content a step's training depends on (repo-relative globs).
# Evaluation code is not listed: changing an evaluator re-judges nothing that
# was already accepted, and never retrains.
# The schedule constants are not a file input: each step hashes only its own
# (STEP_SCHEDULE_KEYS), so a get-up schedule change does not retrain walking.
_COMMON = ["config/home_pose.yaml", "src/mjlab_microban/robot",
           "src/mjlab_microban/tasks/curriculum.py", "src/mjlab_microban/tasks/mdp.py", "uv.lock"]
STEP_INPUTS = {
    "walk": [*_COMMON, "src/mjlab_microban/tasks/microban_velocity_*.py",
             "src/mjlab_microban/tasks/microban_getup_runner.py"],
    "pico": [*_COMMON, "src/mjlab_microban/tasks/microban_velocity_env_cfg.py",
             "src/mjlab_microban/tasks/microban_teleop_*.py",
             "src/mjlab_microban/tasks/microban_policy_export.py"],
    "getup": [*_COMMON, "src/mjlab_microban/tasks/microban_getup_*.py",
              "src/mjlab_microban/tasks/microban_teleop_mdp.py"],
    "export": ["src/mjlab_microban/scripts/export_*.py", "src/mjlab_microban/policy_contract.py",
               "src/mjlab_microban/scripts/teleop_v12_stage.py"],
}


class Pipeline:
    def __init__(self, *, robot: Path, robot_branch: str, training_branch: str | None,
                 state_dir: Path | None, dry: bool, push: bool) -> None:
        if os.environ.get("MJLAB_MICROBAN_HOME_POSE_YAML"):
            raise PipelineError("MJLAB_MICROBAN_HOME_POSE_YAML is set: a release trains at the HOME of "
                                "config/home_pose.yaml only", EXIT_INPUT)
        self.robot = robot.resolve()
        self.robot_branch = robot_branch
        self.training_branch = training_branch
        self.dry = dry
        self.push_enabled = push and not dry
        self.cfg = load_config(dry)
        self.env: dict[str, str] = {}
        if dry:
            self.env[schedules.SCHEDULE_SCALE_ENV] = str(self.cfg["schedule_scale"])
        elif os.environ.get(schedules.SCHEDULE_SCALE_ENV):
            raise PipelineError(f"{schedules.SCHEDULE_SCALE_ENV} is set: a release trains the full schedules",
                                EXIT_INPUT)
        self.sched = self._schedules()
        self.home = self._home_identity()
        prefix = ("dryrun_" if dry else "home_") + self.home["joint_hash"]
        self.prefix = prefix
        directory = state_dir or (REPO / "artifacts" / "home_pipeline" / f"{prefix}_{self.home['tag']}")
        self.state = State(directory.resolve())
        self.jobs = Jobs(self.state, stall_minutes=self.cfg["stall_minutes"], wait_for_gpu=not dry,
                         external_min_envs=int(self.cfg["external_training_min_envs"]), own_prefix=prefix)
        # What a judgment reads besides the trained policy: a change re-judges a failed step.
        self.judge_inputs = hash_inputs(["src", "config/pipeline.yaml"], {"dry": dry})
        self.yaml_sha256 = sha256(HOME_YAML)

    # -- helpers ---------------------------------------------------------------
    def _schedules(self) -> dict[str, Any]:
        """The schedule constants as the training subprocesses see them."""

        code = ("import json; from mjlab_microban import schedules as s, policy_contract as p; "
                "from mjlab_microban.pipeline.steps import stage_tables; print(json.dumps({"
                "'contract': p.POLICY_CONTRACT, 'recipes': dict(p.RECIPES), "
                "'walk_total': s.WALK_TOTAL_UPDATES, 'getup_total': s.GETUP_TOTAL_UPDATES, "
                "'pico_total': s.PICO_TOTAL_UPDATES, 'pico': s.pico_schedule_record(), "
                "'getup': s.GETUP_SCHEDULE, 'stages': stage_tables()}))")
        out = capture([*UV, "python", "-c", code], env=self.env, check=True, timeout=600)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def _home_identity(self) -> dict[str, Any]:
        out = capture([*UV, "python", "-m", "mjlab_microban.pipeline.home_check"], timeout=900)
        try:
            home = json.loads(out.stdout or "{}")
        except ValueError:
            home = {}
        if not home.get("tag"):
            raise PipelineError("cannot load config/home_pose.yaml: "
                                + ("; ".join(home.get("reasons") or []) or out.stderr[-800:]), EXIT_INPUT)
        return home

    def log(self, message: str) -> None:
        self.state.log(message)

    def check_home_yaml(self) -> None:
        if sha256(HOME_YAML) != self.yaml_sha256:
            raise PipelineError(f"{HOME_YAML} changed during the run; restore it or use another "
                                "worktree for another HOME", EXIT_INPUT)

    def inputs(self, step: str, *upstream: str) -> str:
        # Only what the training reads: the judgments (``judgment``, ``gate``,
        # ``evals``, pass lines) judge a run, they never retrain it.
        section = {"seed": self.cfg.get("seed"),
                   step: {key: value for key, value in (self.cfg.get(step) or {}).items()
                          if key in TRAINING_KEYS}}
        schedule = {key: self.sched[key] for key in STEP_SCHEDULE_KEYS.get(step, ())}
        if step in self.sched.get("stages", {}):
            schedule["stages"] = self.sched["stages"][step]
        extra = {"dry": self.dry, "cfg": section, "schedules": schedule,
                 "upstream": {name: self.state.step(name).get("outputs") for name in upstream}}
        return hash_inputs(STEP_INPUTS.get(step, []), extra)

    def begin(self, step: str, inputs: str) -> bool:
        """True when the step must run; False when it is done with these inputs."""

        record = self.state.step(step)
        if record.get("status") == "done" and record.get("inputs") == inputs and \
                self.state.outputs_intact(step):
            self.log(f"[{step}] done earlier with the same inputs: skipped")
            return False
        judge = getattr(self, "judge_inputs", None)
        if record.get("status") == "failed" and record.get("inputs") == inputs:
            if record.get("judge_inputs") == judge:
                raise PipelineError(f"[{step}] failed earlier with the same inputs: {record.get('error')}\n"
                                    "Question the test first: fix a wrong test (evaluation code or judgment "
                                    "config, committed) and rerun to re-judge; a valid failing test is the "
                                    "model's problem.")
            # The judgment changed (a committed fix of a test): judge the
            # trained policy again, its training is kept.
            self.log(f"[{step}] failed earlier ({str(record.get('error'))[:200]}); the judgment changed: "
                     "re-judging without retraining")
            # The walker's source probe (the PICO entry gate of the trained
            # run) is part of the training; it is kept.
            for key in ("error", "ended"):
                record.pop(key, None)
        if record.get("inputs") != inputs:
            record.clear()
        record.update(status="running", inputs=inputs, judge_inputs=judge, started=f"{datetime.now():%F %T}")
        # The training run of these inputs (kept by a rerun that judges it again).
        record.setdefault("label", f"{getattr(self, 'prefix', 'run')}_{step}_{inputs[:8]}")
        self.state.save()
        self.log(f"[{step}] start")
        return True

    def finish(self, step: str, outputs: dict[str, Any], files: list[Path]) -> None:
        record = self.state.step(step)
        record.update(status="done", outputs=outputs, files=record_files(files),
                      ended=f"{datetime.now():%F %T}")
        self.state.save()
        self.log(f"[{step}] done")

    def fail(self, step: str, error: str) -> None:
        record = self.state.step(step)
        record.update(status="failed", error=error, ended=f"{datetime.now():%F %T}")
        self.state.save()

    def train(self, step: str, experiment: str, label: str, task: str, total: int, envs: int,
              extra: list[str], *, fresh_extra: list[str] = ()) -> Path:
        """Train ``label`` from scratch to model_<total - 1>, or keep that checkpoint when it exists.

        Every curriculum stage must start at its update of the table
        (``Monitor``).  A training that stopped is trained again from update 0.
        A run resumed from a checkpoint (``--agent.resume``, for trials only) is
        refused: a release model is one training from update 0.
        """

        def refuse_resumed(what: str) -> None:
            raise JobStopped(f"[{step}] {what} resumed from a checkpoint: a release model is trained from "
                             f"update 0 in one run. Move it out of {LOG_ROOT / experiment} (or stop it); a "
                             "trial resume takes another --agent.run-name", EXIT_INPUT)

        record = self.state.step(step)
        monitor = Monitor(state=self.state, record=record, expected_stages=self.sched["stages"][step],
                          stage_tolerance=int(self.cfg[step]["stage_tolerance"]))
        final = find_checkpoint(experiment, label, total - 1)
        if final is not None:
            if resumed_run(final.parent):
                refuse_resumed(f"run {final.parent.name} was")
            self.log(f"[{step}] {final.parent.name}/{final.name} exists")
            monitor.log_path = self.training_log(step, label)
            monitor.read_log()
        elif (running := running_training(label)) is not None:
            # Started by hand with this step's command (same run name): watch it.
            # Its stages are checked from its own log only.
            pgid, cmdline = running
            if resumes(cmdline):
                refuse_resumed(f"the running training {label} is")
            record["stages_seen"] = {}
            monitor.log_path = self.training_log(step, label)
            self.check_home_yaml()
            self.jobs.watch(f"train_{step}", pgid, monitor.log_path, poll=monitor.poll)
            self.check_home_yaml()
        else:
            record["stages_seen"] = {}
            seed = str(self.cfg["seed"])
            cmd = [*UV, "train", task, "--env.scene.num-envs", str(envs), "--env.seed", seed,
                   "--agent.seed", seed, "--agent.logger", "tensorboard", "--agent.run-name", label,
                   "--agent.upload-model", "False", "--enable-nan-guard", "True",
                   "--agent.max-iterations", str(total), *extra, *fresh_extra]
            self.check_home_yaml()
            self.jobs.run(f"train_{step}", cmd, "train", env=self.env, poll=monitor.poll,
                          on_start=lambda log: setattr(monitor, "log_path", log))
            self.check_home_yaml()
        final = find_checkpoint(experiment, label, total - 1)
        if final is None:
            raise JobStopped(f"[{step}] model_{total - 1}.pt missing after training {label}: the training "
                             "stopped early; rerun to train it again")
        monitor.require_stages(total - 1)
        self.state.save()
        return final

    def training_log(self, step: str, label: str) -> Path:
        """The newest ``*_train_<step>.log`` of run ``label`` in the state's logs."""

        for log in sorted((self.state.dir / "logs").glob(f"*_train_{step}.log"), reverse=True):
            with open(log, errors="replace") as stream:
                if label in stream.read(1 << 20):
                    return log
        raise PipelineError(f"[{step}] no training log of {label} in {self.state.dir / 'logs'} (a training "
                            "started by hand must write its output there, named <time>_train_<step>.log)")

    # -- steps -------------------------------------------------------------------
    def step_home(self) -> None:
        inputs = hash_inputs(["config", "src", "tests", "uv.lock", "pyproject.toml"], {"dry": self.dry})
        if not self.begin("home", inputs):
            return
        home = self.home
        for warning in home.get("warnings", []):
            self.log(f"WARNING HOME: {warning}")
        if home.get("status") == "refuse":
            raise PipelineError("HOME refused: " + "; ".join(home.get("reasons", [])))
        balance = capture([*UV, "python", "config/balance_home_pose.py", "--check", "--no-training-check"],
                          timeout=900)
        if balance.returncode != 0:
            self.log("WARNING HOME: config/balance_home_pose.py --check: not the canonical balanced "
                     "solution (run it with --write to re-centre the COM); continuing")
        show = capture([*UV, "python", "config/home_pose_tool.py", "show"], timeout=1200)
        if show.returncode != 0:
            raise PipelineError("config/home_pose_tool.py show refused the HOME: "
                                + (show.stderr or show.stdout).strip()[-800:])
        training_line = json.loads(show.stdout).get("training_line")
        self.log(f"HOME {home['tag']} (hash {home['joint_hash']}), training line {training_line}")
        out = self.state.dir / "training_suite.out"
        # The suite checks the real schedules: never under the dry-run scale.
        rc, _ = self.jobs.run("training_suite", [*UV, "--with", "pytest", "python", "-m", "pytest", "-q",
                                                 "-p", "no:cacheprovider", "tests"], "eval",
                              env={"CUDA_VISIBLE_DEVICES": ""}, gpu=False, check=False)
        log = sorted((self.state.dir / "logs").glob("*_training_suite.log"))[-1]
        shutil.copyfile(log, out)
        summary = next((line for line in reversed(out.read_text().splitlines()) if " passed" in line
                        or " failed" in line), "(no summary)")
        if rc != 0:
            raise PipelineError(f"the training test suite fails at this HOME ({summary}; see {out}): fix it first",
                                EXIT_INPUT)
        self.log(f"training suite: {summary}")
        self.finish("home", {"tag": home["tag"], "joint_hash": home["joint_hash"],
                             "yaml_sha256": self.yaml_sha256, "training_line": training_line,
                             "suite": summary}, [])

    def step_walk(self) -> None:
        inputs = self.inputs("walk")
        if not self.begin("walk", inputs):
            return
        c = self.cfg["walk"]
        label = self.state.step("walk")["label"]
        final = self.train("walk", WALK_EXP, label, c["task"], self.sched["walk_total"], c["envs"],
                           ["--agent.save-interval", str(c["save_interval"])])
        failures, summary = self.walk_judgment(final)
        if failures and not self.dry:
            raise PipelineError(f"walking judgment failed for {final.parent.name}/{final.name}: {failures} "
                                f"{json.dumps(summary, default=str)[:1200]}")
        if failures:
            self.log(f"dry run: walking judgment failed ({failures}); recorded, not enforced")
        walker = self.install_walker(final)
        self.finish("walk", {"checkpoint": str(walker), "sha256": sha256(walker), "run": final.parent.name,
                             "iteration": int(final.stem.split("_")[1]), "passed": not failures,
                             "failures": failures, "summary": summary}, [walker])

    def walk_judgment(self, checkpoint: Path) -> tuple[list[str], dict[str, Any]]:
        """The held-out walk probe (seeds 101-105) and the 9x300 probe with a held-out seed: (failures, summary)."""

        chk = self.cfg["walk"]["judgment"]
        out = self.state.dir / "walk_judgment"
        out.mkdir(exist_ok=True)
        probe_out = out / f"{checkpoint.parent.name}_{checkpoint.stem}_walk_probe.json"
        receipt = out / f"{checkpoint.parent.name}_{checkpoint.stem}_9x300_seed{chk['probe_seed']}.json"
        for path in (probe_out, receipt):
            path.unlink(missing_ok=True)
        rules = {"max_falls": chk["max_falls"], "still_touchdowns_per_s": chk["still_touchdowns_per_s"]}
        self.jobs.run("walk_judgment_probe", [*UV, "python", "-m", "mjlab_microban.pipeline.walk_probe",
                                              str(checkpoint), str(probe_out), "--seeds", chk["seeds"],
                                              "--rules", json.dumps(rules)], "eval", env=self.env, check=False)
        self.jobs.run("walk_judgment_9x300", [*UV, "python", "-m",
                                              "mjlab_microban.scripts.probe_legacy_actor_in_teleop_env",
                                              "--checkpoint", str(checkpoint), "--expected-sha256",
                                              sha256(checkpoint), "--seed", str(chk["probe_seed"]),
                                              "--output", str(receipt), "--force"], "eval", env=self.env,
                      check=False)
        for path in (probe_out, receipt):
            if not path.is_file():
                raise JobStopped(f"the walking evaluator wrote no report ({path}): not a judgment "
                                 "(rerun to evaluate again)")
        probe = json.loads(probe_out.read_text())
        probe.pop("rows", None)
        nine = probe_verdict(json.loads(receipt.read_text()))
        failures = walk_failures(probe, nine)
        summary = {"rules": probe["checks"],
                   "single_signed": {k: round(v, 3) for k, v in probe["single_signed"].items()},
                   # mean twist (v_x m/s, v_y m/s, w_z rad/s) on the standing command
                   "still_twist": [round(v, 3) for v in probe["still"]],
                   "still_touchdowns_per_s": round(probe["still_touchdowns_per_s"], 2),
                   "falls": probe["falls"], "9x300": nine["ok"],
                   # signed response minus its fixed minimum (m/s or rad/s)
                   "9x300_worst_margin": [nine["worst"], round(nine["worst_margin"], 4)],
                   # signed response along the command (m/s or rad/s)
                   "9x300_signed": {k: round(v, 4) for k, v in nine["responses"].items()},
                   "9x300_twist": {k: [round(x, 4) for x in nine["twists"][k]]
                                   for k in ("neutral", "backward_0p1", "backward_0p2") if k in nine["twists"]},
                   "9x300_soft_limit_rad": round(nine["soft_limit_overshoot_rad"], 3)}
        (out / f"{checkpoint.parent.name}_{checkpoint.stem}_judgment.json").write_text(
            json.dumps({"failures": failures, "summary": summary, "probe": probe, "nine_by_300": nine},
                       indent=1, sort_keys=True, default=str))
        self.log(f"walking judgment of {checkpoint.parent.name}/{checkpoint.name}: "
                 f"{'PASS' if not failures else f'FAIL {failures}'} {json.dumps(summary, default=str)[:1500]}")
        return failures, summary

    def install_walker(self, checkpoint: Path) -> Path:
        """Copy the walker (and its params/) where the PICO provenance re-hashes it."""

        dest_dir = REPO / "checkpoints" / f"{self.prefix}_walk_{sha256(checkpoint)[:12]}"
        dest = dest_dir / checkpoint.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(checkpoint, dest)
        if not (dest_dir / "params").exists() and (checkpoint.parent / "params").is_dir():
            shutil.copytree(checkpoint.parent / "params", dest_dir / "params")
        return dest

    def walker_probe(self, walker: Path) -> Path:
        """The 9x300 source probe of the walker in the PICO env (seed 42); the PICO entry gate."""

        digest = sha256(walker)
        receipt = PROBE_ROOT / f"velocity_{digest[:16]}_teleop81_arm_overlay_9x300.json"
        PROBE_ROOT.mkdir(parents=True, exist_ok=True)
        self.jobs.run("probe_walker", [*UV, "python", "-m",
                                        "mjlab_microban.scripts.probe_legacy_actor_in_teleop_env",
                                        "--checkpoint", str(walker), "--expected-sha256", digest,
                                        "--output", str(receipt), "--force"], "eval", env=self.env)
        verdict = probe_verdict(json.loads(receipt.read_text()))
        self.log(f"walker 9x300 probe (seed 42): {'PASS' if verdict['ok'] else 'FAIL'} "
                 f"worst margin over the fixed minimum {verdict['worst_margin']:+.4f} ({verdict['worst']}), "
                 f"minimums { {k: round(v, 3) for k, v in verdict['minimums'].items()} }, signed responses "
                 f"{ {k: round(v, 3) for k, v in verdict['responses'].items()} }, falls {verdict['falls']}, "
                 f"soft-limit overshoot {verdict['soft_limit_overshoot_rad']:.3f} rad")
        self.state.step("pico")["walker_probe"] = {**verdict, "receipt": str(receipt),
                                                   "receipt_sha256": sha256(receipt)}
        self.state.save()
        if verdict["ok"]:
            return receipt
        if not self.dry:
            raise PipelineError(f"the walker fails the PICO source probe: {verdict['below']} (worst "
                                f"{verdict['worst']} {verdict['worst_margin']:+.4f}, soft-limit overshoot "
                                f"{verdict['soft_limit_overshoot_rad']:.3f} rad, falls {verdict['falls']}; "
                                f"receipt {receipt})")
        forced = PROBE_ROOT / f"DRYRUN_FORCED_PASS_{receipt.name}"
        capture([*UV, "python", "-m", "mjlab_microban.pipeline.dry", "force-probe", str(receipt), str(forced)],
                env=self.env, check=True, timeout=900)
        self.log(f"dry run: the walker's probe failed; the chain bootstraps from {forced.name} (not deployable)")
        return forced

    def step_pico(self) -> None:
        inputs = self.inputs("pico", "walk")
        if not self.begin("pico", inputs):
            return
        c = self.cfg["pico"]
        walker = Path(self.state.step("walk")["outputs"]["checkpoint"])
        label = self.state.step("pico")["label"]
        total = self.sched["pico_total"]
        fresh: list[str] = []
        if find_checkpoint(PICO_EXP, label, total - 1) is None and running_training(label) is None:
            # A fresh run: its walker passes the entry probe and is bootstrapped.
            capture([*UV, "python", "-c", "import sys; from mjlab_microban.scripts.export_walk_onnx import "
                     "require_current_home_walk_checkpoint as c; c(sys.argv[1])", str(walker)],
                    env=self.env, check=True, timeout=900)
            receipt = self.walker_probe(walker)
            walker_sha, receipt_sha = sha256(walker), sha256(receipt)
            self.jobs.run("bootstrap_gate", [*UV, "python", "-m",
                                             "mjlab_microban.scripts.teleop_v12_bootstrap_gate",
                                             "--checkpoint", str(walker), "--checkpoint-sha256", walker_sha,
                                             "--probe-receipt", str(receipt), "--probe-receipt-sha256",
                                             receipt_sha, "--output-dir",
                                             str(BOOTSTRAP_ROOT / f"velocity_{walker_sha[:16]}"), "--force"],
                          "eval", env=self.env, gpu=False)
            fresh = ["--agent.legacy-velocity-checkpoint", str(walker),
                     "--agent.legacy-velocity-checkpoint-sha256", walker_sha,
                     "--agent.legacy-teleop-probe-receipt", str(receipt),
                     "--agent.legacy-teleop-probe-receipt-sha256", receipt_sha,
                     "--agent.save-pristine-checkpoint", "True"]
        final = self.train("pico", PICO_EXP, label, c["task"], total, c["envs"],
                           ["--agent.num-steps-per-env", "24", "--agent.save-interval", str(c["save_interval"])],
                           fresh_extra=fresh)
        report_prefix, passed, failures = self.pico_judgment(final)
        if not passed and not self.dry:
            raise PipelineError(f"PICO judgment failed for {final.parent.name}/{final.name}: {failures} "
                                f"(reports {report_prefix}_*.json)")
        if not passed:
            self.log(f"dry run: PICO judgment failed ({failures}); recorded, not enforced")
        self.finish("pico", {"checkpoint": str(final), "sha256": sha256(final), "run": final.parent.name,
                             "report_prefix": report_prefix,
                             "passed": passed, "failures": failures}, [final])

    def pico_commands(self, checkpoint: Path, prefix: Path, seed: int) -> list[tuple[str, list[str]]]:
        digest = sha256(checkpoint)
        return [
            ("loco", [*UV, "python", "-m", "mjlab_microban.scripts.evaluate_teleop_v12_checkpoint",
                      str(checkpoint), "--expected-sha256", digest, "--seed", str(seed),
                      "--output", f"{prefix}_9x300.json", "--force"]),
            ("tracking", [*UV, "python", "-m", "mjlab_microban.scripts.evaluate_teleop_v12_tracking",
                          str(checkpoint), "--expected-sha256", digest, "--seed", str(seed),
                          "--output", f"{prefix}_tracking.json", "--force"]),
            ("onnx", [*UV, "python", "-m", "mjlab_microban.scripts.teleop_v12_onnx_gate", str(checkpoint),
                      "--expected-sha256", digest, "--onnx", f"{prefix}.onnx", "--output", f"{prefix}_onnx.json",
                      "--force"]),
        ]

    @staticmethod
    def pico_failures(prefix: Path) -> list[str]:
        failures = []
        for name in ("9x300", "tracking", "onnx"):
            path = Path(f"{prefix}_{name}.json")
            if not path.is_file():
                raise JobStopped(f"PICO evaluator {name} wrote no report ({path}): not a judgment "
                                 "(rerun to evaluate again)")
            data = json.loads(path.read_text())
            failures += [f"{name}:{key}" for key, ok in sorted((data.get("checks") or {}).items()) if ok is not True]
            if data.get("status") != "pass" and not any(f.startswith(name + ":") for f in failures):
                failures.append(f"{name}:status")
        return failures

    def pico_judgment(self, checkpoint: Path) -> tuple[str, bool, list[str]]:
        """The three PICO evaluators (seed 42; the PICO judgment also 43) and, when all pass, the gate file."""

        out_dir = self.state.dir / "pico_judgment"
        out_dir.mkdir(exist_ok=True)
        prefix = out_dir / f"{checkpoint.parent.name}_{checkpoint.stem}"
        for name in ("9x300", "tracking", "onnx"):
            Path(f"{prefix}_{name}.json").unlink(missing_ok=True)
        for name, cmd in self.pico_commands(checkpoint, prefix, int(self.cfg["seed"])):
            self.jobs.run(f"pico_judgment_{name}", cmd, "eval", env=self.env, check=False)
        failures = self.pico_failures(prefix)
        passed = not failures
        if passed:
            capture([*UV, "python", "-m", "mjlab_microban.scripts.teleop_v12_stage", "create", str(checkpoint),
                     f"{prefix}_9x300.json", f"{prefix}_tracking.json", f"{prefix}_onnx.json",
                     f"{prefix}_gate.json", "--force"], env=self.env, check=True, timeout=1800)
        self.log(f"PICO judgment of {checkpoint.parent.name}/{checkpoint.name}: "
                 + ("PASS" if passed else f"FAIL {failures}"))
        return str(prefix), passed, failures

    def step_getup(self) -> None:
        inputs = self.inputs("getup")
        if not self.begin("getup", inputs):
            return
        c = self.cfg["getup"]
        label = self.state.step("getup")["label"]
        final = self.train("getup", GETUP_EXP, label, c["task"], self.sched["getup_total"], c["envs"],
                           ["--agent.save-interval", str(c["save_interval"])])
        summary = self.getup_judgment(final, "judgment")
        failures = getup_gate_failures(summary, c["gate"])
        if failures and not self.dry:
            raise PipelineError(f"get-up judgment failed for {final.parent.name}/{final.name}: {failures} "
                                f"({summary})")
        if failures:
            self.log(f"dry run: get-up judgment failed ({failures}); recorded, not enforced")
        self.finish("getup", {"checkpoint": str(final), "sha256": sha256(final), "run": final.parent.name,
                              "summary": summary,
                              "passed": not failures, "failures": failures}, [final])

    def getup_commands(self, checkpoint: Path) -> list[tuple[str, list[str]]]:
        c = self.cfg["getup"]
        commands = []
        for eval_name, mode, seed, extra in c["evals"]:
            cmd = [*UV, "python", "-m", "mjlab_microban.pipeline.getup_eval", mode, c["task"], str(checkpoint),
                   "--seed", str(seed), *extra]
            if "eval_envs" in c:
                cmd += ["--envs", str(c["eval_envs"]), "--steps", str(c["eval_steps"])]
            commands.append((eval_name, cmd))
        return commands

    def getup_judgment(self, checkpoint: Path, name: str) -> dict[str, Any]:
        results = {}
        for eval_name, cmd in self.getup_commands(checkpoint):
            rc, log = self.jobs.run(f"getup_{name}_{eval_name}", cmd, "eval", env=self.env, check=False)
            text = log.read_text(errors="replace")
            if rc != 0 or "RESULT " not in text:
                raise JobStopped(f"the get-up evaluator {eval_name} wrote no result (rc={rc}, log {log}): not "
                                 "a judgment (rerun to evaluate again)")
            results[eval_name] = last_json_line(text, prefix="RESULT ")
        summary = getup_summary(results)
        (self.state.dir / f"getup_{name}.json").write_text(json.dumps({"summary": summary, "results": results},
                                                                       indent=1, sort_keys=True))
        self.log(f"get-up {name} {checkpoint.parent.name}/{checkpoint.name}: {summary}")
        return summary

    def step_export(self) -> None:
        inputs = self.inputs("export", "walk", "pico", "getup")
        if not self.begin("export", inputs):
            return
        out = self.state.dir / "release"
        out.mkdir(exist_ok=True)
        steps = self.state.data["steps"]
        checkpoints = {"walk": Path(steps["walk"]["outputs"]["checkpoint"]),
                       "getup": Path(steps["getup"]["outputs"]["checkpoint"]),
                       "pico": Path(steps["pico"]["outputs"]["checkpoint"])}
        files = {kind: out / name for kind, name in (("walk", "walk.onnx"), ("getup", "getup.onnx"),
                                                     ("pico", "pico_teleop.onnx"))}
        gates = self.write_gate_reports(out, checkpoints)
        for kind, module in (("walk", "export_walk_onnx"), ("getup", "export_getup_onnx")):
            self.jobs.run(f"export_{kind}", [*UV, "python", "-m", f"mjlab_microban.scripts.{module}",
                                             "--checkpoint", str(checkpoints[kind]), "--output", str(files[kind]),
                                             "--gate-report", str(gates[kind]), "--replace",
                                             *(["--dry-run"] if self.dry else [])],
                          "eval", env={**self.env, "CUDA_VISIBLE_DEVICES": ""}, gpu=False)
        pico = steps["pico"]["outputs"]
        prefix = pico["report_prefix"]
        if pico["passed"]:
            cmd = [*UV, "python", "-m", "mjlab_microban.scripts.export_teleop_v12_deployment",
                   "--checkpoint", pico["checkpoint"], "--stage-gate", f"{prefix}_gate.json",
                   "--output", str(files["pico"]), "--force"]
        else:  # dry run only (a real run stopped at the judgment)
            cmd = [*UV, "python", "-m", "mjlab_microban.pipeline.dry", "package", pico["checkpoint"], prefix,
                   str(out / "DRYRUN_pico_gate.json"), str(files["pico"])]
        _, log = self.jobs.run("export_pico", cmd, "eval", env={**self.env, "CUDA_VISIBLE_DEVICES": ""}, gpu=False)
        receipt = last_json_line(log.read_text(errors="replace"))
        if receipt.get("status") != "pass" or receipt.get("output_sha256") != sha256(files["pico"]):
            raise PipelineError(f"PICO packaging failed: {receipt}")
        manifest = self.manifest(files, checkpoints)
        manifest_path = out / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        self.finish("export", {"dir": str(out), "manifest_sha256": sha256(manifest_path)},
                    [*files.values(), manifest_path, *gates.values()])

    def write_gate_reports(self, out: Path, checkpoints: dict[str, Path]) -> dict[str, Path]:
        """The walk and get-up judgments as their exporters check them (policy_contract.gate_report)."""

        steps = self.state.data["steps"]
        walk_probe = steps["pico"].get("walker_probe") or {}
        evidence = {
            "walk": {"passed": bool(steps["walk"]["outputs"]["passed"]) and bool(walk_probe.get("ok")),
                     "summary": steps["walk"]["outputs"]["summary"],
                     "failures": steps["walk"]["outputs"]["failures"],
                     "source_probe_9x300": walk_probe},
            "getup": {"passed": bool(steps["getup"]["outputs"]["passed"]),
                      "summary": steps["getup"]["outputs"]["summary"],
                      "failures": steps["getup"]["outputs"]["failures"],
                      "judgment_sha256": sha256(self.state.dir / "getup_judgment.json")},
        }
        gates = {}
        for kind, record in evidence.items():
            gates[kind] = out / f"{kind}_gate.json"
            code = ("import json, sys; from pathlib import Path; from mjlab_microban import policy_contract as p; "
                    "a = json.loads(sys.argv[1]); p.write_gate_report(Path(a['path']), a['kind'], a['sha'], "
                    "passed=a['passed'], dry_run=a['dry'], evidence=a['evidence'])")
            arguments = {"path": str(gates[kind]), "kind": kind, "sha": sha256(checkpoints[kind]),
                         "passed": record["passed"], "dry": self.dry, "evidence": record}
            result = capture([*UV, "python", "-c", code, json.dumps(arguments, default=str)], env=self.env,
                             timeout=600)
            if result.returncode != 0:
                raise PipelineError(f"no {kind} gate report: {(result.stderr or result.stdout).strip()[-800:]}")
        return gates

    def manifest(self, files: dict[str, Path], checkpoints: dict[str, Path]) -> dict[str, Any]:
        """src/agents/manifest.json (policy_contract.manifest), built in the training env."""

        steps = self.state.data["steps"]
        extra = {
            "home_joint_hash": self.home["joint_hash"],
            "home_yaml_sha256": self.yaml_sha256,
            "training_branch": git(REPO, "rev-parse", "--abbrev-ref", "HEAD"),
            "judgments": {
                "walk": {k: steps["walk"]["outputs"][k] for k in ("summary", "passed", "failures")},
                "walk_source_probe": steps["pico"].get("walker_probe"),
                "pico": {k: steps["pico"]["outputs"][k] for k in ("passed", "failures")},
                "getup": {k: steps["getup"]["outputs"][k] for k in ("summary", "passed", "failures")},
            },
            "checkpoints": {kind: path.name for kind, path in checkpoints.items()},
            "created": f"{datetime.now():%F %T}",
        }
        arguments = {"files": {kind: str(path) for kind, path in files.items()},
                     "checkpoints": {kind: sha256(path) for kind, path in checkpoints.items()},
                     "commit": git(REPO, "rev-parse", "HEAD"), "dry": self.dry, "extra": extra}
        code = ("import json, sys; from mjlab_microban import policy_contract as p; a = json.loads(sys.argv[1]); "
                "print(json.dumps(p.manifest(files=a['files'], checkpoint_sha256=a['checkpoints'], "
                "training_commit=a['commit'], dry_run=a['dry'], extra=a['extra'])))")
        out = capture([*UV, "python", "-c", code, json.dumps(arguments, default=str)], env=self.env, check=True,
                      timeout=600)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def robot_env(self) -> dict[str, str]:
        env = {"PYTHONPATH": str(self.robot / "src"), "CUDA_VISIBLE_DEVICES": ""}
        if self.dry:
            env[DRY_RUN_POLICY_ALLOW_ENV] = "1"
        return env

    def step_install(self) -> None:
        inputs = hash_inputs([], {"export": self.state.step("export").get("outputs"),
                                  "robot": git(self.robot, "rev-parse", "HEAD"), "branch": self.robot_branch})
        if not self.begin("install", inputs):
            return
        release = Path(self.state.step("export")["outputs"]["dir"])
        agents = self.robot / "src" / "agents"
        agents.mkdir(parents=True, exist_ok=True)
        for name in ("walk.onnx", "getup.onnx", "pico_teleop.onnx", "manifest.json"):
            shutil.copyfile(release / name, agents / name)
        capture([*UV, "python", "config/home_pose_tool.py", "write-robot", "--microban-repo", str(self.robot)],
                check=True, timeout=1200)
        validator = self.robot / "tools" / "validate_policies.py"
        out = release / "robot_validation.json"
        valid = capture(["uv", "run", "--project", str(self.robot), "--locked", "python", str(validator),
                         "src/agents"], cwd=self.robot, env=self.robot_env(), timeout=1800)
        out.write_text(valid.stdout or valid.stderr)
        if valid.returncode != 0:
            raise PipelineError("the robot rejected the release (tools/validate_policies.py): "
                                + (valid.stderr or valid.stdout).strip()[-1500:])
        tests = capture(["uv", "run", "--project", str(self.robot), "--locked", "--with", "pytest", "python", "-m",
                         "pytest", "-q", "tests"], cwd=self.robot, env=self.robot_env(), timeout=3600)
        (release / "robot_tests.log").write_text(tests.stdout + tests.stderr)
        summary = (tests.stdout.strip().splitlines() or ["(no output)"])[-1]
        if tests.returncode != 0:
            raise PipelineError(f"the robot test suite fails with the release: {summary} "
                                f"(log {release / 'robot_tests.log'})")
        self.log(f"robot validator: pass; robot tests: {summary}")
        self.finish("install", {"robot_tests": summary}, [])

    def step_commit(self) -> None:
        inputs = hash_inputs([], {"export": self.state.step("export").get("outputs"),
                                  "install": self.state.step("install").get("outputs")})
        if not self.begin("commit", inputs):
            return
        h = self.home
        manifest = json.loads((Path(self.state.step("export")["outputs"]["dir"]) / "manifest.json").read_text())
        steps = self.state.data["steps"]
        body = (f"HOME {h['tag']} (joint hash {h['joint_hash']}), policy contract {manifest['contract']}.\n\n"
                + "".join(f"{entry['file']} sha256 {entry['sha256']}\n"
                          f"  from {steps[name]['outputs']['run']}/{manifest['checkpoints'][name]} "
                          f"sha256 {entry['checkpoint_sha256']}\n"
                          for name, entry in manifest["policies"].items())
                + f"\nRobot validator and tests pass ({steps['install']['outputs']['robot_tests']}).\n"
                  "Generated by mjlab_microban scripts/retrain_all_for_home.py.")
        title = ("DRY RUN (not deployable): " if self.dry else "") + f"Install the policies retrained at HOME {h['tag']}"
        # The robot's contract constants (src/policy_contract.py: POLICY_CONTRACT,
        # RECIPES) and its contract tests' package (tests/fixtures/policies, a
        # dry-run package of this pipeline) change in the same commit as the
        # policies that need them.
        fixtures = [f"tests/fixtures/policies/{name}"
                    for name in ("walk.onnx", "getup.onnx", "pico_teleop.onnx", "manifest.json")]
        robot_paths = ["config/home_pose.yaml", "src/agents/walk.onnx", "src/agents/getup.onnx",
                       "src/agents/pico_teleop.onnx", "src/agents/manifest.json", "src/policy_contract.py",
                       *fixtures]
        robot_commit = commit(self.robot, robot_paths, f"{title}\n\n{body}")
        self.log(f"robot commit {robot_commit[:12]} on {self.robot_branch}")
        self.push(self.robot, self.robot_branch)
        training_commit = None
        if not self.dry:
            # The training side keeps the released HOME; the release files and
            # their judgments are in the robot commit and the state dir.
            training_commit = commit(REPO, ["config/home_pose.yaml"],
                                     f"Release HOME {h['tag']}\n\n{body}\n\nRobot: "
                                     f"{self.robot_branch} {robot_commit}.")
            self.push(REPO, git(REPO, "rev-parse", "--abbrev-ref", "HEAD"))
        self.finish("commit", {"robot": robot_commit, "training": training_commit}, [])

    def push(self, repo: Path, branch: str) -> None:
        if not self.push_enabled:
            self.log(f"not pushing {repo.name} {branch} ({'dry run' if self.dry else '--no-push'})")
            return
        result = capture(["git", "-C", str(repo), "push", "-u", "origin", branch], timeout=600)
        if result.returncode != 0:
            raise PipelineError(f"git push failed for {repo} {branch}: {result.stderr.strip()[-800:]}")
        self.log(f"pushed {repo.name} {branch}")

    # -- flow ----------------------------------------------------------------------
    def preflight(self) -> None:
        if not (self.robot / ".git").exists():
            raise PipelineError(f"--robot-repo is not a git checkout: {self.robot}", EXIT_INPUT)
        if self.dry:
            push_url = git(self.robot, "remote", "get-url", "--push", "origin", check=False)
            if push_url.startswith(("git@", "ssh://", "http://", "https://")):
                raise PipelineError("--dry-run commits DRYRUN policies locally: --robot-repo must be a scratch "
                                    f"clone whose origin push URL is a local path (got {push_url!r})", EXIT_INPUT)
        current = git(REPO, "rev-parse", "--abbrev-ref", "HEAD")
        if self.training_branch and self.training_branch != current:
            if git(REPO, "rev-parse", "--verify", "--quiet", f"refs/heads/{self.training_branch}", check=False):
                raise PipelineError(f"the training repo is on {current}; switch it to {self.training_branch} first",
                                    EXIT_INPUT)
            git(REPO, "switch", "-c", self.training_branch)
        dirty = sorted({line[3:] for line in git(REPO, "status", "--porcelain", "--untracked-files=all").splitlines()
                        if line.strip()} - {"config/home_pose.yaml"})
        if dirty and not self.dry:
            raise PipelineError(f"the training repo has uncommitted or untracked files besides "
                                f"config/home_pose.yaml: {dirty} (a release is trained from committed code)",
                                EXIT_INPUT)
        current = git(self.robot, "rev-parse", "--abbrev-ref", "HEAD")
        if current != self.robot_branch:
            if git(self.robot, "rev-parse", "--verify", "--quiet", f"refs/heads/{self.robot_branch}", check=False):
                git(self.robot, "switch", self.robot_branch)
            else:
                git(self.robot, "switch", "-c", self.robot_branch)
            self.log(f"robot repo: now on {self.robot_branch} (was {current})")
        self.require_robot_contract()

    def require_robot_contract(self) -> None:
        """The robot checkout implements this repository's contract (before any training)."""

        source = self.robot / "src" / "policy_contract.py"
        if not source.is_file() or not (self.robot / "tools" / "validate_policies.py").is_file():
            raise PipelineError(f"{self.robot} does not implement the policy contract (no src/policy_contract.py "
                                "or tools/validate_policies.py): use a robot branch that implements it "
                                "(docs/policies.md)", EXIT_INPUT)
        robot = robot_contract_constants(source)
        ours = {"POLICY_CONTRACT": self.sched["contract"], "RECIPES": self.sched["recipes"]}
        if robot != ours:
            raise PipelineError(f"the robot's contract {robot} differs from this repository's {ours} "
                                f"({source}); change both sides together", EXIT_INPUT)

    def execute(self) -> int:
        self.state.lock()
        self.state.data.update(home=self.home, dry_run=self.dry, prefix=self.prefix)
        self.state.data.setdefault("created", f"{datetime.now():%F %T}")
        self.state.save()
        self.log(f"==== retrain_all_for_home {'DRY RUN ' if self.dry else ''}HOME {self.home['tag']} "
                 f"(hash {self.home['joint_hash']}), state {self.state.dir}")
        self.preflight()
        for name in STEPS:
            try:
                getattr(self, f"step_{name}")()
            except JobStopped as stopped:
                self.log(f"[{name}] stays running: {str(stopped).splitlines()[0][:300]}; rerun to train or "
                         "evaluate it again")
                raise
            except PipelineError as error:
                self.fail(name, str(error).splitlines()[0][:500] if str(error) else repr(error))
                raise
        self.log("==== ALL STEPS DONE")
        return 0


def stage_tables() -> dict[str, dict[str, int]]:
    """Stage name -> start update of each task's curriculum table (imports the tasks).

    Get-up also logs its refine switch of the exploration (the runner's
    GETUP_REFINE_EXPLORATION_STAGE) at the refine update.
    """

    from mjlab_microban.schedules import GETUP_SCHEDULE
    from mjlab_microban.tasks.microban_getup_env_cfg import GETUP_STAGES
    from mjlab_microban.tasks.microban_getup_runner import GETUP_REFINE_EXPLORATION_STAGE
    from mjlab_microban.tasks.microban_teleop_env_cfg import TELEOP_STAGES
    from mjlab_microban.tasks.microban_velocity_env_cfg import WALK_STAGES

    tables = {kind: {stage.name: stage.iteration for stage in stages}
              for kind, stages in (("walk", WALK_STAGES), ("pico", TELEOP_STAGES), ("getup", GETUP_STAGES))}
    tables["getup"][GETUP_REFINE_EXPLORATION_STAGE] = GETUP_SCHEDULE["refine"]
    return tables


def robot_contract_constants(source: Path) -> dict[str, Any]:
    """POLICY_CONTRACT and RECIPES of the robot's src/policy_contract.py (read, not imported)."""

    import ast

    found: dict[str, Any] = {}
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id in ("POLICY_CONTRACT", "RECIPES") and node.value is not None:
            try:
                found[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def commit(repo: Path, paths: list[str], message: str) -> str:
    existing = [p for p in paths if (repo / p).exists() or git(repo, "ls-files", p)]
    git(repo, "add", "--", *existing)
    if capture(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode != 0:
        result = subprocess.run(["git", "-C", str(repo), "commit", "-q", "-F", "-"], input=message, text=True,
                                capture_output=True)
        if result.returncode != 0:
            raise PipelineError(f"git commit failed in {repo}: {result.stderr.strip()}")
    return git(repo, "rev-parse", "HEAD")


def probe_verdict(receipt: dict[str, Any]) -> dict[str, Any]:
    """Pass/fail and worst margin of one 9x300 walker probe receipt.

    Each moving scenario passes when its mean body twist moves the commanded
    axis the commanded way by the fixed minimum (mjlab_microban/twist_pass_line.py:
    0.1 / 0.2 m/s forward 0.04 / 0.08, backward 0.02 / 0.04, lateral 0.02,
    yaw 0.2; the margin is the signed response minus it); the measured joints
    may overshoot the soft limits by ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD.
    The neutral scenario is checked for completion only, as the PICO source
    gate does.
    """

    from mjlab_microban.teleop_v12_safety import ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
    from mjlab_microban.twist_pass_line import axis_minimums, twist_passes, worst_margin

    summary = receipt["summary"]
    responses, minimums, margins, twists, below = {}, {}, {}, {}, []
    for result in receipt["results"]:
        name = result.get("scenario", result.get("name"))
        command = [float(result["command"][axis]) for axis in ("vx_m_s", "vy_m_s", "yaw_rad_s")]
        measured = result.get("measured_velocity_body", {})
        means = [measured.get(axis, {}).get("mean") for axis in ("vx_m_s", "vy_m_s", "yaw_rad_s")]
        twist = [float("nan") if m is None else float(m) for m in means]
        twists[name] = twist
        if all(x == 0.0 for x in command):
            continue
        minimums[name] = min(axis_minimums(command).values())
        margins[name] = worst_margin(command, twist)
        response = result.get("directional_response") or {}
        if response.get("signed_response") is not None:
            responses[name] = float(response["signed_response"])
        if not twist_passes(command, twist):
            below.append(name)
    worst = min(margins, key=lambda key: margins[key] if margins[key] == margins[key] else float("-inf"))
    overshoot = float(summary.get("maximum_actual_soft_limit_violation_rad", float("inf")))
    ok = (
        not below and len(margins) == 8
        and summary.get("completed_scenario_count") == 9 and summary.get("fall_scenario_count") == 0
        and summary.get("nonfinite_scenario_count", 0) == 0
        and summary.get("directionally_correct_scenario_count") == 8
        and summary.get("raw_action_recurrence_all_steps") is True
        and overshoot <= ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
    )
    return {"ok": ok, "worst_margin": margins[worst], "worst": worst, "below": below,
            "falls": summary.get("fall_scenario_count"), "responses": responses, "minimums": minimums,
            "margins": margins, "twists": twists, "soft_limit_overshoot_rad": overshoot}


WALK_PROBE_RULES = ("falls", "direction", "standing_still")


def walk_failures(probe: dict[str, Any], nine: dict[str, Any]) -> list[str]:
    """The failed items of a walking judgment: the walk probe's rules and the 9x300 probe."""

    failures = [name for name in WALK_PROBE_RULES if not (probe.get("checks") or {}).get(name)]
    return failures + ([] if nine.get("ok") else ["9x300"])


def getup_summary(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    stand = [results[name] for name in ("stand_s11", "stand_s5", "push_s11")]
    push = results["push_s11"]
    return {
        "min_fallen_standing_fraction": min(r["fallen_standing_fraction"] for r in stand),
        "push_fall_fraction": push.get("push_fell_within_3s", 0) / max(1, push.get("push_standing_before", 0)),
        "standing_joint_abs_vel_rad_s": max(r["standing_joint_abs_vel_rad_s"] for r in stand),
        "standing_targets_on_clip": max(r["standing_targets_on_clip"] for r in stand),
        "posture_standing_fraction": results["posture_s11"]["standing_fraction"],
        "final_tilt_deg": max(r["final_tilt_deg"] for r in stand),
    }


def getup_gate_failures(summary: dict[str, float], gate: dict[str, float]) -> list[str]:
    failures = []
    if summary["min_fallen_standing_fraction"] < gate["min_fallen_standing_fraction"]:
        failures.append(f"fallen-start standing {summary['min_fallen_standing_fraction']:.2f} < "
                        f"{gate['min_fallen_standing_fraction']}")
    if summary["push_fall_fraction"] > gate["max_push_fall_fraction"]:
        failures.append(f"push falls {summary['push_fall_fraction']:.2f} > {gate['max_push_fall_fraction']}")
    if summary["standing_joint_abs_vel_rad_s"] > gate["max_standing_joint_abs_vel_rad_s"]:
        failures.append(f"standing tremble {summary['standing_joint_abs_vel_rad_s']:.2f} rad/s > "
                        f"{gate['max_standing_joint_abs_vel_rad_s']}")
    if summary["posture_standing_fraction"] < gate["min_posture_standing_fraction"]:
        failures.append(f"posture standing {summary['posture_standing_fraction']:.2f} < "
                        f"{gate['min_posture_standing_fraction']}")
    if summary["standing_targets_on_clip"] > gate["max_standing_targets_on_clip"]:
        failures.append(f"standing targets on the +-pi clip {summary['standing_targets_on_clip']:.2f} > "
                        f"{gate['max_standing_targets_on_clip']}")
    return failures


def print_status(state_dir: Path) -> int:
    path = state_dir / "state.json"
    if not path.exists():
        print(f"no state in {state_dir}")
        return 1
    data = json.loads(path.read_text())
    print(json.dumps({k: data.get(k) for k in ("home", "dry_run", "prefix", "created")}, indent=1, default=str))
    for name in STEPS:
        record = data.get("steps", {}).get(name, {})
        line = f"{name:8s} {record.get('status', '-'):8s} {record.get('started', '')} -> {record.get('ended', '')}"
        if record.get("error"):
            line += f"  {record['error']}"
        print(line)
    return 0


__all__ = ["Pipeline", "STEPS", "print_status", "LOG_ROOT"]
