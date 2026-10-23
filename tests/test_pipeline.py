"""The release pipeline (src/mjlab_microban/pipeline) on the CPU, without training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mjlab_microban.pipeline import core, steps
from mjlab_microban.pipeline.core import PipelineError, State, hash_inputs, is_training_command, load_config

GATE = {**load_config(dry=False)["getup"]["gate"],
        **steps.stand_still_lines(steps.stand_still_config(load_config(dry=False), "getup"))}


def fake_pipeline(state_dir: Path) -> steps.Pipeline:
    pipeline = steps.Pipeline.__new__(steps.Pipeline)
    pipeline.state = State(state_dir)
    pipeline.cfg = load_config(dry=False)
    pipeline.dry = False
    pipeline.sched = {"walk_total": 30000}
    return pipeline


class StateTest(unittest.TestCase):
    def test_done_with_same_inputs_is_skipped_and_changed_inputs_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = fake_pipeline(root / "state")
            output = root / "model_9.pt"
            output.write_bytes(b"walker")
            self.assertTrue(pipeline.begin("walk", "a" * 64))
            pipeline.finish("walk", {"checkpoint": str(output)}, [output])
            reloaded = fake_pipeline(root / "state")
            self.assertFalse(reloaded.begin("walk", "a" * 64))
            self.assertTrue(reloaded.begin("walk", "b" * 64))  # inputs changed
            self.assertEqual(reloaded.state.step("walk")["status"], "running")

    def test_a_changed_output_file_reruns_the_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = fake_pipeline(root / "state")
            output = root / "model_9.pt"
            output.write_bytes(b"walker")
            pipeline.begin("walk", "a" * 64)
            pipeline.finish("walk", {}, [output])
            output.write_bytes(b"other")
            self.assertTrue(fake_pipeline(root / "state").begin("walk", "a" * 64))

    def test_a_failure_stops_until_the_inputs_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.begin("getup", "a" * 64)
            pipeline.fail("getup", "push falls 0.20 > 0.1")
            again = fake_pipeline(Path(directory) / "state")
            with self.assertRaisesRegex(PipelineError, "push falls"):
                again.begin("getup", "a" * 64)
            self.assertTrue(again.begin("getup", "c" * 64))

    def test_inputs_cover_the_training_keys_not_the_judgment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.cfg = {"seed": 42, "walk": {"task": "T", "envs": 4096, "save_interval": 500,
                                                 "stage_tolerance": 1, "judgment": {"seeds": "101"}}}
            before = pipeline.inputs("walk")
            pipeline.cfg["walk"]["judgment"] = {"seeds": "101,102", "line": 0.5}
            pipeline.cfg["walk"]["stage_tolerance"] = 3
            pipeline.cfg["walk"]["gate"] = {"x": 1}
            self.assertEqual(pipeline.inputs("walk"), before)
            pipeline.cfg["walk"]["envs"] = 2048
            self.assertNotEqual(pipeline.inputs("walk"), before)

    def test_the_walking_judgment(self) -> None:
        probe = {"checks": {"falls": True, "direction": True, "standing_still": True}}
        self.assertEqual(steps.walk_failures(probe, {"ok": True}), [])
        self.assertEqual(steps.walk_failures({"checks": {"falls": True, "direction": False, "standing_still": True}}, {"ok": False}),
                         ["direction", "9x300"])

    def test_the_walking_judgment_on_soft_and_slippery_floors(self) -> None:
        good = {"checks": {"falls": True, "direction": True, "standing_still": True}}
        probe = {"floors": {"soft": good, "slippery": {"checks": {**good["checks"], "falls": False}}}}
        self.assertEqual(steps.floor_failures(["soft", "slippery"], probe, {"soft": {"ok": True}, "slippery": {"ok": True}}),
                         ["slippery:falls"])
        self.assertEqual(steps.floor_failures(["soft", "slippery", "corner"], probe,
                                              {"soft": {"ok": False}, "slippery": {"ok": True}}),
                         ["soft:9x300", "slippery:falls", "corner:missing"])

    def test_the_judged_floors(self) -> None:
        from mjlab_microban.pipeline import floors
        from mjlab_microban.robot.microban_constants import FLOOR_FRICTION, FLOOR_TIME_CONSTANT_S

        judged = floors.parse(load_config(dry=False)["walk"]["judgment"]["floors"])
        # Inside training's range, each softer or more slippery than the play floor (0.02 s, 0.3-1.2), or giving
        # sideways at training's two ends.
        self.assertEqual(judged, {"soft_0.05": (None, 0.05), "slippery_0.3": (0.3, 0.02),
                                  "soft_slippery": (0.3, 0.05), "plate_soft": (None, 0.02),
                                  "plate_stiff": (None, 0.02)})
        self.assertEqual({name: floor.plate for name, floor in judged.items() if floor.plate},
                         {"plate_soft": (500.0, 0.3), "plate_stiff": (5000.0, 1.0)})
        self.assertEqual(floors.RIGID_TIME_CONSTANT_S, FLOOR_TIME_CONSTANT_S[0])
        for bad in ({"time_constant_s": 0.2}, {"time_constant_s": 0.01}, {"friction": FLOOR_FRICTION[0] - 0.01},
                    {"friction": 2.0}, {"softness": 0.1}, {"plate": {"stiffness_n_m": 400.0, "damping_ratio": 0.3}},
                    {"plate": {"stiffness_n_m": 500.0, "damping_ratio": 1.5}}, {"plate": {"stiffness_n_m": 500.0}}):
            with self.assertRaises(ValueError):
                floors.parse({"bad": bad})
        cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=2), events={})
        floors.add_sole_floors(cfg, [(None, 0.05), (0.3, 0.1)])
        params = cfg.events["judgment_floor"].params
        self.assertEqual(params["time_constants"], [0.05, 0.1])
        self.assertTrue(params["frictions"][0] != params["frictions"][0] and params["frictions"][1] == 0.3)
        self.assertEqual(params["asset_cfg"].geom_names, floors.SOLES)
        # The soles whose friction the play env draws (and which have priority over the floor).
        from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

        play = make_microban_velocity_env_cfg(play=True).events["foot_friction"].params["asset_cfg"]
        self.assertEqual(tuple(play.geom_names), floors.SOLES)
        with self.assertRaises(ValueError):
            floors.add_sole_floors(cfg, [(None, 0.05)])

    def test_the_pico_judgment_on_soft_and_slippery_floors(self) -> None:
        from mjlab_microban.scripts import evaluate_teleop_v12_tracking as tracking

        cfg = load_config(dry=False)
        # PICO trains on walking's floors and is judged, as the get-up, on the floors up to 0.05 s.
        floors = cfg["pico"]["judgment"]["floors"]
        self.assertEqual(floors, cfg["getup"]["floors"])
        self.assertTrue(all(f.get("time_constant_s", 0.02) <= 0.05 for f in floors.values()))
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.env = {}
            ran: list[list] = []

            def run(name, cmd, kind, **_):  # the evaluators write their own checks (the rigid floor's lines)
                ran.append(cmd)
                output = Path(cmd[cmd.index("--output") + 1])
                bad = output.name.endswith("soft_slippery_tracking.json")
                output.write_text(json.dumps({"status": "fail" if bad else "pass", "checks": {"push_falls": not bad}}))

            def stand_still(kind, checkpoint, report, time_constants=None, floor=None):
                ran.append(["stand_still", kind, time_constants, floor])
                report.write_text(json.dumps({"status": "pass", "checks": {"falls": True}}))

            pipeline.jobs = SimpleNamespace(run=run)
            pipeline.stand_still = stand_still
            checkpoint = Path(directory) / "model_9.pt"
            checkpoint.write_bytes(b"pico")
            prefix = Path(directory) / "run_model_9"
            self.assertEqual(pipeline.pico_floor_judgment(checkpoint, prefix), ["soft_slippery_tracking:push_falls"])
            evaluated = [cmd for cmd in ran if cmd[0] != "stand_still"]
            self.assertEqual([json.loads(cmd[cmd.index("--floor") + 1]) for cmd in evaluated],
                             [floor for floor in floors.values() for _ in range(2)])
            # Standing still on the soft floors, then on each floor that gives sideways.
            plates = [floor for floor in floors.values() if "plate" in floor]
            self.assertEqual([cmd for cmd in ran if cmd[0] == "stand_still"],
                             [["stand_still", "pico", [0.05], None]]
                             + [["stand_still", "pico", None, floor] for floor in plates])
            self.assertEqual(len(plates), 2)
            Path(f"{prefix}_slippery_0.3_9x300.json").unlink()
            with self.assertRaises(core.JobStopped):  # a report not written is no judgment
                pipeline.pico_failures(prefix, ("slippery_0.3_9x300",))
            pipeline.cfg = {**cfg, "pico": {**cfg["pico"], "judgment": {}}}
            self.assertEqual(pipeline.pico_floor_judgment(checkpoint, prefix), [])
        # The tracking evaluator puts the floor on every environment; the rigid floor has none.
        rigid, _ = tracking._cfg(42, 10)
        soft, _ = tracking._cfg(42, 10, (0.3, 0.1))
        self.assertNotIn("judgment_floor", rigid.events)
        self.assertEqual(soft.events["judgment_floor"].params["time_constants"], [0.1] * soft.scene.num_envs)
        self.assertEqual(soft.events["judgment_floor"].params["frictions"], [0.3] * soft.scene.num_envs)

    def test_the_getup_floors(self) -> None:
        # The walking judgment's floors, on every collision geom of training's robot: named, with priority over the
        # floor, and otherwise the play robot's contacts.
        import re

        import mujoco
        from mjlab.entity import Entity

        from mjlab_microban.pipeline import floors
        from mjlab_microban.tasks.microban_getup_env_cfg import make_microban_getup_env_cfg

        cfg = load_config(dry=False)
        # A soft, a slippery and a soft-and-slippery floor, up to 0.05 s (where the get-up's stands hold), and the
        # floors that give sideways.
        judged_floors = floors.parse(cfg["getup"]["floors"])
        self.assertEqual(sorted(((tuple(f), f.plate) for f in judged_floors.values()), key=str),
                         sorted(((tuple(f), f.plate) for f in floors.parse({
                             "a": {"time_constant_s": 0.05}, "b": {"friction": 0.3},
                             "c": {"time_constant_s": 0.05, "friction": 0.3},
                             "d": {"plate": {"stiffness_n_m": 500.0, "damping_ratio": 0.3}},
                             "e": {"plate": {"stiffness_n_m": 5000.0, "damping_ratio": 1.0}}}).values()), key=str))
        play = make_microban_getup_env_cfg(play=True)
        judged = make_microban_getup_env_cfg(play=True)
        judged.scene.num_envs = 2
        floors.add_body_floors(judged, [(None, 0.05), (0.3, 0.1)])
        params = judged.events["judgment_floor"].params
        self.assertEqual((params["asset_cfg"].geom_names, params["time_constants"]), (floors.BODY, [0.05, 0.1]))
        self.assertEqual(list(judged.events)[-1], "judgment_floor")  # after foot_friction's draw
        self.assertEqual(judged.scene.entities["robot"].articulation, play.scene.entities["robot"].articulation)
        before, after = (Entity(c.scene.entities["robot"]).spec.compile() for c in (play, judged))
        colliding = [g for g in range(after.ngeom) if after.geom_contype[g] or after.geom_conaffinity[g]]
        names = [mujoco.mj_id2name(after, mujoco.mjtObj.mjOBJ_GEOM, g) for g in colliding]
        self.assertEqual(len(colliding), 36)
        self.assertTrue(all(re.match(floors.BODY[0], name) for name in names))
        self.assertTrue((after.geom_priority[colliding] == 1).all())
        soles = [name for name in names if any(re.match(pattern, name) for pattern in floors.SOLES)]
        self.assertEqual(len(soles), 12)
        self.assertEqual(tuple(judged.events["foot_friction"].params["asset_cfg"].geom_names), floors.SOLES)
        for field in ("geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_contype"):
            self.assertTrue((getattr(before, field) == getattr(after, field)).all(), field)
        # A judgment, not a training input.
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.sched = {"getup": {"effort": 10000}, "getup_total": 18000}
            label = pipeline.inputs("getup")
            del pipeline.cfg["getup"]["floors"]
            self.assertEqual(pipeline.inputs("getup"), label)

    def test_the_getup_judgment_on_soft_and_slippery_floors(self) -> None:
        # The fallen starts, the push and the tremble (the "stand" evaluations) and standing still run again on each
        # floor of getup.floors, judged by the rigid floor's lines.
        cfg = load_config(dry=False)
        floors_cfg = cfg["getup"]["floors"]
        stand = {"fallen_standing_fraction": 0.9, "standing_joint_abs_vel_rad_s": 0.1, "standing_targets_on_clip": 0.02,
                 "final_tilt_deg": 3.0, "push_fell_within_3s": 1, "push_standing_before": 60, "standing_fraction": 0.9}
        commands = []
        with tempfile.TemporaryDirectory() as directory:

            def run(name, cmd, kind, env=None, check=True):
                commands.append(cmd)
                log = Path(directory) / f"{len(commands)}.log"
                floor = json.loads(cmd[cmd.index("--floor") + 1]) if "--floor" in cmd else None
                if "mjlab_microban.pipeline.stand_still" in cmd:
                    report = Path(cmd[cmd.index("mjlab_microban.pipeline.stand_still") + 4])
                    rms = 0.2 if floor == floors_cfg["soft_0.05"] else 0.05
                    report.write_text(json.dumps({"leg_speed_rms_rad_s": rms, "worst_delay_ms": 40,
                                                  "falls": 0, "judged_envs": 48, "envs": 48, "status": "pass"}))
                    log.write_text("")
                else:
                    bad = floor == floors_cfg["soft_slippery"]
                    shaking = floor == floors_cfg["slippery_0.3"]
                    log.write_text("RESULT " + json.dumps({**stand, "fallen_standing_fraction": 0.5 if bad else 0.9,
                                                           "standing_joint_abs_vel_rad_s": 0.31 if shaking else 0.1}))
                return 0, log

            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.jobs, pipeline.env, pipeline.log = SimpleNamespace(run=run), {}, lambda message: None
            summary = pipeline.getup_judgment(Path("model_1.pt"), "judgment")
            written = json.loads((Path(directory) / "state" / "getup_judgment.json").read_text())
        rigid = len(cfg["getup"]["evals"]) + 1
        self.assertFalse(any("--floor" in cmd for cmd in commands[:rigid]))
        on_floors = [cmd for cmd in commands[rigid:]]
        for name, floor in floors_cfg.items():
            mine = [cmd for cmd in on_floors if json.loads(cmd[cmd.index("--floor") + 1]) == floor]
            modules = [cmd[cmd.index("-m") + 1].rsplit(".", 1)[1] for cmd in mine]
            self.assertEqual(modules, ["getup_eval"] * 3 + ["stand_still"], name)
            self.assertFalse(any("posture" in cmd for cmd in mine))
        self.assertEqual(len(on_floors), 4 * len(floors_cfg))
        # The get-up's standing still: 360 environments per floor, at most 4 falls (1 in 90).
        still = [cmd for cmd in commands if "mjlab_microban.pipeline.stand_still" in cmd]
        self.assertEqual(len(still), 1 + len(floors_cfg))
        for cmd in still:
            self.assertEqual((cmd[cmd.index("--reps") + 1], cmd[cmd.index("--max-falls") + 1]), ("15", "4"))
        self.assertEqual(GATE["max_stand_still_falls"], 4)
        self.assertEqual((list(summary["floors"]), written["summary"]), (list(floors_cfg), summary))
        self.assertEqual(steps.getup_gate_failures(summary, GATE), [])
        failures = steps.getup_floor_failures(summary["floors"], GATE)
        self.assertEqual(failures, ["soft_0.05: standing still: leg speed RMS 0.2 rad/s > 0.15",
                                    "slippery_0.3: standing tremble 0.31 rad/s > 0.3",
                                    "soft_slippery: fallen-start standing 0.50 < 0.85"])
        # Each of the five items fails on its own with the rigid floor's line.
        good = {**summary["floors"]["soft_0.05"], "stand_still_leg_speed_rms_rad_s": 0.05}
        self.assertEqual(steps.getup_floor_failures({"f": good}, GATE), [])
        for key, value in (("min_fallen_standing_fraction", 0.84), ("push_fall_fraction", 0.11),
                           ("standing_joint_abs_vel_rad_s", 0.31), ("stand_still_leg_speed_rms_rad_s", None),
                           ("stand_still_falls", GATE["max_stand_still_falls"] + 1)):
            with self.subTest(key=key):
                self.assertEqual(len(steps.getup_floor_failures({"f": {**good, key: value}}, GATE)), 1)

    def test_a_step_hashes_only_its_own_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.cfg = {"seed": 42, "walk": {"task": "W"}, "getup": {"task": "G"}}
            pipeline.sched = {"walk_total": 30000, "getup": {"effort": 10000},
                              "getup_total": 18000, "stages": {"walk": {"widen": 3000}, "getup": {"effort": 10000}}}
            walk, getup = pipeline.inputs("walk"), pipeline.inputs("getup")
            pipeline.sched["getup"] = {"effort": 10000, "push": 15000}
            pipeline.sched["stages"]["getup"] = {"effort": 10000, "push": 15000}
            self.assertEqual(pipeline.inputs("walk"), walk)
            self.assertNotEqual(pipeline.inputs("getup"), getup)
            pipeline.sched["walk_total"] = 20000
            self.assertNotEqual(pipeline.inputs("walk"), walk)
            self.assertNotIn("src/mjlab_microban/schedules.py", steps.STEP_INPUTS["walk"])

    def test_the_run_label_is_kept_by_the_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.prefix = "home_h"
            pipeline.begin("pico", "a" * 64)
            self.assertEqual(pipeline.state.step("pico")["label"], "home_h_pico_aaaaaaaa")
            pipeline.state.step("pico")["label"] = "home_h_pico_kept"
            pipeline.state.save()
            again = fake_pipeline(Path(directory) / "state")
            again.prefix = "home_h"
            again.begin("pico", "a" * 64)
            self.assertEqual(again.state.step("pico")["label"], "home_h_pico_kept")

    def test_a_failed_step_is_judged_again_only_when_its_judgment_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.judge_inputs = "j1"
            self.assertTrue(pipeline.begin("walk", "a" * 64))
            pipeline.fail("walk", "direction fails")
            again = fake_pipeline(Path(directory) / "state")
            again.judge_inputs = "j1"  # the same test: a valid failing test stays failed
            with self.assertRaisesRegex(PipelineError, "Question the test first"):
                again.begin("walk", "a" * 64)
            again.judge_inputs = "j2"  # the test was fixed: re-judge, keep the training
            self.assertTrue(again.begin("walk", "a" * 64))
            record = again.state.step("walk")
            self.assertEqual((record["status"], record["inputs"], record["judge_inputs"]),
                             ("running", "a" * 64, "j2"))
            self.assertNotIn("error", record)
            # The trained run's entry gate stays.
            record["walker_probe"] = {"ok": True}
            again.fail("walk", "fails again")
            again.judge_inputs = "j3"
            again.begin("walk", "a" * 64)
            self.assertEqual(again.state.step("walk")["walker_probe"], {"ok": True})

    def test_a_training_started_by_hand_is_found_and_watched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = State(Path(directory) / "state")
            log = state.dir / "logs" / "1007-000000_train_walk.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("run lbl_x\n")
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)", "train", "--agent.run-name",
                                     "lbl_x", "end"], start_new_session=True)
            try:
                self.assertEqual(core.running_training("lbl_x")[0], os.getpgid(proc.pid))
                self.assertIsNone(core.running_training("lbl"))
                jobs = core.Jobs(state, stall_minutes={"train": 20, "eval": 30}, wait_for_gpu=False,
                                 external_min_envs=1024, own_prefix="lbl")
                self.assertIn("--agent.run-name lbl_", jobs.own_marker)
                polls = []
                with mock.patch.object(core.time, "sleep", lambda _s: proc.wait()):
                    jobs.watch("train_walk", os.getpgid(proc.pid), log, poll=lambda: polls.append(1))
                self.assertTrue(polls)
                self.assertFalse(core.group_alive(proc.pid))
            finally:
                proc.kill()
                proc.wait()

    def test_running_steps_are_entered_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.begin("pico", "a" * 64)  # then the process died
            again = fake_pipeline(Path(directory) / "state")
            self.assertTrue(again.begin("pico", "a" * 64))
            self.assertEqual(again.state.step("pico")["inputs"], "a" * 64)

    def test_a_crashed_or_stalled_training_is_run_again_and_a_refusal_fails(self) -> None:
        def run(state_dir: Path, cmd: list[str], kind: str) -> steps.Pipeline:
            pipeline = fake_pipeline(state_dir)
            pipeline.home = {"tag": "t", "joint_hash": "h"}
            pipeline.prefix = "home_h"
            pipeline.jobs = core.Jobs(pipeline.state, stall_minutes={"train": 20, "eval": 30},
                                      wait_for_gpu=False, external_min_envs=1024)
            pipeline.preflight = lambda: None
            for name in steps.STEPS:
                setattr(pipeline, f"step_{name}", lambda: None)

            def step_walk() -> None:
                if pipeline.begin("walk", "a" * 64):
                    pipeline.jobs.run("train_walk" if kind == "train" else "export_walk", cmd, kind, gpu=False)

            pipeline.step_walk = step_walk
            return pipeline

        def execute(pipeline: steps.Pipeline) -> None:
            try:
                pipeline.execute()
            finally:
                pipeline.state._lock_file.close()  # the process ended

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            crash = ["bash", "-c", "exit 1"]  # e.g. CUDA out of memory on the shared GPU
            with self.assertRaises(core.JobStopped) as raised:
                execute(run(state_dir, crash, "train"))
            self.assertEqual(raised.exception.code, core.EXIT_STALL)
            self.assertEqual(State(state_dir).step("walk")["status"], "running")
            with self.assertRaises(core.JobStopped):  # the rerun enters the step again
                execute(run(state_dir, crash, "train"))
            self.assertEqual(State(state_dir).step("walk")["status"], "running")

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            with self.assertRaises(PipelineError) as raised:  # an exporter refusing is a verdict
                execute(run(state_dir, ["bash", "-c", "exit 1"], "eval"))
            self.assertNotIsInstance(raised.exception, core.JobStopped)
            self.assertEqual(State(state_dir).step("walk")["status"], "failed")

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            pipeline = run(state_dir, ["bash", "-c", "sleep 600"], "train")
            pipeline.jobs.stall_s["train"] = 0.0
            with self.assertRaisesRegex(core.JobStopped, "STALL"):
                execute(pipeline)
            self.assertEqual(State(state_dir).step("walk")["status"], "running")

    def test_a_stopped_training_is_trained_again_from_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = fake_pipeline(root / "state")
            pipeline.env = {}
            pipeline.check_home_yaml = lambda: None
            pipeline.sched = {"stages": {"walk": {"widen": 3}}}
            stopped = root / "logs" / "exp" / "2026-10-07_01-00-00_lbl"
            stopped.mkdir(parents=True)
            (stopped / "model_5.pt").write_bytes(b"x")  # a run that stopped at update 5
            commands = []

            def run(name, cmd, kind, **kwargs):
                commands.append(cmd)
                log = pipeline.state.dir / "logs" / "0000-000000_train_walk.log"
                log.write_text("run lbl\nCurriculum stage 1 widen at step 72 (update 3)\n")
                kwargs["on_start"](log)
                kwargs["poll"]()
                fresh = root / "logs" / "exp" / "2026-10-07_02-00-00_lbl"
                (fresh / "params").mkdir(parents=True)
                (fresh / "params" / "agent.yaml").write_text("resume: false\n")
                (fresh / "model_9.pt").write_bytes(b"x")
                return 0, log

            pipeline.jobs = SimpleNamespace(run=run)
            with mock.patch.object(core, "LOG_ROOT", root / "logs"), \
                    mock.patch.object(steps, "running_training", lambda label: None):
                final = pipeline.train("walk", "exp", "lbl", "T", 10, 64, ["--agent.save-interval", "1"])
                self.assertEqual(final.parent.name, "2026-10-07_02-00-00_lbl")
                (command,) = commands
                self.assertIn("--agent.max-iterations", command)
                self.assertEqual(command[command.index("--agent.max-iterations") + 1], "10")
                self.assertFalse(any("resume" in word or "load-checkpoint" in word for word in command))
                self.assertEqual(pipeline.state.step("walk")["stages_seen"], {"widen": 3})
                # The finished run is kept: a rerun judges it again without training.
                pipeline.jobs = SimpleNamespace(run=lambda *a, **k: self.fail("trained again"))
                self.assertEqual(pipeline.train("walk", "exp", "lbl", "T", 10, 64, []), final)

    def test_a_resumed_run_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = fake_pipeline(root / "state")
            pipeline.sched = {"stages": {"walk": {}}}
            pipeline.jobs = SimpleNamespace(run=lambda *a, **k: self.fail("trained"),
                                            watch=lambda *a, **k: self.fail("watched"))
            run = root / "logs" / "exp" / "2026-10-07_01-00-00_lbl"
            (run / "params").mkdir(parents=True)
            (run / "model_9.pt").write_bytes(b"x")
            self.assertTrue(core.resumed_run(run))  # no params/agent.yaml: not known to start at update 0
            (run / "params" / "agent.yaml").write_text("resume: true\nload_run: x\n")
            self.assertTrue(core.resumed_run(run))
            with mock.patch.object(core, "LOG_ROOT", root / "logs"):
                with self.assertRaisesRegex(core.JobStopped, "resumed from a checkpoint") as raised:
                    pipeline.train("walk", "exp", "lbl", "T", 10, 64, [])
                self.assertEqual(raised.exception.code, core.EXIT_INPUT)
                (run / "model_9.pt").unlink()
                cmd = "uv run --locked train T --agent.run-name lbl --agent.resume True --agent.load-run x "
                with mock.patch.object(steps, "running_training", lambda label: (1, cmd)):
                    with self.assertRaisesRegex(core.JobStopped, "running training lbl is resumed"):
                        pipeline.train("walk", "exp", "lbl", "T", 10, 64, [])
            (run / "params" / "agent.yaml").write_text("resume: false\n")
            self.assertFalse(core.resumed_run(run))
            # mjlab writes python tags into params (a PICO run's wandb_tags, for one).
            (run / "params" / "agent.yaml").write_text("wandb_tags: !!python/tuple []\nresume: false\n")
            self.assertFalse(core.resumed_run(run))
            self.assertTrue(core.resumes("train T --agent.resume=true"))
            self.assertFalse(core.resumes("train T --agent.resume False --agent.run-name lbl"))

    def test_one_instance_per_state_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = State(Path(directory))
            first.lock()
            with self.assertRaises(PipelineError) as raised:
                State(Path(directory)).lock()
            self.assertEqual(raised.exception.code, core.EXIT_BUSY)


class InputsTest(unittest.TestCase):
    def test_step_inputs_cover_only_their_training_files(self) -> None:
        walk = set(steps.STEP_INPUTS["walk"])
        self.assertIn("src/mjlab_microban/tasks/microban_velocity_*.py", walk)
        self.assertFalse(any("teleop" in pattern or "getup_env" in pattern for pattern in walk))
        self.assertFalse(any("velocity" in pattern for pattern in steps.STEP_INPUTS["getup"]))
        # PICO trains on the walking task's config (its base) and rewards.
        self.assertIn("src/mjlab_microban/tasks/microban_velocity_env_cfg.py", steps.STEP_INPUTS["pico"])

    def test_hash_changes_with_a_listed_file_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=core.REPO / "logs" if (core.REPO / "logs").is_dir() else None) \
                as directory:
            root = Path(directory)
            listed = root / "listed.py"
            other = root / "other.py"
            listed.write_text("a = 1\n")
            other.write_text("b = 1\n")
            pattern = str(listed.relative_to(core.REPO)) if listed.is_relative_to(core.REPO) else None
            if pattern is None:
                self.skipTest("needs a directory inside the repository")
            before = hash_inputs([pattern], {"x": 1})
            other.write_text("b = 2\n")
            self.assertEqual(hash_inputs([pattern], {"x": 1}), before)
            listed.write_text("a = 2\n")
            self.assertNotEqual(hash_inputs([pattern], {"x": 1}), before)
            self.assertNotEqual(hash_inputs([pattern], {"x": 2}), hash_inputs([pattern], {"x": 1}))


class ConfigTest(unittest.TestCase):
    def test_dry_section_overrides_only_a_dry_run(self) -> None:
        real, dry = load_config(dry=False), load_config(dry=True)
        self.assertEqual(real["walk"]["envs"], 4096)
        self.assertEqual(dry["walk"]["envs"], 64)
        self.assertEqual(dry["walk"]["judgment"], real["walk"]["judgment"])
        self.assertEqual((real["walk"]["stage_tolerance"], dry["walk"]["stage_tolerance"]), (1, 3))
        self.assertNotIn("dry", real)
        self.assertLess(dry["schedule_scale"], 1.0)

    def test_judgment_thresholds(self) -> None:
        self.assertEqual(GATE["max_standing_targets_on_clip"], 0.05)
        self.assertEqual(GATE["min_fallen_standing_fraction"], 0.85)


class JudgmentTest(unittest.TestCase):
    def test_pico_tracking_waiver_covers_accuracy_only(self) -> None:
        failures = ["tracking:foot_error", "soft_0.05_tracking:foot_lift", "slippery_0.3_tracking:push_falls"]
        waiver = {"checks": ["foot_error", "foot_lift"], "note": "why"}
        self.assertEqual(steps.waived_failures(failures, waiver), failures[:2])
        self.assertEqual(steps.waived_failures(failures, {}), [])
        self.assertEqual(steps.waived_failures(["9x300:foot_error"], waiver), [])  # only the tracking reports
        for broken in ({"checks": ["foot_error"], "note": " "}, {"checks": ["push_falls"], "note": "why"},
                       {"checks": ["foot_error", "no_falls_without_push"], "note": "why"},
                       {"checks": ["support_foot_still"], "note": "why"}):
            with self.subTest(broken=broken), self.assertRaises(PipelineError):
                steps.waived_failures(failures, broken)
        # standing_still: only a report whose standing rows neither step nor drift (the ZMP alone).
        thresholds = {"still_touchdowns_per_s_max": 0.5}
        still = {"lifted": [True, True], "mean_twist": [0.0, 0.0, 0.0], "touchdowns_per_s": 0.0}
        arm = {"twist": [0.0, 0.0, 0.0], "mean_twist": [0.0, 0.01, 0.0], "touchdowns_per_s": 0.1}
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            for rows, waived in (([still], True), ([{**still, "touchdowns_per_s": 0.6}], False),
                                 ([{**still, "mean_twist": [0.06, 0.0, 0.0]}], False), ([still, arm], True)):
                feet = [r for r in rows if "lifted" in r]
                arms = [r for r in rows if "twist" in r]
                report.write_text(json.dumps({"thresholds": thresholds, "results": {"feet": feet, "arms": arms}}))
                with self.subTest(rows=rows):
                    self.assertEqual(steps.waived_failures(["tracking:standing_still"],
                                                           {"checks": ["standing_still"], "note": "why"},
                                                           lambda source: report), ["tracking:standing_still"] if waived else [])

    def test_pico_gate_records_the_waiver_of_the_rigid_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.env, pipeline.log = {}, lambda message: None
            pipeline.cfg = {**pipeline.cfg, "pico": {**pipeline.cfg["pico"],
                                                     "tracking_waiver": {"checks": ["foot_error"], "note": "why"}}}
            pipeline.jobs = SimpleNamespace(run=lambda *a, **k: (0, None))
            pipeline.stand_still = lambda *a, **k: None
            created = []
            checkpoint = Path(directory) / "run" / "model_9.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"pico")
            for failures, passed, flags in (
                (["tracking:foot_error", "soft_0.05_tracking:foot_error"], True,
                 ["--waive-tracking-checks", "foot_error", "--waiver-note", "why"]),
                (["soft_0.05_tracking:foot_error"], True, []),
                (["tracking:foot_error", "tracking:push_falls"], False, None),
            ):
                created.clear()
                with mock.patch.object(steps, "capture", lambda cmd, **k: created.append(cmd)), \
                        mock.patch.object(pipeline, "pico_failures", lambda prefix: [f for f in failures if
                                                                                     f.startswith("tracking:")]), \
                        mock.patch.object(pipeline, "pico_floor_judgment", lambda c, prefix: [
                            f for f in failures if not f.startswith("tracking:")]):
                    _, result, reported = pipeline.pico_judgment(checkpoint)
                self.assertEqual((result, reported), (passed, failures))
                if flags is None:
                    self.assertEqual(created, [])
                else:
                    self.assertEqual(created[0][created[0].index("--force") + 1:], flags)

    def test_getup_gate(self) -> None:
        good = {"min_fallen_standing_fraction": 0.9, "push_fall_fraction": 0.05,
                "standing_joint_abs_vel_rad_s": 0.1, "standing_targets_on_clip": 0.03,
                "posture_standing_fraction": 0.9, "final_tilt_deg": 4.0,
                "stand_still_leg_speed_rms_rad_s": 0.05, "stand_still_falls": 0}
        self.assertEqual(steps.getup_gate_failures(good, GATE), [])
        for key, value in (("min_fallen_standing_fraction", 0.8), ("push_fall_fraction", 0.2),
                           ("standing_joint_abs_vel_rad_s", 0.4), ("standing_targets_on_clip", 0.06),
                           ("posture_standing_fraction", 0.7), ("stand_still_leg_speed_rms_rad_s", 0.3),
                           ("stand_still_leg_speed_rms_rad_s", None),
                           ("stand_still_falls", GATE["max_stand_still_falls"] + 1)):
            with self.subTest(key=key):
                self.assertEqual(len(steps.getup_gate_failures({**good, key: value}, GATE)), 1)

    def test_walker_probe_verdict(self) -> None:
        # Each moving scenario passes at the fixed signed-response minimum
        # (0.1 / 0.2 forward 0.04 / 0.08, backward 0.02 / 0.04, lateral 0.02,
        # yaw 0.2); the joints may overshoot 0.25 rad.
        commands = {
            "forward_0p1": (0.1, 0.0, 0.0), "forward_0p2": (0.2, 0.0, 0.0),
            "backward_0p1": (-0.1, 0.0, 0.0), "backward_0p2": (-0.2, 0.0, 0.0),
            "lateral_left_0p1": (0.0, 0.1, 0.0), "lateral_right_0p1": (0.0, -0.1, 0.0),
            "yaw_left_0p5": (0.0, 0.0, 0.5), "yaw_right_0p5": (0.0, 0.0, -0.5),
        }
        axes = ("vx_m_s", "vy_m_s", "yaw_rad_s")

        def result(name, twist, measured):
            index = next((i for i, v in enumerate(twist) if v != 0.0), None)
            return {"scenario": name, "command": dict(zip(axes, twist, strict=True)),
                    "measured_velocity_body": {a: {"mean": m} for a, m in zip(axes, measured, strict=True)},
                    "directional_response": None if index is None else {
                        "signed_response": measured[index] * (1 if twist[index] > 0 else -1)}}

        def receipt(measured, overshoot=0.2):
            return {
                "summary": {"completed_scenario_count": 9, "fall_scenario_count": 0, "nonfinite_scenario_count": 0,
                            "directionally_correct_scenario_count": 8, "raw_action_recurrence_all_steps": True,
                            "maximum_actual_soft_limit_violation_rad": overshoot},
                "results": [result("neutral", (0.0, 0.0, 0.0), (0.03, 0.0, 0.0))]
                + [result(name, twist, measured(twist)) for name, twist in commands.items()],
            }

        # Half of every command: 0.01 above the backward / lateral minimums.
        verdict = steps.probe_verdict(receipt(lambda c: [0.5 * x for x in c]))
        self.assertTrue(verdict["ok"])
        self.assertAlmostEqual(verdict["worst_margin"], 0.01, places=6)
        # A tenth of every command is below them.
        self.assertFalse(steps.probe_verdict(receipt(lambda c: [0.1 * x for x in c]))["ok"])
        self.assertTrue(steps.probe_verdict(receipt(lambda c: [2.0 * x for x in c]))["ok"])
        # A walker that answers 0.1 m/s forward by moving back (-0.0396 m/s).
        bad = receipt(lambda c: [0.5 * x for x in c])
        bad["results"][1]["measured_velocity_body"]["vx_m_s"]["mean"] = -0.0396
        verdict = steps.probe_verdict(bad)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["below"], ["forward_0p1"])
        # Drift on the other axes does not count.
        drift = receipt(lambda c: [0.5 * x for x in c])
        drift["results"][5]["measured_velocity_body"]["yaw_rad_s"]["mean"] = 1.2
        self.assertTrue(steps.probe_verdict(drift)["ok"])
        # The joints may overshoot the soft limits by 0.25 rad.
        self.assertTrue(steps.probe_verdict(receipt(lambda c: [0.5 * x for x in c], 0.25))["ok"])
        self.assertFalse(steps.probe_verdict(receipt(lambda c: [0.5 * x for x in c], 0.2501))["ok"])

    def test_getup_summary(self) -> None:
        stand = {"fallen_standing_fraction": 0.9, "standing_joint_abs_vel_rad_s": 0.1,
                 "standing_targets_on_clip": 0.02,
                 "final_tilt_deg": 3.0}
        summary = steps.getup_summary({
            "stand_s11": stand, "stand_s5": {**stand, "fallen_standing_fraction": 0.88},
            "push_s11": {**stand, "push_fell_within_3s": 3, "push_standing_before": 60},
            "posture_s11": {"standing_fraction": 0.95},
            "stand_still": {"leg_speed_rms_rad_s": 0.04, "worst_delay_ms": 50, "falls": 0, "judged_envs": 80,
                            "envs": 80},
        })
        self.assertEqual(summary["min_fallen_standing_fraction"], 0.88)
        self.assertAlmostEqual(summary["push_fall_fraction"], 0.05)
        self.assertEqual((summary["stand_still_leg_speed_rms_rad_s"], summary["stand_still_falls"]), (0.04, 0))


class GpuTest(unittest.TestCase):
    def test_what_counts_as_another_training(self) -> None:
        self.assertTrue(is_training_command("/x/.venv/bin/python /x/.venv/bin/train Mjlab-Velocity-Microban", 1024))
        self.assertTrue(is_training_command("python -m x --env.scene.num-envs 4096", 1024))
        self.assertFalse(is_training_command("python -m x --env.scene.num-envs 64", 1024))
        self.assertFalse(is_training_command("python -m mjlab_microban.pipeline.getup_eval stand", 1024))

    def test_a_dry_run_does_not_wait(self) -> None:
        jobs = core.Jobs(SimpleNamespace(log=print), stall_minutes={"train": 1, "eval": 1}, wait_for_gpu=False,
                         external_min_envs=1024)
        with mock.patch.object(core.Jobs, "external_training", side_effect=AssertionError("queried")):
            jobs.wait_for_free_gpu("x")


class FindCheckpointTest(unittest.TestCase):
    def test_the_newest_run_of_the_label_with_that_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run, models in (("2026-10-07_01-00-00_label", (0, 500, 1000)),
                                ("2026-10-07_02-00-00_label", (0, 500)),
                                ("2026-10-07_03-00-00_other", (9000,))):
                (root / "exp" / run).mkdir(parents=True)
                for index in models:
                    (root / "exp" / run / f"model_{index}.pt").write_bytes(b"x")
            with mock.patch.object(core, "LOG_ROOT", root):
                self.assertEqual(core.find_checkpoint("exp", "label", 500).parent.name,
                                 "2026-10-07_02-00-00_label")
                self.assertEqual(core.find_checkpoint("exp", "label", 1000).parent.name,
                                 "2026-10-07_01-00-00_label")
                self.assertIsNone(core.find_checkpoint("exp", "label", 9000))


class StatusTest(unittest.TestCase):
    def test_status_lists_every_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = State(Path(directory))
            state.data["steps"]["walk"] = {"status": "done"}
            state.save()
            with mock.patch("builtins.print") as printed:
                self.assertEqual(steps.print_status(Path(directory)), 0)
            text = "\n".join(str(call.args[0]) for call in printed.call_args_list)
            for name in steps.STEPS:
                self.assertIn(name, text)
            self.assertIn(json.dumps(None), text)


class DryPackageRowsTest(unittest.TestCase):
    def test_forced_tracking_report_keeps_late_arm_rows(self) -> None:
        import tempfile

        from mjlab_microban.pipeline import dry

        def row(foot: float, arm: float) -> list[float]:
            values = [0.0] * 81
            values[5] = -1.0
            values[71] = foot
            values[75] = arm
            return values

        probe = {"summary": {}, "results": [{} for _ in range(9)]}
        for count, expected in ((40, 40), (5, dry.SMOKE_ROWS)):
            rows = [row(0.01, 0.0)] * (count // 2) + [row(0.0, 0.2)] * (count - count // 2)
            with tempfile.TemporaryDirectory() as directory:
                prefix = Path(directory) / "run"
                Path(f"{prefix}_9x300.json").write_text(json.dumps(probe))
                Path(f"{prefix}_tracking.json").write_text(
                    json.dumps({"status": "fail", "runtime_smoke_observations": rows})
                )
                _locomotion, tracking = dry.forced_reports(str(prefix), Path(directory), 0.25)
                kept = json.loads(tracking.read_text())["runtime_smoke_observations"]
            self.assertEqual(len(kept), expected)
            self.assertTrue(any(value[75] != 0.0 for value in kept))


if __name__ == "__main__":
    unittest.main()
