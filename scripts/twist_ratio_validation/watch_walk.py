"""Probe the twist-ratio walking runs every 1000 iterations (rules: walk_status.txt).

usage: watch_walk.py RUN_NAME [RUN_NAME ...]   (e.g. trw_s42 trw_s43)
For each run's model_K.pt with K % 1000 == 0 (and the final checkpoint): the held-out probe
(vprobe_walk.py + rules_walk.py, appended to walk_status.txt) and, for reference only, the
official 9x300 source probe (seed 42).  Exits when every trainer has exited and its last
checkpoint is probed.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

SPT = os.path.dirname(os.path.abspath(__file__))
WT = f"{SPT}/wt"
M = "/home/rtx-server/Git-projects/mjlab_microban_lean2"
LOGDIR = f"{WT}/logs/rsl_rl/mjlab_microban_velocity"
STATUS = f"{SPT}/walk_status.txt"
THRESH = {"forward_0p1": 0.04, "forward_0p2": 0.08, "backward_0p1": 0.02, "backward_0p2": 0.04,
          "lateral_left_0p1": 0.02, "lateral_right_0p1": 0.02, "yaw_left_0p5": 0.2, "yaw_right_0p5": 0.2}


def log(msg: str) -> None:
    line = f"{datetime.now():%F %T} {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def run_dir(name: str):
    runs = sorted(d for d in os.listdir(LOGDIR) if d.endswith("_" + name))
    return f"{LOGDIR}/{runs[-1]}" if runs else None


def trainer_alive(name: str) -> bool:
    out = subprocess.run(["pgrep", "-f", f"agent.run-name {name}$"], capture_output=True, text=True).stdout
    return bool(out.strip())


def source_probe(ckpt: str, tag: str) -> str:
    sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    out = f"{SPT}/probes/src_{tag}.json"
    env = {**os.environ, "PYTHONPATH": f"{WT}/src", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [f"{M}/.venv/bin/python", "-m", "mjlab_microban.scripts.probe_legacy_actor_in_teleop_env",
         "--checkpoint", ckpt, "--expected-sha256", sha, "--output", out, "--force"],
        cwd=WT, capture_output=True, text=True, env=env)
    if not os.path.exists(out):
        return f"9x300 probe failed rc={proc.returncode}: {proc.stderr[-200:]!r}"
    d = json.load(open(out))
    s = d["summary"]
    resp, below = {}, []
    for r in d["results"]:
        dr = r.get("directional_response")
        name = r.get("scenario", r.get("name"))
        if dr is not None:
            resp[name] = dr["signed_response"]
            if dr["signed_response"] < THRESH.get(name, 0.0):
                below.append(name)
    ok = not below and s["fall_scenario_count"] == 0 and s["completed_scenario_count"] == 9
    cells = " ".join(f"{k.replace('_0p', '')}={v:.3f}" for k, v in resp.items())
    return f"{'PASS' if ok else 'FAIL'} falls={s['fall_scenario_count']} {cells} below={below}"


def main() -> None:
    names = sys.argv[1:]
    done = {n: set() for n in names}
    while True:
        alive = {n: trainer_alive(n) for n in names}
        todo = []
        for n in names:
            d = run_dir(n)
            if d is None:
                continue
            cks = sorted(int(m.group(1)) for f in os.listdir(d) if (m := re.fullmatch(r"model_(\d+)\.pt", f)))
            for it in cks:
                last = (not alive[n]) and it == cks[-1]
                if it not in done[n] and it > 0 and (it % 1000 == 0 or last):
                    todo.append((it, n, d))
        for it, n, d in sorted(todo):
            ck = f"{d}/model_{it}.pt"
            time.sleep(10)
            tag = f"{n}_{it}"
            rc = subprocess.run([f"{SPT}/run_walk_probe.sh", ck, tag]).returncode
            if rc == 0:
                line = subprocess.run(["python3", f"{SPT}/rules_walk.py", f"{SPT}/probes/w_{tag}.json", "--line"],
                                      capture_output=True, text=True).stdout.strip()
            else:
                line = f"held-out probe rc={rc} (probes/w_{tag}.log)"
            log(f"{n} model_{it}: {line}")
            log(f"{n} model_{it}: reference 9x300 source probe (seed 42, not a criterion): {source_probe(ck, tag)}")
            done[n].add(it)
        if not any(alive.values()) and not todo:
            log("watch_walk DONE: all trainers exited and their checkpoints are probed")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
