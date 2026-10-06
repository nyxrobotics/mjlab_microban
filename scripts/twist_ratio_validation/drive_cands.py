"""Run the ratio-angle candidates in series (rules: AB_result.md, written before the runs).

usage: drive_cands.py   (C2 must already be running as trw8C2_s42; then C1, C3)
Applies the early-stop rules at 3000 and the 08:55 deadline; logs to walk_status.txt.
"""
import json, os, signal, subprocess, sys, time
from datetime import datetime

SPT = os.path.dirname(os.path.abspath(__file__))
WT = f"{SPT}/wt"
M = "/home/rtx-server/Git-projects/mjlab_microban_lean2"
LOGDIR = f"{WT}/logs/rsl_rl/mjlab_microban_velocity"
STATUS = f"{SPT}/walk_status.txt"
DEADLINE = datetime(2026, 10, 7, 8, 55)
sys.path.insert(0, SPT)
from rules_walk import evaluate  # noqa: E402

PLAN = [("trw8C2_s42", "C2"), ("trw9C1_s42", "C1"), ("trw10C3_s42", "C3")]


def log(msg):
    line = f"{datetime.now():%F %T} {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def pgid_of(run):
    out = subprocess.run(["bash", "-c", f"ps -eo pgid,cmd | grep -E 'run-name {run}$' | grep -v grep | awk '{{print $1}}' | head -1"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else None


def stop(run, why):
    g = pgid_of(run)
    if g:
        os.killpg(g, signal.SIGTERM)
        time.sleep(15)
    for d in os.listdir(LOGDIR):
        if d.endswith("_" + run):
            open(f"{LOGDIR}/{d}/DO_NOT_USE.txt", "w").write(f"Stopped {datetime.now():%F %T}: {why}. Candidate run of the ratio-angle search; do not use.\n")
    log(f"{run} stopped: {why}")


def result(run, k):
    path = f"{SPT}/probes/w_{run}_{k}.json"
    have = os.path.exists(path) and any(f"{run} model_{k}: it" in line for line in open(STATUS))
    if not have:
        return None
    nd, pd, pure, wn, wp, ch = evaluate(json.load(open(path)))
    return {"passes": sum(ch.values()), "none_worst": wn, "W5": ch["W5"]}


def start(run, cand):
    env = {**os.environ, "PYTHONPATH": f"{WT}/src", "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.Popen([f"{M}/.venv/bin/train", f"Mjlab-Velocity-TwistRatio{cand}-Microban", "--env.scene.num-envs", "4096",
                      "--env.seed", "42", "--agent.seed", "42", "--agent.logger", "tensorboard", "--agent.upload-model", "False",
                      "--agent.max-iterations", "5001", "--agent.run-name", run],
                     cwd=WT, stdout=open(f"{SPT}/train_{run}.log", "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    subprocess.Popen(["python3", f"{SPT}/watch_walk.py", run], cwd=SPT, stdout=open(f"{SPT}/watch_{run}.out", "w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    subprocess.Popen([f"{SPT}/auto_append_runs.sh", run], cwd=SPT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    head = subprocess.run(["git", "-C", WT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    log(f"start {run} ({cand}, form B3 settings per MICROBAN_VELOCITY_TWIST_RATIO_CANDIDATES, commit {head}, max 5001)")


def watcher_alive(run):
    return subprocess.run(["pgrep", "-f", f"watch_walk.py {run}$"], capture_output=True).returncode == 0


def main():
    for index, (run, cand) in enumerate(PLAN):
        if index > 0:
            if datetime.now() > DEADLINE:
                break
            start(run, cand)
        while True:
            time.sleep(60)
            if datetime.now() > DEADLINE:
                stop(run, "the 08:55 deadline of the search")
                log("deadline reached; search ends")
                return
            r2, r3 = result(run, 2000), result(run, 3000)
            if r3 is not None:
                reasons = []
                if r3["passes"] <= 1:
                    reasons.append(f"3000 passes {r3['passes']} <= 1")
                if r3["none_worst"] >= 31.1:
                    reasons.append(f"3000 none worst angle {r3['none_worst']:.1f} >= 31.1")
                if r2 is not None and not r2["W5"] and not r3["W5"]:
                    reasons.append("W5 failed at 2000 and 3000")
                if reasons and pgid_of(run):
                    stop(run, "early rule: " + "; ".join(reasons))
                    break
            if pgid_of(run) is None and not watcher_alive(run):
                log(f"{run} finished (trainer exited and its checkpoints are probed)")
                break
    log("candidate search driver done")


if __name__ == "__main__":
    main()
