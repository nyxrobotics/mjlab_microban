"""After the candidate driver ends: continue C1 from model_3000 to 5000 (declared 04:06 in AB_result.md)."""
import os, subprocess, time
from datetime import datetime

SPT = os.path.dirname(os.path.abspath(__file__))
WT = f"{SPT}/wt"
M = "/home/rtx-server/Git-projects/mjlab_microban_lean2"
LOGDIR = f"{WT}/logs/rsl_rl/mjlab_microban_velocity"
STATUS = f"{SPT}/walk_status.txt"
LATEST_START = datetime(2026, 10, 7, 7, 50)  # 2000 updates + probes must fit before 08:55


def log(msg):
    with open(STATUS, "a") as f:
        f.write(f"{datetime.now():%F %T} {msg}\n")


while subprocess.run(["pgrep", "-f", "drive_cands.py"], capture_output=True).returncode == 0:
    time.sleep(30)
while subprocess.run(["pgrep", "-f", "agent.run-name trw10C3_s42$"], capture_output=True).returncode == 0:
    time.sleep(30)
if datetime.now() > LATEST_START:
    log("C1 continuation not started: too late for 2000 updates before 08:55")
    raise SystemExit(0)
parent = sorted(d for d in os.listdir(LOGDIR) if d.endswith("_trw9C1_s42"))[-1]
env = {**os.environ, "PYTHONPATH": f"{WT}/src", "PYTHONDONTWRITEBYTECODE": "1"}
subprocess.Popen([f"{M}/.venv/bin/train", "Mjlab-Velocity-TwistRatioC1-Microban", "--env.scene.num-envs", "4096",
                  "--env.seed", "42", "--agent.seed", "42", "--agent.logger", "tensorboard", "--agent.upload-model", "False",
                  "--agent.resume", "True", "--agent.load-run", parent, "--agent.load-checkpoint", "model_3000.pt",
                  "--agent.max-iterations", "2001", "--agent.run-name", "trw11C1c_s42"],
                 cwd=WT, stdout=open(f"{SPT}/train_trw11C1c_s42.log", "w"), stderr=subprocess.STDOUT, env=env,
                 start_new_session=True)
time.sleep(60)
subprocess.Popen(["python3", f"{SPT}/watch_walk.py", "trw11C1c_s42"], cwd=SPT, stdout=open(f"{SPT}/watch_trw11C1c_s42.out", "w"),
                 stderr=subprocess.STDOUT, start_new_session=True)
subprocess.Popen([f"{SPT}/auto_append_runs.sh", "trw11C1c_s42"], cwd=SPT, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, start_new_session=True)
log(f"start trw11C1c_s42 (C1 continued from {parent}/model_3000 to 5000, declared 04:06)")
deadline = datetime(2026, 10, 7, 8, 55)
while subprocess.run(["pgrep", "-f", "agent.run-name trw11C1c_s42$"], capture_output=True).returncode == 0:
    if datetime.now() > deadline:
        out = subprocess.run(["bash", "-c", "ps -eo pgid,cmd | grep -E 'run-name trw11C1c_s42$' | grep -v grep | awk '{print $1}' | head -1"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            os.killpg(int(out), 15)
        log("trw11C1c_s42 stopped at the 08:55 deadline")
        break
    time.sleep(30)
log("C1 continuation driver done")
