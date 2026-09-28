# Learning a disturbance-resistant HOME

`learn_robust_home.py` learns the **HOME angles themselves** in one simulation
run. It samples continuous, left/right symmetric hip, ankle, and shoulder pitch
angles, starts the robot at each pose, perturbs all 21 joints, applies body
velocity pushes, and updates a Gaussian distribution toward the poses that
remain upright. This is the cross-entropy method (CEM), a derivative-free
optimization method. There is no separate PPO training run for each pose and
no manually enumerated angle grid.

The knee remains at 0°, the trunk starts vertical, and other HOME angles come
from `HOME_FRAME`. Hip search is −20° to +5°, ankle −10° to +10°, and shoulder
−5° to +20°. Each sampled HOME is placed with the lowest foot collision
corner on the floor. Every candidate receives the same seed-specific initial
21-joint angle error, distinct low-frequency target waveform on **all 21
joints**, and body push. A quiet hold with no waveform or push is also part of
the score. By default, both P=125 and P=900 are evaluated with the same position-hold
control law, actuator voltage/drop/delay, collision model, and trial seeds.
Startup model randomization is disabled, so parallel candidates do not get
different actuator or friction draws. The score prioritizes survival and
uprightness; it does not reward closeness to either historical HOME.

Run a search on a CUDA-capable PC with the project dependencies installed:

```bash
uv run --locked python -m mjlab_microban.scripts.learn_robust_home \
  --population=32 --generations=30 --train-seeds=3 \
  --heldout-seeds=12 --test-seeds=12 --duration-s=5 \
  --output=artifacts/home_search/learned_home.json
```

`training_history` records every sampled pose and score. Each generation gets
fresh, paired training seeds. Old (−10°, 0°, +10°) and centered
(+1.198384°, −1.198384°, 0°) homes are included as fixed reference points.
The best pose from each generation is ranked on **different** selection seeds.
`selected_by_validation` may be either a learned pose or one of the baselines.
`selected_home_joint_deg` expands that choice into all 21 joint angles for
machine-readable use. The best learned pose and both baselines are then compared using independent
test seeds that were never used for CEM updates or finalist selection. The
report includes individual trial outcomes, quiet-hold outcomes, model hashes,
and simulation settings. A short smoke run is possible with
`--population=4 --generations=1 --train-seeds=1 --heldout-seeds=1
--test-seeds=1 --duration-s=2 --push-at-s=0.6`.

This optimizes **static standing under perturbations** in the current MuJoCo
model. It does not prove which HOME makes the deployed walking policy most
stable. In the corrected zero-perturbation diagnostic, the old physical HOME
fell in simulation in 1.56–2.36 seconds despite being observed to walk better
on the robot. That measured sim-to-real disagreement must be resolved before
using a learned angle on hardware. The experiment writes only a JSON report;
it does not change `HOME_FRAME`, retrain `walk.onnx`, or send commands to the
robot.

## 2026-09-28 search result

An actual search used 24 parallel candidates for 8 generations. Each
generation used two fresh training seeds; six different seeds selected the
finalist, and another six independent seeds tested it. All trials ran for 5 s
at both P=125 and P=900. The exact command was:

```bash
PYTHONPATH=src /home/dev/Git-projects/mjlab_microban/.venv/bin/python \
  -m mjlab_microban.scripts.learn_robust_home \
  --population=24 --generations=8 --train-seeds=2 \
  --heldout-seeds=6 --test-seeds=6 --duration-s=5 \
  --kp-values=125,900 \
  --output=artifacts/home_search/learned_home_8gen.json
```

The selected simulation HOME was **hip −1.6515°, ankle −1.6333°, shoulder
+2.0987°**, applied symmetrically to both sides. Knees stayed at 0°. On the
independent test, each pose had 26 trials: six seeds × two body push levels ×
two gains, plus a quiet hold at each gain.

| HOME | Falls / 26 | Mean survival | Quiet hold at P=125 / P=900 |
| --- | ---: | ---: | --- |
| Old physical (−10°, 0°, +10°) | 26 | 2.00 s | Fell at 1.56 s / 2.36 s |
| Centered (+1.1984°, −1.1984°, 0°) | 13 | 4.14 s | Fell at 3.66 s / survived 5 s |
| Learned (−1.6515°, −1.6333°, +2.0987°) | 6 | 4.82 s | Survived 5 s / survived 5 s |

The learned pose was better in this **simulation standing** experiment; it
still fell in 6 of 26 trials, all at P=125. The old HOME's poor simulation
result conflicts with its observed walking behavior on the physical robot.
Therefore these numbers do not establish a walking-optimal or physical
robot-optimal HOME. The full local result is in the ignored file
`artifacts/home_search/learned_home_8gen.json`. No physical HOME or walking
policy was changed.

## Longer standing horizon

A follow-up repeated the automatic search with 10-second trials and P=125,
the gain at which the first learned pose eventually fell. It began from the
5-second result and used 24 candidates × 8 generations, two fresh training
seeds per generation, six finalist-selection seeds, and six independent test
seeds. The selected angles were **hip −2.7852°, ankle −1.5259°, shoulder
+2.0940°** (both sides; knees 0°). Its independent P=125 test averaged
5.96 seconds of survival, compared with 3.28 seconds for the centered HOME;
it still fell in 12 of 13 trials. A quiet, unshaken hold lasted 7.72 seconds.
The local full report is `artifacts/home_search/learned_home_10s_8gen.json`.
The search command was:

```bash
uv run --locked python -m mjlab_microban.scripts.learn_robust_home \
  --population=24 --generations=8 --train-seeds=2 \
  --heldout-seeds=6 --test-seeds=6 --duration-s=10 \
  --kp-values=125 --initial-mean=-1.6514866926214435,-1.633292297383672,2.098672410301831 \
  --initial-sigma=3,3,4 --seed=20260929 \
  --output=artifacts/home_search/learned_home_10s_8gen.json
```

A further paired 10-second check gave the old, centered, 5-second learned,
and 10-second learned poses the same six new disturbance seeds at both P=125
and P=900. Results were:

| HOME | Falls / 26 | Mean survival | Quiet hold at P=125 / P=900 |
| --- | ---: | ---: | --- |
| Old physical | 26 | 2.03 s | 1.56 s / 2.36 s |
| Centered | 13 | 7.04 s | 3.66 s / survived 10 s |
| 5-second search | 13 | 7.73 s | 5.50 s / survived 10 s |
| 10-second search | 12 | 8.02 s | 7.72 s / survived 10 s |

The 10-second search pose ranks highest under these static simulation
conditions, but even it does not stand indefinitely at P=125. The paired
records are in the ignored local file
`artifacts/home_search/learned_home_paired_10s.json`. No angle was written to
the robot or walking training configuration because the static simulator
still disagrees with the observed physical walking behavior.
