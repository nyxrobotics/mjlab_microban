"""The release pipeline behind scripts/retrain_all_for_home.py.

One command trains, judges, exports, installs and commits the three policies
at the HOME of config/home_pose.yaml (docs/home_pose_workflow.md):

* ``core``: configuration (config/pipeline.yaml), the state file, input
  hashes, the job runner (stall detection, own process groups, waiting for
  another training on the GPU);
* ``steps``: home -> walk -> pico -> getup -> export -> install -> commit;
* ``dry``: the stand-ins a --dry-run uses where a from-scratch policy of a
  few updates cannot pass a check (never part of a release);
* ``home_check`` / ``getup_eval``: the HOME balance report and the get-up
  evaluations, run as subprocesses.
"""
