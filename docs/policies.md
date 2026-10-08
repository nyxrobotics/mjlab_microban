# 方策と実機の契約（`microban-policy-1`）

学習側（このリポジトリ）が作り、実機側（microban）が受け取るものの取り決め。契約を決めているのは実機側
（microban の `src/policy_contract.py`）で、学習側はそれと同じものを
`src/mjlab_microban/policy_contract.py` の1か所で書く。互換性は次の3つで確かめ、ソースのハッシュはどちらの
側も使わない。

1. 契約の版 `microban_policy_contract` と、方策ごとのレシピ id `microban_recipe`（観測・行動・目標の意味や
   報酬の系統が変わったら、両側で手で上げる。学習側の `RECIPES` と実機側の `RECIPES` は同じ値で、
   `retrain_all_for_home.py` は学習を始める前に実機側の値と照合する）。
2. HOME のスタンプ `home_pose`（全精度。実機の `config/home_pose.yaml` と 1e-6 で一致すること）。
3. 起動時の自己テスト: 書き出したチェックポイントのロールアウトで actor に入った観測（8〜64 行）と、
   それぞれに対する torch の actor の決定的な出力を ONNX に入れ、実機は起動のたびに ONNX Runtime で
   同じ出力が出ることを確かめる（各行 `max|ORT − 記録| ≤ 1e-4 + 1e-5 × max|記録|`）。書き出しの道具も
   公開の前に同じ規則で自分のファイルを確かめる（`policy_contract.check_self_test`）。

## リリースの中身

`scripts/retrain_all_for_home.py` の export 段階が `<状態>/release/` に書き、install 段階が実機の
`src/agents/` にコピーする。実機の `config/home_pose.yaml`（`schema_version: 2`、HOME ごとの契約文字列は
持たない）は同じ段階で `config/home_pose_tool.py write-robot` が書く。

| ファイル | 中身 |
| --- | --- |
| `walk.onnx` | 歩行（入力 `obs` [1, 63]、出力 `actions` [1, 18]） |
| `getup.onnx` | 起き上がり（入力 [1, 60]） |
| `pico_teleop.onnx` | PICO（入力 [1, 81]） |
| `manifest.json` | 3つのファイルを1つのリリースに結びつける（下） |
| `walk_gate.json`、`getup_gate.json` | 歩行と起き上がりの判定の記録（`gate_report_sha256` の元。実機にはコピーしない） |

### manifest.json

```json
{
  "contract": "microban-policy-1",
  "home_tag": "<config/home_pose.yaml の tag>",
  "training_commit": "<学習リポジトリの 40 桁の commit>",
  "dry_run": false,
  "policies": {
    "walk":  {"file": "walk.onnx",        "sha256": "...", "checkpoint_sha256": "..."},
    "getup": {"file": "getup.onnx",       "sha256": "...", "checkpoint_sha256": "..."},
    "pico":  {"file": "pico_teleop.onnx", "sha256": "...", "checkpoint_sha256": "..."}
  },
  "home_joint_hash": "...", "home_yaml_sha256": "...", "training_branch": "...",
  "judgments": {...}, "checkpoints": {"walk": "model_N.pt", ...}, "created": "..."
}
```

実機は `contract`、`home_tag`、`dry_run`、各ファイルの SHA-256、ONNX の `checkpoint_sha256` との一致、
PICO の `pico_walk_checkpoint_sha256 == policies.walk.checkpoint_sha256` を確かめる。最後の行のキーは記録用で、
実機は読まない。

## 3つの ONNX に共通のメタデータ（値はすべて文字列）

| キー | 値 |
| --- | --- |
| `microban_policy_contract` | `microban-policy-1` |
| `microban_policy_kind` | `walk` / `getup` / `pico` |
| `microban_recipe` | `microban-walk-track-velocity-1` / `microban-getup-single-run-1` / `microban-pico-arm-overlay-residual-track-velocity-1` |
| `home_pose` | `{"joint_pos_rad":{21 関節},"root_pos_m":[3],"root_quat_wxyz":[4]}`（全精度） |
| `joint_names`、`default_joint_pos` | 21 関節と、その順の HOME（`repr` の全精度） |
| `action_joint_names` | 18 関節（実機の `OBSERVATION_DOF_ORDER`） |
| `action_scale` | `1.0` |
| `action_clip_lower` / `action_clip_upper` | 18 個の −π / +π（全精度） |
| `observation_schema_json`、`observation_joint_names` | 観測の項と幅、関節の並び（PICO は首の3関節が先頭） |
| `previous_action_semantics` | `raw_policy_output` |
| `base_ang_vel_frame`、`control_hz` | `imu_sensor_xyz`、`50` |
| `checkpoint_filename`、`checkpoint_iteration`、`checkpoint_sha256` | `model_N.pt`、`N`、その SHA-256 |
| `gate_status`、`gate_report_sha256` | `pass` と判定の記録の SHA-256（歩行・起き上がりは `*_gate.json`、PICO はステージゲート） |
| `self_test_observations_json`、`self_test_actions_json` | 起動時の自己テストの観測と記録した出力 |
| `dry_run_not_deployable` | 試運転のパッケージだけ `true` |

自己テストの観測は、実在しうる状態の行だけを選ぶ（重力のノルム 1 ± 0.05、体の 18 関節が MJCF の範囲 + 5° の
内側、関節速度 12.1 rad/s 以下）。歩行と起き上がりは書き出しのときに再生の env で種 0 のロールアウトを
400 ステップ行い、10 ステップごとの観測から最大 32 行を取る（同じチェックポイントからの書き出しは
バイト単位で同じになる）。PICO は判定の追従評価のロールアウトの観測から取る。

PICO だけのキー: `pico_walk_checkpoint_sha256`（凍結した歩行器）、`pico_target_frame`、
足・両足の目標の範囲（`pico_*_target_lower_json` / `_upper_json`）、`pico_arm_target_json`（腕の目標の
取り決め: 名前 `microban_pico_arm_target_rel_home_v1`、関節の名前、箱 `lower_rad`/`upper_rad`、速さ 4.0 rad/s）、
`pico_raw_action_guard_json`（最終の追従評価から max(v12, 元の歩行器 + 差) × 6）、`pico_curriculum_json`
（`critic_warmup`、`arm_start`、`foot_start`、`foot_tighten`、`total`。更新回数）、
`pico_active_adapter_columns_json`。

PICO の観測は 81 列で、最後の 6 列（75〜80）は直前の制御周期で腕の 6 サーボに書いた目標角 − HOME
（左 pitch, roll, elbow、右 pitch, roll, elbow）。腕のサーボには方策の出力を使わない。PICO の自己テストの
行には、足の目標と腕の目標が 0 でない行を含め、腕の目標は箱の中（± 1e-6 rad）でなければならない。

## 実機側の検証

install 段階は実機の worktree で次を実行し、終了コード 0 を合格とする（試運転では
`MICROBAN_ALLOW_DRYRUN_POLICY=1` を付ける）。

```bash
PYTHONPATH=src uv run --project <robot> --locked python tools/validate_policies.py src/agents
PYTHONPATH=src uv run --project <robot> --locked --with pytest python -m pytest -q tests
```

## サーボのゲイン（実機の制御）

- 学習した方策（歩行・起き上がり・PICO）が動いている間は、頭と首を含む全 21 サーボを P125
  （学習の `SERVO_KP_POLICY`）。
- 学習した方策を使わずに静止しているときは全サーボ P900。
