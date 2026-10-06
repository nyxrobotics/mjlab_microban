# 方策と実機の契約（policy contract `microban-policy-1`）

学習側（このリポジトリ）が作り、実機側（microban）が受け取るものの取り決め。1つの版番号
`policy_contract` で全体を表し、意味が変わるときは版番号を上げる。実機は自分が実装した版だけを受け付ける。
実機側のソースのハッシュはパッケージに入れない（入れていたのは v12 の PICO パッケージまで）。代わりに、
導入のときに実機のバリデータとテストを、起動のたびにパッケージに入った観測で自己テストを行う。

## リリースの中身

`scripts/retrain_all_for_home.py` の export 段階が `<状態>/release/` に書き、install 段階が実機の
`src/agents/` にそのままコピーする。

| ファイル | 中身 |
| --- | --- |
| `walk.onnx` | 歩行（入力 63、出力 18） |
| `getup.onnx` | 起き上がり（入力 60、出力 18） |
| `pico_teleop.onnx` | PICO（入力 83、出力 18） |
| `manifest.json` | 下の表。3 つのファイルの SHA-256 と、それを作ったチェックポイントと判定の記録 |

実機の `config/home_pose.yaml` は同じ段階で `config/home_pose_tool.py write-robot` が書く（HOME と、
HOME に結びついた契約文字列）。

### manifest.json（`schema_version` 1）

```json
{
  "schema_version": 1,
  "policy_contract": "microban-policy-1",
  "dry_run_not_deployable": false,
  "home": {"tag": "...", "joint_hash": "...", "yaml_sha256": "..."},
  "policies": {
    "walk":        {"file": "walk.onnx",        "sha256": "...", "checkpoint": "model_N.pt", "checkpoint_sha256": "..."},
    "getup":       {"file": "getup.onnx",       "sha256": "...", "checkpoint": "...", "checkpoint_sha256": "..."},
    "pico_teleop": {"file": "pico_teleop.onnx", "sha256": "...", "checkpoint": "...", "checkpoint_sha256": "..."}
  },
  "judgments": {"walk_source_probe": {...}, "pico": {"passed": true, "failures": []},
                "getup": {"summary": {...}, "passed": true, "failures": []}},
  "schedules": {"pico": {...}, "getup": {...}, "walk_max_updates": 20000},
  "training": {"commit": "...", "branch": "..."},
  "created": "YYYY-MM-DD hh:mm:ss"
}
```

実機は manifest の `policy_contract` が自分の版と同じこと、`home.tag` / `home.joint_hash` が自分の
`config/home_pose.yaml` の HOME と同じこと、各ファイルの SHA-256 が一致すること、`dry_run_not_deployable`
が false（または `MICROBAN_ALLOW_DRYRUN_POLICY=1`）であることを確かめる。

## 3 つの ONNX に共通のメタデータ

| キー | 値 |
| --- | --- |
| `policy_contract` | `microban-policy-1` |
| `servo_kp` | 学習した全 21 サーボのファームウェア P ゲイン（`125`、`SERVO_KP_POLICY`）。実機は方策を動かしている間、頭と首を含む全サーボをこの値にする |
| `home_tag`、`home_joint_hash` | 学習した HOME（`config/home_pose.yaml`） |

それ以外の歩行・起き上がりのキー（関節の並び、HOME、±π の目標クリップ、前回行動の意味、契約文字列
`walk_contract_version` / `microban_getup_contract` など）は今までと同じ。

## PICO（`pico_teleop.onnx`）で変わったキー

- 足した: 共通の 4 キー、`pico_schedule_json`（`critic_warmup`、`hand_start`、`hand_tighten_start`、
  `foot_start`、`foot_tighten_start`、`total`、`min_final`。単位は学習の更新回数）。
- 消した: `microban_*_source_sha256`、`microban_runtime_lock_sha256`、`microban_walk_fallback_onnx_sha256`
  （実機のソースのハッシュ）、`v12_boundary_stage_gates_*`、`v12_stage_gate_canonical_boundary`、
  `v12_lr_order_migration_revision`。
- 値が変わった:
  - `microban_teleop_recipe_revision`: twist-ratio の速度の報酬と 1 本の学習の新しいレシピ
    （`..._active_hand_arm_pose_release_twist_ratio_one_run_warmup1000_total9000_v1`。実機の
    `config/home_pose.yaml` の `v12_hand_pose_release_recipe_revision` と同じ値）。
  - `v12_deployment_packager_revision`: `microban_pico_packager_one_run_v1_<tag>_servo_range`。
  - `adapter_gradient_schedule_revision`: `freeze_extra_to1000_then_hmd_hand_to4000_then_all_v2`。
  - `checkpoint_iteration` / `checkpoint_completed_updates`: 学習の終わり（9000 回）か、途中確認で採った
    保存点（最後の段の半分以上を学習した後、`min_final` 以上）。15000 固定ではない。
  - `v12_stage_gate_schema_version`: 3。判定は 1 回だけで、プロファイルは今までの最終プロファイル
    `full_body_reachable_performance_perturbation_v2_deployed_accuracy_v1`。
- そのまま: 観測と行動の並び、目標の範囲と座標系、HMD、実行時のガード
  （`runtime_raw_action_guard_*`）、起動時の自己テストの観測（`v12_runtime_smoke_observations_json`、
  判定の追従評価の実際の観測。実機は起動のたびにこれを ONNX Runtime に通し、ガードの範囲に収まることを
  確かめる）。

## 実機側の検証ツール

install 段階は実機の worktree で次を実行し、終了コード 0 を合格とする。

```bash
uv run --project <robot> --locked python tools/validate_policies.py src/agents/manifest.json
uv run --project <robot> --locked --with pytest python -m pytest -q tests
```

`tools/validate_policies.py` は manifest と 3 つの ONNX を上の取り決めで確かめ（各 ONNX の読み込み、
入出力の形、メタデータ、PICO の自己テストの観測による ONNX Runtime の実行）、結果を JSON で標準出力に書く。

## サーボのゲイン（実機の制御、2026-10-06 のユーザー判断）

- 学習した方策（歩行・起き上がり・PICO）が動いている間は、頭と首を含む全 21 サーボを `servo_kp`（125）。
- 学習した方策を使わずに静止しているとき（A ボタンの初期姿勢、起き上がり後に歩行が始まらないままの静止、
  起き上がりの方策が使えないとき、学習した動作が終わって HOME に戻った後）は全サーボ P900。
- ゲインの書き込みは、前回そのサーボに送った値と違うときだけ。起動時に全サーボの値を一度合わせる。
  サーボが通信から外れて戻ったときの再送は残す。
