# Microban teleop: 歩行器を保ったままの PICO 方策の学習

PICO 方策（学習の取り決め 13、腕は外から動かす）は、`scripts/retrain_all_for_home.py` の歩行段階が作った歩行器
（`Mjlab-Velocity-Microban` の checkpoint、`checkpoints/<prefix>_walk_<sha>/`）を凍結した source として、
その上に PICO の入力の列と、出力に足す小さな補正の網を足して学習する。

- source の SHA-256・iteration・normalizer count は bootstrap provenance（schema 2）に記録され、save の
  たびに再ハッシュされる。source ファイルは動かさないこと。
- PICO 段階の入口で source の 9x300 probe（腕は HOME に固定。`artifacts/legacy_teleop_probe/velocity_<sha16>_teleop81_arm_overlay_9x300.json`、
  `mjlab_microban.scripts.probe_legacy_actor_in_teleop_env`）を作り、9/9 完走・転倒0・方向8/8・実関節の
  soft-limit 超過 ≤ 0.25 rad を満たさなければ bootstrap を拒否する。
- action target は脚の12関節で `HOME + raw_action * 1.0`（腕は下の「腕」）。ソフトウェア clip は無い。唯一の上下限は XC330 の
  goal position 範囲（1回転、raw 0..4095 = `[-π, π - 2π/4096]` rad）で、学習側はこれを絶対 target の ±π 飽和
  （`SERVO_TARGET_RANGE_RAD`）として模擬する。previous-action 観測は actor の raw 出力のまま。
  checkpoint/receipt/gate の `action_clip` は `[-π, π]`。
- 左右の site の順（`MICROBAN_BILATERAL_SITE_ORDER_REVISION`）は bootstrap の時点から checkpoint に記録され、
  パッケージャがそれを確かめる。

## actor と観測の対応

- actor: `81 -> 512 -> 256 -> 128 -> 18`（凍結した歩行器）＋補正の網 `81 -> 64 -> 64 -> 18`（ELU）。
  補正の網は歩行器と同じ正規化した 81 列を入力に取り、その出力を歩行器の出力（Gaussian の平均）に足す。
  最後の層は 0 で始めるので、学習の最初は歩行器と同じ出力になる。隠れ層は専用の乱数（seed 20261009）で
  初期化し、ほかの初期化と環境の乱数の流れを変えない。ONNX は両方を含む 1 つのグラフ
- scalar unbounded Gaussian（std は歩行器のまま凍結）
- action: clip なし。脚の 12 関節の target は `default_joint_pos + raw_action * scale`。腕の 6 関節は方策の
  出力を使わず、外からの腕の目標を書く（下の「腕」）
- previous action observation: actor の raw output（18 個）をそのまま再入力

63 個の歩行器の observation は名前で 81 列へ移植する。対応は `0:6 -> 0:6`, `6:24 -> 9:27`,
`24:42 -> 30:48`, `42:60 -> 48:66`, `60:63 -> 66:69`。新しい列は `6:9`（head/neck の位置）、`27:30`
（head/neck の速度）、`69:75`（足先の目標 6）、`75:81`（腕の目標 6）。

EmpiricalNormalization の歩行器の 63 列、全 trunk、bias、head、Gaussian std は凍結する。foot位置列 `69..74` の
有効な分母（`stored_std + eps`）は左右とも `(0.03, 0.03, 0.05)m`。腕の列 `75..80` の分母は HOME から腕の
箱の端までの最大の距離（pitch 100°、roll 110°、elbow 90°）で、HOME はちょうど 0、箱の中は 1 以下になる。
`eps=0.01` を引いた値を stored std、さらにその二乗を var として保存する。HMD 6列は identity 状態を保つ。
normalizer 全体は親 module が train mode になっても更新されない。

## 腕（実機の pico_arms と同じく外から動かす）

実機では PICO の間、`pico_arms`（直接の IK）が腕の 6 関節を上書きし、方策の腕の出力はサーボに届かない。
学習も同じにする（`mjlab_microban/tasks/microban_teleop_mdp.py`）。

- `PicoArmOverlayJointPositionAction`: 腕の 6 関節には `arm_target_rad` を書く。`process_actions` で
  `arm_goal_rad` へ関節ごとに 4 rad/s（1 step 0.08 rad）まで近づける。リセットでは HOME。
- `PicoArmTargetMotion`（step の事象）: 0.5〜3 秒ごとに目標を引き直す。確率 `home_probability` で両腕とも
  HOME（右トリガーを離した状態）、それ以外は関節ごとに箱（肩 pitch ±100°、肩 roll 外向き 10〜120°、
  肘 −110〜0°。実機の `pico_arm_contract`）から一様に引き、手先が胴の中か体の中心の面の反対側に入るものは
  引き直す。1000 回までは HOME に固定、そのあと 0.3。
- 観測 `arm_target`（75〜80 列）: 直前の物理で使った腕の目標 − HOME。左 pitch, roll, elbow、右 pitch,
  roll, elbow の順。
- HOME の姿勢を保つ報酬（pose 項）は脚の 12 関節だけを見る。足の目標がある行（足の報酬の速度による弱めが
  0 でない行）では 1.0 にして、脚を足の目標に使えるようにする（`microban_teleop_v13_arm_overlay.py`）。
- 足の目標は teleop と同じく 0.12 m/s で動かし、z が 2.5 mm 以下なら (0,0,0) にする
  （`microban_teleop_foot_command.py`）。両足とも 0 でない間は速度の指令を 0 に保つ。

## 列ごとの学習の開始

学習可能なのは第一層 weight の追加列と補正の網だけである。inactive target のノイズ学習で adapter が暴走しないよう、
更新可能列を段階的に開く（`mjlab_microban/schedules.py`）。

- completed updates `<= 1000`（critic の準備）: 追加18列をすべて exact zero に固定し、補正の網も動かさない
- `1001..4000`: HMD 6列と腕 6列と補正の網を許可
- `>= 4001`: foot 6列も許可し、追加18列すべてを許可

境界 rollout の古い batch で新しい列を更新しないため、gradient hook が見る `common_step_counter == 1000*24` と
`4000*24` はまだ lock する。lock 中の weight と Adam moment は、update・save の各時点で exact zero を
検証する（補正の網は、lock 中は最後の層が 0 で Adam moment が 0）。ONNX の判定の「歩行器との一致」は、
補正の網を除いた actor で確かめる（補正の網はすべての列を見るため）。

## 1本の学習と判定

`scripts/retrain_all_for_home.py` の PICO 段階が、歩行器の probe と bootstrap ゲートのあと、
`Mjlab-Teleop-V13-ArmOverlay-Microban` を1つのプロセスで 9000 回学習する（2048 env、seed 42）。

| 回数 | 段 |
| --- | --- |
| 0-999 | critic の準備（歩行器は凍結、追加列は 0。全指令範囲・押し ±0.5 m/s） |
| 1000- | 腕を動かす、動く HMD、立ち止まりの足踏み罰 |
| 4000- | 足先の目標 |
| 6000-8999 | 足先を絞る |

同じことを手で回すときは、次のとおり（`W` は歩行器の checkpoint）:

```bash
S=$(sha256sum "$W" | cut -d' ' -f1)
R=artifacts/legacy_teleop_probe/velocity_${S:0:16}_teleop81_arm_overlay_9x300.json
uv run --locked python -m mjlab_microban.scripts.probe_legacy_actor_in_teleop_env \
  --checkpoint "$W" --expected-sha256 "$S" --output "$R"
RS=$(sha256sum "$R" | cut -d' ' -f1)
uv run --locked python -m mjlab_microban.scripts.teleop_v12_bootstrap_gate \
  --checkpoint "$W" --checkpoint-sha256 "$S" --probe-receipt "$R" --probe-receipt-sha256 "$RS"
uv run --locked train Mjlab-Teleop-V13-ArmOverlay-Microban --env.scene.num-envs 2048 \
  --env.seed 42 --agent.seed 42 --agent.logger tensorboard \
  --agent.legacy-velocity-checkpoint "$W" --agent.legacy-velocity-checkpoint-sha256 "$S" \
  --agent.legacy-teleop-probe-receipt "$R" --agent.legacy-teleop-probe-receipt-sha256 "$RS" \
  --agent.save-pristine-checkpoint True
```

途中で止まったら最初から学習し直す。判定は学習の最後の保存点（`model_8999`）で1回だけ、9x300・最終
プロファイルの追従評価・ONNX パリティ（どれも seed 42）（docs/teleop_v12_deployment.md）。

どの report でも status/check/hash/schema/必要 scenario/evidence が欠けたら停止する。空の `checks={}` は pass と
見なさない。ONNX は `[1,81] -> [1,18]`、neutral 10,000 samples と全 81 列の 64 samplesの誤差が許容内で、
`CPUExecutionProvider` を実際に使うことを要求する。
