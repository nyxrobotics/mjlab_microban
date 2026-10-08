# Microban teleop v12: 歩行器を保ったままの PICO 方策の学習

PICO 方策（v12）は、`scripts/retrain_all_for_home.py` の歩行段階が作った歩行器
（`Mjlab-Velocity-Microban` の checkpoint、`checkpoints/<prefix>_walk_<sha>/`）を凍結した source として、
その上に PICO の入力の列を足して学習する。

- source の SHA-256・iteration・normalizer count は bootstrap provenance（schema 2）に記録され、save の
  たびに再ハッシュされる。source ファイルは動かさないこと。
- PICO 段階の入口で source の 9x300 probe（`artifacts/legacy_teleop_probe/velocity_<sha16>_teleop83_raw_9x300.json`、
  `mjlab_microban.scripts.probe_legacy_actor_in_teleop_env`）を作り、9/9 完走・転倒0・方向8/8・実関節の
  soft-limit 超過 ≤ 0.25 rad を満たさなければ bootstrap を拒否する。
- action target は全18関節で `HOME + raw_action * 1.0`。ソフトウェア clip は無い。唯一の上下限は XC330 の
  goal position 範囲（1回転、raw 0..4095 = `[-π, π - 2π/4096]` rad）で、学習側はこれを絶対 target の ±π 飽和
  （`SERVO_TARGET_RANGE_RAD`）として模擬する。previous-action 観測は actor の raw 出力のまま。
  checkpoint/receipt/gate の `action_clip` は `[-π, π]`。
- 左右の site の順（`MICROBAN_BILATERAL_SITE_ORDER_REVISION`）は bootstrap の時点から checkpoint に記録され、
  パッケージャがそれを確かめる。

## actor と観測の対応

- actor: `83 -> 512 -> 256 -> 128 -> 18`、ELU、scalar unbounded Gaussian
- action: clip なし。target は `default_joint_pos + raw_action * scale`
- previous action observation: actor の raw output をそのまま再入力

63 個の歩行器の observation は名前で 83 列へ移植する。対応は `0:6 -> 0:6`, `6:24 -> 9:27`,
`24:42 -> 30:48`, `42:60 -> 48:66`, `60:63 -> 66:69`。新しい列は `6:9`（head/neck の位置）、`27:30`
（head/neck の速度）、`69:83`（足先の目標 6 と手先の目標 8）。

EmpiricalNormalization の歩行器の 63 列、全 trunk、bias、head、Gaussian std は凍結する。foot位置列 `69..74` の
有効な分母（`stored_std + eps`）は左右とも `(0.03, 0.03, 0.05)m`。hand位置列 `75..80` は後述の到達可能FK box
を401点/軸で走査した最大絶対offset `(0.062894644, 0.038751220, 0.060477221)m` を外向きに丸め、左右とも
`(0.0630, 0.0388, 0.0605)m` とする。`eps=0.01` を引いた値をstored std、さらにその二乗をvarとして保存する。
HMD 6列とhand active flag 2列はidentity状態を保つ。normalizer全体は親 module がtrain modeになっても更新されない。
per-axis定数とFK provenanceはactor metadataに記録される。

trainingのhand targetはCartesian cubeから直接サンプルしない。左右独立に `(shoulder_pitch, shoulder_roll, elbow)`
をuniform joint sampleし、`robot.xml`のbody/joint/site transformと同じvectorized FKでsoftware HOMEからの
trunk-frame XYZ offsetへ変換する。boxはpitch `[-25,+25]deg`、left roll `[10,30]deg`、right roll `[-30,-10]deg`、
elbow `[-50,-10]deg`。HOMEはpitch `0deg`、roll `left +10/right -10deg`、elbow `-20deg`である。inactive handは
HOME tupleを使い、offsetをexact zeroにする。FK helperはMuJoCoのreference site位置とfloat64で照合する。
体幹が傾いたHOMEでは、実機receiverで確認済みの各軸 `±0.064m` の箱から出る目標を棄却する
（`mjlab_microban/robot/microban_hand_fk.py`）。ONNX/wireの `hand_target_lower/upper` は各軸 `±0.08m` で、
実際のreachable subsetは別の `hand_target_fk` metadataに記録する。

## 列ごとの学習の開始

学習可能なのは第一層 weight の追加列だけである。inactive target のノイズ学習で adapter が暴走しないよう、
更新可能列を段階的に開く（`mjlab_microban/schedules.py`）。

- completed updates `<= 1000`（critic の準備）: 追加20列をすべて exact zero に固定
- `1001..4000`: HMD 6列と hand 8列だけを許可
- `>= 4001`: foot 6列も許可し、追加20列すべてを許可

境界 rollout の古い batch で新しい列を更新しないため、gradient hook が見る `common_step_counter == 1000*24` と
`4000*24` はまだ lock する。lock 中の weight と Adam moment は、update・save の各時点で exact zero を
検証する。

## 1本の学習と判定

`scripts/retrain_all_for_home.py` の PICO 段階が、歩行器の probe と bootstrap ゲートのあと、
`Mjlab-Teleop-V12-HandPoseRelease-Microban` を1つのプロセスで 9000 回学習する（2048 env、seed 42）。

| 回数 | 段 |
| --- | --- |
| 0-999 | critic の準備（歩行器は凍結、追加列は 0。全指令範囲・押し ±0.5 m/s） |
| 1000- | 手先の目標、動く HMD、立ち止まりの足踏み罰 |
| 2500- | 手先を絞る |
| 4000- | 足先の目標 |
| 6000-8999 | 足先を絞る |

同じことを手で回すときは、次のとおり（`W` は歩行器の checkpoint）:

```bash
S=$(sha256sum "$W" | cut -d' ' -f1)
R=artifacts/legacy_teleop_probe/velocity_${S:0:16}_teleop83_raw_9x300.json
uv run --locked python -m mjlab_microban.scripts.probe_legacy_actor_in_teleop_env \
  --checkpoint "$W" --expected-sha256 "$S" --output "$R"
RS=$(sha256sum "$R" | cut -d' ' -f1)
uv run --locked python -m mjlab_microban.scripts.teleop_v12_bootstrap_gate \
  --checkpoint "$W" --checkpoint-sha256 "$S" --probe-receipt "$R" --probe-receipt-sha256 "$RS"
uv run --locked train Mjlab-Teleop-V12-HandPoseRelease-Microban --env.scene.num-envs 2048 \
  --env.seed 42 --agent.seed 42 --agent.logger tensorboard \
  --agent.legacy-velocity-checkpoint "$W" --agent.legacy-velocity-checkpoint-sha256 "$S" \
  --agent.legacy-teleop-probe-receipt "$R" --agent.legacy-teleop-probe-receipt-sha256 "$RS" \
  --agent.save-pristine-checkpoint True
```

途中で止まったら最初から学習し直す。判定は学習の最後の保存点（`model_8999`）で1回だけ、9x300・最終
プロファイルの追従評価・ONNX パリティ（どれも seed 42）（docs/teleop_v12_deployment.md）。

どの report でも status/check/hash/schema/必要 scenario/evidence が欠けたら停止する。空の `checks={}` は pass と
見なさない。ONNX は `[1,83] -> [1,18]`、neutral 10,000 samples と full83 64 samplesの誤差が許容内で、
`CPUExecutionProvider` を実際に使うことを要求する。
