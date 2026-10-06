# Microban teleop v12: legacy-preserving policy training

## centered HOME + servo 範囲（±π）版（branch `track-centered-home-clip`）

このブランチの v12 は旧HOME（股ピッチ -10°）の `xc330_velocity/model_14999.pt` ではなく、
共通の centered HOME（`HOME_FRAME`: 股ピッチ `+1.198384259489°`、足首ピッチ `-1.198384259489°`、肩ピッチ `0°`）で学習した
`Mjlab-Velocity-Microban` の checkpoint を凍結 source とする。

- source は `scripts/train_microban_teleop_v12.sh start --source PATH` で指定する。SHA-256・iteration・normalizer count は
  bootstrap provenance（schema 2）に記録され、save/resume/gate のたびに再ハッシュされる。source ファイルは動かさないこと。
- start は source の 9x300 probe（`artifacts/legacy_teleop_probe/velocity_<sha16>_teleop83_raw_9x300.json`）を毎回生成し、
  9/9 完走・転倒0・方向8/8・実関節 soft-limit overshoot ≤ 5° を満たさなければ bootstrap を拒否する。
- action target は全18関節で `HOME + raw_action * 1.0`。ソフトウェア clip は無い。唯一の上下限は XC330 の
  goal position 範囲（1回転、raw 0..4095 = `[-π, π - 2π/4096]` rad）で、学習側はこれを絶対 target の ±π 飽和
  （`SERVO_TARGET_RANGE_RAD`）として模擬する。previous-action 観測は actor の raw 出力のまま。
  checkpoint/receipt/gate の `action_clip` は `[-π, π]`（`[-3.141592653589793, 3.141592653589793]`）。
  以前の ±1.57 clip は XC330 のトルクを約70%に制限し歩行学習が停滞したため廃止（training commit `94d2946`）。
- recipe `centered_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_raw_prev_action_servo_range_pi_v11`、
  HOME revision `centered_home_hip_plus1p198384259489_ankle_minus1p198384259489_shoulder_zero_v5`。旧 checkpoint（旧HOME、±1.57 clip の v10 recipe を含む）は再開できない。
- 新規 chain は bilateral site order 修正後に bootstrap するため model-9200 の LR migration を持たない。exporter は
  `v12_lr_order_migration_revision=none_corrected_site_order_from_bootstrap_v1` を書く。
- 以下の節は旧HOMEで作った配備済みモデルの履歴手順である（stage 境界・gate の内容は同じ）。

## 固定した契約

- source: `checkpoints/xc330_velocity/model_14999.pt`
- source SHA-256: `b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141`
- source probe: `artifacts/legacy_teleop_probe/model_14999_teleop83_raw_9x300.json`
- probe SHA-256: `f51378d59ff4d68fb1185a91eb2a863749e5c7be6ec4cd0ab4a0b08f1565e69d`
- recipe: `legacy_velocity_model14999_staged_mask_reachable_fk_elbow_minus10_raw_actions_v5`
- bootstrap mapping: `normalized_legacy_velocity_63_to_teleop83_reachable_fk_elbow_minus10_v4`
- adapter schedule: `freeze_extra_to7000_then_hmd_hand_to10000_then_all_v1`
- actor: `83 -> 512 -> 256 -> 128 -> 18`, ELU、scalar unbounded Gaussian
- action: clip なし。target は `default_joint_pos + raw_action * scale`
- previous action observation: actor の raw output をそのまま再入力

63 個の legacy observation は名前で 83 列へ移植する。対応は `0:6 -> 0:6`, `6:24 -> 9:27`, `24:42 -> 30:48`, `42:60 -> 48:66`, `60:63 -> 66:69`。EmpiricalNormalization の legacy 63 列、全 trunk、bias、head、Gaussian std は凍結する。foot位置列 `69..74` の有効な分母（`stored_std + eps`）は左右とも `(0.03, 0.03, 0.05)m`。hand位置列 `75..80` は後述の到達可能FK boxを401点/軸で走査した最大絶対offset `(0.062894644, 0.038751220, 0.060477221)m` を外向きに丸め、左右とも `(0.0630, 0.0388, 0.0605)m` とする。`eps=0.01` を引いた値をstored std、さらにその二乗をvarとして保存する。HMD 6列とhand active flag 2列は従来のidentity状態を保つ。normalizer全体は親 module がtrain modeになっても更新されない。per-axis定数とFK provenanceはactor/sanitizer metadataに記録される。

trainingのhand targetはCartesian cubeから直接サンプルしない。左右独立に `(shoulder_pitch, shoulder_roll, elbow)` をuniform joint sampleし、`robot.xml`のbody/joint/site transformと同じvectorized FKでsoftware HOMEからのtrunk-frame XYZ offsetへ変換する。boxはpitch `[-25,+25]deg`、left roll `[10,30]deg`、right roll `[-30,-10]deg`、elbow `[-50,-10]deg`。HOMEはpitch `0deg`、roll `left +10/right -10deg`、elbow `-20deg`である。inactive handはHOME tupleを使い、offsetをexact zeroにする。FK helperはMuJoCoのreference site位置とfloat64で照合する。このboxの全FK offsetは実機receiverで確認済みの各軸 `±0.064m` の内側に収まる。ONNX/wireの `hand_target_lower/upper` は既存runtime互換の各軸 `±0.08m` を維持し、実際のreachable subsetは別の `hand_target_fk` metadataに記録する。

学習可能なのは第一層 weight の追加列だけである。ただし inactive target のノイズ学習で adapter が暴走しないよう、更新可能列を段階的に開く。

- completed updates `<= 7000`: 追加20列をすべて exact zero に固定
- `7001..10000`: HMD 6列と hand 8列だけを許可
- `>= 10001`: foot 6列も許可し、追加20列すべてを許可

境界 rollout の古い batch で新しい列を更新しないため、gradient hook が見る `common_step_counter == 7000*24` と `10000*24` はまだ lock する。再開後の次 rollout が最初の学習対象になる。lock 中の weight と Adam moment は、update・save・resume の各時点で exact zero を検証する。

## canonical training と必須 gate

新規開始は次の wrapper だけを使う。

```bash
scripts/train_microban_teleop_v12.sh start --source VELOCITY_MODEL.pt
```

再開は gate を作ってから行う（`RUN_NAME` と `ITERATION` は再開元）。

```bash
scripts/evaluate_microban_teleop_v12_stage.sh RUN_NAME ITERATION
scripts/train_microban_teleop_v12.sh resume RUN_NAME --agent.run-name NEW_RUN
```

wrapper は checkpoint iteration ではなく completed updates を使い、次の状態機械を強制する。

```text
0 -> 3000 -> 3100 -> 7000 -> 7100 -> 10000 -> 10100 -> 15000
```

3000/7000/10000 の各境界後は100 update activation canaryを省略できない。電源断した任意 checkpoint も hash-bound gate 後に再開できるが、canary 区間で中断した場合は同じ canary endpoint までしか進めない。

各再開前に次の3 reportを同じ checkpoint SHAへ束縛した schema 2 stage gate が必要である。

1. neutral teleop input で legacy locomotion 9 scenarios x 300 steps
2. moving HMD と bounded nonzero hand/foot observation を使う stage別 tracking/exposure gate
3. neutral legacy parity と full random 83-column PyTorch/ONNX/ONNX Runtime CPU parity

tracking gate は early termination、fall、non-finite、実 joint の動的soft-limit overshootが `5deg`（`0.08726646259971647rad`）を超える場合、raw previous-action 不一致、HMD 3軸 motion 不足、必要な hand/foot observation が zero、全 nonzero twist 軸の方向/応答不足を拒否する。この5deg許容は測定された実joint stateだけに適用する。direct-IK等のcommanded joint targetはsoft limitを`1e-7rad`より超えてはならない。raw policy targetのhypothetical soft-limit excessはlegacy source自体でも起こるためreport-onlyであり、このcommanded-target許容とは別である。

target-column ablation は同じ83列 observation の hand または foot 対象列だけを zero にし、raw action の最大絶対差が `1e-4` を厳密に超えることを因果応答として判定する。未学習性能を先取りしないため、3000/7000までのprofileではresponseを要求せず、7001..10000はhand、10001以降はhandとfootの両方を要求する。ただし全profileで各scenarioの対象有無、測定値、閾値、pass booleanの整合性を厳密に検証し、欠落・非有限・負値・偽装booleanは拒否する。

stage profile は「次 stage を学習する前の policy に未学習性能を要求しない」順序である。activation canaryは新しく開いた入力に対する安全性・coverage・同一observation ablationの因果応答を認証するが、100 updatesだけで最終追従精度を要求しない。

- 3000まで: locomotion + HMD/hand/foot exposure safety
- 3001..7000: expanded locomotion + pre-HMD exposure safety
- 7001..7100: HMD/hand activation canary。安全性、HMD/hand coverage、hand ablationを要求し、hand RMS/P95はまだ要求しない
- 7101..10000: HMD/hand performance + foot exposure safety。10000境界でhand RMS/P95を要求する
- 10001..10100: foot activation canary。10000で認証済みのstrict hand品質を維持し、foot coverage/ablationを要求するがfoot RMS/P95はまだ要求しない
- 10101..14999: whole-body performance
- 15000: full-body performance + perturbation

どの report でも status/check/hash/schema/必要 scenario/evidence が欠けたら停止する。空の `checks={}` は pass と見なさない。ONNX は `[1,83] -> [1,18]`、neutral 10,000 samples と full83 64 samplesの最大誤差が `2e-5` 以下で、`CPUExecutionProvider` を実際に使うことを要求する。

## 所要時間と停止条件

このPCで100 updatesは約91〜95秒だった。目安は次の通り。

- fresh 0 -> 3000: 約48分
- 15,000 updatesの学習部分: 約4時間（評価・再試行を除く保守値）

次のどれかで直ちに停止する: non-finite、fall、actual jointの動的soft-limit overshootが5deg超、commanded targetのsoft-limit excessが`1e-7rad`超、方向反転/応答不足、locked weight/Adam mutation、normalizer/legacy tensor drift、provenance/hash不一致、ONNX parity超過、tracking observation coverage欠落。camera/tracker/learned policyの異常時にも、実行側は別系統の legacy joystick walkへ同一cycleでfallbackする。
