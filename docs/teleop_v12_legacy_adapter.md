# Microban teleop v12: legacy-preserving policy training

## centered HOME + servo 範囲（±π）版（branch `track-centered-home-clip`）

このブランチの v12 は旧HOME（股ピッチ -10°）の `xc330_velocity/model_14999.pt` ではなく、
共通の centered HOME（`HOME_FRAME`: 股ピッチ `+1.198384259489°`、足首ピッチ `-1.198384259489°`、肩ピッチ `0°`）で学習した
`Mjlab-Velocity-Microban` の checkpoint を凍結 source とする。

- source は `scripts/retrain_all_for_home.py` の歩行段階が作った歩行器（`checkpoints/<prefix>_walk_<sha>/`）。SHA-256・iteration・normalizer count は
  bootstrap provenance（schema 2）に記録され、save/resume のたびに再ハッシュされる。source ファイルは動かさないこと。
- PICO 段階の入口で source の 9x300 probe（`artifacts/legacy_teleop_probe/velocity_<sha16>_teleop83_raw_9x300.json`）を生成し、
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
- 以下の「固定した契約」は旧HOMEで作った配備済みモデルの履歴である。学習の段と判定は「1本の学習と判定」の節（2026-10-07 以降）。

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

学習可能なのは第一層 weight の追加列だけである。ただし inactive target のノイズ学習で adapter が暴走しないよう、更新可能列を段階的に開く（`mjlab_microban/schedules.py`、2026-10-07 以降）。

- completed updates `<= 1000`（critic の準備）: 追加20列をすべて exact zero に固定
- `1001..4000`: HMD 6列と hand 8列だけを許可
- `>= 4001`: foot 6列も許可し、追加20列すべてを許可

境界 rollout の古い batch で新しい列を更新しないため、gradient hook が見る `common_step_counter == 1000*24` と `4000*24` はまだ lock する。lock 中の weight と Adam moment は、update・save・resume の各時点で exact zero を検証する。

## 1本の学習と判定（2026-10-07 以降）

`scripts/retrain_all_for_home.py` の PICO 段階が、歩行器の probe と bootstrap ゲートのあと、`Mjlab-Teleop-V12-HandPoseRelease-Microban` を1つのプロセスで 9000 回学習する（2048 env、seed 42）。

| 回数 | 段 |
| --- | --- |
| 0-999 | critic の準備（歩行器は凍結、追加列は 0。全指令範囲・押し ±0.5 m/s） |
| 1000- | 手先の目標、動く HMD、立ち止まりの足踏み罰 |
| 2500- | 手先を絞る |
| 4000- | 足先の目標 |
| 6000-8999 | 足先を絞る |

途中で落ちたら最後のチェックポイントから続ける（runner の厳密な再開）。判定は学習の終わり（または途中確認で採った保存点）で1回だけ、9x300・最終プロファイルの追従評価・ONNX パリティ（どれも seed 42）。区切りの再起動、100 回のカナリア、段ごとのゲート、救済はない（docs/teleop_v12_deployment.md）。

どの report でも status/check/hash/schema/必要 scenario/evidence が欠けたら停止する。空の `checks={}` は pass と見なさない。ONNX は `[1,83] -> [1,18]`、neutral 10,000 samples と full83 64 samplesの誤差が許容内で、`CPUExecutionProvider` を実際に使うことを要求する。
