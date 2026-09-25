# Microban teleop v12: legacy-preserving policy training

この文書は、実績のある joystick 歩行 actor を壊さずに PICO の HMD・手・足の目標を追加する手順を固定する。v12 の canonical policy と、早期確認専用の simulation preview は別物である。preview は実機配備できない。

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

## v1 run の棄却と sanitizer

旧 v1 は HMD/target が無効な期間にも追加列を学習した。model300 の forced-HMD 評価で actual soft-limit violation と forward 符号反転を確認したため、旧 run は採用しない。sanitizer v4は旧model600のbootstrap mapping v1と追加20列のidentity normalizerを先に認証し、actorの追加20列と対応するAdam momentsをzeroへ戻し、12個のtarget位置normalizerとbootstrap provenanceをelbow上限-10degのreachable-FK mapping v4へ一回で移行する。sourceは上書きしない。

以下の旧sanitized v1 checkpointはidentity target normalizerの履歴資料であり、recipe v5では再開・preview sourceとして使用できない。

```text
logs/rsl_rl/mjlab_microban_teleop_v12/2026-09-25_v12_sanitized601_reachable_fk_v5_grid401/model_600.pt
SHA-256 ab0dbe0db9cadd6bb937e5eebf3d2b8f6bb3fadcc5e025aa15d28fb87e85d3a2
iteration 600 / completed updates 601 / common_step_counter 14424
```

再現する場合は、旧 v1 checkpoint を source にして次を実行する。destination は必ず新規 path にする。

```bash
uv run --locked python -m \
  mjlab_microban.scripts.sanitize_teleop_v12_adapter_checkpoint \
  PATH_TO_V1_MODEL_600.pt PATH_TO_NEW_V5_MODEL_600.pt \
  --output artifacts/teleop_v12_sanitization/model600_to_v5_reachable_fk.json
```

sanitizer revisionは`zero_pre7000_extra_w0_adam_and_reachable_fk_elbow_minus10_v4`、receipt schemaは4である。actor shared columns、trunk、critic、optimizerの他stateとparam groupsがbit-identicalであること、旧extra normalizerがmean=0/var=1/std=1であること、移行後の分母・std・var・joint box・FK revisionを検証する。advanced indexingのcopyを誤ってzeroにする回帰を防ぐため、resetは`index_fill_`に固定した。

2026-09-25に上記one-pass migrationを実行した再開用checkpointは
`logs/rsl_rl/mjlab_microban_teleop_v12/2026-09-25_v12_sanitized601_reachable_fk_v5_grid401/model_600.pt`
（SHA-256 `ab0dbe0db9cadd6bb937e5eebf3d2b8f6bb3fadcc5e025aa15d28fb87e85d3a2`）、
receiptは`artifacts/teleop_v12_sanitization/model600_to_v5_reachable_fk_elbow_minus10_grid401.json`
（SHA-256 `6cd838c8fcc974323bc25d12b91f7e04b680133899fb035d10bec9e9d7a53c0a`）である。

## canonical training と必須 gate

新規開始は次の wrapper だけを使う。

```bash
scripts/train_microban_teleop_v12.sh start
```

新しいscaled-target sanitized601からの正式な再開例（`RUN_NAME`は生成先に置き換える）:

```bash
scripts/evaluate_microban_teleop_v12_stage.sh \
  RUN_NAME 600

scripts/train_microban_teleop_v12.sh resume \
  RUN_NAME \
  --agent.run-name v12_stage_601_to3000
```

wrapper は checkpoint iteration ではなく completed updates を使い、次の状態機械を強制する。

```text
601 -> 3000 -> 3100 -> 7000 -> 7100 -> 10000 -> 10100 -> 15000
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

どの report でも status/check/hash/schema/必要 scenario/evidence が欠けたら停止する。空の `checks={}` は pass と見なさない。ONNX は `[1,83] -> [1,18]`、neutral 10,000 samples と full83 64 samplesの最大誤差が canonical では `2e-5` 以下、期限用 final profile では float32 実測差を含む `2.5e-5` 以下で、`CPUExecutionProvider` を実際に使うことを要求する。

## simulation-only early preview

手足の挙動を canonical 15k 完了前に見る場合だけ、canonical chainから完全に分離した staged-v2 previewを使う。最初からfull20列を同時学習した旧v1 artifactはlegacy read-onlyであり、live候補にはならない。

phase 1 は sanitized601 のactor/critic/optimizerを変えずclockだけ7001へliftし、HMD6+hand8列を100 updates学習する。foot6列・そのAdam moments・foot command/rewardはexact zero/inactiveのままである。hand commandは全stageで上記の同じreachable joint-box/FK samplerを使う。preview専用curriculumは7001..7049をhand reward weight 4.0/std 0.12 m、7050以降をweight 4.0/std 0.08 mとし、measured joint soft-limit guardも両区間で-10.0に強化する。8500以降はweight 2.0/std 0.05 mへ復帰する。

```bash
scripts/train_microban_teleop_v12_preview.sh \
  v12_staged_preview_v2_hmd_hand100 --updates 100
```

model7100を専用evaluatorへ通す。strict hand 3cm/5cmだけがfailし、hard safety/locomotion/HMD/coverage/directionalがすべてpassした場合に限り、simulation表示用の暫定hand基準 RMS 0.13m/P95 0.15mとlearned-source差分coverageを再計算するpromotion toolを使える。strict receipt自体はfailのまま保持され、promotionも実機deploymentには使えない。

```bash
uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_preview \
  PATH/model_7100.pt --expected-sha256 MODEL_SHA --device cuda:0 \
  --output artifacts/teleop_v12_preview/phase1_strict.json
uv run --locked python -m mjlab_microban.scripts.promote_teleop_v12_preview_visual \
  PATH/model_7100.pt --expected-checkpoint-sha256 MODEL_SHA \
  --strict-evaluation-receipt artifacts/teleop_v12_preview/phase1_strict.json \
  --expected-receipt-sha256 STRICT_RECEIPT_SHA \
  --output artifacts/teleop_v12_preview/phase1_visual_promotion.json
```

phase 2 はaccepted phase1のactor/critic/optimizerを一切変えず10001へliftする。learned HMD/hand列を保持し、foot6列/momentsがexact zeroであることを検証してからfull20列とfoot command/rewardを有効にする。

```bash
scripts/train_microban_teleop_v12_preview_fullbody.sh \
  v12_staged_preview_v2_fullbody100 PATH/model_7100.pt MODEL_SHA \
  artifacts/teleop_v12_preview/phase1_visual_promotion.json PROMOTION_SHA
```

phase2学習前のHMD/hand actor自体がfinal mixed perturbationで安全か切り分ける場合は、lifted `model_10000.pt`だけに使える診断モードを使う。この出力はchecksが全trueでも必ず`status=diagnostic`、`pico_live_accepted=false`であり、acceptance receiptにはならない。

```bash
uv run --locked python -m \
  mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck \
  PATH/model_10000.pt --expected-sha256 CHECKPOINT_SHA256 \
  --device cuda:0 --diagnose-phase2-seed \
  --output artifacts/teleop_v12_preview/phase2_seed_diagnostic.json
```

seed診断がhard PASSのときだけ、100 updatesで悪化した候補を同じmodel10000 seedから`--save-interval 10`で10/20 update候補へ分け、最初のhard PASSをfull評価する。seed自体がactual joint-limit等に違反する場合、update数探索を安全修正として扱ってはならずphase1を再学習・再評価する。LRやhard基準は緩めない。

preview checkpoint は永久に次を持つ。

- `preview_non_deployable=true`
- `teleop_v12_preview.revision=sanitized601_staged_hmd_hand_then_full20_sim_only_v2`
- phase、source/receipt SHA、lift前後clock、active/inactive columns、phase1 quality class

canonical runner、stage gate、通常 ONNX export、実機 deployment はこの marker を必ず拒否する。PICO simulation は明示的な preview option と専用 task/consumer APIを使った場合だけ読み込める。preview の結果を canonical 3000/7000/10000/15000 chainへ戻してはならない。

fullbody候補はまずtargeted precheck、次に専用full evaluatorへ通す。

```bash
uv run --locked python -m \
  mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck \
  PATH_TO_PREVIEW_MODEL.pt --expected-sha256 CHECKPOINT_SHA256 \
  --device cuda:0 --output artifacts/teleop_v12_preview/precheck.json
uv run --locked python -m \
  mjlab_microban.scripts.evaluate_teleop_v12_preview \
  PATH_TO_PREVIEW_MODEL.pt \
  --expected-sha256 CHECKPOINT_SHA256 \
  --device cuda:0 \
  --output artifacts/teleop_v12_preview/preview_acceptance.json
```

full evaluatorはCUDA固定でlegacy locomotion 9x300とmoving HMD・hand・foot・mixed twist・perturbation 8x300を実行し、全`src/mjlab_microban` Python/XML/JSON、robot model、motion prior、`pyproject.toml`、`uv.lock`を含むhash-bound source manifestも記録する。評価開始時と終了時のmanifestが一致しないrunは破棄する。strict PASS、またはhard checksすべてPASSかつvisual基準（hand RMS/P95 0.13/0.15m、foot 0.08/0.13m）とtarget-column ablationを満たすself-contained final visual receiptだけがPICO simulation候補である。learned-source差分はcoverageでありtarget因果証拠とは呼ばず、因果応答は同一83-observationの対象列だけをzeroにしたablationで判定する。live launcherにはcheckpointだけでなく最終receipt pathとSHAも必須。phase1 promotion、precheck、旧v1 receiptでは起動できない。source/evaluator変更後は以前のreceiptを再利用せず、phase1評価・promotion・phase2 liftから再発行する。

## 所要時間と停止条件

このPCで100 updatesは約91〜95秒だった。目安は次の通り。

- sanitized601 -> 3000: 約38分
- fresh 0 -> 3000: 約48分
- 15,000 updatesの学習部分: 約4時間（評価・再試行を除く保守値）
- preview 100 updates: 約1.5〜2分

次のどれかで直ちに停止する: non-finite、fall、actual jointの動的soft-limit overshootが5deg超、commanded targetのsoft-limit excessが`1e-7rad`超、方向反転/応答不足、locked weight/Adam mutation、normalizer/legacy tensor drift、provenance/hash不一致、ONNX parity超過、tracking observation coverage欠落。camera/tracker/learned policyの異常時にも、実行側は別系統の legacy joystick walkへ同一cycleでfallbackする。
