# Microban teleop v12: legacy-preserving policy training

この文書は、実績のある joystick 歩行 actor を壊さずに PICO の HMD・手・足の目標を追加する手順を固定する。v12 の canonical policy と、早期確認専用の simulation preview は別物である。preview は実機配備できない。

## 固定した契約

- source: `checkpoints/xc330_velocity/model_14999.pt`
- source SHA-256: `b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141`
- source probe: `artifacts/legacy_teleop_probe/model_14999_teleop83_raw_9x300.json`
- probe SHA-256: `f51378d59ff4d68fb1185a91eb2a863749e5c7be6ec4cd0ab4a0b08f1565e69d`
- recipe: `legacy_velocity_model14999_staged_mask_raw_actions_v2`
- adapter schedule: `freeze_extra_to7000_then_hmd_hand_to10000_then_all_v1`
- actor: `83 -> 512 -> 256 -> 128 -> 18`, ELU、scalar unbounded Gaussian
- action: clip なし。target は `default_joint_pos + raw_action * scale`
- previous action observation: actor の raw output をそのまま再入力

63 個の legacy observation は名前で 83 列へ移植する。対応は `0:6 -> 0:6`, `6:24 -> 9:27`, `24:42 -> 30:48`, `42:60 -> 48:66`, `60:63 -> 66:69`。EmpiricalNormalization の legacy 63 列、全 trunk、bias、head、Gaussian std は凍結する。追加20列の normalizer は identity であり、normalizer 自体も親 module が train mode になっても更新されない。

学習可能なのは第一層 weight の追加列だけである。ただし inactive target のノイズ学習で adapter が暴走しないよう、更新可能列を段階的に開く。

- completed updates `<= 7000`: 追加20列をすべて exact zero に固定
- `7001..10000`: HMD 6列と hand 8列だけを許可
- `>= 10001`: foot 6列も許可し、追加20列すべてを許可

境界 rollout の古い batch で新しい列を更新しないため、gradient hook が見る `common_step_counter == 7000*24` と `10000*24` はまだ lock する。再開後の次 rollout が最初の学習対象になる。lock 中の weight と Adam moment は、update・save・resume の各時点で exact zero を検証する。

## v1 run の棄却と sanitizer

旧 v1 は HMD/target が無効な期間にも追加列を学習した。model300 の forced-HMD 評価で actual soft-limit violation と forward 符号反転を確認したため、旧 run は採用しない。保存済み model600 は actor の追加20列と対応する Adam moments だけを zero に戻した別 checkpoint へ変換した。source は上書きしていない。

canonical sanitized checkpoint:

```text
logs/rsl_rl/mjlab_microban_teleop_v12/2026-09-25_16-30-00_v12_sanitized601/model_600.pt
SHA-256 d31d5362dc6776a47395bcefc446bb4324d87361e423edc4a575b2b972d269a3
iteration 600 / completed updates 601 / common_step_counter 14424
```

再現する場合は、旧 v1 checkpoint を source にして次を実行する。destination は必ず新規 path にする。

```bash
uv run --locked python -m \
  mjlab_microban.scripts.sanitize_teleop_v12_adapter_checkpoint \
  PATH_TO_V1_MODEL_600.pt PATH_TO_NEW_V2_MODEL_600.pt \
  --output artifacts/teleop_v12_sanitization/model600_to_v2_sanitized.json
```

sanitizer は actor shared columns、trunk、critic、optimizer の他 state と param groups が bit-identical であることを検証する。advanced indexing の copy を誤って zero にする回帰を防ぐため、reset は `index_fill_` に固定した。

## canonical training と必須 gate

新規開始は次の wrapper だけを使う。

```bash
scripts/train_microban_teleop_v12.sh start
```

sanitized601 からの正式な再開例:

```bash
scripts/evaluate_microban_teleop_v12_stage.sh \
  2026-09-25_16-30-00_v12_sanitized601 600

scripts/train_microban_teleop_v12.sh resume \
  2026-09-25_16-30-00_v12_sanitized601 \
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

tracking gate は early termination、fall、non-finite、実 joint soft-limit violation、raw previous-action 不一致、HMD 3軸 motion 不足、必要な hand/foot observation が zero、全 nonzero twist 軸の方向/応答不足を拒否する。raw target の hypothetical soft-limit excess は legacy source 自体でも起こるため report-only であり、実 joint limit を判定する。

stage profile は「次 stage を学習する前の policy に未学習性能を要求しない」順序である。

- 3000まで: locomotion + HMD/hand/foot exposure safety
- 3001..7000: expanded locomotion + pre-HMD exposure safety
- 7001..10000: HMD/hand performance + foot exposure safety
- 10001..14999: whole-body performance
- 15000: full-body performance + perturbation

どの report でも status/check/hash/schema/必要 scenario/evidence が欠けたら停止する。空の `checks={}` は pass と見なさない。ONNX は `[1,83] -> [1,18]`、neutral 10,000 samples と full83 64 samplesの最大誤差がともに `2e-5` 以下で、`CPUExecutionProvider` を実際に使うことを要求する。

## simulation-only early preview

手足の挙動を canonical 15k 完了前に見る場合だけ、次を使う。

```bash
scripts/train_microban_teleop_v12_preview.sh \
  v12_fullbody_preview_canary100 --updates 100
```

これは sanitized601 の actor/critic/optimizerを一切変更せず、iteration/common clockだけを `10000/10001` に liftして、HMD・hand・foot curriculumと追加20列を即時有効化する。その後1〜1000 updatesだけを専用 taskで学習する。

preview checkpoint は永久に次を持つ。

- `preview_non_deployable=true`
- `teleop_v12_preview.revision=sanitized601_clock_lift_to10001_full20_sim_only_v1`
- source sanitized checkpoint SHA、clock-lift lineage、full20 active columns

canonical runner、stage gate、通常 ONNX export、実機 deployment はこの marker を必ず拒否する。PICO simulation は明示的な preview option と専用 task/consumer APIを使った場合だけ読み込める。preview の結果を canonical 3000/7000/10000/15000 chainへ戻してはならない。

## 所要時間と停止条件

このPCで100 updatesは約91〜95秒だった。目安は次の通り。

- sanitized601 -> 3000: 約38分
- fresh 0 -> 3000: 約48分
- 15,000 updatesの学習部分: 約4時間（評価・再試行を除く保守値）
- preview 100 updates: 約1.5〜2分

次のどれかで直ちに停止する: non-finite、fall、actual joint-limit violation、方向反転/応答不足、locked weight/Adam mutation、normalizer/legacy tensor drift、provenance/hash不一致、ONNX parity超過、tracking observation coverage欠落。camera/tracker/learned policyの異常時にも、実行側は別系統の legacy joystick walkへ同一cycleでfallbackする。
