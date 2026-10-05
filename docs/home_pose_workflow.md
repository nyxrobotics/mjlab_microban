# HOMEを変えたときの全ポリシー再学習（`scripts/retrain_all_for_home.py`）

HOMEは `config/home_pose.yaml` だけで決まる（`config/README.md`）。HOMEを変えたら、
歩行・起き上がり・PICO v12 をすべて学習し直し、ロボット側に入れて、両リポジトリにコミットする。
その一連の流れを1つのコマンドにしたのが `scripts/retrain_all_for_home.py`。

```text
home_pose.yaml を編集 → (任意) balance_home_pose.py --write → retrain_all_for_home.py → 終わったら両リポジトリにコミット・push（自動）
```

## 使い方

```bash
# 学習リポジトリ: home-config ブランチ（またはそこから作ったブランチ）のチェックアウトで実行する。
# 2026-10 時点では ../mjlab_microban_homecfg。mjlab_microban（getup-on-pico-v12-trunk）や
# mjlab_microban_track（track-centered-home-clip）にはこのスクリプトも config/ もない。
cd ../mjlab_microban_homecfg
# ロボットリポジトリ: config/home_pose.yaml を読む home-config 系のブランチの、専用の worktree を使う。
# デプロイ用の ../microban（feature/neck-roll-pitch-camera）は HOME を手書きで持っているので
# パイプラインが最初に拒否する。パイプラインはこの worktree のブランチを切り替え、書き換える。
git -C ../microban fetch origin
git -C ../microban worktree add -b home-knee15 ../microban_home-knee15 origin/home-config
# 1. HOMEを編集する（変えられる値は config/README.md の表）。name / label も新しい姿勢に合わせる。
# 2. 重心を足裏の前後中央に戻す（任意、股・足首ピッチだけを書き換える）。体幹ピッチも値の1つで、
#    --trunk-pitch-deg 10 なら前傾HOME（前傾ラインの仕組みで学習する。config/README.md）
uv run python config/balance_home_pose.py --write
# 3. 全部やり直す（数十時間。中断しても同じコマンドで続きから）
python3 scripts/retrain_all_for_home.py --commit-trailer "Co-Authored-By: ..." \
    --robot-repo ../microban_home-knee15 --robot-branch home-knee15 --training-branch home-knee15
# 進み具合
python3 scripts/retrain_all_for_home.py --status --state-dir artifacts/home_pipeline/<tag>_<hash>
tail -f artifacts/home_pipeline/<tag>_<hash>/STATUS.log
```

- `--robot-repo`: ロボット（microban）のチェックアウト。`src/home_pose.py` を `src/constants.py` が
  importしている（`config/home_pose.yaml` を読む）ものでなければ、学習を始める前に終了コード3で拒否する。
  ポリシー・HOME yaml・ピンをここに書き込み、`--robot-branch` にコミットする（なければ今のHEADから作る）。
  段階5で止まると、このツリーには新しい walk/getup ONNX・ピン・テストの値と古い pico_teleop.onnx が
  コミットされずに残る（PICOバリデータが落ちる）。同じコマンドで再開するか、`git -C <robot> checkout -- .`
  で戻す。デプロイ中のチェックアウトを使わないのはこのため。
- `--training-branch`: 学習リポジトリのコミット先（省略時は今のブランチ。なければ今のHEADから作る）。
  学習中のコードは切り替えないので、既存の別ブランチへは自分で切り替えてから実行する。
- 学習リポジトリで変更してよい追跡ファイルは `config/home_pose.yaml` だけ（ほかの変更は先にコミット、
  または `--allow-dirty`）。
- `--no-push`: コミットまでしてpushしない。`--sequential`: 起き上がりを歩行と並列にしない。
- `--commit-trailer TEXT`: コミットメッセージの末尾に付ける行（既定は空。エージェントが実行するときは
  `Co-Authored-By: ...` を渡す）。
- 実行中に `config/home_pose.yaml` が変わったら（同じ worktree で次のHOMEを試した、など）、次のジョブ・
  コマンドの前に終了コード3で止まる。次のHOMEは別の worktree で試す。

## 何をするか

| 段階 | 内容 | 合格条件（満たさなければ止まる） |
| --- | --- | --- |
| 1. HOME | `scripts/home_pipeline/home_check.py`（重心と足裏接地面）、`balance_home_pose.py --check`、`home_pose_tool.py show`（学習ライン）、`write-robot`（ロボット側yaml） | 重心が足裏の前後範囲の外なら拒否。中心から 0.5 mm 以上ずれている、足裏がロールしていて床に触れる角が減る、正準の解でない、は警告。yamlは書き換えない |
| 2. 歩行 | `Mjlab-Velocity-Microban` 15000回（4096 env）→ 続きを30000まで。1000回ごとのチェックポイントを、GPUに空きがあれば学習中に、なければ学習後にPICOテレオペ環境の 9×300 プローブで評価。合格した上位6候補を計3回ずつプローブし、全回の最悪マージンが最大のものを選ぶ | 全回合格の候補がなければ、最悪マージン最大のものを使い **FALLBACK** と記録。v12の開始時プローブ（新しく1回）で落ちたら、次の候補に替えて開始し直す（候補がなくなったら止まる）。選んだものは `checkpoints/<prefix>_walk/` にコピー |
| 3. 起き上がり | 段階1〜5（2500 → ImuDelay +1500 → std 0.5 にリセット → CalmRoll +8000 → CalmEffortStrong +6500 → CalmPush +3000、段階3以降 entropy 0.001）。各段階のあと `scripts/home_pipeline/getup_eval.py` で評価（遅延0-3+ノイズの2シード、0.3 m/s 押し、姿勢）。歩行と並列 | 最終段階: 倒れた状態からの起立 ≥ 85 %、押しで転倒 ≤ 10 %、立位の関節速度 ≤ 0.3 rad/s、姿勢評価で立ち続け ≥ 80 % |
| 4. PICO v12 | 選んだ歩行から pose-release の新規チェーン（`train_microban_teleop_v12.sh start --hand-pose-release`）、0→3000→3100→7000→7100→10000→10100→15000、境界ごとに `evaluate_microban_teleop_v12_stage.sh` | 各ゲート。自動救済: カナリア（3099/7099/10099）が精度だけで落ちたら1回だけ再学習。10000 境界は人手なしで段階的に進める（2026-10 の前傾チェーンで手でやった順序）: 9999 が落ちたら model_9900 の pose-release コーナー救済を `--pr-corner-rescue-mixes`（既定 lf60,lf90,lf72,lf72、同じ mix の繰り返しは別の run＝GPU の非決定性で結果が変わる）の順に1本ずつ学習して 9999 でゲート（`train_microban_teleop_v12_corner_rescue.sh --hand-pose-release --mix M`。親は新しい pose-release チェーンの model_9900 で、厳密評価が手先精度だけで落ちたものに限る。それ以外で落ちた親は救済を飛ばす）→ 全部落ちたらゲート済みの model_7099 から 7100→10000 を新しい試行として学習し直す（run `<prefix>_v12_7100_to10000_a2`、ゲートし、落ちたらその model_9900 から同じ救済）→ `--v12-9999-attempts`（既定 2）回で尽きたら止まる。最初の救済の前に `config/home_pose.yaml` を学習ブランチにコミットする（救済の起動スクリプトはクリーンなツリーでしか学習しない）。通った救済の model_9999（系譜 `fresh_chain_model9900_corner_rescue`）または再学習した試行を、通常の pose-release として続きを学習する。15000 は pose-release レシピの completion-allowance プロファイルで評価される（評価スクリプトが自動で選ぶ） |
| 5. 書き出し・導入 | walk.onnx / getup.onnx を書き出してロボットへ。ロボット側ピン（`pico_hybrid.py` の歩行ソースSHA・反復数・プローブSHA、`validate_pico_policy.py` の walk.onnx SHA）、歩行テスト用フィクスチャの再生成、テストに固定されたHOMEのピン（`test_shared_home.py` の `TRAINING_HOME_DEG` と `test_pico_hybrid.py` の `PACKAGER_V12_HOME_POSE_JSON`・実行ピン）を更新。それ以外のHOMEに依存する期待値は、ロボットのテストが `config/home_pose.yaml` から読む（robot home-config 31cfafa 以降）。そのロボットツリーに対してPICOをパッケージ（実ロボットのバリデータ込み）して導入 | `tools/validate_pico_policy.py` が pass、ロボットのテストスイートが全部通る |
| 6. コミット | ロボット: `--robot-branch` にコミットしてpush。学習: `config/home_pose.yaml`、記録 `config/releases/<tag>.json`、リリース一式（中心HOMEの 5b5a9d0 と同じ: PICO model_14999.pt と params/git、ゲート報告とONNX、パッケージONNXと受領書。加えてロボットがSHAで固定する選択歩行チェックポイントとその 9×300 プローブ受領書、起き上がり最終チェックポイント。gitignore対象なので `git add -f`）をコミットしてpush | 1〜5がすべて通ったときだけ |

ロボットのテストは、HOMEに依存する値（契約文字列、目標の座標系、立位の重力、get-up の契約など）を
読み込んだ `config/home_pose.yaml` から作るので、どのHOMEのブランチでも手で直す必要はない。中心HOMEの
リテラル（中心ブランチとの同一性の確認）は中心HOMEのときだけ確かめる。それでもテストが落ちたら段階5で
「robot test suite failed」と失敗したテストの一覧を出して止まる（直して同じコマンドを再実行すれば、
学習・書き出しは飛ばされ、検証とテストからやり直す）。

## 再開・停止・GPU

- 各段階は出力があり検証が通れば飛ばす。途中で止まった学習は最後のチェックポイントから続ける。
  同じコマンドをもう一度実行すればよい。
- 状態: `artifacts/home_pipeline/<tag>_<hash>/`（`STATUS.log` に1行ずつ、各ジョブの出力は `logs/`、
  機械用の `state.json`、歩行の選択表 `walk_selection.txt`、起き上がり評価 `getup_eval/`、
  書き出し物 `export/`）。HOMEが変われば別の状態ディレクトリになる。
- GPUジョブは、必要な空きメモリ（4096 env 学習 11000 MiB、v12学習 17000 MiB（2048 env で約 16.1 GB 使う）、
  評価 3000 MiB。`--train-gpu-mib` などで変更可）ができるまで待ってから始める。待っている間もロックは持たない
  ので、大きなジョブの待ちが、いま入る小さな評価や、走っているジョブの停滞検知を止めることはない。
  始めたジョブの分は 120 秒間「予約」として空きから差し引き、自分のジョブ2つが同じ空きで同時に始まらないようにする。
  学習中の歩行プローブは空きがなければ待たずに後回しにする（歩行の停滞検知を止めないため）。
- ログが一定時間（学習 20 分、ゲート 60 分）書かれないジョブは停止させ、終了コード 2 で止まる。
  止めるのは自分が起動したジョブのプロセスグループだけで、他のプロセスには一切シグナルを送らない。
  Ctrl-C / SIGTERM でも自分のジョブだけ止めて終わる。
- 終了コード: 0 完了、1 チェック/ゲート不合格（想定外の例外も、STOPPED を記録し自分のジョブを止めてから 1）、
  2 停滞、3 入力・事前確認の拒否、4 同じ状態ディレクトリを別のインスタンスが使用中（このとき実行中の
  インスタンスの `state.json` と `STATUS.log` には何も書かない）、130 中断。
- `--status` は止まった理由（`stopped`）も表示する。

## ドライラン

配線だけを数十分〜1時間で確かめる。どのHOMEでも通る。各段階 2〜3 反復・64 env、PICOの境界はクロックを
持ち上げて越える、ゲートは評価して記録するだけ（plumbing mode: 強制しない。救済の経路を決めるのは
`--dry-run-simulate-failures` だけ）、最終パッケージは `scripts/home_pipeline/dry_run_tools.py`（`DRYRUN` と表示、
配備不可）。GPUジョブは1本ずつ（`--serial-gpu` がドライランの既定。起き上がりも歩行のあと）。ロボット側は
スクラッチのクローン（originのpush URLがローカルパス）でなければ拒否し、ロボットのコミットはローカルだけ、
学習リポジトリにはコミットもpushもしない（ドライランの救済は yaml をコミットしない）。次のどちらかが必要:

- `--dry-run-walk-init WALKER`: そのHOMEで学習した（HOMEスタンプが一致する）歩行チェックポイントから続ける。
  最初に照合し、違えば終了コード3で止まる。v12 の開始時プローブが本物で通るので、ロボットのバリデータと
  テストも強制する。例: 前傾の yaml なら前傾の cont2 `model_29000.pt`、中心なら `model_20000.pt`。
- `--dry-run-plumbing`: 歩行から全部ゼロから（編集したばかりのHOME、例えば膝15度）。3反復の歩行は歩けないので、
  v12 の開始時プローブ・ロボットのバリデータ・ロボットのテストは実行して記録するが止めない。落ちた
  開始時プローブは `artifacts/legacy_teleop_probe/DRYRUN_FORCED_PASS_<受領書>` に合否欄だけ合格に書き換えて
  複製し（測った値は `dry_run_original_*` に残す）、`train_microban_teleop_v12.sh start --dry-run-probe-receipt`
  でそこからチェーンを始める。

```bash
# 前傾 yaml、前傾の歩行から（別の worktree で yaml を差し替え、コミットはしない）
git worktree add --detach ../train_lean origin/home-config
cp tests/fixtures/home_pose_forward_lean.yaml ../train_lean/config/home_pose.yaml
git clone ../microban_homecfg /tmp/robot_lean
cd ../train_lean && python3 scripts/retrain_all_for_home.py --dry-run \
    --dry-run-walk-init /path/to/forward_lean/model_29000.pt \
    --dry-run-simulate-failures --dry-run-simulate-9999 retrain \
    --robot-repo /tmp/robot_lean --robot-branch dryrun --state-dir /tmp/home_dry_lean

# 膝15度（balance_home_pose.py --write でつり合わせた yaml）、ゼロから
python3 scripts/retrain_all_for_home.py --dry-run --dry-run-plumbing \
    --dry-run-simulate-failures --dry-run-simulate-9999 rescue \
    --robot-repo /tmp/robot_k15 --robot-branch dryrun --state-dir /tmp/home_dry_k15
```

`--dry-run-simulate-failures` は各カナリアの最初のゲートと 9999 のゲートを不合格とみなす。
`--dry-run-simulate-9999` で 9999 の経路を選ぶ: `retrain`（試行1の救済が全部落ち、学習し直した試行2が通る。
前傾チェーンが実際にたどった経路）、`rescue`（2つ目の mix の救済が通る）、`stop`（全部落ちて止まる）。
ドライランの救済は、本物の救済バリデータ（`teleop_v12_corner_rescue validate-parent`）を dry の model_9900
（model_9999 のクロックを 9900 に下げたもの）にかけて判定を記録し（Adam のステップ数が違うので必ず拒否される）、
2048 env の救済学習の代わりに `dry_run_tools.py stamp-corner-rescue` で dry の model_9999 に本物と同じ
pose-release コーナー救済の系譜マーカーを付ける。そのあとの通常の再開・ゲート・パッケージャーは本物の
系譜バリデータでそれを受け付ける。

ドライランの歩行は3反復の続きなので、v12 の開始時プローブのマージンがプローブのばらつき（±0.02 程度）と
同じ大きさになる。`--dry-run-walk-init` で開始時プローブに落ちたら、候補を順に替えて計 `DRY_START_ATTEMPTS`（4）回まで
開始し直す（本番は各候補1回だけで、候補がなくなったら止まる）。
PICOの最終パッケージでは、転倒した DRYRUN チェックポイントの短い smoke コーパス（16行未満）を16行に繰り返して
パッケージャーに渡す（本番の最終ゲートは全シナリオ完走が条件なので常に16行）。

DRYRUN_RESULTS_PLACEHOLDER

## 所要時間の目安（RTX 5000 Ada 1枚、他の学習と共用だった 2026-10 の実測から）

| 段階 | 実測 | 
| --- | --- |
| 歩行 15000（4096 env） | 7.0〜10.7 h |
| 歩行 続き 15000 | 4.5 h |
| 歩行プローブ（約30回 + 選択の再プローブ12回、学習と並行） | 約 1 h |
| 起き上がり 段階1〜5 | 約 11.4 h（1.3 + 0.85 + 4.5 + 3.3 + 1.6） |
| PICO v12 0→15000 + ゲート7回 | 約 7.5〜12 h（単独だと短い） |
| 書き出し・パッケージ・検証・テスト | 約 0.2 h（CPU） |

GPUを使う時間の合計は約 32〜40 GPU時間。GPUに歩行と起き上がりの両方が入る空きがあれば並列に走り、
壁時計ではおよそ 21〜28 時間（歩行 12〜15 h の間に起き上がりが終わり、そのあとPICO 8〜12 h）。
ほかの学習とGPUを共用していて片方しか入らないとき（例: 前傾の v12 ジョブが 16 GB 使っている間）は
順番に走るので、壁時計もほぼ 32〜40 時間になる。
