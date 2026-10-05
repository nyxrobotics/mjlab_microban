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
| 4. PICO v12 | 選んだ歩行から pose-release の新規チェーン（`train_microban_teleop_v12.sh start --hand-pose-release`）、0→3000→3100→7000→7100→10000→10100→15000、境界ごとに `evaluate_microban_teleop_v12_stage.sh` | 各ゲート。自動救済: カナリア（3099/7099/10099）が精度だけで落ちたら1回だけ再学習。9999が落ちたら model_9900 の pose-release コーナー救済を試す（`train_microban_teleop_v12_corner_rescue.sh --hand-pose-release --mix <--pr-corner-rescue-mix、既定 lf60>`。新しい pose-release チェーンの model_9900 で、厳密評価が手先精度だけで落ちたものを親として受け付ける。救済後の model_9999 は通常の段階評価でゲートし、pose-release として続きを学習する）。15000 は pose-release レシピの completion-allowance プロファイルで評価される（評価スクリプトが自動で選ぶ） |
| 5. 書き出し・導入 | walk.onnx / getup.onnx を書き出してロボットへ。ロボット側ピン（`pico_hybrid.py` の歩行ソースSHA・反復数・プローブSHA、`validate_pico_policy.py` の walk.onnx SHA）、歩行テスト用フィクスチャの再生成、テストに固定されたHOME値（`TRAINING_HOME_DEG`、タグ、ルート高さ、全桁の角度、リビジョン文字列、`PACKAGER_V12_HOME_POSE_JSON`）を更新。そのロボットツリーに対してPICOをパッケージ（実ロボットのバリデータ込み）して導入 | `tools/validate_pico_policy.py` が pass、ロボットのテストスイートが全部通る |
| 6. コミット | ロボット: `--robot-branch` にコミットしてpush。学習: `config/home_pose.yaml`、記録 `config/releases/<tag>.json`、リリース一式（中心HOMEの 5b5a9d0 と同じ: PICO model_14999.pt と params/git、ゲート報告とONNX、パッケージONNXと受領書。加えてロボットがSHAで固定する選択歩行チェックポイントとその 9×300 プローブ受領書、起き上がり最終チェックポイント。gitignore対象なので `git add -f`）をコミットしてpush | 1〜5がすべて通ったときだけ |

ロボットのテストのうち、HOMEの値を前提に作られた変異ケース（例: `test_home_pose_config.py` の
`rad_not_deg` は膝 0 のyaml文字列を置換する）は自動では直せない。そのときは段階5で
「robot test suite failed」と失敗したテストの一覧を出して止まるので、テストを手で直して同じコマンドを
もう一度実行する（学習・書き出しは飛ばされ、検証とテストからやり直す。手で直したテストは上書きしない）。

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

配線だけを数十分で確かめる。各段階 2〜3 反復・64 env、歩行は既存の歩行チェックポイントから続ける
（3反復の歩行ではv12開始時のプローブに通らないため）、PICOの境界はクロックを持ち上げて越える、
ゲートは評価して表示するだけ（強制しない）、最終パッケージは `scripts/home_pipeline/dry_run_tools.py`
（`DRYRUN` と表示、配備不可）。ロボット側はスクラッチのクローン（originのpush URLがローカルパス）でなければ
拒否し、ロボットのコミットはローカルだけ、学習リポジトリにはコミットもpushもしない。

```bash
git clone ../microban_homecfg /tmp/robot_scratch && git -C /tmp/robot_scratch remote set-url --push origin /nonexistent
python3 scripts/retrain_all_for_home.py --dry-run \
    --dry-run-walk-init ../mjlab_microban_track/checkpoints/centered_home_velocity_cont/model_20000.pt \
    --dry-run-simulate-failures \
    --robot-repo /tmp/robot_scratch --robot-branch dryrun --state-dir /tmp/home_dry
```

ドライランの歩行は3反復の続きなので、v12 の開始時プローブのマージンがプローブのばらつき（±0.02 程度）と
同じ大きさになる。開始時プローブで落ちたら、候補を順に替えて計 `DRY_START_ATTEMPTS`（4）回まで
開始し直す（本番は各候補1回だけで、候補がなくなったら止まる）。

`--dry-run-simulate-failures` は最初のカナリアゲートと9999ゲートを不合格とみなして、
再学習とコーナー救済の経路も通す。

**ドライランは中心HOMEでしか最後まで通らない。** `--dry-run-walk-init` の歩行チェックポイントは、続きの学習で
読み込むときに HOME スタンプが照合される（`require_current_home_walk_checkpoint`）。中心HOMEで学習した
`model_20000.pt` は、編集したHOMEでは「Walking checkpoint was not trained at the current HOME」で拒否される。
省くと3反復の歩行になり、v12 の開始時プローブで止まる。編集したHOMEで確かめられるのは、
`home_pose_tool.py show`、`balance_home_pose.py`、各タスクの数反復の学習（`uv run train <task> --env.scene.num-envs 16
--agent.max-iterations 3`）まで。配線そのものはHOMEに依存しないので、中心HOMEのドライランで確かめる。
PICOの最終パッケージでは、転倒した DRYRUN チェックポイントの短い smoke コーパス（16行未満）を16行に繰り返して
パッケージャーに渡す（本番の最終ゲートは全シナリオ完走が条件なので常に16行）。

2026-10-05 の確認（中心HOME、他の学習とGPU共用）: 約55分で最後まで通った（ゲート評価12回が大半）。
歩行の選択（3チェックポイント、上位2候補を2回ずつ）、起き上がり5段階と評価、v12 の開始時プローブ・
ブートストラップ・全境界のクロック持ち上げとゲート、カナリア再学習、コーナー救済の経路（親の model_9900 が
ないので記録だけ）、15000 で completion-allowance プロファイルが選ばれること、書き出し・ピン更新・
実ロボットバリデータ付きパッケージ・ロボットのテスト（268 passed）・ローカルコミット。
再実行では学習と評価をすべて飛ばし、`pico_hybrid.py` が変わったときだけ再パッケージする。

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
