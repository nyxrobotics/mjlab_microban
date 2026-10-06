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
- 学習リポジトリで変更してよいファイルは `config/home_pose.yaml` だけ。追跡外（.gitignore されていない）の
  ファイルも数える（9999 のコーナー救済はクリーンなツリーでしか学習しないので、本番では最初に拒否する）。
  `--allow-dirty` はドライランでだけ効く。
- `--no-push`: コミットまでしてpushしない。`--sequential`: 起き上がりを歩行と並列にしない。
- `--commit-trailer TEXT`: コミットメッセージの末尾に付ける行（既定は空。エージェントが実行するときは
  `Co-Authored-By: ...` を渡す）。
- 実行中に `config/home_pose.yaml` が変わったら（同じ worktree で次のHOMEを試した、など）、次のジョブ・
  コマンドの前に終了コード3で止まる。次のHOMEは別の worktree で試す。

## 何をするか

| 段階 | 内容 | 合格条件（満たさなければ止まる） |
| --- | --- | --- |
| 1. HOME | `scripts/home_pipeline/home_check.py`（重心と足裏接地面）、`balance_home_pose.py --check`、`home_pose_tool.py show`（学習ライン）、`write-robot`（ロボット側yaml） | 重心が足裏の前後範囲の外なら拒否。中心から 0.5 mm 以上ずれている、足裏がロールしていて床に触れる角が減る、正準の解でない、は警告。yamlは書き換えない |
| 1b. テスト | そのHOMEで学習側のテスト一式を CPU で実行（`uv run --locked --with pytest python -m pytest -q tests`、約2分、出力は `training_suite.out`）。コードのコミットと yaml が同じなら再開時は飛ばす。`--skip-training-suite` はドライランだけ | 失敗が `scripts/home_pipeline/known_test_failures.txt`（どのHOMEでも同じ既存の失敗）の中だけ。それ以外が1つでもあれば GPU を使う前に終了コード 3 |
| 2. 歩行 | `Mjlab-Velocity-Microban` 15000回（4096 env）→ 続きを30000まで。1000回ごとのチェックポイントを、GPUに空きがあれば学習中に、なければ学習後にPICOテレオペ環境の 9×300 プローブで評価。合格した上位6候補を計3回ずつプローブし、全回の最悪マージンが最大のものを選ぶ | 全回合格の候補がなければ、最悪マージン最大のものを使い **FALLBACK** と記録。v12の開始時プローブ（新しく1回）で落ちたら、次の候補に替えて開始し直す（候補がなくなったら止まる）。選んだものは `checkpoints/<prefix>_walk/` にコピー |
| 3. 起き上がり | 段階1〜5（2500 → ImuDelay +1500 → std 0.5 にリセット → CalmRoll +8000 → CalmEffortStrong +6500 → CalmPush +3000、段階3以降 entropy 0.001）。各段階のあと `scripts/home_pipeline/getup_eval.py` で評価（遅延0-3+ノイズの2シード、0.3 m/s 押し、姿勢）。歩行と並列 | 最終段階: 倒れた状態からの起立 ≥ 85 %、押しで転倒 ≤ 10 %、立位の関節速度 ≤ 0.3 rad/s、姿勢評価で立ち続け ≥ 80 % |
| 4. PICO v12 | 選んだ歩行から pose-release の新規チェーン（`train_microban_teleop_v12.sh start --hand-pose-release`）、0→3000→3100→7000→7100→10000→10100→15000、境界ごとにステージゲート（`evaluate_microban_teleop_v12_stage.sh` と同じ3つの評価を1本ずつ最後まで走らせ、全部合格なら `teleop_v12_stage create` でゲートを作る。あのスクリプトは `set -e` なので、追従で不合格になると ONNX の報告を書かずに終わり、評価が落ちたのと区別できない。パイプラインでは報告が書かれていれば合否の判定、書かれていなければ評価の異常終了として扱う） | 各ゲート。同じ親からの再学習は学習シードを変える（`train_microban_teleop_v12.sh --seed`、forward-lean-v2 eb02a05 から移植。前傾チェーンではシード 42 の再学習2回が同じ形で落ちた）。自動救済: カナリア（3099/7099/10099）が精度だけで落ちたらシード 43 で1回だけ再学習（再学習が model を保存する前に中断されても、run ディレクトリを作る前でも後でも、再開時にその再学習をシード 43 でやり直す）。評価が報告を書かずに落ちた（GPU の OOM など）ゲートは判定として記録せず、その場で止まり、再実行で評価し直す。10000 境界は人手なしで段階的に進める（2026-10 の前傾チェーンで手でやった順序）: 9999 が落ちたら model_9900 の pose-release コーナー救済を `--pr-corner-rescue-mixes`（既定 lf60,lf90,lf72,lf65。コーナー救済の起動スクリプトにはシード指定が無いので、同じ mix を繰り返し並べても違いは GPU の非決定性だけ。既定には繰り返しは無い）の順に1本ずつ学習して 9999 でゲート（`train_microban_teleop_v12_corner_rescue.sh --hand-pose-release --mix M`。親は新しい pose-release チェーンの model_9900 で、厳密評価（`--profile hmd_hand_reachable_performance_foot_exposure_v2`。救済のバリデータはこのプロファイルの報告しか受け付けない。指定しないと評価器は 9901 で deployed-accuracy プロファイルを選び、本番では救済が一度も走らなかった。古いプロファイルで記録された報告は評価し直す）が手先精度だけで落ちたものに限る。それ以外で落ちた親は救済を飛ばす）→ 全部落ちたらゲート済みの model_7099 から 7100→10000 を新しい試行として学習し直す（run `<prefix>_v12_7100_to10000_a2`、ゲートし、落ちたらその model_9900 から同じ救済）→ `--v12-9999-attempts`（既定 2。試行 k はシード 41+k）回で尽きたら止まる。15000 境界も同じように進める（forward-lean-v2 fc1c313〜cd0ea78 から移植した pose-release 最終シナリオ救済）: 14999 が落ち、歩行と ONNX の検査は合格で、追従の不合格が救済できるもの（手先・足先精度、`actual_soft_limits`、`twist_directional_response`）だけなら、同じ run の model_14900 から `--pr-final-rescue-mixes`（既定 pr_v1,…,pr_v6。pr_v5/pr_v6 は再生エピソードに評価器の固定の押し 0.35/-0.20 m/s・1.0 s 間隔も与える、005f55c）の順に `train_microban_teleop_v12_hand_pose_release_final_rescue.sh MODEL_14900 落ちたtracking報告 --mix M --seed S` を1本ずつ学習して 14999 でゲート（落ちたシナリオを全部再生しない mix はバリデータが拒否するので飛ばす）→ 全部落ちたら（または救済できない落ち方なら）ゲート済みの model_10099 から 10100→15000 を次のシードで学習し直す（run `<prefix>_v12_10100_to15000_a2`、ゲートし、落ちたらその model_14900 から同じ救済）→ `--v12-15000-attempts`（既定 2）回で尽きたら止まる。前傾HOMEでは前傾チェーン自身がこの境界を 10 回（再学習4回・救済6回）試して全部落ちているので、前傾 yaml ではここで止まる見込みが高い（通すには 10100→15000 のレシピ変更が要る）。最初の救済の前に `config/home_pose.yaml` を学習ブランチにコミットする（救済の起動スクリプトはクリーンなツリーでしか学習しない）。通った救済の model_9999（系譜 `fresh_chain_model9900_corner_rescue`）または再学習した試行を、通常の pose-release として続きを学習する。10000 境界（model_9999）と 10100 カナリアは、中心HOME以外の pose-release レシピでは手先 RMS 0.040 m まで許容するプロファイル（`*_hand_rms_40mm_v1`、forward-lean-v2 e3271de / ec67f1e から移植、ユーザー判断「手は 0.04mまで許容でいいんじゃない？」）、中心HOMEでは中心ブランチと同じ 0.035 m。15000 は completion-allowance プロファイルで評価される（どれも評価スクリプトが自動で選ぶ）。パッケージには続きを学習した 10000/10100 のゲートを `--boundary-gate` で渡し、どのプロファイルで判定したかをメタデータに残す。中心HOME以外ではパッケージャーが両方を resume 系譜上に必須とする（forward-lean-v2 7ceb280）ので、どちらかが無い・検証が通らなければ段階5で止まる |
| 5. 書き出し・導入 | walk.onnx / getup.onnx を書き出してロボットへ。ロボット側ピン（`pico_hybrid.py` の歩行ソースSHA・反復数・プローブSHA、`validate_pico_policy.py` の walk.onnx SHA）、歩行テスト用フィクスチャの再生成、テストに固定されたHOMEのピン（`test_shared_home.py` の `TRAINING_HOME_DEG` と `test_pico_hybrid.py` の `PACKAGER_V12_HOME_POSE_JSON`・実行ピン）を更新。それ以外のHOMEに依存する期待値は、ロボットのテストが `config/home_pose.yaml` から読む（robot home-config 31cfafa 以降）。そのロボットツリーに対してPICOをパッケージ（実ロボットのバリデータ込み）して導入 | `tools/validate_pico_policy.py` が pass、ロボットのテストスイートが全部通る |
| 6. コミット | ロボット: `--robot-branch` にコミットしてpush。学習: `config/home_pose.yaml`、記録 `config/releases/<tag>.json`、リリース一式（中心HOMEの 5b5a9d0 と同じ: PICO model_14999.pt と params/git、ゲート報告とONNX、パッケージONNXと受領書。加えてロボットがSHAで固定する選択歩行チェックポイントとその 9×300 プローブ受領書、起き上がり最終チェックポイント。gitignore対象なので `git add -f`）をコミットしてpush | 1〜5がすべて通ったときだけ |

ロボットのテストは、HOMEに依存する値（契約文字列、目標の座標系、立位の重力、get-up の契約など）を
読み込んだ `config/home_pose.yaml` から作るので、どのHOMEのブランチでも手で直す必要はない。中心HOMEの
リテラル（中心ブランチとの同一性の確認）は中心HOMEのときだけ確かめる。それでもテストが落ちたら段階5で
「robot test suite failed」と失敗したテストの一覧を出して止まる（直して同じコマンドを再実行すれば、
学習・書き出しは飛ばされ、検証とテストからやり直す）。

## 再開・停止・GPU

- 各段階は出力があり検証が通れば飛ばす。同じコマンドをもう一度実行すればよい。途中で止まった歩行・起き上がりの
  学習は最後のチェックポイントから続ける。PICO v12 の区間は、終わりの model を保存する前に止まったら、
  ゲート済みの親から（カナリアの再学習なら同じシード 43 で）その区間を学習し直す。
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
- `--status` は止まった理由（`stopped`）も表示する。再開すると前回の停止は `previous_stops` に移り、
  `stopped` には今の実行の停止だけが出る（件数と最後の時刻も表示）。
- `config/home_pose.yaml` は各ジョブの開始前と終了後に確認する。ジョブは起動の数秒後に yaml を読むので、
  ジョブの実行中に yaml が変わると（同じ worktree で次のHOMEを `balance_home_pose.py --write` した、など）、
  そのジョブの終わりで止まり（終了コード 3）、編集時点で動いていた（開始から 10 分以内の）ジョブが開始して
  以降に作られた run ディレクトリ・ゲート・プローブ受領書・評価報告を `<状態ディレクトリ>/quarantine/<時刻>/` に移す
  （state の `quarantined` に記録）。yaml を戻して再実行すると、それらは使われず学習し直される。

## ドライラン

配線だけを数十分〜1時間で確かめる。どのHOMEでも通る。各段階 2〜3 反復・64 env、PICOの境界はクロックを
持ち上げて越える（カナリアも含め各区間を本来の終わりの3反復前に持ち上げるので、ゲートは本来のクロック
3099/7099/10099 などで評価される。持ち上げたコピーは `params/agent.yaml` に親を resume と同じ形で記録する）、ゲートは評価して記録するだけ（plumbing mode: 強制しない。救済の経路を決めるのは
`--dry-run-simulate-failures` だけ）、最終パッケージは `scripts/home_pipeline/dry_run_tools.py`（`DRYRUN` と表示、
配備不可）。GPUジョブは1本ずつ（`--serial-gpu` がドライランの既定。起き上がりも歩行のあと）。ロボット側は
スクラッチのクローン（originのpush URLがローカルパス）でなければ拒否し、ロボットのコミットはローカルだけ、
学習リポジトリにはコミットもpushもしない（ドライランの救済は yaml をコミットしない）。次のどちらかが必要:

- `--dry-run-walk-init WALKER`: そのHOMEで学習した歩行チェックポイントから続ける（`train_microban_teleop_v12.sh start`
  と同じ確認: run の `params/env.yaml` の HOME・±π クリップ・生の前回行動と、HOMEスタンプ）。
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
`--dry-run-simulate-15000` は 14999 の経路を同じように選ぶ（既定 `pass` は不合格にしない）。dry の最終救済は
本物のバリデータ（`teleop_v12_hand_pose_release_final_rescue validate-parent`）を dry の model_14900 にかけて
判定を記録し、学習の代わりに `dry_run_tools.py stamp-final-rescue` で dry の model_14999 に本物と同じ最終救済の
系譜マーカーを付け、起動スクリプトと同じ `pr_final_rescue_seed_<sha16>/` の配置（親・落ちた報告・親 run の
params/agent.yaml）を作るので、パッケージャーの系譜チェックと resume 系譜の探索がそれに対して走る。
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

最終パッケージには 10000 境界を越えた model_9999（スタンプした救済、または学習し直した試行）と、そこから続けた
10100 カナリアの model_10099 にも同じ強制合格のゲート（`DRYRUN_boundary_gate_*.json`）を作り `--boundary-gate` で
渡すので、パッケージャー本物の境界チェック（resume 系譜上にあるか、救済マーカーを最終チェックポイントが引き継いで
いるか、中心HOME以外では両方そろっているか）が走り、境界のプロファイルがパッケージに記録される。
ドライランのパッケージはメタデータ `dry_run_not_deployable` を持ち、ロボットは `MICROBAN_ALLOW_DRYRUN_POLICY=1`
（ドライラン自身のバリデータ・テスト呼び出しだけが設定する）がなければ拒否する。本物のパッケージャーは
ドライランの証跡（ゲート・チェックポイントの `dry_run*` キー、`DRYRUN_*` の強制合格プローブ受領書）を拒否する。
ゼロからの policy は全シナリオで転ぶので、ロボットのバリデータがメタデータで再確認する歩行の合格欄
（転倒数・完走数・方向数・ソフトリミット超過）も force-probe と同様に合格値に書き換える（測った値は
`dry_run_original_*`）。

### 実測（2026-10-05、home-config 27b6a64、64 env、GPU は前傾 PICO 学習と共用、ジョブは1本ずつ）

| ドライラン | 9999 の経路 | 所要 | 終了コード | 結果 |
| --- | --- | --- | --- | --- |
| 前傾 yaml、`--dry-run-walk-init` 前傾 cont2 `model_29000.pt`、`--dry-run-simulate-9999 retrain` | 試行1 → 救済 lf60/lf90/lf72/lf65 全部不合格 → gated model_7099 から試行2 → 合格 | 56.5 分 | 0 | 開始時プローブ本物で合格（最悪マージン +0.0445）、境界ゲート 9999 は `..._hand_rms_40mm_v1` で記録、ロボットのバリデータ pass、ロボットのテスト 271 passed / 2 skipped |
| 膝15度（つり合わせた yaml）、`--dry-run-plumbing`、`--dry-run-simulate-9999 rescue` | 試行1 → 救済 lf60 不合格 → lf90 合格（`fresh_chain_model9900_corner_rescue`） | 23.9 分 | 0 | 開始時プローブは落ちて強制合格の複製から開始、起き上がりゲートは記録のみ、パッケージは救済の境界ゲートつきで robot バリデータ pass、ロボットのテスト 271 passed / 2 skipped |

どちらもカナリア 3100/7100/10100 の最初のゲートを不合格とみなして1回だけ学習し直す経路も通った。

再実測（2026-10-06、home-config 1059fe3、ロボット 2ffe8d2、同条件）: 前傾 yaml・`--dry-run-walk-init` 前傾 cont2
`model_29000.pt`・`--dry-run-simulate-9999 retrain` で 3598 s、終了コード 0。カナリアは本来のクロックで評価
（3099 / 7099 は `hmd_hand_activation_canary_*` など、10099 は `..._hand_rms_40mm_v1`）、試行1と救済4本が落ちて試行2が通り、
パッケージは 10000（試行2の model_9999）と 10100（model_10099）の境界ゲートを resume 系譜で検証して記録
（semantics `..._resume_ancestor_..._v2`）、`dry_run_not_deployable` 付き。ロボットのバリデータ pass、ロボットの
テスト 272 passed / 2 skipped。同じパッケージを `MICROBAN_ALLOW_DRYRUN_POLICY` なしで `tools/validate_pico_policy.py`
にかけると「DRY RUN package」で拒否される。膝15度の plumbing ドライランはこの版では再実行していない。

再々実測（2026-10-06、home-config 3b2dabf = 落ちたゲートの判定・再学習のシード・15000 境界の救済を入れた版、
ロボット 2ffe8d2、64 env、GPU は空き、ジョブは1本ずつ）:

| ドライラン | 9999 / 15000 の経路 | 所要 | 終了コード | 結果 |
| --- | --- | --- | --- | --- |
| 前傾 yaml、`--dry-run-walk-init` 前傾 cont2 `model_29000.pt`、`--dry-run-simulate-9999 retrain --dry-run-simulate-15000 rescue` | 9999: 試行1と救済 lf60/lf90/lf72/lf65 が落ち、model_7099 からシード 43 の試行2が合格。15000: 14999 不合格 → 最終救済 pr_v1 不合格 → pr_v2 合格 | 4357 s | 0 | カナリア 3099/7099/10099 はシード 43 で1回再学習。パッケージに `v12_final_rescue_marker_*`・`v12_final_rescue_training_replay = evaluator_scenario_commands`・`v12_final_rescue_final_gate_held_out = false` と 10000/10100 の境界ゲートが入り、ロボットのバリデータ pass、テスト 272 passed / 2 skipped |
| 膝15度、`--dry-run-plumbing`、`--dry-run-simulate-9999 rescue --dry-run-simulate-15000 retrain` | 9999: 救済 lf60 不合格 → lf90 合格。15000: 14999 不合格 → pr_v1〜pr_v6 の6本すべて不合格 → model_10099 からシード 43 の試行2が合格 | 1994 s | 0 | ロボットのバリデータ pass、テスト 272 passed / 2 skipped |

2026-10-06、home-config 57be9f6（膝15度・体幹5° の yaml、新しいクローン、`--dry-run-plumbing --dry-run-simulate-failures
--dry-run-simulate-9999 rescue`）: 最初にテスト一式（21 failed / 708 passed / 16 errors、既知の失敗だけ、2.1 分）。
カナリア 3099 の再学習が run ディレクトリ（`params/` だけ）を作った直後に SIGTERM（終了コード 130）→ 同じコマンドで再開すると
テストは記録から飛ばし、「canary 3099: resuming the interrupted retry ... with training seed 43」で `--seed 43` の再学習から続けた。
9999: 救済の親報告は `--profile hmd_hand_reachable_performance_foot_exposure_v2` で書かれ（以前は deployed-accuracy
プロファイルで、本番ではバリデータが必ず拒否していた）、lf60 不合格 → lf90 合格。再開から 55 分で終了コード 0、
リリース記録に `training_suite`、ロボットのバリデータ pass、テスト 272 passed / 2 skipped、ロボットのコミットはローカルだけ。

ドライランの救済バリデータは dry の親を「optimizer clock drifted」で拒否する（記録のみ、plumbing では強制しない）。
本番の判定経路（追従で落ちて ONNX の報告が無いゲートを判定として扱い、カナリアの再学習や救済に進む）は
ドライランでは通らないので、単体テスト（`RealGateVerdictTest`、`GateCrashTest`）と、本物の評価スクリプトに偽の評価器を
つないだ再現（落ちたゲートが `(False, ['hand_tracking_rms'], [])` と判定され記録される）で確かめた。

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
