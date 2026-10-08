# HOMEを変えたときの全ポリシー再学習（`scripts/retrain_all_for_home.py`）

HOMEは `config/home_pose.yaml` だけで決まる（`config/README.md`）。HOMEを変えたら、歩行・PICO・
起き上がりをすべて最初から学習し直し、判定し、ロボット側に入れて、両リポジトリにコミットする。
その一連の流れを1つのコマンドにしたのが `scripts/retrain_all_for_home.py`
（中身は `src/mjlab_microban/pipeline/`）。

```text
home_pose.yaml を編集 → (任意) balance_home_pose.py --write → retrain_all_for_home.py → 両リポジトリにコミット・push（自動）
```

## 使い方

```bash
# 学習リポジトリ: home-config ブランチ（またはそこから作ったブランチ）のチェックアウトで実行する。
cd ../mjlab_microban_homecfg
# ロボットリポジトリ: 方策の契約 microban-policy-1（docs/policies.md、runtime-cleanup から作ったブランチ）を
# 実装したブランチの、専用の worktree を使う（契約の版とレシピ id が合わなければ学習の前に止まる）。パイプラインはこの worktree のブランチを切り替え、書き換える。
git -C ../microban worktree add -b home-lean ../microban_home-lean <契約を実装したロボットのブランチ>
# 1. HOMEを編集する（変えられる値は config/README.md の表）。name / label も新しい姿勢に合わせる。
# 2. 重心を足裏の前後中央に戻す（任意）。--trunk-pitch-deg 10 なら前傾HOME。
uv run python config/balance_home_pose.py --write
# 3. 全部やり直す（中断しても同じコマンドで続きから）
uv run --locked python scripts/retrain_all_for_home.py \
    --robot-repo ../microban_home-lean --robot-branch home-lean [--training-branch home-lean]
# 進み具合
uv run --locked python scripts/retrain_all_for_home.py --status --state-dir artifacts/home_pipeline/<prefix>_<tag>
tail -f artifacts/home_pipeline/<prefix>_<tag>/STATUS.log
```

- 引数は `--robot-repo`、`--robot-branch`、`--training-branch`、`--state-dir`、`--no-push`、`--status`、
  `--dry-run` だけ。学習の回数・環境数・閾値は `config/pipeline.yaml`、学習の時刻（段の切り替え）は
  `src/mjlab_microban/schedules.py` に1か所ずつある。
- 学習リポジトリで変更してよいファイルは `config/home_pose.yaml` だけ（本番は追跡外のファイルも含めて
  拒否する。学習したコードと記録が必ず一致するように）。実行中に yaml が変わると止まる（終了コード 3）。

## 何をするか

| 段階 | 内容 | 止まる条件 |
| --- | --- | --- |
| home | `mjlab_microban.pipeline.home_check`（重心と足裏接地面）、`balance_home_pose.py --check`（警告のみ）、`home_pose_tool.py show`、学習側のテスト一式（CPU、約1分） | 重心が足裏の外、テストが1つでも落ちる |
| walk | `Mjlab-Velocity-Microban` を最初から1本（4096 env、seed 42、最大 `WALK_MAX_UPDATES` = 8000）。続けて2回合格しなければ、5000 回以降に確かめたチェックポイントから一番良いもの（合格した項目の数、いちばん苦手な指令の報酬の余裕、後のもの、の順）を採る。選んだチェックポイントを `checkpoints/<prefix>_walk_<sha>/` に置く（PICO の来歴がそこを再ハッシュする） | 採ったチェックポイントが確認の項目に落ちる |
| pico | 入口: 歩行器の契約の確認、9×300 プローブ（seed 42、合格ラインは `twist_pass_line.py`）、bootstrap ゲート。`Mjlab-Teleop-V12-HandPoseRelease-Microban` を1本（2048 env、critic の準備 1000 → 手 1000、手を絞る 2500、合計 5500。足の追従はしない: 足の目標は常に中立、足の報酬は重み 0）。最後に判定1回: 歩行 9×300・追従（最終プロファイル、手先 RMS 0.040 m、足の項目なし、mixed の場面は指令の範囲の半分）・ONNX（どれも seed 42）。合格ならゲートファイルを作る | 入口のプローブ不合格、判定の不合格 |
| getup | `Mjlab-Getup-Microban` を1本（4096 env、18000 回。IMU 遅延 2500、calm と探索の切り替え 4000、押しなしの effort 10000、押し 15000）。最後に判定1回（`mjlab_microban.pipeline.getup_eval`、遅延 0-3 とノイズの2シード、0.3 m/s 押し、姿勢） | 倒れた状態からの起立 < 0.85、押しで転倒 > 0.10、立位の関節速度 > 0.30 rad/s、姿勢 < 0.80、立位で目標が切り詰め（±π）に張り付く割合 > 0.05 |
| （足） | PICO の足の追従は外した（2026-10-08、ユーザーの判断）。v12 の PICO は凍結した歩行器に入力の列を足すだけの作りで、一度も足を上げられなかった。足は、歩行の部分も一緒に学習させる作りなど、別の取り組みとして残す。観測の足の列と実機の受け取りはそのまま（常に中立が入る） | |
| export | walk.onnx、getup.onnx、pico_teleop.onnx と manifest.json（`docs/policies.md`）を `<状態>/release/` に書く | 書き出しの検査（パリティ、グラフ、メタデータ） |
| install | ロボットの worktree に 3 つの ONNX と manifest.json、ロボット用 `config/home_pose.yaml` を書き、ロボットの `tools/validate_policies.py src/agents` とテスト一式を実行 | バリデータかテストが落ちる |
| commit | ロボットのブランチにコミットして push、学習側は `config/home_pose.yaml` と `config/releases/<tag>/`（manifest.json と記録）をコミットして push | |

判定に落ちたら、その場で止まって報告する（終了コード 1）。まずテストの妥当性を疑う（合格ラインが報酬の
考え方と合っているか、評価のコードに誤りがないか、指令・シード・押し・測り方が妥当か。ユーザーの方針、
2026-10-07）。テストが誤りなら、評価のコードか `config/pipeline.yaml` の確認の設定を直してコミットし、
同じコマンドを回す。学習はし直さず、学習済みの方策を判定し直す。テストが妥当で落ちているなら、それは
モデルの問題なので止めて報告する。テストに合わせて報酬を変えることはしない。救済、乱数の種を変えた学習の
やり直しもしない。

## 学習中の確認と早期停止（`pipeline/monitor.py`、閾値は `config/pipeline.yaml` の各 `check:`）

学習は止めずに、保存されたチェックポイントを別のプロセスで 1 本ずつ、古い順にすべて確認する（学習が先に
終わったら、残りを順に確認する）。確認の結果は `state.json` と `STATUS.log` に残る。確認の不合格で学習を
途中で止めることはない（学習は予定の回数まで回す）。

| 方策 | いつ | 確認 | 早期停止 | 中止 |
| --- | --- | --- | --- | --- |
| 歩行 | 1000 回ごと | 保持プローブ W1-W5（`pipeline/walk_probe.py`、シード 101-105）と 9×300（シード 101）。合格ラインは歩行の速度報酬そのもの（`twist_pass_line.py`：指令があれば平均の速度での報酬が止まっているとき（0.5）より高い、止まれなら動いた量が 0.1 m/s 前進の指令より小さい）、関節の限界の超過は 0.25 rad まで | N と N+1000 が続けて合格したら学習を止め、model_N+1000 を採る。ただし唯一の段（3000 回）の後 2000 回（`WALK_MIN_FINAL_UPDATES`、5000）より前では止めない。最後の 7999 も確認する | 段の切り替えが表の時刻から 1 回以上ずれた |
| PICO | 手を絞った後、1000 回ごと | 判定と同じ 3 つの評価をシード 101 で | 同じ。ただし最後の段の半分（`PICO_MIN_FINAL_UPDATES`）より前では止めない | 段の切り替えが表の時刻から 1 回以上ずれた |
| 起き上がり | 3500 と 9500（記録だけ）、effort の後 1000 回ごと | 判定と同じ 4 つの評価を別のシードで | 同じ（`GETUP_MIN_FINAL_UPDATES` より前では止めない） | 段の切り替えが表の時刻から 1 回以上ずれた |

採ったチェックポイント（または学習の最後）に対して、判定をシード 42（起き上がりは 11 と 5）で 1 回だけ行う。
どの方策でも、段の切り替えのログ（`Curriculum stage ... (update U)`）が表の時刻から 1 回より遅れたら中止する。

## 再開・停止・GPU

- `state.json` に、段階ごとの状態（done / running / failed）、入力のハッシュ、出力のハッシュを持つ。
  - done で入力も出力ファイルも変わっていない段階は飛ばす。
  - running の段階（学習のプロセスが落ちた、ジョブが停滞して止めた、Ctrl-C・SIGTERM）は、学習なら最後の
    チェックポイントから続ける。これらは判定ではないので failed にしない（`core.JobStopped`）。
    判定の不合格、中止の規則、書き出しやゲートの拒否は failed にする。
  - failed の段階は、入力も判定（`src` と `config/pipeline.yaml`）も変わっていなければ理由を表示して止まる。
    入力が同じで判定だけが変わったとき（テストを直したとき）は、学習はそのままで判定をやり直す。
  - 入力は段階ごとに、学習が読むものだけ（`pipeline/steps.py` の `STEP_INPUTS`、`config/pipeline.yaml` の
    その段階の節のうち `TRAINING_KEYS`（task、envs、save_interval）、上流の段階の出力）。PICO のコードを
    直しても歩行はやり直しにならず、確認や判定の設定を直しても学習はやり直しにならない。
  - 段階の学習と同じ run 名の学習が動いていれば（同じコマンドで手で始めたもの。出力は
    `<状態>/logs/<時刻>_train_<段階>.log`）、新しく始めずにそれを見守る。
- GPU の学習と評価は1本ずつ。本番は、別の学習（`train` コマンド、または 1024 env 以上）が GPU を使っている
  あいだは次の GPU ジョブを始めずに待ち、待った時間を `STATUS.log` に書く（ドライランは待たない）。この本番の
  run 名（`<prefix>_`）の学習は別の学習に数えない。
- ログが一定時間（学習 20 分、評価 30 分）書かれないジョブは止め、終了コード 2 で終わる。学習のプロセスが
  0 以外で終わったときも終了コード 2。どちらも同じコマンドで回し直せば続きから再開する。止めるのは自分が
  起動したジョブのプロセスグループだけ。
- 終了コード: 0 完了、1 チェック不合格、2 判定なしで止まった（学習の異常終了、停滞）、3 入力・事前確認の拒否、4 同じ状態ディレクトリを別の
  インスタンスが使用中、130 中断。

## ドライラン

```bash
# 学習リポジトリの別の worktree に試したい yaml を置き、ロボットはスクラッチのクローンにする
git worktree add --detach ../train_dry origin/home-config
cp tests/fixtures/home_pose_forward_lean.yaml ../train_dry/config/home_pose.yaml
git clone ../microban ../robot_dry && git -C ../robot_dry remote set-url --push origin /dev/null/no-push
cd ../train_dry && uv sync --locked
uv run --locked python scripts/retrain_all_for_home.py --dry-run --robot-repo ../robot_dry --robot-branch dryrun
```

- 64 env、すべての時刻を `config/pipeline.yaml` の `dry.schedule_scale`（0.001）倍にして、全段階・全部の
  切り替えを数十分で通す。数回しか学習しない方策は判定に通らないので、ドライランでは判定を記録だけして
  続け（`pipeline/dry.py`: 入口のプローブと PICO の判定報告を合否欄だけ合格にした `DRYRUN_FORCED_PASS_*` の
  複製で先へ進める。測った値は `dry_run_original_*` に残す）、PICO のパッケージには
  `dry_run_not_deployable` を付ける（ロボットは `MICROBAN_ALLOW_DRYRUN_POLICY=1` のときしか受け付けない）。
- ロボット側はスクラッチのクローン（origin の push URL がローカルパス）でなければ拒否する。ロボットの
  コミットはローカルだけ、学習リポジトリにはコミットも push もしない。
