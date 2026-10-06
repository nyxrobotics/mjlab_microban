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
# ロボットリポジトリ: 方策の契約（docs/policies.md、src/agents/manifest.json と tools/validate_policies.py）を
# 実装したブランチの、専用の worktree を使う。パイプラインはこの worktree のブランチを切り替え、書き換える。
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
| walk | `Mjlab-Velocity-Microban` を最初から1本（4096 env、seed 42、最大 `WALK_MAX_UPDATES`）。選んだチェックポイントを `checkpoints/<prefix>_walk_<sha>/` に置く（PICO の来歴がそこを再ハッシュする） | 学習の異常終了・停滞 |
| pico | 入口: 歩行器の契約の確認、9×300 プローブ（seed 42、`config/pipeline.yaml` の閾値）、bootstrap ゲート。`Mjlab-Teleop-V12-HandPoseRelease-Microban` を1本（2048 env、critic の準備 1000 → 手 → 足、合計 9000）。最後に判定1回: 歩行 9×300・追従（最終プロファイル、手先 RMS 0.040 m）・ONNX（どれも seed 42）。合格ならゲートファイルを作る | 入口のプローブ不合格、判定の不合格 |
| getup | `Mjlab-Getup-Microban` を1本（4096 env、16500 回。IMU 遅延 2500、calm と探索の切り替え 4000、effort と押し 10000）。最後に判定1回（`mjlab_microban.pipeline.getup_eval`、遅延 0-3 とノイズの2シード、0.3 m/s 押し、姿勢） | 倒れた状態からの起立 < 0.85、押しで転倒 > 0.10、立位の関節速度 > 0.30 rad/s、姿勢 < 0.80、立位でクリップに張り付く割合 > 0.05 |
| export | walk.onnx、getup.onnx、pico_teleop.onnx と manifest.json（`docs/policies.md`）を `<状態>/release/` に書く | 書き出しの検査（パリティ、グラフ、メタデータ） |
| install | ロボットの worktree に 3 つの ONNX と manifest.json、ロボット用 `config/home_pose.yaml` を書き、ロボットの `tools/validate_policies.py src/agents/manifest.json` とテスト一式を実行 | バリデータかテストが落ちる |
| commit | ロボットのブランチにコミットして push、学習側は `config/home_pose.yaml` と `config/releases/<tag>/`（manifest.json と記録）をコミットして push | |

判定に落ちたら、その場で止まって報告する（終了コード 1）。救済、乱数の種を変えた学習のやり直し、
閾値の変更はしない。原因を直してレシピの変更としてコミットし、同じコマンドを回す。

## 再開・停止・GPU

- `state.json` に、段階ごとの状態（done / running / failed）、入力のハッシュ、出力のハッシュを持つ。
  - done で入力も出力ファイルも変わっていない段階は飛ばす。
  - running の段階（プロセスが落ちた、停滞で止めた、Ctrl-C）は、学習なら最後のチェックポイントから続ける。
  - failed の段階は、入力が変わっていなければ理由を表示して止まる（直してから回す）。
  - 入力は段階ごとに関係するファイルだけ（`pipeline/steps.py` の `STEP_INPUTS`、`config/pipeline.yaml` の
    その段階の節、上流の段階の出力）。PICO のコードを直しても歩行はやり直しにならない。
- GPU の学習と評価は1本ずつ。本番は、別の学習（`train` コマンド、または 1024 env 以上）が GPU を使っている
  あいだは次の GPU ジョブを始めずに待ち、待った時間を `STATUS.log` に書く（ドライランは待たない）。
- ログが一定時間（学習 20 分、評価 30 分）書かれないジョブは止め、終了コード 2 で終わる。止めるのは自分が
  起動したジョブのプロセスグループだけ。
- 終了コード: 0 完了、1 チェック不合格、2 停滞、3 入力・事前確認の拒否、4 同じ状態ディレクトリを別の
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
