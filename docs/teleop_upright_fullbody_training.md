# Upright full-body teleop training

この学習タスクは、**実機のAボタン初期姿勢と21関節のHOME角を一致させて**、全身トラッキング歩行ポリシーをゼロから学習する。左右の肩ピッチと膝ピッチは `0°`、股ピッチは `+1.198384259489°`、足首ピッチは `-1.198384259489°`。MuJoCoの体幹ピッチは `0°`、root高さは `0.170554885633559 m`。タスク名は `Mjlab-Teleop-Upright-Fullbody-Microban`、保存先は `logs/rsl_rl/mjlab_microban_teleop_upright_fullbody/` である。ほかの学習PCのrunと混同しないよう、run labelには `upright_fullbody` を含める。

Aボタンの実機ニュートラルと学習HOMEを上記の同じ関節角にする。肩ピッチは、腕を前後に動かす肩関節の角度を指す。新規タスクの起動時に21関節すべてとroot姿勢を共通HOMEと照合する。稼働中のPICOモデルは旧HOMEに合わせたままとし、新しいモデルの完成時にPICO側のdefault姿勢とONNXを同時に切り替える。

MuJoCoの `robot.xml` で左右の股ピッチを同じ角 `h`、足首ピッチを `-h` にすると、膝0°・体幹垂直のまま両足裏が水平になる。肩ピッチ0°で全身重心の前後座標と左右の足裏collision box群の中心座標との差を計算し、この差が0となる `h` を数値的に求めた。`h=+1.198384259489°` では差が約 `2.4e-16 m`。足裏collision boxの最下点を床の `z=0` に置くroot高さは `0.170554885633559 m` で、全関節0°時の接地root高さ `0.170644236955448 m` より約 `0.08935 mm` 低い。これはMuJoCoモデル上の幾何計算であり、学習や実機の重心実測ではない。

学習環境は現行の83観測・18関節raw actionと、歩行→HMD/手→足の15,000更新カリキュラムを使う。actorの63個の旧歩行入力も含め、すべてのMLP重みと正規化統計を更新する。旧v12の固定チェックポイント、旧歩行モデル、旧HOME由来のwalk004モーションpriorは読み込まない。criticからもpriorの39観測と関連rewardを除いたため、旧v12チェックポイントとの互換性はない。

専用runnerはcheckpointにレシピ版とHOME版、root位置・姿勢、21関節の名前・初期角を記録する。再開時は現在のソースと完全一致するかをモデル読込前に確認し、旧v12やほかのHOMEのcheckpointを拒否する。

## 別PCでの起動

学習PCではこのリポジトリの学習ブランチをチェックアウトする。初回のcheckoutと依存関係のセットアップは次のとおり。学習はこのPCでは実行しない。

| 用途 | リポジトリ | ブランチ |
| --- | --- | --- |
| 全身テレオペの再学習 | `mjlab_microban`（このPCのworktreeは `mjlab_microban_v12`） | `pico-v12-centered-home` |
| 歩行・転倒復帰・追従の共通学習 | `mjlab_microban` | `centered-home-training` |

```bash
git fetch origin
git switch --track origin/pico-v12-centered-home
uv sync --locked
git rev-parse HEAD
scripts/train_microban_teleop_upright_fullbody.sh start pico_v12_physical_neutral_fullbody_v4
```

初回の試行を100更新で止める場合は、末尾に `100` を付ける。途中から再開する場合は、保存先の実際のrunディレクトリ名、`model_<番号>.pt` の番号、追加更新数、新しいrun labelを指定する。例は100更新後の `model_99.pt` から残り14,900更新する場合である。

```bash
scripts/train_microban_teleop_upright_fullbody.sh start pico_v12_centered_home_fullbody_pilot 100
scripts/train_microban_teleop_upright_fullbody.sh resume \
  RUN_DIRECTORY_NAME 99 14900 pico_v12_centered_home_fullbody_continued
```

学習の入力、optimizer、ログ、checkpoint名は旧v12のrunと別である。**旧v12のstage gate、finalize、ONNX exporterを新runに適用しない。** 旧v12は既存配備モデルの履歴用レシピとして残す。新runは専用の歩行・転倒・追従評価とONNX/実機メタデータ検証ができるまでシミュレーション用であり、ロボットには反映しない。
