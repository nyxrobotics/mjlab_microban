# HOME姿勢の設定

全ポリシー（歩行・起き上がり・PICO v12）が使うHOME姿勢は、このディレクトリの
`home_pose.yaml` だけで決まる。ほかの値はすべて `src/mjlab_microban/robot/home_pose.py`
が `robot.xml` の MuJoCo FK で計算する。

| ファイル | 役割 |
| --- | --- |
| `home_pose.yaml` | 唯一の入力。21関節の角度（度）、体幹ピッチ、名前、ラベル |
| `home_pose_tool.py` | `show`: 派生値と「この学習ラインで再学習できるか」の表示。`write-robot`: ロボット側 `config/home_pose.yaml` の生成 |
| `balance_home_pose.py` | 股・足首ピッチだけを動かして重心を足裏の前後中央に合わせ、`home_pose.yaml` を書き換える |

## yamlから計算される値

- root姿勢: 体幹ピッチからクォータニオン、足裏collision boxの最下点が z=0 になる root z
- HOMEでの projected gravity（体幹座標系）
- 全身重心、足裏接地面の前後範囲、かかと側・つま先側の余裕
- 起き上がりの頭高さ `HEAD_STANDING_HEIGHT`（0.1 mm 丸め）、両足の横間隔 `HOME_FEET_LATERAL_M`
- 腕HOMEに依存する手先FKのオフセット・範囲・正規化値
- HOMEの識別子: 関節値と体幹ピッチの正規形のSHA-256先頭10桁（`joint_hash`）と、
  契約文字列に埋め込む `tag`

`tag` は現在の中心HOMEでは `centered_home` で、今日までの文字列
（`v3_centered_home_servo_range`、`centered_home_hip_plus1p198384259489_...` など）と完全に一致する。
値を1つでも変えると `tag` は `<label>_<joint_hash>` になる。古いチェックポイント・ゲート・ONNXは
文字列やHOMEスタンプが合わないため拒否される。中心HOMEのときだけ、手で丸めた足の横間隔 0.094 m
（FKでは 0.0935 m）も維持される（`LEGACY_HOME_OVERRIDES`）。

ローダーは、関節名21個がそろっていること、左右対称であること（左右ペアの一致・反転に加えて、
首ヨー `head` と `neck_roll` が 0）、MJCFの可動範囲内であること、`trunk_pitch_deg` で両足裏が水平
（1e-9 rad 以内）であることを確認し、満たさなければ読み込みを拒否する。`-0.0` は `0.0` と同じ値として扱う
（同じハッシュ・`tag`）。

## この学習ラインで再学習できるHOME

ローダーが読めるHOMEでも、このチェックアウトの学習タスクが受け付けないものがある。
`home_pose_tool.py show` と `balance_home_pose.py` は、候補のHOMEを別プロセスで
`mjlab_microban.robot.home_pose.HOME` に入れて `mjlab_microban.tasks` をimportし（歩行・起き上がり各段階・
PICO v12 と救済段階の全タスクの環境設定が組み立てられる、GPU不要、数秒）、結果を「training line」として表示する
（`mjlab_microban/robot/home_pose_training.py`）。規則の一覧を手で持たず、タスク自身に聞いている。

| 変えたい値 | この学習ライン |
| --- | --- |
| 膝（左右同じ） | 可。`balance_home_pose.py --write` で股・足首を合わせればよい。膝 20° で歩行と起き上がりの数回の学習、v12 の開始時HOME照合とテレオペ環境のプローブが動くことを確認済み |
| 股・足首ロール（左右反転）、首ピッチ | 可 |
| 体幹ピッチ | 0 だけ。前傾HOMEの座標系は `forward-lean-v2` ブランチ |
| 肩ピッチ | 0 だけ（PICO contract v12 のHOME revision） |
| 肘・肩ロール | 手先の到達範囲が PICO 受信側で検証済みの ±0.064 m の箱に収まる範囲だけ。今の腕HOMEで x がすでに 0.0630 m なので、1つだけ動かす場合の目安は 肘 −22.4°〜−19.1°、肩ロール（左）−2.6°〜44.7°（MJCFの範囲内で） |
| 首ヨー `head`、`neck_roll` | 0 だけ（左右対称でなくなるため、ローダーが拒否） |

この表の外へ進めるには、そのタスク側の制約（受信箱の検証、v12 contract など）を先に変える必要がある。

## HOMEを変える手順

```text
home_pose.yaml を編集 → balance_home_pose.py（任意）→ home_pose_tool.py show / write-robot → 全ポリシー再学習 → 両リポジトリでコミット
```

手順3〜7は1つのコマンドでまとめて実行できる（中断しても同じコマンドで再開、詳細は
[`docs/home_pose_workflow.md`](../docs/home_pose_workflow.md)）:

```bash
python3 scripts/retrain_all_for_home.py --robot-repo ../microban --robot-branch home-<label> \
    --training-branch home-<label>
```

以下はそのコマンドが行う内容（手で行う場合の手順）。

1. `home_pose.yaml` を編集する（膝など。変えられる値は上の表）。
2. （任意）`uv run python config/balance_home_pose.py` で重心を合わせる。既定は確認だけ（dry run）で、
   `--write` を付けると股・足首ピッチの4つの値だけを書き換える（後述）。学習タスクが受け付けないHOMEは
   書き込まない。
3. `name` と `label` を新しい姿勢に合わせて直し（ツールは変えない。`tag` は `<label>_<hash>` になる）、
   `uv run python config/home_pose_tool.py show` で根元高さ・余裕・頭高さ・`tag`・`training_line` を確認する
   （学習タスクが受け付けないHOME、足裏が水平でないyamlは `error: ...` の1行で終了コード1）。
4. `uv run python config/home_pose_tool.py write-robot --microban-repo ../microban` でロボット側の
   `config/home_pose.yaml` を書き出す（`--check` で最新か確認できる）。学習タスクが受け付けないHOMEは
   書き出さない（`--force` で強制）。失敗はどれも `error: ...` の1行。
5. すべてを最初から学習し直す: 歩行（`Mjlab-Velocity-Microban` 15000回とその続き、プローブで選択）、
   起き上がり5段階、PICO v12（`scripts/train_microban_teleop_v12.sh start --source ... --hand-pose-release`）。
   歩行チェックポイントには `microban_walk_home_pose` が記録され、v12の開始時に現在のHOMEと照合される。
6. 3つのポリシーをロボットの `src/agents/` に入れる。ロボット側では `tests/test_shared_home.py` の固定値と、
   ランごとに変わる値（`pico_hybrid.py` の歩行ソースSHA、`tools/validate_pico_policy.py` の `walk.onnx` SHA）
   も更新する。PICOのパッケージは、そのロボット側ツリーに対して作る（`--microban-repo`）。
7. 両方のリポジトリでテストを通し、コミットする。

## 重心合わせツール（`balance_home_pose.py`）

```bash
uv run python config/balance_home_pose.py                         # 確認だけ: 変更前後の表を表示
uv run python config/balance_home_pose.py --write                 # yamlを書き換える
uv run python config/balance_home_pose.py --trunk-pitch-deg 10    # 体幹ピッチも変える（既定はyamlの値のまま）
uv run python config/balance_home_pose.py --check                 # yamlが正準の解でなければ終了コード1
uv run python config/balance_home_pose.py --yaml path/to/home.yaml
uv run python config/balance_home_pose.py --write --force         # 学習タスクが拒否しても書く（別ラインのHOME用）
uv run python config/balance_home_pose.py --no-training-check     # 学習ラインの確認（数秒）を省く
```

- **動かすのは股ピッチと足首ピッチだけ**（左右同じ値）。膝・腕・ロール・体幹ピッチなど他の値はそのまま。
- **満たす条件は2つ**（`home_pose.py` の `analyze_pose` と同じ定義、`robot.xml` の MuJoCo FK）:
  1. 体幹ピッチで両足裏が水平（`flat_sole_trunk_pitch_rad` = 目標の体幹ピッチ）
  2. 全身重心のxが、足裏接地面（地面に触れている足裏collision boxの角）の前後範囲の中央に一致
- **解き方**: 足裏を水平にした状態の2残差に対するNewton法（中心差分ヤコビアン）。収束しなければ股ピッチの
  範囲を走査し、各点で足首が足裏を水平に保つ条件のもとでBrent法を使う。収束判定は 1e-13 rad、1e-13 m。
- **答えは一意（正準形）**: 解は常に 股 = 足首 = 0 から始め（yamlに入っている股・足首ピッチの値は使わない）、
  浮動小数点の精度まで詰めてから小数12桁（度）に丸める（丸めによる重心のずれは 1e-14 m 未満）。
  同じ姿勢なら必ず同じ文字列・同じHOMEハッシュ・同じ `tag` になるので、たとえば膝を15°にして `--write`、
  0°に戻して `--write` すると、ファイルはバイト単位で元に戻り `centered_home`（`bbef07cab8`）のままになる。
  許容誤差内で釣り合っていても正準形でない値（手で書いた全桁の値など）は、`--write` で正準形に直す
  （ハッシュが変わるので表示で知らせる）。`--check` は「`--write` しても何も変わらない」ときだけ 0 を返す。
- **「水平」はピッチ方向の意味**: 体幹を傾けると、固定した股・足首ロールのため足裏にわずかなロールと
  つま先の内向き（体幹 +10° でロール約 0.08°、ヨー約 0.87°）が残る。表に「sole roll / sole yaw」として表示する。
  変更前のyamlで足裏が水平でない場合、接地を前提にした行（重心と足裏中央の差、かかと・つま先余裕、接地角数）は
  意味を持たないので `n/a` と表示する。
- **拒否する場合**: 左右のピッチ値が違う、股・足首ピッチがない・数値でない（`true` や空欄も不可）・有限でない、
  他の関節が左右非対称・範囲外、体幹ピッチが ±90° 以上、収束しない、解がMJCFの可動範囲や学習のソフトリミット
  （範囲の中央90 %）の外、重心が足裏中央に届かない、ファイルがない・YAMLとして読めない・値の行が見つからない
  （flow形式など）。どの場合も `error: cannot balance FILE: 理由` の1行を出して終了コード1、yamlは変更しない。
- **学習ラインの確認**: 書き換えが必要なとき、解いたHOMEでこのチェックアウトの学習タスクがimportできるかを
  確認して「training line: OK / REFUSED: 理由」を表示する（上の表）。`--write` は REFUSED なら書かずに
  `error: cannot write FILE: ...` で終了コード1（`--force` で書く。そのときは警告を出す）。`--check` は重心だけを見る。
- **書き換え**: `--write` は4つのピッチ値（`--trunk-pitch-deg` で変えたときは `trunk_pitch_deg` も）だけを、
  コメント・行の順序・改行コード（CRLFも）・キーの引用符を残したまま書き換える。値は小数点を必ず含む最短の表記
  （`1.0e-05` など。PyYAMLは `1e-05` を文字列として読むため）で、`-0.0` は `0.0` と書く。その後ローダーで読み直して
  検査し、失敗したら元のバイト列に戻す。正準の解がすでに入っているyamlは変更しない（何度実行しても同じ）。
  `name` と `label` は変えない（書き込み後の案内に現在の値と新しい `tag` を表示する）。
- **再現できる今日の解**:
  - 体幹 0°: 股 +1.198384259489°、足首 −1.198384259489°、root z 0.170554885633559、余裕 30.83/30.83 mm（現在のyamlがそのまま解）
  - 体幹 +10°: 股 −14.166561199931°、足首 +4.127976841869°（前傾ブランチの全桁の解 −14.166561199931119 /
    +4.127976841869204 を12桁に丸めた値）、root z 0.170430569776402、余裕 30.96/30.96 mm
- 体幹 +10° の解はこの学習ラインでは REFUSED になる（体幹ピッチ 0 だけに対応）。書くには `--force` が要る。

## Pythonからの利用（重心合わせなどのツール向け）

```python
from mjlab_microban.robot.home_pose import (
    HOME,                    # 読み込み済みのHOME（HomePose）
    analyze_pose,            # 任意の関節角（rad）と体幹ピッチからFK解析（1回 約50 us）
    home_pose_from_values,   # 値からHomePoseを作る（yamlと同じ検査）
    load_home_pose,          # yamlを読む
    rewrite_home_pose_yaml,  # コメントを残したまま値だけ書き換える
)

a = analyze_pose(dict(HOME.joint_pos_rad), trunk_pitch_rad=HOME.trunk_pitch_rad)
a.com_offset_x                 # 重心x - 足裏接地面の前後中央 [m]
a.sole_pitch_rad               # 左右の足裏の傾き（0で水平）
a.heel_margin_m, a.toe_margin_m
a.flat_sole_trunk_pitch_rad    # 足裏が水平になる体幹ピッチ
```
