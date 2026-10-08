# HOME姿勢の設定

全ポリシーが使うHOME姿勢は、このディレクトリの
`home_pose.yaml` だけで決まる。ほかの値はすべて `src/mjlab_microban/robot/home_pose.py`
が `robot.xml` の MuJoCo FK で計算する。

| ファイル | 役割 |
| --- | --- |
| `home_pose.yaml` | 唯一の入力。21関節の角度（度）、体幹ピッチ、名前、ラベル |
| `home_pose_tool.py` | `show`: 派生値と「この学習ラインで再学習できるか」の表示 |
| `balance_home_pose.py` | 股・足首ピッチだけを動かして重心を足裏の前後中央に合わせ、`home_pose.yaml` を書き換える |

## yamlから計算される値

- root姿勢: 体幹ピッチからクォータニオン、足裏collision boxの最下点が z=0 になる root z
- HOMEでの projected gravity（体幹座標系）
- 全身重心、足裏接地面の前後範囲、かかと側・つま先側の余裕
- 頭の高さ（0.1 mm 丸め）、両足の横間隔
- HOMEの識別子: 関節値と体幹ピッチの正規形のSHA-256先頭10桁（`joint_hash`）と `tag`

## 公開済みのHOME

方策を学習して公開したHOMEは前傾HOME（体幹 +10°、joint hash `481503d292`）で、`home_pose.py` の
`PUBLISHED_HOME_OVERRIDES`（joint hash がキー）に、そのHOMEで公開した値を持つ。それ以外のHOMEの `tag` は
`<label>_<joint_hash>` になる。

root z は FK 値を 1e-12 m に丸めて公開する。前傾HOMEは公開済みの値（0.170430569776402）を
`PUBLISHED_HOME_OVERRIDES` の `root_z_m` に固定し、読み込み時に FK との差が 1e-12 m 以内かを確かめる。
股・足首ピッチも同様に、yaml の12桁の正準値（−14.166561199931 / 4.127976841869、ハッシュはこの値）から
1.2e-13° 以内であることを確かめた上で、全桁の値（−14.166561199931119 / 4.127976841869204）を
`HOME.joint_pos_deg` / `joint_pos_rad` として使う（`HOME.input_joint_pos_deg` がyamlの値）。

## 読み込みの検査

ローダーは、関節名21個がそろっていること、左右対称であること（左右ペアの一致・反転に加えて、
首ヨー `head` と `neck_roll` が 0）、MJCFの可動範囲内であること、`trunk_pitch_deg` で両足裏が水平
（1e-9 rad 以内）であること、さらに足裏が床に平らに着くこと（足裏面の角48個すべてが、ワールド高さで
床から 0.5 mm 以内、ロール約 0.7° まで。ロールした足裏は縁だけで立つので拒否する）を確認し、満たさなければ読み込みを拒否する。`-0.0` は `0.0` と同じ値として扱う
（同じハッシュ・`tag`）。

## この学習ラインで再学習できるHOME

ローダーが読めるHOMEでも、このチェックアウトの学習タスクが受け付けないものがある。
`home_pose_tool.py show` と `balance_home_pose.py` は、候補のHOMEを別プロセスで
`mjlab_microban.robot.home_pose.HOME` に入れて `mjlab_microban.tasks` をimportし（登録した全タスクの
環境設定が組み立てられる、GPU不要、数秒）、結果を「training line」として表示する
（`mjlab_microban/robot/home_pose_training.py`）。規則の一覧を手で持たず、タスク自身に聞いている。

| 変えたい値 | この学習ライン |
| --- | --- |
| 膝（左右同じ） | 可。`balance_home_pose.py --write` で股・足首を合わせればよい |
| 股・足首ロール（左右反転）、股ヨー | 足裏が床に平らに着く組み合わせだけ（ふつうは 足首ロール = −股ロール、股ヨー 0）。股ロールだけを変えると足裏がロールしてローダーが拒否する |
| 首ピッチ | 可 |
| 体幹ピッチ | 可（±90° 未満） |
| 首ヨー `head`、`neck_roll` | 0 だけ（左右対称でなくなるため、ローダーが拒否） |

この表の外へ進めるには、そのタスク側の制約を先に変える必要がある。

## HOMEを変える手順

```text
home_pose.yaml を編集 → balance_home_pose.py（任意）→ home_pose_tool.py show → 全ポリシー再学習
```

1. `home_pose.yaml` を編集する（膝など。変えられる値は上の表）。
2. （任意）`uv run python config/balance_home_pose.py` で重心を合わせる。既定は確認だけ（dry run）で、
   `--write` を付けると股・足首ピッチの4つの値だけを書き換える（後述）。学習タスクが受け付けないHOMEは
   書き込まない。
3. `name` と `label` を新しい姿勢に合わせて直し（ツールは変えない。`tag` は `<label>_<hash>` になる）、
   `uv run python config/home_pose_tool.py show` で根元高さ・余裕・頭高さ・`tag`・`training_line` を確認する
   （学習タスクが受け付けないHOME、足裏が水平でないyamlは `error: ...` の1行で終了コード1）。

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
  0°に戻して `--write` すると、ファイルはバイト単位で元に戻り、HOMEハッシュも元のままになる。
  許容誤差内で釣り合っていても正準形でない値（手で書いた全桁の値など）は、`--write` で正準形に直す
  （ハッシュが変わるので表示で知らせる）。`--check` は「`--write` しても何も変わらない」ときだけ 0 を返す。
- **「水平」**: ツールが合わせるのはピッチ方向。ロールは動かさないので、足裏がロールしていて床に縁でしか
  触れないHOME（股ロールだけを変えた、股ヨーと膝を組み合わせた、など）は「HOME soles are not flat on the floor」
  で拒否する。体幹を傾けると、
  固定した股・足首ロールのため足裏にわずかなロールとつま先の内向き（体幹 +10° でロール約 0.08°、ヨー約 0.87°）が
  残るが、角48個すべてが床から 0.5 mm 以内なので受け付ける。表に「sole roll / sole yaw」として表示する。
  変更前のyamlで足裏が水平でない場合、接地を前提にした行（重心と足裏中央の差、かかと・つま先余裕、接地角数）は
  意味を持たないので `n/a` と表示する。
- **拒否する場合**: 左右のピッチ値が違う、股・足首ピッチがない・数値でない（`true` や空欄も不可）・有限でない、
  他の関節が左右非対称・範囲外、体幹ピッチが ±90° 以上、収束しない、解がMJCFの可動範囲や学習のソフトリミット
  （範囲の中央90 %）の外、重心が足裏中央に届かない、ファイルがない・YAMLとして読めない・値の行が見つからない
  （flow形式など）。どの場合も `error: cannot balance FILE: 理由` の1行を出して終了コード1、yamlは変更しない。
- **学習ラインの確認**: 解いたHOME（書き換え不要なときはyamlのHOME）でこのチェックアウトの学習タスクがimportできるかを
  確認して「training line: OK / REFUSED: 理由」を表示する（上の表）。`--write` は REFUSED なら書かずに
  `error: cannot write FILE: ...` で終了コード1（`--force` で書く。そのときは警告を出す）。`--check` は重心だけを見る。
- **書き換え**: `--write` は4つのピッチ値（`--trunk-pitch-deg` で変えたときは `trunk_pitch_deg` も）だけを、
  コメント・行の順序・改行コード（CRLFも）・キーの引用符を残したまま書き換える。値は小数点を必ず含む最短の表記
  （`1.0e-05` など。PyYAMLは `1e-05` を文字列として読むため）で、`-0.0` は `0.0` と書く。その後ローダーで読み直して
  検査し、失敗したら元のバイト列に戻す。正準の解がすでに入っているyamlは変更しない（何度実行しても同じ）。
  `name` と `label` は変えない（書き込み後の案内に現在の値と新しい `tag` を表示する）。
- **解の例**:
  - 体幹 0°: 股 +1.198384259489°、足首 −1.198384259489°、余裕 30.83/30.83 mm（`tests/fixtures/home_pose_centered.yaml`）
  - 体幹 +10°: 股 −14.166561199931°、足首 +4.127976841869°（全桁の解 −14.166561199931119 /
    +4.127976841869204 を12桁に丸めた値）、root z 0.170430569776402、余裕 30.96/30.96 mm
- この体幹 +10° の解に `name`/`label` を付けたものが `config/home_pose.yaml` と
  `tests/fixtures/home_pose_forward_lean.yaml`。

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

## テスト

テスト一式はどのHOMEのチェックアウトでも通る（`tests/home_cases.py`）:

- ツール（重心合わせ・yaml 編集）のテストは、チェックアウトの `config/home_pose.yaml` ではなく
  `tests/fixtures/home_pose_centered.yaml`（体幹 0°）/ `home_pose_forward_lean.yaml`（体幹 +10°）を入力にする。
- 1つの fixture HOME の値を固定するテストは `centered_home_only` / `forward_lean_home_only` で
  そのHOMEでだけ実行し、pytest マーカー（`centered_home_pinned` / `forward_lean_home_pinned`）を付ける。
  `tests/test_home_pose_any_trunk.py` の `HomePinnedTestsTest` が、もう一方の fixture のマーカー付きテストを
  その yaml の子プロセスで実行するので、どのチェックアウトでも両方の仕組みを毎回確かめる。
- 残りは HOME から期待値を計算する。
- `MJLAB_MICROBAN_HOME_POSE_YAML=<yaml>` で、そのプロセスの HOME を別の yaml にできる（テスト用）。
