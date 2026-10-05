# HOME姿勢の設定

全ポリシー（歩行・起き上がり・PICO v12）が使うHOME姿勢は、このディレクトリの
`home_pose.yaml` だけで決まる。ほかの値はすべて `src/mjlab_microban/robot/home_pose.py`
が `robot.xml` の MuJoCo FK で計算する。

| ファイル | 役割 |
| --- | --- |
| `home_pose.yaml` | 唯一の入力。21関節の角度（度）、体幹ピッチ、名前、ラベル |
| `home_pose_tool.py` | `show`: 派生値の表示。`write-robot`: ロボット側 `config/home_pose.yaml` の生成 |
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

ローダーは、関節名21個がそろっていること、左右対称であること、MJCFの可動範囲内であること、
`trunk_pitch_deg` で両足裏が水平（1e-9 rad 以内）であることを確認し、満たさなければ読み込みを拒否する。
この学習ラインは体幹が垂直なHOME（`trunk_pitch_deg: 0.0`）だけに対応している。前傾HOMEに必要な
座標系の変更は `forward-lean-v2` ブランチにある。

## HOMEを変える手順

1. `home_pose.yaml` を編集する。重心合わせだけなら `uv run python config/balance_home_pose.py` を使う。
2. `uv run python config/home_pose_tool.py show` で、根元高さ・余裕・頭高さ・`tag` を確認する。
3. すべてを最初から学習し直す: 歩行（`Mjlab-Velocity-Microban` 15000回とその続き、プローブで選択）、
   起き上がり5段階、PICO v12（`scripts/train_microban_teleop_v12.sh start --source ... --hand-pose-release`）。
   歩行チェックポイントには `microban_walk_home_pose` が記録され、v12の開始時に現在のHOMEと照合される。
4. `uv run python config/home_pose_tool.py write-robot --microban-repo ../microban` でロボット側の
   `config/home_pose.yaml` を書き出し、3つのポリシーをロボットの `src/agents/` に入れる。
   ロボット側では `tests/test_shared_home.py` の固定値と、ランごとに変わる値
   （`pico_hybrid.py` の歩行ソースSHA、`tools/validate_pico_policy.py` の `walk.onnx` SHA）も更新する。
   PICOのパッケージは、そのロボット側ツリーに対して作る（`--microban-repo`）。
5. 両方のリポジトリでテストを通し、コミットする。

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
