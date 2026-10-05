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

## 学習済みの2つのHOMEと契約文字列

学習済みの成果物があるHOMEは2つあり、どちらも `home_pose.py` の `LEGACY_HOME_OVERRIDES`（joint hash
がキー）で、そのHOMEのブランチが使っていた値と文字列をそのまま再現する。それ以外のHOMEは
`<label>_<joint_hash>` を埋め込んだ文字列になり（`src/mjlab_microban/robot/home_contracts.py` に全部まとめてある）、
古いチェックポイント・ゲート・ONNXは文字列やHOMEスタンプが合わないため拒否される。

| | 中心HOME（`bbef07cab8`） | 前傾HOME（`481503d292`） | その他のHOME |
| --- | --- | --- | --- |
| yaml | `config/home_pose.yaml`（体幹 0°） | `tests/fixtures/home_pose_forward_lean.yaml`（体幹 +10°） | 編集したyaml |
| 再現するブランチ | 学習 `track-centered-home-clip`、ロボット `feature/neck-roll-pitch-camera` | 学習 `forward-lean-centered-home` / `forward-lean-v2`、ロボット `forward-lean-home` | — |
| `tag` | `centered_home` | `forward_lean_home` | `<label>_<hash>` |
| 歩行 ONNX 契約 | `v3_centered_home_servo_range` | `v4_forward_lean_home_servo_range` | `v3_<tag>_servo_range`（体幹 0°）/ `v4_<tag>_servo_range` |
| 起き上がり契約 | `v5`（`v4` スタンプも記録envで確認して受理） | `v6` | `v5_<tag>` / `v6_<tag>` |
| PICO v12 HOME / recipe | `centered_home_hip_plus1p198..._v5`、recipe `..._v11`、pose-release `..._v12` | `forward_lean10_hip_minus14p166561199931_..._v6`、recipe `..._v17`、pose-release `..._v18` | `<tag>_hip_..._v5`/`_v6`、`<tag>_...` |
| パッケージャ | `..._packager_v6_centered_home_servo_range` | `..._packager_v7_forward_lean_home_servo_range` | `v6`/`v7` + `<tag>` |
| 救済段階・upright full-body の revision | 中心ブランチのまま | 前傾ブランチのまま | `<tag>` 入り |
| 固定値 | root z 0.170554885633559、足の横間隔 0.094 m（FK 0.0935） | root z 0.170430569776402、股・足首は前傾ブランチの全桁の値（yamlは12桁の正準値） | FK値 |
| 互換 | 起き上がりの near-HOME リセット既定 (0.2, 0.6)、HOMEスタンプのない歩行チェックポイント、model_7099 からの pose-release 切替 | （なし。pose-release は新しいチェーンで学習） | （なし） |
| pose-release の 10000 境界 / 10100 カナリア | 手先 RMS 0.035 m（中心ブランチのまま。0.040 m 許容プロファイルは存在しない）。パッケージの境界ゲートは渡したときだけ記録 | 手先 RMS 0.040 m 許容（e3271de / ec67f1e）。パッケージは 10000 と 10100 のゲートを resume 系譜（`params/agent.yaml`）でたどって必須（7ceb280） | 前傾HOMEと同じ |
| HOMEスタンプの比較 | 完全一致（記録envの root は atol 1e-12） | 完全一致（同） | 1e-9 の許容（FK の最終桁の揺れ） |

体幹ピッチ 0° のHOMEは中心ラインの仕組み（体幹座標系の目標、元の手先FK箱）、0° 以外は前傾ラインの仕組み
（HOME水平化座標系の目標、水平なヘッドセットのHMD中立、受信箱 ±64 mm に収めた手先目標、F 評価姿勢 (−20, 25, −50)°）
で学習する。どちらも同じコードで、体幹ピッチの値から決まる（次節）。

root z は FK 値を 1e-12 m に丸めて公開する。中心HOME（FK 0.17055488563355944、15桁丸めの境界から
2 ulp）と前傾HOME（`481503d292`）は、公開済みの値（0.170554885633559 / 0.170430569776402）を
`LEGACY_HOME_OVERRIDES` の `root_z_m` に固定し、読み込み時に FK との差が 1e-12 m 以内かを確認する。
MuJoCo の更新などで FK が最終桁で揺れても、HOMEスタンプ・ロボット側yaml・テストの固定値は変わらない。
前傾HOMEの股・足首ピッチも同様に、yaml の12桁の正準値（−14.166561199931 / 4.127976841869、ハッシュはこの値）
から 1.2e-13° 以内であることを確かめた上で、前傾ブランチが公開した全桁の値（−14.166561199931119 /
4.127976841869204）を `HOME.joint_pos_deg` / `joint_pos_rad` として使う（`HOME.input_joint_pos_deg` がyamlの値）。

## 体幹ピッチに依存するもの（すべて yaml の `trunk_pitch_deg` から計算）

体幹ピッチ p は yaml の1つの値で、次がすべてそこから決まる。p = 0 では中心ラインと同じ式・同じ設定になる
（テストで確認、後述）。

| 項目 | p ≠ 0 のとき | p = 0 のとき |
| --- | --- | --- |
| HOME の root 姿勢・重力 | クォータニオン (cos p/2, 0, sin p/2, 0)、重力 (sin p, 0, −cos p) | 単位クォータニオン、(0, 0, −1) |
| 歩行の upright 報酬 | 目標ピッチ p | 0 |
| 歩行・テレオペの速度報酬 | HOME水平化座標系 R_trunk·R_y(−p) で速度を読む（`track_*_home_frame`、`trunk_pitch` パラメータ） | mjlab の項そのもの |
| リセットのヨー | ワールド z 軸まわり（`reset_root_state_uniform_world_yaw`。起き上がりの near-HOME リセットも） | mjlab の `reset_root_state_uniform` |
| 起き上がりの直立報酬 | `upright_standing` の目標重力 (sin p, 0)（`pitch` パラメータ） | (0, 0) |
| 頭高さ・足の横間隔 | FK（前傾 0.2953 / 0.0941 m） | FK（中心 0.2965 / 0.094 m） |
| PICO 足・手目標 | HOME水平化座標系、ラベル `robot_home_levelled_trunk_xyz_forward_left_up`、手先FK v4（受信箱 ±64 mm で棄却サンプリング） | 体幹座標系、`robot_trunk_xyz_forward_left_up`、手先FK v2 |
| HMD 中立 | neck_pitch = −p（水平なヘッドセット） | HOME（default_joint_pos） |
| エクスポータ | 重力 = HOME 重力（歩行・起き上がりのスモーク、PICO の parity 入力） | (0, 0, −1) |
| ロボット | 起き上がりの立ち上がり判定・引き渡し前の静定判定を HOME 重力からの傾きで測る、首の安定化の基準を p だけずらす、シミュレータの IMU 遅延を HOME 姿勢で初期化 | 鉛直基準の元の判定 |

転倒判定（重力 z > −0.5、鉛直から 60°）は物理的な姿勢なので、どの p でも鉛直基準のまま。
記録済みでない p（+10° 以外）では手先目標の箱を 401^3 格子で初回に計算する（数十秒、`~/.cache/mjlab_microban/` に保存）。

ローダーは、関節名21個がそろっていること、左右対称であること（左右ペアの一致・反転に加えて、
首ヨー `head` と `neck_roll` が 0）、MJCFの可動範囲内であること、`trunk_pitch_deg` で両足裏が水平
（1e-9 rad 以内）であること、さらに足裏が床に平らに着くこと（足裏面の角48個すべてが、ワールド高さで
床から 0.5 mm 以内、ロール約 0.7° まで。ロールした足裏は縁だけで立つので拒否する）を確認し、満たさなければ読み込みを拒否する。`-0.0` は `0.0` と同じ値として扱う
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
| 股・足首ロール（左右反転）、股ヨー | 足裏が床に平らに着く組み合わせだけ（ふつうは 足首ロール = −股ロール、股ヨー 0）。股ロールだけを変えると足裏がロールしてローダーが拒否する |
| 首ピッチ | 可 |
| 体幹ピッチ | 可（±90° 未満）。0 は中心ラインの仕組み、それ以外は前傾ラインの仕組みで学習する（上の表）。前傾HOME（+10°）は前傾ブランチを全桁で再現する |
| 肩ピッチ | 0 だけ（PICO contract v12 のHOME revision） |
| 肘・肩ロール | 体幹 0° では、手先の到達範囲が PICO 受信側で検証済みの ±0.064 m の箱に収まる範囲だけ。今の腕HOMEで x がすでに 0.0630 m なので、1つだけ動かす場合の目安は 肘 −22.4°〜−19.1°、肩ロール（左）−2.6°〜44.7°（MJCFの範囲内で）。体幹 ≠ 0° では箱の外の目標を棄却して学習するので、評価姿勢（F/B/f/b）が箱に入る範囲 |
| 首ヨー `head`、`neck_roll` | 0 だけ（左右対称でなくなるため、ローダーが拒否） |

この表の外へ進めるには、そのタスク側の制約（受信箱の検証、v12 contract など）を先に変える必要がある。

## HOMEを変える手順

```text
home_pose.yaml を編集 → balance_home_pose.py（任意）→ home_pose_tool.py show / write-robot → 全ポリシー再学習 → 両リポジトリでコミット
```

手順3〜7は1つのコマンドでまとめて実行できる（中断しても同じコマンドで再開、詳細は
[`docs/home_pose_workflow.md`](../docs/home_pose_workflow.md)）:

```bash
# 学習側: home-config ブランチ（このディレクトリがあるチェックアウト）で実行する
# ロボット側: home-config ブランチ（src/home_pose.py があるもの）の作業ツリー。
#   デプロイ用のチェックアウトをそのまま使わず、専用の worktree を作る:
git -C ../microban fetch origin
git -C ../microban worktree add -b home-<label> ../microban_home-<label> origin/home-config
python3 scripts/retrain_all_for_home.py --robot-repo ../microban_home-<label> --robot-branch home-<label> \
    --training-branch home-<label>
```

`--robot-repo` が `config/home_pose.yaml` を読まないロボットのチェックアウト（`src/home_pose.py` がない、
たとえば `feature/neck-roll-pitch-camera`）なら、パイプラインも `write-robot` も最初に拒否する。

人手の介入なしで最後まで進むように、PICO v12 の 10000 境界は自動で段階的に救済する（9999 ゲート不合格 →
model_9900 の pose-release コーナー救済を mix lf60, lf90, lf72, lf65 の順に → それも全部落ちたらゲート済みの
model_7099 から 7100→10000 を学習し直す → 尽きたら止まる）。15000 境界も同じ（14999 ゲート不合格 → model_14900 の
pose-release 最終シナリオ救済を mix pr_v1〜pr_v6 の順に（pr_v5/pr_v6 は評価器の押しも再生）→ 全部落ちたらゲート済みの model_10099 から 10100→15000 を
学習し直す → 尽きたら止まる）。同じ親からの再学習は学習シードを変える（42, 43, ...）。
ただし前傾HOME（体幹 +10°）では、前傾チェーン自身が 15000 境界で 10 回（シード 42/42/43/44 の再学習4回と
救済6回、pr_v5/pr_v6 を含む）試して全部 14999 ゲートに落ちている（forward-lean-v2 の
`docs/teleop_v12_hand_pose_release_final_rescue.md`）。前傾 yaml でこのパイプラインを回すと、同じ仕組みなので
15000 境界で止まる見込みが高い。通すには 10100→15000 のレシピ自体の変更が要る（前傾ブランチ側でもまだ無い）。学習を回す前の配線確認は、どのHOMEでも
`--dry-run` で数十分（そのHOMEの歩行があれば `--dry-run-walk-init`、編集したばかりのHOMEなら
`--dry-run-plumbing`）。詳細は [`docs/home_pose_workflow.md`](../docs/home_pose_workflow.md) の「ドライラン」。

以下はそのコマンドが行う内容（手で行う場合の手順）。

1. `home_pose.yaml` を編集する（膝など。変えられる値は上の表）。
2. （任意）`uv run python config/balance_home_pose.py` で重心を合わせる。既定は確認だけ（dry run）で、
   `--write` を付けると股・足首ピッチの4つの値だけを書き換える（後述）。学習タスクが受け付けないHOMEは
   書き込まない。
3. `name` と `label` を新しい姿勢に合わせて直し（ツールは変えない。`tag` は `<label>_<hash>` になる）、
   `uv run python config/home_pose_tool.py show` で根元高さ・余裕・頭高さ・`tag`・`training_line` を確認する
   （学習タスクが受け付けないHOME、足裏が水平でないyamlは `error: ...` の1行で終了コード1）。
4. `uv run python config/home_pose_tool.py write-robot --microban-repo ../microban_home-<label>` でロボット側の
   `config/home_pose.yaml` を書き出す（`--check` で最新か確認できる）。学習タスクが受け付けないHOMEは
   書き出せない（契約文字列と手先FKはその学習タスクのコードが作るため。`--force` はない）。
   `config/home_pose.yaml` を読まないロボットのチェックアウトも拒否する。失敗はどれも `error: ...` の1行。
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
- **「水平」**: ツールが合わせるのはピッチ方向。ロールは動かさないので、足裏がロールしていて床に縁でしか
  触れないHOME（股ロールだけを変えた、股ヨーと膝を組み合わせた、など）は「HOME soles are not flat on the floor」
  で拒否する（接地面の定義が床の高さ基準の `scripts/home_pipeline/home_check.py` と一致する）。体幹を傾けると、
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
- **再現できる今日の解**:
  - 体幹 0°: 股 +1.198384259489°、足首 −1.198384259489°、root z 0.170554885633559、余裕 30.83/30.83 mm（現在のyamlがそのまま解）
  - 体幹 +10°: 股 −14.166561199931°、足首 +4.127976841869°（前傾ブランチの全桁の解 −14.166561199931119 /
    +4.127976841869204 を12桁に丸めた値）、root z 0.170430569776402、余裕 30.96/30.96 mm
- 体幹 +10° の解は training line: OK（前傾ラインの仕組みで学習する）。この解に `name`/`label` を付けたものが
  `tests/fixtures/home_pose_forward_lean.yaml`（前傾ブランチを全桁で再現する yaml）。

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

## 直立版と前傾版の切り替え（ブランチ）

コードは1つで、HOME は yaml の値だけで決まる。直立版と前傾版はブランチで持つ:
各ブランチ = 同じコード + そのブランチの `config/home_pose.yaml` + そのHOMEで学習したモデル。

- `home-config`（学習・ロボットとも）: 中心HOME（体幹 0°）。今の成果物（歩行 cont 20000、起き上がり段階5、
  PICO pose-release v12）がそのまま有効。
- 前傾版: `home-config` から作ったブランチで `config/home_pose.yaml` を前傾HOMEにする
  （`tests/fixtures/home_pose_forward_lean.yaml` をコピーするか、`balance_home_pose.py --trunk-pitch-deg 10 --write`
  の後で `name`/`label` を直す）。ロボット側は `write-robot` で生成し、前傾ブランチのモデル
  （walk.onnx b33cd9ea、getup.onnx ce6cdc04）とピン（歩行ソース・プローブ・walk.onnx の SHA-256）を入れる。
  学習側の前傾チェックポイントはそのまま再エクスポートでき（バイト単位で同じ ONNX になる）、PICO はこのコードで
  新しく学習・パッケージする。新しく学習し直すなら上の「HOMEを変える手順」（`retrain_all_for_home.py`）でよい。

切り替えは `git checkout <branch>` だけ（ロボット側も同様）。

## 等価性の確認（テスト）

- `tests/test_home_pose_any_trunk.py`: 中心 yaml で `track-centered-home-clip`（5b5a9d0）、前傾 yaml で
  `forward-lean-v2`（7ceb280）のモジュール定数・HOME 由来の関数値・登録された全タスクの env/play/rl 設定と runner が
  一致すること（参照は `tests/fixtures/home_equivalence/*.json`、`tests/home_equivalence.py` で記録）。
  中心HOMEで許す差は、新しいコマンド設定フィールドの既定値（`trunk_pitch=0.0`、`lf_rb_probability=0.9`）と
  HOMEスタンプを付ける歩行 runner だけ（追跡プロファイル表も中心ブランチと同じ）。前傾HOMEにしかない仕組み
  （0.040 m 許容、必須の境界ゲート）のテストは中心 yaml では skip され、`ForwardLeanOnlyTestsTest` が前傾 yaml で
  実行する。
- `MJLAB_MICROBAN_EXPORT_EQUIVALENCE=1` で、両HOMEの歩行・起き上がりチェックポイントを CPU で再エクスポートし、
  公開済み ONNX（前傾 b33cd9ea / ce6cdc04、中心 c9cdd852 / 80cd7ddb）とバイト単位で一致することを確認する。
- `MJLAB_MICROBAN_HOME_POSE_YAML=<yaml>` で、そのプロセスの HOME を別の yaml にできる（テスト・比較用。
  `retrain_all_for_home.py` はこれが設定されていると拒否する）。

## 中心HOMEで元のブランチより厳しくなった確認（意図的）

中心HOMEの値・文字列・学習設定・ゲートの判定基準は元のブランチと同じだが、次の確認は前傾ブランチの仕組みを
全HOME共通にしたため、中心HOMEでも元のブランチ（学習 5b5a9d0 / ロボット a62a793）より厳しい。どれも
拒否する側への変更で、デプロイ済みの成果物（walk.onnx c9cdd852、getup.onnx 80cd7ddb、pico_teleop.onnx）は通る。

- 学習 `train_microban_teleop_v12.sh start`: 歩行ソースの run の `params/env.yaml`（HOME 関節・root、±π クリップ、
  生の前回行動）と HOME スタンプを `export_walk_onnx.require_current_home_walk_checkpoint` で確認する
  （5b5a9d0 は確認なし、前傾ブランチと同じ）。
- 学習パッケージャ: `--boundary-gate` のチェックポイントは最終チェックポイントの resume 系譜上でなければ拒否
  （兄弟 run は不可）。ドライランの証跡（`dry_run*` キー、`DRYRUN_*` の強制パスのプローブ受領書）を含むものは
  `dry_run=True`（`scripts/home_pipeline/dry_run_tools.py package`）以外では拒否する。
- ロボット `walk.py`: `home_pose` スタンプのない walk.onnx を拒否（a62a793 は受理）。`getup.py`: スタンプの
  root 位置を z だけでなく x, y も比較。`pico_hybrid.py`: `v12_deployment_packager_revision` の一致を要求
  （a62a793 は未確認。デプロイ済みと前回のパッケージはどちらも持つ）、学習HOMEマーカーは 1e-9 許容ではなく
  JSON の完全一致。`dry_run_not_deployable` を持つパッケージ（ドライランの PICO）は
  `MICROBAN_ALLOW_DRYRUN_POLICY=1`（ドライラン自身の検証・テストだけが設定する）がなければ拒否。
