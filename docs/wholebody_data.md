# 全身の遠隔操作の学習データ（`wholebody_make_data.py`）

人の動き（LAFAN1、CMU の BVH）を Microban に写し、学習タスク `Mjlab-Wholebody-Microban` の学習用・評価用の
HDF5 を作る。PICO の公開録画からは評価用のデータだけを作る。すべて CPU で動き、GPU は使わない。

```text
取得（fetch）→ 写し（retarget）→ 腕の反転を切る（finalize）→ 除外の規則と HDF5（build）→ 混ぜる（mix）
```

## 作り直しのコマンド

学習リポジトリのチェックアウトで次の 1 つを実行する。

```bash
uv run --locked python src/mjlab_microban/scripts/wholebody_make_data.py all --workers 12
```

- LAFAN1、CMU、G1 のモデル（PICO の録画を読むため）を、決まった版で `data/wholebody/raw/` に落とす。
  すでにあれば落とさない。LAFAN1 の zip は SHA-256 で、CMU は git のコミットで版を確かめる。
- 写し → 腕の反転の切り → 学習側の除外の規則と HDF5 → `mix` のリンク、の順に進む。
- `data/wholebody/raw/pico/` があれば PICO の評価用（`pico_eval`）も、
  `data/wholebody/raw/amass/smplx_g/` があれば AMASS（`docs/wholebody_amass_smplx.md`）も作る。
- かかる時間（このマシン、12 並列の実測）：取得 3.5 分（CMU 4.2 GB、LAFAN1 0.5 GB）、写し 約 8 分（LAFAN1 と CMU の 2625 本）、
  除外と HDF5 約 2 分。

段ごとにも回せる（`--sets` で対象を選ぶ。既定は lafan1 と cmu、元のファイルがあれば amass と pico_eval）。

```bash
S=src/mjlab_microban/scripts/wholebody_make_data.py
uv run --locked python $S fetch
uv run --locked python $S retarget --sets cmu [--only names.txt]   # names.txt: 写し直すクリップの名前
uv run --locked python $S finalize --sets lafan1 cmu pico_eval
uv run --locked python $S build --sets lafan1 cmu pico_eval
uv run --locked python $S mix --sets lafan1 cmu
```

1 本だけ写して見るとき、写しの品質の数字を並べるとき：

```bash
uv run --locked python $S file data/wholebody/raw/cmu/data/007/07_01.bvh --format bvh_cmu --out-dir /tmp/check
uv run --locked python $S render /tmp/check/07_01.npz        # 07_01.mp4、_sheet.png、_joints.png（CPU で描く）
uv run --locked python $S quality data/wholebody/gmr/cmu data/wholebody/cmu/quality.json
```

## 置き場所

すべて `data/wholebody/` の下で、git の対象外（`.gitignore`）。LAFAN1 は CC BY-NC-ND 4.0 なので、
写したもの（pkl、npz、HDF5）を含めて配らない。

| 場所 | 中身 |
|---|---|
| `raw/lafan1/` | LAFAN1 の BVH 77 本、`lafan1.zip`、`LICENSE_lafan1.txt`、`README_lafan1.md` |
| `raw/cmu/` | cmu-mocap の写し（`data/<被験者>/<被験者>_<番号>.bvh` 2548 本と索引 `cmu-mocap-index-text.txt`） |
| `raw/g1/` | G1 のモデル `g1_mocap_29dof.xml` と `LICENSE`（PICO の録画の順運動学だけに使う。メッシュは読まない） |
| `raw/pico/` | PICO の録画（手で置く。下記） |
| `gmr/<set>/` | 写しの出力（クリップごとの `.pkl`、`.npz`、`_report.json`、`_ik.json`）と `manifest.json`（名前、元のファイル、説明）、`errors.json` |
| `<set>/retarget/` | 腕の反転を切った pkl。切った量は `<set>/finalize.json` |
| `<set>/{train,eval}/motion.h5`、`<set>/report.json` | 学習側の HDF5 と、種類ごとの量・外した割合 |
| `pico_eval/eval/motion.h5` | PICO の評価用（学習用はない） |
| `mix/{train,eval}/<set>.h5` | 学習用のセットの HDF5 へのリンク。学習も評価もこのフォルダを読む |

学習と評価：

```bash
# 学習
--env.commands.motion.motion-file data/wholebody/mix/train
# 評価（学習の種類がすべて評価にあることを求める）
uv run --locked python src/mjlab_microban/scripts/wholebody_benchmark.py --checkpoint <model.pt> \
    --motion-file data/wholebody/mix/eval --train-motion-file data/wholebody/mix/train
# PICO の評価用だけで評価する（評価用の種類だけを、同じ線で判定する）
uv run --locked python src/mjlab_microban/scripts/wholebody_benchmark.py --checkpoint <model.pt> \
    --motion-file data/wholebody/pico_eval/eval --eval-only-set --output-root artifacts/wholebody_pico
```

進み具合の線（胴の進んだ量を参照の進んだ量に射影した比が 0.5 以上）は、種類の名前によらず、参照の胴が片（10 秒）の間に
0.12 m 以上進む片を持つ種類すべてに掛かる。0.12 m は、片の終わりに胴の位置の報酬の std（0.06 m）以内にいれば 0.5 に届き、
立ち止まれば届かない長さ（0.06 / (1 - 0.5)）。PICO の歩き（`picoteleopit`、`picotwist`）や CMU の歩き（`cmuwalk`）にも掛かる。

### 評価の始め方と、暴れと倒れの線

評価（`src/mjlab_microban/wholebody/benchmark.py`）は、片（区間を 10 秒ずつに切ったもの）ごとに、実機と同じく
止まった HOME からロボットを始める。

- **置き方**：関節は HOME で速さ 0、胴は HOME の高さと傾きで、参照の胴の水平の位置に置き、向き（yaw）を参照に合わせる。
  参照は片の始めから進める。
- **最初の 1 秒は判定しない**：HOME から参照の姿勢へ寄る間なので、終わりの判定で失敗にせず、誤差も取らない。
  HOME から始めると、最初の歩で片の 49〜59% が終わりの判定（胴・足・手の高さが参照から 5 cm）に掛かる。
  参照の姿勢が HOME と違うためで、0.6 秒までに 14〜15% に下がり、そのあとは倒れとともにゆっくり増えるだけ（2 秒で 15〜21%）
  （PICO の評価用データと学習の評価用データ、小さな学習と本番の学習のチェックポイントで測った）。
  1 秒以下の片は判定するところがないので外す。判定する時間は、PICO の評価用データで 3541 秒のうち 3030 秒、
  区間の短い学習の評価用データで 8586 秒のうち 5807 秒。
- **前の始め方**（`--start reference`）：参照の最初のフレームに、関節の速さも含めて置き、最初の歩から判定する。
  前の報告と比べるために残している。実機ではこの始め方は起きない。動いている姿勢に置かれたロボットは、
  最初の 1 秒で、PICO の評価用データの片の 15%、学習の評価用データの片の 17% が倒れる
  （HOME から始めると 0.5% と 4%。小さな学習のチェックポイントで測った）。

成功、誤差、進み具合の線は、判定する歩（最初の 1 秒のあと）で取る。これに 2 つの線を足した。

- **暴れの線**（種類ごと、`leg_action_rate`）：脚 12 関節の、1 歩ごとのサーボの目標の変化の二乗和
  （報酬の `action_rate_l2` と同じ形、rad²）の、歩の平均が 3.36 以下。
  - 根拠：実機で暴れないと分かっている PICO の方策
    （`mjlab_microban_teleop_v12` の `2026-10-11_23-19-20_home_481503d292_pico_v13_v13g2`、`model_14999`）が、
    歩き（止まれと 8 方向、腕は HOME、10 秒を 128 回）で 3.355。
  - 測る歩も同じにする：片の始めから 1 秒たったあと、倒れるまで。失敗した片も含める。
- **倒れの線**（PICO の評価用データ全体、`falls_per_minute`）：立っていた 1 分あたりの倒れの回数が 0.425 以下。
  - 倒れの定義：胴の傾き（鉛直から）が 45° を超え、そのとき参照の胴の傾きが 30° 未満（参照が寝ているときは倒れにしない）。
    片の始めから終わりまで見る。終わりの判定のあとも片の終わりまで回し続けて見る。
  - 時間あたりで数える理由：片の長さは 1〜10 秒とばらばら（PICO の評価用データで平均 7.2 秒）で、根拠の押しの判定は 1 回 6 秒。
    片の数で割ると、割合が片の長さで変わる。そこで、倒れの数を、片が倒れるまでに回った時間（倒れなければ片の長さ）の和で割る。
  - 根拠：上の PICO の方策の押しの判定（止まれと 8 方向の歩き、毎秒 0.4 m/s で押す、1 回 6 秒を 144 回）で、
    同じ傾きの決まりで 6 回倒れた（144 回のうち 6 回）。立っていた時間の和は 846.58 秒なので、1 分あたり 0.425 回。
    実機で動かしている方策と同じ頻度までを認める。
  - 線は評価だけのデータ（`--eval-only-set`、操作者の動きである PICO の録画）にだけ掛ける。
    学習の評価用データの倒れの割合は報告するが、線にしない。操作では使わない動き（全力の走り、跳び、倒れ込み）を含み、
    終わりの判定は成功率の線ですでに見ているため。

報告の表（`<model>-kinds.csv`）には、種類ごとに `judged_seconds`（判定した秒数）、`falls_per_minute`（立っていた 1 分あたりの倒れ）、
`leg_action_rate`（暴れの線の数字）を足した。判定（`verdicts.final`）には `kinds_above_action_rate_line`、`falls_per_minute`、
`fall_ok` が入る。

## PICO の録画（手で置く）

PICO の全身推定を G1 に写した公開録画。評価だけに使い、配らない。データの条件は 3 つとも確認できていない。
置き方（`src/mjlab_microban/wholebody/retarget/pico.py`）：

| 場所 | 出どころ | 中身 |
|---|---|---|
| `raw/pico/teleopit/pico_record_shard_000.h5` | Teleopit（https://github.com/BotRunner64/Teleopit、コードは Apache-2.0）の PICO 録画。配布元の場所は未確認（HuggingFace の `12e21/Teleopit-datasets` と推測） | 9 本、31 分。`teleopit_bench_pico_shard_000.h5` は同じ録画の一部なので使わない |
| `raw/pico/twist2/*.pkl` | TWIST2（https://github.com/YanjieZe/TWIST2、コードは MIT）のデータセット（作者の Google Drive）のうち作者の PICO 録画（`0807_yanjie_walk` 50 本、`0811_static` 23 本）。使うときは引用を求めている | GMR の pkl、29.7〜29.9 fps（30 fps に補間する） |
| `raw/pico/any4hdmi/manifest.json`、`*.npz` | EGalahad/sim2real の real_pico の録画。配布元の場所は未確認 | 13 本、30 fps |

根が 1 フレームで 0.15 m を超えて動く所（瞬間移動）で切り、1 秒以上の部分を別のクリップにする（名前に `_p<番号>`）。
基準の立ち姿は、録画の元ごとに静かに立つフレームから作る。

## 写し方（`src/mjlab_microban/wholebody/retarget/`）

- GMR の IK（`gmr/`、Teleopit 経由の写し。GMR は MIT、Teleopit は Apache-2.0。ライセンスは `gmr/LICENSE`、変更点は `gmr/motion_retarget.py` の冒頭）で解く。
- BVH の読み込みは自前（`skeleton.py`）。GMR 同梱の BVH 読み込み（`lafan_vendor`、CC BY-NC-ND 4.0）は入れていない。
- 人の基準の立ち姿（骨格から作る。LAFAN1 は被験者ごとの静かに立つフレームから）を、HOME（`config/home_pose.yaml`）に重ねる。基準の立ち姿を写すと HOME に一致する。
- 胴の傾きは骨盤から首の付け根への向き、向き（yaw）は左右の肩の線から取る。手足は胴の座標でねじれのない回転で回すので、体の向きによらない。
- 床は、学習側の衝突形状の最も低い点を、人の体の浮き（足、または床に着きうる部位）× 縮尺の高さに置く。
- 首は写さない（HMD から別に動かす）。
- 詳しい決め方は `core.py` の冒頭にある。IK のロボットのモデルは `robot.xml` に関節の中心の site を足したもので、実行時に作る（`model.py`）。

確かめ（`tests/test_wholebody_retarget.py`、合成の骨格）：基準の立ち姿が HOME に写る、人の左腕と左足を上げるとロボットの左手と左足首が上がる
（右は動かない）、左右に映した動きが左右に映したロボットになる（2 mm 以内）、動きを回して動かしても関節角が変わらない（1e-6 rad 以内）、
SMPL-X の読み込みが体のモデルの式と一致する。

作り直しの再現性：

- リポジトリの環境（`uv.lock`：mujoco 3.13.0、numpy 2.3.5、scipy 1.18.1、mink 1.3.0）では、同じ元からは同じ写しになる
  （並列の数 1 と 4 で同じ）。
- いまの学習データ（作業場所で作り、小さな学習に使ったもの）は、版の違う環境（mujoco 3.15.0、numpy 2.2.6、scipy 1.15.3）で作った。
  このコマンドで写し直すと、ほとんどのクリップは一致する（CMU は 1e-9° 以内、PICO は 103 本中 101 本が 0.0011° 以内）が、
  IK の解の分かれ目にあるクリップは変わる。LAFAN1 push1_subject2 は最大 0.11°、PICO picotwist_0807_yanjie_walk_037 は最大 2.1°、
  picotwist_0807_yanjie_walk_044_p0 は腕の解が入れ替わり、412 フレーム中約 280 フレームで最大 190° 違う
  （腕の反転のフレームも 168 → 170 になり、`finalize` で切ったあとの区間の長さが変わる）。
- そのため、作業場所で作ったデータは、このコマンドでは完全には再現できない。このコマンドで作り直したデータで学習し直せば、
  そのあとは同じ環境で再現できる。

## 外すもの

1. **腕の反転**（`finalize`）：肩を 180° 回して肘を逆に曲げた、人と違う腕の解。3 フレーム以上続くものを、学習側と同じ前後 0.2 秒と一緒に切る。
   学習側の規則では反転したフレームの一部しか落ちないため。
2. **学習側の除外の規則**（`build`、`src/mjlab_microban/wholebody/data/preprocess.py`）：床へのめり込み、体どうしの入り込み、
   関節の範囲の外、関節の跳び、浮き、と次の支えのない姿勢。当たったフレームの前後 0.2 秒も落とし、22 フレーム未満の残りは捨てる。
3. **支えのない姿勢**（`unsupported`）：胴が立っていて（鉛直から 45° 以内）、重心がほぼ止まっていて（床に沿った速さが 0.1 m/s 未満）、
   ロボットの重心の床への投影が、床から 2.5 cm 以内にある衝突形状の角の凸包から 1 cm を超えて外れた状態が、1 秒を超えて続く区間。
   - 椅子や台に座る（人が座るとロボットは足だけが床に着き、胴が宙に浮く）、床に座る（脚が人のように畳めず、胴が床に届かない）、
     手すりや壁、相手に寄りかかる、などの sim にない支えを前提にした区間。
   - 重心がほぼ止まっている条件：ZMP（支えの中にしかない）は、重心から（重心の高さ / g）× 加速度だけずれる。
     速さが 1 秒の間 0.1 m/s 未満なら、両者の平均の差は (0.18 m / g) × (2 × 0.1 m/s) / 1 s = 3.6 mm 以内なので、
     重心が支えの外に 1 秒とどまる参照は追えない（この高さの倒立振子は 0.13 秒で離れていく）。
     重心が動いている区間（走る、跳ぶ、倒れる、踏み出す）は判定しない。この条件がないと、幅を狭めたときに跳びや走りにも当たる。
   - 幅 1 cm の根拠（今の LAFAN1、CMU、PICO の写しで測った）：

     | 幅 | LAFAN1 の支えのいらない動き（walk、run、sprint、dance、fight、jumps など）で、重心が止まった 1 秒の区間のうち幅を超えて外れる割合 | CMU の座り（cmusit）で印が付く割合 |
     |---|---|---|
     | 0.5 cm | 0.65% | 17.2% |
     | 1 cm | 0.28% | 16.3% |
     | 1.5 cm | 0.03% | 13.0% |
     | 2 cm（前の規則、速さの条件なし） | — | 9.0% |

     1 cm は、写し自体の誤差（支えのいらない動きで重心が足の外に出る量）より大きく、座っている区間のほとんどより小さい。
     2 cm では、たとえば CMU 114_05 の最初の 1.4 秒（重心が足の外に 1.8〜1.9 cm、椅子に座っている）が残っていた。
   - 支えの範囲を床から 2.5 cm 以内の角まで広く取るのは、写しで立ち足が浮くため。LAFAN1 の支えのいらない動きで止まって立つとき、
     高いほうの足が床から 1〜2.5 cm にあるフレームが 7% ある。1.5 cm に詰めると、立っている区間にも印が付く
     （cmudance 0.5 → 4.0%、cmumisc 1.1 → 2.7%）。
   - 種類の名前ではなく姿勢で決めるので、座りでも重心が足の上にある区間（立ち上がる直前など）は sim でもできるので残る。
   - 寝る・這う動き（胴が 45° を超えて倒れている）は判定しない。胴が床の近くにあり、床に降りても姿勢がほとんど変わらないため。
   - 宙にある区間（接地点がない）は、浮きの規則が受け持つ。
   - PICO の膝立ち（`picoany_kneel*`、`picoany_balance`、`picoany_squat_kneel_and_lounge`）が外れるのは、写しで膝が床に届かないため。
     元の録画（G1）では膝が床に着いている（G1 の膝の関節の中心が床から約 4 cm）が、写した姿勢では足が最も低く、
     膝（tibia）が床から 2〜5 cm 浮いて、片足だけで立つ参照になる。床合わせは最も低い点（足）を床に置いていて、そのとおりに動いている。
     膝立ちそのものは sim でもできるが、膝を床に着けるように写す方法は今はない。
   - この規則で新しく外れた量（それまでの規則で残っていた分から。今の写しで測った）：

     | データ | 学習用（分） | 評価用（分） | 主な種類（外れた分） |
     |---|---|---|---|
     | LAFAN1 | 177.2 → 175.6 | 46.63 → 46.56 | obstacles 0.89、ground 0.35、aiming 0.15、fightAndSports 0.12、fallAndGetUp 0.11 |
     | CMU | 418.3 → 414.0 | 99.4 → 98.4 | cmumisc 2.64（動物のまね：panda、bird など）、cmuclimb 1.28（台や椅子に座る）、cmusit 0.89（外れる割合 2.2 → 19.4%）、cmuwalk 0.24、cmuground 0.22 |
     | PICO（評価用） | — | 62.1 → 59.2 | picoany 2.18（膝立ち）、picoteleopit 0.60、picotwist 0.13 |

種類ごとに外れた割合が 30% を超えると、`report.json` と `build` の表に `REVIEW` が付く。学習の前に写しを見直す（`quality` の内訳、`render` の動画）。

## データの出どころと条件

| データ | 出どころ（取得の版） | 条件 |
|---|---|---|
| LAFAN1 | https://github.com/ubisoft/ubisoft-laforge-animation-dataset の `lafan1/lafan1.zip`（git LFS、SHA-256 `ea918082…a977`）。5 人、15 種類、30 fps、4.6 時間 | CC BY-NC-ND 4.0（非商用、改変物の配布不可、表示）。研究目的の学習にだけ使い、写したものは配らない |
| CMU | https://github.com/una-dinosauria/cmu-mocap（コミット `09a07f5`）。CMU Graphics Lab Motion Capture Database（http://mocap.cs.cmu.edu）の BVH 変換版（cgspeed / Bruce Hahne）。120 fps、9.9 時間 | 研究にも商用にも使ってよい。データそのものの再販は不可（変換後も）。発表時は謝辞を推奨（同リポジトリの READMEFIRST.txt） |
| G1 のモデル | GMR（https://github.com/YanjieZe/GMR、コミット `bb1bbe4`）の `assets/unitree_g1/g1_mocap_29dof.xml` | BSD 3-Clause（Unitree Robotics） |
| PICO の録画 | 上の表 | 未確認。評価だけに使い、配らない |

種類の名前：LAFAN1 は元のファイル名のまま（`walk1_subject1` なら walk）。CMU は索引の説明の語で 9 つに分け、
`cmu<種類>_<被験者>_<番号>` にする（`kinds.py`）。PICO は `picoteleopit_`、`picotwist_`、`picoany_` で始まる。
評価の分け：subject5 の録画（LAFAN1 の標準の分け）、subject5 のない種類は各ファイルの最後の 20%。
