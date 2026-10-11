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

確かめ（`tests/test_wholebody_retarget.py`、合成の骨格）：基準の立ち姿が HOME に写る、左右に映した動きが左右に映したロボットになる（2 mm 以内）、
動きを回して動かしても関節角が変わらない（1e-6 rad 以内）、SMPL-X の読み込みが体のモデルの式と一致する。
この手順の写しは、前に作業場所で作ったものと、見本の 36 本（CMU 3、LAFAN1 4、PICO 29）で関節角の差 0.03° 以内で一致した。

## 外すもの

1. **腕の反転**（`finalize`）：肩を 180° 回して肘を逆に曲げた、人と違う腕の解。3 フレーム以上続くものを、学習側と同じ前後 0.2 秒と一緒に切る。
   学習側の規則では反転したフレームの一部しか落ちないため。
2. **学習側の除外の規則**（`build`、`src/mjlab_microban/wholebody/data/preprocess.py`）：床へのめり込み、体どうしの入り込み、
   関節の範囲の外、関節の跳び、浮き、と次の支えのない姿勢。当たったフレームの前後 0.2 秒も落とし、22 フレーム未満の残りは捨てる。
3. **支えのない姿勢**（`unsupported`）：胴が立っていて（鉛直から 45° 以内）、ロボットの重心の床への投影が、
   床から 2.5 cm 以内にある衝突形状の角の凸包から 2 cm（足の長さの 1/3）を超えて外れた状態が、1 秒を超えて続く区間。
   - 椅子や台に座る（人が座るとロボットは足だけが床に着き、胴が宙に浮く）、床に座る（脚が人のように畳めず、胴が床に届かない）、
     手すりや壁、相手に寄りかかる、などの sim にない支えを前提にした区間。立ったロボットがここまで外れると 1 秒もたずに倒れる。
   - 種類の名前ではなく姿勢で決めるので、座りでも重心が足の上にある区間（立ち上がる直前など）は sim でもできるので残る。
   - 寝る・這う動き（胴が 45° を超えて倒れている）は判定しない。胴が床の近くにあり、床に降りても姿勢がほとんど変わらないため。
   - 宙にある区間（接地点がない）は、浮きの規則が受け持つ。
   - この規則で新しく外れた量（それまでの規則で残っていた分から）：

     | データ | 学習用（分） | 評価用（分） | 主な種類（外れた分） |
     |---|---|---|---|
     | LAFAN1 | 177.2 → 176.1 | 46.6 → 46.4 | obstacles 0.7、jumps 0.2、aiming 0.2、ground 0.2 |
     | CMU | 418.3 → 415.9 | 99.4 → 98.8 | cmumisc 1.6、cmuclimb 0.7、cmusit 0.5（外れる割合 2.2 → 11.7%）、cmuground 0.2 |
     | PICO（評価用） | — | 62.1 → 60.3 | picoany 1.3（膝立ち、片足立ち）、picoteleopit 0.3、picotwist 0.2 |

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
