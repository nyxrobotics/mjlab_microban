# AMASS と SMPL-X を学習データに足す手順

2026-10-12 時点の状態です。**まだデータを取っていません。** SMPL-X を読むコードは写しに入れてあり
（`src/mjlab_microban/wholebody/retarget/smplx.py`、形式 `smplx`）、合成のモデルと動きでだけ確かめました
（`tests/test_wholebody_retarget.py` の `test_amass_smplx_motion_matches_the_body_model_and_retargets`）。
- 関節の位置は、SMPL-X の式と 1e-9 m 以内で一致します。
- 今の写し（`core.retarget_frames`）を最後まで通り、30 fps の pkl が出ます。
- 基準の立ち姿を写すと HOME（ロボットの初期姿勢。`config/home_pose.yaml`）に一致します（関節の差 0°、根の差 0 mm）。

本物のファイルで試していない所は「未確認」と書いています。データ全体の作り方は `docs/wholebody_data.md` にあります。

## 1. 何のために使うか

- **量と種類を増やすため**：今の学習データは LAFAN1（77 本、4.6 時間）と CMU（2548 本、9.9 時間）です。
  - AMASS は 15 を超える光学式モーションキャプチャのデータを、同じ体のモデルにそろえたものです。サイトの記載では 40 時間以上、300 人以上、11,000 以上の動きです。
  - 踊り（DanceDB）、ハンドボール（BMLhandball）、日常の動き（BMLmovi、KIT）、動きのつなぎ（Transitions）、極端な姿勢のヨガ（MOYO）などが加わります。
- **体型から基準の姿勢を作れるため**：AMASS の各ファイルには、その人の体型（SMPL-X の betas）が入っています。
  - SMPL-X の rest（全関節 0）は、脚と背骨がまっすぐの立ち姿です。
  - そのため、LAFAN1 のように「被験者ごとに静かに立つフレームを選ぶ」作業は要りません。
- **CMU の確かめに使えるため**：AMASS には CMU も入っています（同じ収録を、マーカーから合わせ直したもの）。
  - 同じクリップを今の BVH の経路と比べられます。
  - フレームの速さ（`mocap_frame_rate`）がファイルごとに入っているので、「CMU の BVH は全部 120 fps として扱った」点も確かめられます。

### 利用の条件

AMASS も SMPL-X も、使う人がそれぞれ登録して同意します。以下は 2026-10-12 にサイトで読んだ内容です。

| | AMASS（https://amass.is.tue.mpg.de/license.html） | SMPL-X のモデル（https://smpl-x.is.tue.mpg.de/modellicense.html） |
|---|---|---|
| 使ってよいこと | 非商用の科学研究だけ（"for the sole purpose of performing non-commercial scientific research"）。単一の利用者への許諾（"single-user"） | 非商用の研究・教育・芸術 |
| 配ること | 不可。保管用に 1 部の写しを持つことだけ認められる | 不可。保管用に 1 部の写しを持つことだけ認められる |
| 学習 | 商用のための学習は不可（"to train methods/algorithms/neural networks/etc. for commercial use"） | 商用・ポルノ・軍事・監視・中傷に使う学習は不可 |
| 発表するとき | AMASS の論文（Mahmood ら、ICCV 2019）を引用する。各元データの引用は https://download.is.tue.mpg.de/amass/amass.bib | SMPL-X の論文（Pavlakos ら、CVPR 2019）を引用する |

このプロジェクトでの扱い：
- **研究目的だけに使います。** AMASS を混ぜて学習した方策は、商用（製品として出す実機など）には使いません。LAFAN1（CC BY-NC-ND 4.0）と同じ扱いです。
- **配りません。** 元のファイルも、写したもの（pkl、npz、HDF5）も配りません。`data/wholebody/` は git の対象外です（`.gitignore`）。
- 学習した方策の重みを外へ出すときは、その前に条件を読み直します。非商用の研究なら許されると読めますが、これは推測です。
- 許諾は単一の利用者向けです。共有のマシンに置いて、登録していない人が使うと条件に触れるおそれがあります（推測）。使う人がそれぞれ登録してください。

## 2. 登録とダウンロード

**どちらのサイトも、ダウンロードのページはログインしないと見えません。** そのため、ファイルの一覧と名前はこの手順書を書いた時点では見ておらず、未確認です。下に書いたものは、公開されているページ、GMR の README、smplx パッケージの README、検索で見た説明から組み立てています。

### SMPL-X の体のモデル

1. https://smpl-x.is.tue.mpg.de/ で Sign up し、メールで確かめてからログインします。
2. Model License に同意します（Body License もあり、ページでは 2 つに分かれています）。
3. Download（`download.php`）から **SMPL-X v1.1 の NPZ+PKL**（Python 用の束）を落とします（未確認）。
   - 使うのは `SMPLX_MALE.npz`、`SMPLX_FEMALE.npz`、`SMPLX_NEUTRAL.npz` です。
   - **体型の成分が 16 以上あるモデルが要ります**（AMASS の betas は 16 個）。
     - サイトのお知らせ（2020-08-20）に、体型と表情の全成分を含む版が出たとあります。v1.1 がそれにあたり、体型は 300 成分です（smplx パッケージの `SHAPE_SPACE_DIM = 300` からの推測）。
     - 体型が 10 成分の古い版は使えません。
   - 要らないもの：FBX、Blender・Unity・Unreal の追加機能、UV の地図。
   - ほかに、表情を FLAME 2020 にした版（お知らせ 2021-01-11）や、頭を固定した版など、別の版もあるようです（中身は未確認）。AMASS の SMPL-X N がどのモデルで合わせたかは確かめていません。この点が、下で SMPL-X G を選ぶ理由です。

### AMASS

1. https://amass.is.tue.mpg.de/ で register し、メールで確かめてからログインします。
2. License に同意します。
3. Downloads で **SMPL-X G**（男女別のモデルで合わせたもの）を、元データごとの束（`<元データ>.tar.bz2`）で落とします。
   - サイトのお知らせ（2022-02-14）によると、SMPL-X の中立（N）と男女別（G）の形式は、元のマーカーから MoSh++ を回し直して作ったものです。
   - **G を選ぶ理由（推測）**：男女のモデルは v1.1 の 1 つずつだけと見ています。一方、N をどの中立のモデルで合わせたかは確かめられていません。合わせたときと違うモデルで読むと、関節の位置（体の比率）がずれます。
     - 読み込みはファイルの `gender` でモデルを選ぶので、N を使う場合も直す所はありません。
   - **SMPL+H G は落としません。** 形式が違い（`poses` だけで、`pose_body` などがない）、この経路では読めません。GMR の README も「SMPL+H は落とさない」と書いています。
   - 束の名前：SMPL+H 版では `ACCAD`、`BMLhandball`、`BMLmovi`、`BMLrub`、`CMU`、`DFaust`、`DanceDB`、`EKUT`、`EyesJapanDataset`、`GRAB`、`HDM05`、`HUMAN4D`、`HumanEva`、`KIT`、`MoSh`、`PosePrior`、`SFU`、`SOMA`、`SSM`、`TCDHands`、`TotalCapture`、`Transitions` などです（facebookresearch/humenv の README の一覧）。SMPL-X G にどれがあるかは未確認です。
   - **CMU の束は、比べるためにだけ落とします。** 学習には入りません（作り方の道具が元データ `CMU` を除きます。今の CMU の BVH と同じ収録のため）。
4. 引用のための `amass.bib`（https://download.is.tue.mpg.de/amass/amass.bib）も落とします。

**ファイルの中身**（検索で見た説明による。未確認）：
- 1 つの動きが `<元データ>/<被験者>/<名前>_stageii.npz` です。
- 中のキーは `gender`、`surface_model_type`、`mocap_frame_rate`、`trans`、`poses`、`betas`、`root_orient`、`pose_body`、`pose_hand`、`pose_jaw`、`pose_eye` などです。
- このうち `gender`、`betas`（16）、`root_orient`（T×3）、`pose_body`（T×63 = 21 関節）、`trans`（T×3）、`mocap_frame_rate` だけを使います。
- 世界は Z 上向き、長さは m と見ています。GMR が回さずにロボット（Z 上向き）へ写していることからの推測です。
- フレームの速さは元データで違います。下の issue の例だけでも 60、120、150 fps があります。
  - nghorbani/amass の issue #49 では、SMPL+H 版と SMPL-X 版で速さが違うものが報告されています。例：TotalCapture は 60 と 120、TCD_handMocap は 150 と 120。どちらが正しいかは決まっていません。

## 3. 置き場所

作り方の道具（`src/mjlab_microban/scripts/wholebody_make_data.py`）は、次の場所を探します。

```
data/wholebody/raw/body_models/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz    SMPL-X v1.1 の npz
data/wholebody/raw/amass/smplx_g/<元データ>/<被験者>/<名前>_stageii.npz  AMASS の SMPL-X G を展開したもの
data/wholebody/raw/amass/amass.bib                                       引用
```

展開の例：

```bash
mkdir -p data/wholebody/raw/amass/smplx_g data/wholebody/raw/body_models/smplx
for f in ~/Downloads/amass_smplx_g/*.tar.bz2; do tar -xjf "$f" -C data/wholebody/raw/amass/smplx_g; done
```

取ったら `docs/wholebody_data.md` の「データの出どころと条件」の表に、取った日、取った人、束の名前と次の条件を書き足します。
- AMASS：AMASS License（非商用の科学研究だけ、単一の利用者。配布・改変物の提供は不可。商用のための学習は不可。発表時は AMASS の論文と各元データを引用：`data/wholebody/raw/amass/amass.bib`）。写したもの（pkl、npz、HDF5）は配らない。
- SMPL-X：SMPL-X Model License（非商用の研究・教育・芸術だけ。配布は不可。商用・軍事・監視などに使う学習は不可。発表時は Pavlakos ら CVPR 2019 を引用）。

出力の場所（ほかのデータと同じ）：
- `data/wholebody/gmr/amass/`：写した結果（`<名前>.pkl`、`.npz`、`_report.json`、`_ik.json`）と `manifest.json`
- `data/wholebody/amass/retarget/`：腕の反転を切った pkl
- `data/wholebody/amass/{train,eval}/motion.h5`、`report.json`：学習と評価のデータ
- `data/wholebody/mix/{train,eval}/amass.h5`：上の HDF5 へのリンク

## 4. 写し方

### BVH の経路と同じ流れ

```
入口（BVH: skeleton.read_bvh、AMASS: smplx.load_amass）→ standing_reference（基準の立ち姿）
    → motion_frames（30 fps、顔を +x、床の傾きを直し、床を z=0）→ core.retarget_frames（GMR の IK で Microban に写す）
    → data/wholebody/gmr/<データ>/*.pkl → finalize（腕の反転を切る）→ build（学習側の除外の規則、HDF5）
```

**方針**：AMASS の 1 ファイルを、`skeleton.Bvh`（BVH を読んだときと同じ形）に直します（`smplx.load_amass`、numpy と scipy だけ）。
そうすると、`standing_reference`、`motion_frames`、`retarget_frames`、`render` をそのまま使えます。
形式 `smplx` のときは `core.load_motion` がこの入口を呼びます。

1. `gender` に合うモデル（`SMPLX_<GENDER>.npz`）を開き、体型だけを入れた rest の関節の位置を出します：`J = J_regressor @ (v_template + shapedirs[:, :, :16] @ betas)`
   - SMPL-X では、関節の位置は rest の J と各関節の回転だけで決まります。姿勢による形の補正は表面の頂点にだけ効きます。そのため、体のモデルを使うのは 1 ファイルに 1 回で済みます。
   - 使う関節は、体の 22 個（0〜21：pelvis、腰・膝・足首・足指の付け根、spine1〜3、neck、head、鎖骨、肩、肘、手首）です。指・顎・目は使いません。
   - npz の中のキーの名前と、`J_regressor` が疎行列で入っているかは未確認です（疎なら密にする処理を入れてあります）。
2. ちょうど 30 fps に補間します（回転は球面線形補間、根の位置は線形）。
   - 学習側の `load_retarget_pkl` は、整数の fps でないと止まります。
   - BVH の間引き（`round(元の fps / 30)` おき）では、100 fps の元が 33.3 fps になります。
3. 骨の長さを `J[i] - J[親]`、回転を各関節の局所の回転（オイラー角 ZYX）、根の位置を `J[0] + trans` にして、Bvh を作ります。
   - 世界は Y 上向きに直してから渡します。`motion_frames` が Z 上向きに戻します。
4. `skeleton.FORMATS["smplx"]` が、人の骨の名前と役割の対応です。
   - 骨盤＝`pelvis`、上体の根＝`spine1`、首の付け根＝`neck`、足指＝`left_foot` などです。
   - **胴の傾きは、BVH と同じく骨盤から首の付け根への向きで取ります。** SMPL-X には首の付け根の関節（`neck`）があるので、G1 のときのように左右の肩の中点で代える必要はありません。

**GMR の SMPL-X の読み込み（`general_motion_retargeting/utils/smpl.py` の `load_smplx_file` と `get_smplx_data_offline_fast`）を使わない理由**：
- torch と smplx パッケージが要ります。
- 同じファイルが GMR の `lafan_vendor`（CC BY-NC-ND 4.0）を読み込みます。写しでは、これを使わないことにしています。
- 30 fps への合わせで、fps が整数になりません（`aligned_fps = 長さ / 元の長さ × 元の fps`）。
- `smplx.create` に `num_betas` を渡していません。smplx の既定は 10 なので、16 個の betas でそのまま動くかは未確認です。

なお、関節の位置の出し方は GMR と同じです。GMR も SMPL-X のモデルの関節を使い、回転を親から順に掛けています。

### 基準の姿勢を betas から作る

- そのファイルの betas と性別のモデルで、rest（全関節 0）の関節の位置を出します。SMPL-X の rest は、腕を横に伸ばし、脚と背骨をまっすぐにした立ち姿です。
- `standing_reference` が、その姿勢から次のように基準の立ち姿を作ります。
  - 背骨（骨盤→首の付け根）をまっすぐ上に、脚（股→足首）と腕（肩→手首）をまっすぐ下に向けます。
  - 顔を +x に向け、最も低い関節を z=0 にします。
- これを HOME に重ねて、IK の向きのずれと根の縮尺（脚の長さの比）を決めます。CMU の BVH の rest と同じやり方です。
- AMASS の betas は被験者ごとなので、体の比率の差はファイルごとに入ります。
- 確かめ方：各ファイルの `<名前>_report.json` で、`ref_to_home_max_joint_err_deg` と `ref_to_home_root_err_mm` が 0 になることを見ます。

LAFAN1 との違い：
- LAFAN1 は、被験者の自然な立ち姿（膝が 10〜27° 曲がる）を HOME に重ねています。
- SMPL-X は rest から作るので、膝はまっすぐの立ち姿を HOME に重ねます（CMU と同じ）。
- 立ち姿の膝の曲がりを人ごとに入れる必要があるかは、写した結果を見て決めます。

### BVH の経路との違い

| | BVH（LAFAN1・CMU） | SMPL-X（AMASS） |
|---|---|---|
| 入力 | BVH（Y 上向き、BVH 独自の長さ、オイラー角） | `*_stageii.npz`（Z 上向き、m、回転ベクトル）と SMPL-X のモデル |
| 骨の長さ | BVH の OFFSET | betas と性別のモデルから出す |
| 基準の立ち姿 | CMU は rest、LAFAN1 は被験者ごとに選んだ立つフレーム | rest（betas だけ）。フレームを選ばない |
| 速さ | CMU 120 fps → 4 つおき、LAFAN1 30 fps | 元データで違う（60〜150 fps など）→ ちょうど 30 fps に補間 |
| 最初のフレーム | CMU は 1 フレーム目を捨てる | 捨てない（捨てる必要があるかは未確認） |
| 床に着く部位の高さ（`core.CONTACT_RADIUS`） | LAFAN1 の寝る・倒れる動きで測った値 | **測り直しが要るかもしれない**（下記） |
| 床・向き・傾き | `motion_frames`（3° 以下だけ傾きを直す） | 同じ |
| 名前 | `cmu<種類>_<被験者>_<番号>` など | `amass<種類>_<元データ>_<通し番号>` |

**床に着く部位の高さ**について：
- `core.human_clearance` は、骨盤・背骨・首・肩・肘・手首・膝が床に着いたときの関節の高さを、脚の長さの比で持っています。この値は LAFAN1 の骨格で測ったものです（寝る・倒れる 13 本で、各部位の高さの 0.5% 点の最小）。
- SMPL-X の関節は体の中の位置が違う（例：骨盤や首の付け根の関節の高さ）ので、値が合わないおそれがあります。
- まず同じ測り方を、AMASS の床の動き（名前の種類が ground のもの）で行います。
- 値が LAFAN1 と目立って違うときだけ、`CONTACT_RADIUS` を形式ごとの値（`FORMATS` の中）に移します。
- SMPL-X では体の表面（メッシュ）の最も低い点も出せます。ただし、関節の規則で足りないと測って分かるまでは使いません。

### コマンド

すべて CPU で回します。動いている学習を止めないよう、並列は 12 までにします（このマシンは 20 コア）。

```bash
S=src/mjlab_microban/scripts/wholebody_make_data.py
```

**1. 1 本で試す**
- CMU の 07_01（歩き）は、AMASS と今の BVH の両方にあります。
- AMASS 側のパスは未確認です（`CMU/07/07_01_stageii.npz` のような形と推測）。

```bash
uv run --locked python $S file data/wholebody/raw/amass/smplx_g/CMU/07/07_01_stageii.npz --format smplx --out-dir /tmp/amass_check
uv run --locked python $S file data/wholebody/raw/cmu/data/007/07_01.bvh --format bvh_cmu --out-dir /tmp/amass_check
uv run --locked python $S render /tmp/amass_check/07_01_stageii.npz /tmp/amass_check/07_01.npz
```

見ること：
- `_report.json` で、基準から HOME への差が 0 であること。
- 根の高さ（`root_z_m_min_max`）と、床に最も近い点（`lowest_point_mm_min_p50_p99_max`）がおかしくないこと（体が横倒しなら世界の向きの読み違い）。
- BVH の経路の `07_01` と並べて、関節の角度の図と動画が同じ動きに見えること。

合わせ方（MoSh++ と BVH 変換）が違うので、差は 0 になりません。どこまでの差を許すかの線は、まだ決めていません。

**2. 全部を写す**
- `data/wholebody/raw/amass/smplx_g/**/*_stageii.npz` を集めます。元データが `CMU` のものは除きます。
- 説明は元データからの相対パス（`_stageii` を除く）にします。種類は、その説明の `_`、`-`、`/` を空白にしてから `kinds.kind` で決めます。
- 学習側は、名前の先頭の英字（最初の数字や `_` の前まで）を種類として使います。そのため、種類の部分は英字だけにしています。
- `manifest.json` に、名前・元のファイル・説明が残ります。

```bash
uv run --locked python $S retarget --sets amass --workers 12
```

かかる時間の見込み：CMU（107 万フレーム）は 12 並列で約 400 秒でした（約 2,700 フレーム/秒）。
- 動き 10 時間あたり、CPU で約 7 分です。
- AMASS の SMPL-X G の量は未確認です。40 時間なら約 27 分です（推測）。

**3. 品質の数字、腕の反転の切り、学習側の HDF5**

```bash
uv run --locked python $S quality data/wholebody/gmr/amass data/wholebody/amass/quality.json --workers 8
uv run --locked python $S finalize --sets amass
uv run --locked python $S build --sets amass
```

**4. 混ぜる**（6 節）

```bash
uv run --locked python $S mix --sets lafan1 cmu amass
```

`all` と `mix` は、`data/wholebody/raw/amass/smplx_g/` があると AMASS も混ぜます。確かめる前に混ぜないときは `mix --sets lafan1 cmu` にします。

## 5. 品質での選び方

外すのは、写した結果がおかしい区間と、sim でできない区間だけです。規則は LAFAN1・CMU と同じにし、AMASS 用の規則は先に足しません。

- **学習側の規則**（`src/mjlab_microban/wholebody/data/preprocess.py` の `ExclusionSpec`）。当たったフレームの前後 0.2 秒も落とし、22 フレーム未満の残りは捨てます。
  - 床へのめり込み 10 mm 超
  - 体どうしの入り込み 5 mm 超
  - 関節の範囲を 0.1 rad 超えて出る
  - 関節の跳び（15 rad/s を超え、かつ前後の 2 倍を超える 1 フレームの動き）
  - 浮き 0.05 m 超、または 0.025 m 超が 0.8 秒以上続く
  - 支えのない姿勢（胴が立ち、重心が接地点の外に 2 cm を超えて出る状態が 1 秒を超えて続く。椅子に座る、寄りかかるなど）
- **腕の反転**：肩を 180° 回して肘を逆に曲げた、人と違う腕の解に入ることです。3 フレーム以上続くものを `finalize` が切ります。学習側の規則では、反転したフレームの一部しか落ちないためです。
- **見直しの目安**：学習側の `report.json` で、種類ごとに外れた割合が 30% を超えると `REVIEW` が付きます。これが付いた種類は、学習の前に写しを見直します（`quality.json` の理由の内訳と、`render` の動画）。

AMASS で気をつける所（どれも未確認。測ってから決める）：
- **CMU の重なり**：AMASS の CMU は今の CMU の BVH と同じ収録です。学習には入れません（道具が除きます）。
  - AMASS 版に置き換えるかは、両方で写した CMU の品質の数字（外れる割合、手の誤差、反転）を比べて決めます。
  - 置き換えるなら、`mix` から `cmu` を外し、AMASS の CMU を `amass` に入れます（道具の除外を外す）。両方は入れません。
- **フレームの速さ**：元データごとの `mocap_frame_rate` の分布を見ます。CMU は、同じクリップの長さ（秒）を BVH（120 fps として）と比べます。
  - 長さが合わなければ、どちらかの速さが違います。今の「CMU を全部 120 fps とした」点の確かめにもなります。
- **トレッドミルの歩き**：BMLrub などに、名前に treadmill を含むものがあるはずです（推測）。
  - 接地した足が床ごと後ろへ動くので、写すと止まっている足が滑ります。
  - 今のどの規則にも掛からないと見ています。接地中の人の足の速さを測って、滑るものだけ外すかを決めます。
- **支えのある動き**：椅子に座る、台に乗るなどです。胴が立って重心が接地点の外にある区間は、支えのない姿勢の規則で外れます。重心が足の上にある区間は残ります。
- **手だけ・物を持つ動き**（TCDHands、GRAB など）：体はほぼ立っているだけです。写しがおかしくなければ外しません。
- **種類の分け**：パスの語で分けると、語のないもの（例：BMLmovi、HDM05 の番号だけの名前）は `amassmisc` に入ります。
  - misc が多すぎて評価の内訳が読めないときにだけ、BABEL（AMASS の動きにフレームごとの動作の名前を付けたデータ。別に登録が要る）の名前を使うかを考えます。

## 6. 学習への混ぜ方

- **一覧（manifest）**：`data/wholebody/gmr/amass/manifest.json` に、名前・元のファイル・説明が並びます。
  - 学習側は、`data/wholebody/amass/report.json` に種類ごとのファイル（学習と評価）と、外した割合を書きます。
- **置き場所**：学習は `data/wholebody/mix/train`、評価は `data/wholebody/mix/eval` を読みます。
  - 学習側の `find_motion_shards` は、そのフォルダの下の `*.h5` を全部読みます。そのため、`amass.h5` のリンクを足すだけで混ざります。
  - 学習：`--env.commands.motion.motion-file data/wholebody/mix/train`
  - 評価：`--motion-file data/wholebody/mix/eval --train-motion-file data/wholebody/mix/train`
- **評価の分け**：AMASS には `subject5` がないので、各ファイルの最後の 20% が評価になります（CMU と同じ）。
  - 短いファイルでは評価の分が 22 フレーム未満になり、残りません。
  - 評価は、学習した種類が評価データにないと不合格にします。そのため、`report.json` で全部の種類に評価のファイルがあることを確かめてから学習します。
- **混ざる割合**：学習で動きを選ぶ重みは、長さに比例します（`motion.py` の `sample_weights = 長さ / 合計`）。
  - AMASS が 40 時間なら、今の LAFAN1 と CMU（学習用 約 590 分）の約 4 倍になり、学習のほとんどが AMASS になります。
  - 重みを変えるかは、混ぜて学習した評価の結果を種類ごとに見てから決めます。先に重みの仕組みは足しません。
- 量の目安：CMU の学習用の HDF5 は 418 分で 0.99 GB でした。40 時間なら約 5.7 GB です（推測）。

## 7. 未確認の一覧

- SMPL-X と AMASS のダウンロードのページの中身（ファイルの名前、束の一覧、大きさ）。ログインが要り、見ていません。
- SMPL-X v1.1 の npz のキー（`v_template`、`shapedirs`、`J_regressor`、`kintree_table`）と形。体型の成分が 16 以上あること。
- AMASS の `*_stageii.npz` のキーと形、`gender` の型（bytes か文字列か）、世界が Z 上向きであること。
- AMASS の SMPL-X N がどの中立のモデルで合わせたか（G を選んだ理由）。
- SMPL-X G に入っている元データの一覧と、合計の時間。
- 本物のデータでの写しの結果。
  - 07_01 で BVH の経路と比べること。
  - 許す差の線（決めていない）。
  - 床に着く部位の高さを測り直す必要があるか。
  - 最初のフレームを捨てる必要があるか。
- 元データごとのフレームの速さの正しさ（issue #49 の食い違い）。
- トレッドミルの歩きが含まれるか、それが規則に掛からないか。
- GMR の `load_smplx_file` が 16 個の betas でそのまま動くか（使わないので、確かめる必要は低い）。
- 共有のマシンでの単一利用者の許諾の扱い。
