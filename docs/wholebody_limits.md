# 全身の写しの関節の制限

値と出典は `config/retarget_limits.yaml`、コードは `src/mjlab_microban/wholebody/retarget/core.py`
（`ServoLimit`、`HIP_YAW_OUT_MAX_DEG`）にある。値はどれも、設定ファイルの `src` に出典の鍵を持つ（下の「出典」）。

## どの段で、何のために当てるか

| 段 | 対象 | 制限 | 目的 |
|---|---|---|---|
| 写す前（`human`） | PICO の元データ（G1）の腰のヨー | 人の体幹の回旋の範囲（±64.6°） | 超えたフレームを、根の瞬間移動と同じく切る（TWIST2 の G1 の腰が 90° の端に張りついた誤り） |
| 写す前（`human`） | 人の骨格の関節角（解剖学の軸ごと、ほかの軸） | 可動範囲、最大角速度、最大角加速度 | 値は設定ファイルにあるが、写しには入れていない（下の「人の側の制限」） |
| IK（`robot`） | 股のヨー | 脚を外向きに 105° まで | 横から後ろへ振った腿を、ヨーを 240° の端まで回して追わない |
| IK（`robot`） | Microban の全関節 | サーボの速度と加速度、範囲の端で止まれる速さ | サーボが出せない動きを写さない（速すぎる動きは遅れて追う） |
| IK の後 | Microban の全関節 | MJCF の範囲に丸める | IK の誤差の分（最大 5e-6 rad）だけ範囲の外に出た値を戻す |

IK の中の制限は、各フレームを前のフレームの解から解くときの、関節ごとの箱（下限と上限）として掛ける。腕の解の枝
（肩の前後を 180° 回し、肘を逆に曲げた解）を選び直すのは、リセット直後の最初のフレームだけにする。あとのフレームで乗り換えると、
サーボが出せない跳び（肩の前後で ±180° の境をまたぐものを含む）になるため。

## サーボの速度と加速度

フレームの間隔を h（30 fps で 1/30 s）として、関節は前のフレームから「速さ × h」まで動き、前のフレームの動きから
「加速度 × h²」までしか動きを変えない。範囲の端へは、その加速度で端までに止まれる速さまでしか近づかない
（1 フレームに s 動くなら、s + s²/(2a) が端までの距離以内）。

- 速さ：XC330-T288-T の無負荷の速さ。3S の電池の公称 11.1 V を kt（1.0425 Nm/A）で割って 10.65 rad/s（30 fps で 1 フレームに 20.3°）。
  評価（`benchmark.servo_no_load_speed_rad_s`）と同じ。
- 加速度：関節ごとに、学習のサーボの電流の上限でのトルク（0.91 A × kt = 0.949 Nm）を、HOME での関節まわりの慣性で割った値。
  慣性は、サーボの減速機の慣性（BAM の armature 0.00828 kg·m²）と、その関節が動かすリンクの慣性の和で、MuJoCo の質量行列の対角
  （ほかの関節は止める）。95〜114 rad/s²（股のロールとピッチ 95〜96、ほかは 109〜114）。armature が慣性の 83〜99% を占める。
  測り方は `retarget/model.py` の `servo_accelerations`。テストが設定ファイルの値と照らし合わせる。

## 股のヨーの 105°

ロボットの股のヨーは内向き 60° から外向き 240° まで回る。股のロールは 25° までなので、上げた腿を横へ向けられる関節はヨーだけで、
腿を横から後ろへ振る動きでは、IK がヨーを外向きに回し続ける。CMU 132_56（足を振り回す歩き）では、制限がないと右の股のヨーが
0.2 秒（6 フレーム）で −20° から −240° まで約 220° 回った（1 フレームで最大 91°）。

105° は、LAFAN1 の動き（全 77 本）を制限なしで写したときの、脚を外向きに回す最大（104°）。LAFAN1 の写しは変わらない。
サーボの制限と合わせると、132_56 の右の股のヨーは 8 フレームかけて −105° まで回って止まり、関節の跳びは 30 → 0 フレーム。

## 効くフレームの割合と、お手本の遅れ

49d3b83 で全データを写した結果（`build_wholebody_data_49d3b83`）。割合は写した全フレームに対するもの（学習側で外す前）。
ずれは、サーボの制限が効いたフレームでの、膝・足首・肘・手の目標からの距離の平均（括弧はサーボの制限なしの写し）。
遅れは、サーボの制限が効いた区間ごとに、制限なしの写しの関節角に最もよく重なるずらし（区間の長さで重み。制限なしの写しは
腕の解の枝を途中で乗り換えるので、99% 点にはその分も入る）。

| データ・種類 | 速度 | 加速度 | どちらか | 股のヨー 105° | ずれ (mm) | 遅れ 中央値 / 90% / 99% (ms) |
|---|---|---|---|---|---|---|
| LAFAN1 全体 | 4.5% | 31.5% | 32.2% | 0% | 13.2 (11.0) | 0 / 33 / 100 |
| LAFAN1 run | 16.2% | 69.7% | 70.8% | 0% | 13.7 (9.8) | 33 / 33 / 33 |
| LAFAN1 sprint | 11.8% | 32.3% | 32.5% | 0% | 22.3 (9.0) | 33 / 67 / 67 |
| LAFAN1 jumps | 2.0% | 37.5% | 37.8% | 0% | 9.6 (8.8) | 0 / 33 / 33 |
| LAFAN1 dance | 7.6% | 51.7% | 53.1% | 0% | 12.3 (10.9) | 0 / 33 / 100 |
| CMU 全体 | 2.0% | 16.1% | 16.5% | 0.12% | 11.9 (9.9) | 0 / 33 / 267 |
| CMU cmurun | 11.5% | 65.5% | 66.2% | 0% | 13.4 (7.3) | 33 / 33 / 67 |
| CMU cmujump | 3.6% | 24.5% | 25.0% | 0.02% | 11.2 (8.2) | 0 / 33 / 133 |
| CMU cmudance | 5.9% | 41.9% | 42.3% | 0.29% | 14.3 (10.7) | 33 / 33 / 267 |
| PICO 全体 | 0.1% | 7.1% | 7.1% | 0.04% | 6.1 (5.9) | 0 / 0 / 33 |

外すフレーム（写した全フレームに対する割合。左が制限なし、右が今の写し）：

| データ | 腕の反転で切る（finalize） | 学習側の規則で外す | うち関節の跳び | うち体どうしの入り込み | 残る長さ（学習用と評価用） |
|---|---|---|---|---|---|
| LAFAN1 | 3.6% → 4.7% | 16.5% → 12.3% | 1.17% → 0% | 0.94% → 1.55% | 222.1 → 230.5 分 |
| CMU | 1.8% → 5.1% | 11.9% → 7.5% | 1.00% → 0% | 0.54% → 0.67% | 512.3 → 519.8 分 |
| PICO | 1.2% → 0.7% | 8.3% → 7.0% | 0.16% → 0% | 0.43% → 0.18% | 59.2 → 59.6 分 |

学習用と評価用の HDF5 に、サーボの速さを超える歩と、±180° の境をまたぐ跳びは残っていない（制限なしでは LAFAN1 の学習用に 12,162 歩と 121 歩）。

## 首

首（head、neck_roll、neck_pitch）は、人の頭の向き（胴の座標で、基準の立ち姿からの回転）を写した角度を、IK の首の 3 関節の
目標にする（`core.py`）。サーボの速度・加速度と関節の範囲は、ほかの関節と同じく掛かる。LAFAN1 は、被験者ごとに胴が立っている
フレームの頭の向きの平均を首の 0 にする。PICO の評価用の録画（G1）は頭がないので 0。f55bb95 で全データを写した結果
（`build_wholebody_data_f55bb95`、写した全フレーム）：

| データ | head (°) 1% / 中央値 / 99% | neck_roll (°) | neck_pitch (°、+ が下向き) | 範囲の端 | サーボの速さ | 加速度 |
|---|---|---|---|---|---|---|
| LAFAN1 | −63 / −2 / 58 | −25 / 0 / 24 | −56 / −4 / 25 | 10.0% | 0.02% | 0.23% |
| CMU | −60 / −4 / 57 | −25 / 3 / 25 | −55 / 11 / 25 | 32.7% | 0.10% | 0.80% |

範囲の端は、ほとんどが下を向く側（neck_pitch の 25°、LAFAN1 7.9%、CMU 27.0%）と首を傾ける側（neck_roll の ±25°）。
学習と評価の環境では、首の目標をお手本の首の角度にし、実機と同じく HMD の範囲（head ±85°、neck_roll ±23°、neck_pitch −85〜23°）に抑えて
2.5 rad/s で追う（`motion.ReferenceNeckTarget`、`contract.py`）。方策の行動では動かさない。

## 人の側の制限

写す前の人の骨格の関節角に当てる値で、解剖学の軸ごと（解剖学的基本肢位を 0°、左右は同じ値）。`config/retarget_limits.yaml` の
`human.joints` に、採った値（`adopted`）と、その候補の記録（出典の鍵 `src`、確かめ方 `verified`）がある。

値の選び方（ありえない値だけを除くためなので、健常者で大きい側を採る。余裕は足していない）：

- 可動範囲：標準値（日整会・日本リハビリテーション医学会 2022、AAOS 1965、渡辺 1979）と、健常な若い成人の平均 + 2SD・97.5 または 95 パーセンタイルのうち最大。
- 最大角速度：速い動き（競技を含む）で報告された群の値のうち、原典で確かめた最大。二次引用だけの値、1 人だけの値、けがの瞬間の値は採らない。
- 最大角加速度：速い動きで測った値があればそれを採る。なければ、角速度の採用値に 13.8 /s を掛けて推定する（両方の値がある 8 軸の
  比 角加速度 / 角速度 が 5.4〜13.8 /s で、その最大）。表で「推定」と書いた値。首の屈曲・伸展と側屈は角速度も推定（回旋の 780 °/s）。

| 関節・軸（正の向き） | 可動範囲 (°) | 最大角速度 (°/s) | 最大角加速度 (°/s²) |
|---|---|---|---|
| 股 flexion_extension（屈曲） | -35 〜 155 | flexion 964, extension 850 | both 13271 |
| 股 abduction_adduction（外転） | -45.1 〜 52.8 | abduction 422.5, adduction 852 | abduction 5831, adduction 11758（推定） |
| 股 rotation（内旋） | -74.7 〜 71 | internal 499.7, external 636 | internal 6896, external 8777（推定） |
| 膝 flexion_extension（屈曲） | -12.1 〜 156.6 | flexion 1185, extension 1627.2 | both 10842.3 |
| 足首 dorsi_plantar（背屈） | -80 〜 44.2 | dorsiflexion 989.79, plantarflexion 1041.1 | both 7736.15 |
| 足首 inversion_eversion（内がえし（内反）） | -37.9 〜 49.1 | inversion 638, eversion 484 | inversion 8804, eversion 6679（推定） |
| 肩 flexion_extension（屈曲（前方挙上）） | -73.5 〜 185.65 | flexion 1369.7, extension 1071.4 | flexion 7367, extension 14785（推定） |
| 肩 abduction_adduction（外転（側方挙上）） | -75 〜 200.7 | abduction 1360, adduction 1300 | abduction 18768, adduction 13990.4（推定） |
| 肩 rotation（内旋） | -130.65 〜 91 | internal 7250, external 2400 | internal 95246.3, external 33120（推定） |
| 肘 flexion_extension（屈曲） | -14.9 〜 160 | flexion 1492.5, extension 4595 | extension 47335.1, flexion 20597（推定） |
| 体幹 flexion_extension（屈曲（前屈）） | -30 〜 80 | flexion 506.7, extension 253.3 | flexion 6992, extension 3496（推定） |
| 体幹 lateral_flexion（右側屈） | -50 〜 50 | each_side 424 | each_side 5851（推定） |
| 体幹 rotation（右回旋） | -64.6 〜 64.6 | each_side 954.8 | each_side 11600 |
| 首 flexion_extension（屈曲（前屈）） | -82.74 〜 66.67 | both 780（推定） | both 10764（推定） |
| 首 lateral_flexion（右側屈） | -50 〜 50 | each_side 780（推定） | each_side 10764（推定） |
| 首 rotation（右回旋） | -81 〜 81 | each_side 780 | each_side 8500 |

写しに入れるかを決めるため、写す前の骨格（LAFAN1・CMU の BVH、PICO の G1 の関節から作る骨格、30 fps）で、関節角を出して
この値と比べた。角度は、子の体節の向きを基準の立ち姿からの回転として親の体節の軸で分けたもの（膝と肘は 2 つの体節の間の角度、
肩の屈曲と外転は胸郭に対する上腕の向き）、角速度と角加速度は中心差分。「跳び」は、1 フレームの変化が 10° を超え、前後の変化の
3 倍を超えるもの（なめらかな動きでは起きない、モーションキャプチャや追跡の誤り）で、その前後 2 フレームを「跳びの近く」とした。

| データ | 範囲を超える | 角速度を超える | 角加速度を超える |
|---|---|---|---|
| LAFAN1 | 3.0%（跳びの近く 1.6%） | 0.27%（28%） | 11.3%（18%） |
| CMU | 9.5%（3.8%） | 0.55%（40%） | 5.7%（19%） |
| PICO | 2.8%（0.4%） | 0.01%（56%） | 1.2%（39%） |

（写した全フレームのうち、どれかの軸で超えたフレームの割合。括弧はそのうち跳びの近くにある割合。）

範囲を超えるフレームのほとんどは跳びの近くになく、臨床の測り方（ゴニオメーター、肩甲骨や骨盤の動きを含む）とモーションキャプチャの
体節の角度の決め方の違いや、本物の大きな動きで超えている（多いのは肩の回旋、体幹の屈曲、足首）。角加速度は、本物の速い動き
（LAFAN1 では着地の足首が多い）で広く超える。角速度を超えるフレームは跳びの近くが多いが、全体の 0.6% 以下にとどまる。
そのため、写しには入れていない。

例外は、PICO の体幹の回旋（PICO で範囲を超える 1.3% の大半）で、元データの誤り。TWIST2 の録画 4 本（`walk_030`、`037`、`044`、`045`）で、
G1 の腰のヨーが範囲の端の 90° に張りつき、上体が骨盤から横を向いている（TWIST2 のフレームの 4.7%）。Microban の胴は肩の線に合わせるので、
足の向きを保つために股のヨーが外向きに 105° の上限まで回っていた（PICO で股のヨーの上限がかかった 329 フレームは、すべてこの中）。
この範囲だけを写す前に当て、G1 の腰のヨーが人の体幹の回旋の範囲を超えるフレームを、根の瞬間移動と同じく切る（`pico.py` の
`trunk_turn_kept`）。

## 出典

| 鍵 | 出典 |
|---|---|
| `lafan1` | Harvey FG, Yurick M, Nowrouzezahrai D, Pal C. Robust Motion In-betweening. ACM Transactions on Graphics 39(4), 2020（LAFAN1、https://github.com/ubisoft/ubisoft-laforge-animation-dataset） |
| `retarget_2aa2f61` | コード 2aa2f61 の写し（股のヨーの制限なし）の `data/wholebody/gmr/lafan1/*.npz`。両脚の外向きのヨーの全フレームの最大 |
| `xc330_bam` | XC330-T288-T の BAM の同定（30 本の記録、m6 モデル）。`src/mjlab_microban/robot/xc330_params.json`、https://github.com/rhoban/bam |
| `xc330_current_limit` | XC330-T288-T のファームウェアの電流の上限 0.91 A（学習のアクチュエータの `max_current`）。`src/mjlab_microban/robot/microban_constants.py`、https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/ |
| `pack_3s` | Microban の 3S の電池（公称 11.1 V、9.0〜12.6 V。学習のアクチュエータの `vin_range`）。`src/mjlab_microban/robot/microban_constants.py` |
| `microban_mjcf` | Microban のモデルと HOME（リンクの質量と慣性、HOME の関節角）。`src/mjlab_microban/robot/microban/robot.xml`、`config/home_pose.yaml` |
| `joa_jarm2022` | 日本リハビリテーション医学会・日本整形外科学会・日本足の外科学会 (2021). 関節可動域表示ならびに測定法改訂について（2022年4月改訂）. Jpn J Rehabil Med 58(10):1188-1200（日本足の外科学会雑誌 42:S372-S385、日整会誌 96:75-86 にも掲載）、doi:10.2490/jjrmc.58.1188 |
| `aaos1965` | American Academy of Orthopaedic Surgeons, Committee on Joint Motion (1965). Joint Motion: Method of Measuring and Recording. AAOS（1966 年に British Orthopaedic Association が再版） |
| `watanabe1979` | 渡辺英夫 ほか (1979). 健康日本人における四肢関節可動域について. 年齢による変化. 日本整形外科学会雑誌 53:275-291 |
| `botman2020` | Botman M ほか (2020). The Effectiveness of Burn Scar Contracture Release Surgery in Low- and Middle-income Countries. Plast Reconstr Surg Glob Open 8(7):e2907（Supplementary Digital Content 1）、doi:10.1097/GOX.0000000000002907 |
| `soucie2011` | Soucie JM ほか (2011). Range of motion measurements: reference values and a database for comparison studies. Haemophilia 17(3):500-507、doi:10.1111/j.1365-2516.2010.02399.x |
| `boone1979` | Boone DC ほか (1979). Normal range of motion of joints in male subjects. J Bone Joint Surg Am 61(5):756-759、doi:10.2106/00004623-197961050-00017 |
| `roaas1982` | Roaas A ほか (1982). Normal range of motion of the hip, knee and ankle joints in male subjects, 30-40 years of age. Acta Orthop Scand 53(2):205-208、doi:10.3109/17453678208992202 |
| `svenningsen1989` | Svenningsen S ほか (1989). Hip motion related to age and sex. Acta Orthop Scand 60(1):97-100、doi:10.3109/17453678909150103 |
| `mckay2017` | McKay MJ ほか (2017). Normative reference values for strength and flexibility of 1,000 children and adults. Neurology 88(1):36-43、doi:10.1212/WNL.0000000000003466 |
| `han2015` | Han H ほか (2015). Hip rotation range of motion in sitting and prone positions in healthy Japanese adults. J Phys Ther Sci 27(2):441-445、doi:10.1589/jpts.27.441 |
| `menadue2006` | Menadue C ほか (2006). Reliability of two goniometric methods of measuring active inversion and eversion range of motion at the ankle. BMC Musculoskelet Disord 7:60、doi:10.1186/1471-2474-7-60 |
| `medina2009` | Medina McKeon JM ほか (2009). Sex differences and representative values for 6 lower extremity alignment measures. J Athl Train 44(3):249-255、doi:10.4085/1062-6050-44.3.249 |
| `vairo2012` | Vairo GL ほか (2012). Clinical descriptive measures of shoulder range of motion for a healthy, young and physically active cohort. Sports Med Arthrosc Rehabil Ther Technol 4:33、doi:10.1186/1758-2555-4-33 |
| `fleisig2023` | Fleisig GS ほか (2023). Active range of motion of the shoulder: a cross-sectional study of 6635 subjects. JSES Int 7(1):132-137、doi:10.1016/j.jseint.2022.09.008 |
| `zwerus2019` | Zwerus EL ほか (2019). Normative values and affecting factors for the elbow range of motion. Shoulder Elbow 11(3):215-224、doi:10.1177/1758573217728711 |
| `gunal1996` | Gunal I ほか (1996). Normal range of motion of the joints of the upper extremity in male subjects, with special reference to side. J Bone Joint Surg Am 78(9):1401-1404、doi:10.2106/00004623-199609000-00017 |
| `moromizato2016` | Moromizato K ほか (2016). Whole-body patterns of the range of joint motion in young adults: masculine type and feminine type. J Physiol Anthropol 35:23、doi:10.1186/s40101-016-0112-8 |
| `macedo2007` | Macedo LG (2007). Normal range of motion of joints in female subjects in different age groups（修士論文、University of Alberta）. 、doi:10.7939/r3-5xgt-1w55 |
| `mousavi2018` | Mousavi SJ ほか (2018). Between-session reliability of opto-electronic motion capture in measuring sagittal posture and 3-D ranges of motion of the thoracolumbar spine. J Biomech 79:248-252、doi:10.1016/j.jbiomech.2018.08.033 |
| `zarate2023` | Zárate-Tejero C ほか (2023). Association between Age, Sex and Cervical Spine Sagittal Plane Motion: A Descriptive and Correlational Study in Healthy Volunteers. Life 13(2):461、doi:10.3390/life13020461 |
| `chen1999` | Chen J ほか (1999). Meta-analysis of normative cervical motion. Spine 24:1571-1578、doi:10.1097/00007632-199908010-00011 |
| `ferrario2002` | Ferrario VF ほか (2002). Active range of motion of the head and cervical spine: a three-dimensional investigation in healthy young adults. J Orthop Res 20(1):122-129、doi:10.1016/S0736-0266(01)00079-1 |
| `sforza2002` | Sforza C ほか (2002). Three-dimensional analysis of active head and cervical spine range of motion: effect of age in healthy male subjects. Clin Biomech 17(8):611-614、doi:10.1016/s0268-0033(02)00071-2 |
| `cheng2026` | Cheng G ほか (2026). Biomechanical adaptations of kicks following vertical and single-leg plyometric training in competitive Wushu sanda athletes. Front Physiol 17:1888868、doi:10.3389/fphys.2026.1888868 |
| `orloff2007` | Orloff H ほか (2007). Comparison Between Ground Reaction Force Patterns and Angular, Approach, and Ball Velocities for In-Step Kicking. ISBS 2007（25th International Symposium on Biomechanics in Sports） |
| `miyashiro2019` | Miyashiro K ほか (2019). Kinematics of Maximal Speed Sprinting With Different Running Speed, Leg Length, and Step Characteristics. Front Sports Act Living 1:37、doi:10.3389/fspor.2019.00037 |
| `kivi2001` | Kivi D ほか (2001). Lower extremity kinematics during high speed treadmill sprinting over a range of velocities. ISBS 2001, pp.102-105 |
| `alt2026` | Alt T ほか (2026). Maximum Sprints Elicit Higher Peak Knee Joint Power than Resistance Training Exercises. Sports 14(6):221、doi:10.3390/sports14060221 |
| `ho2019` | Ho M ほか (2019). Foot orthoses alter lower limb biomechanics but not jump performance in basketball players with and without flat feet. J Foot Ankle Res 12:24、doi:10.1186/s13047-019-0334-1 |
| `branco2019` | Branco MAC ほか (2019). Effect of Aging on the Lower Limb Kinematics in Karate Practitioners: Comparing Athletes and Their Senseis. J Healthc Eng 2019:2672185、doi:10.1155/2019/2672185 |
| `grimmer2020` | Grimmer M ほか (2020). Human Lower Limb Joint Biomechanics in Daily Life Activities. Front Robot AI 7:13、doi:10.3389/frobt.2020.00013 |
| `detjareny2013` | Detjareny T ほか (2013). （テコンドーの後ろ蹴りの運動学。題は未確認）. ISBS 2013 |
| `gavagan2017` | Gavagan CJ ほか (2017). A biomechanical analysis of the roundhouse kicking technique of expert practitioners: A comparison between the martial arts disciplines of Muay Thai, Karate, and Taekwondo. PLoS ONE 12:e0182645、doi:10.1371/journal.pone.0182645 |
| `kageyama2015` | Kageyama M ほか (2015). Difference Between Adolescent and Collegiate Baseball Pitchers in the Kinematics and Kinetics of the Lower Limbs and Trunk During Pitching Motion. J Sports Sci Med 14:246-255 |
| `radzak2020` | Radzak KN ほか (2020). Fatigue-Induced Hip-Abductor Weakness and Changes in Biomechanical Risk Factors for Running-Related Injuries. J Athl Train 55(12)、doi:10.4085/1062-6050-531-19 |
| `nunome2002` | Nunome H ほか (2002). Three-dimensional kinetic analysis of side-foot and instep soccer kicks. Med Sci Sports Exerc 34(12):2028-2036、doi:10.1097/00005768-200212000-00025 |
| `jessop2016` | Jessop DM ほか (2016). Maximum Velocities in Flexion and Extension Actions for Sport. J Hum Kinet 50:37-44、doi:10.1515/hukin-2015-0139 |
| `kellis2007` | Kellis E ほか (2007). Biomechanical characteristics and determinants of instep soccer kick. J Sports Sci Med 6(2):154-165 |
| `manolopoulos2006` | Manolopoulos E ほか (2006). （インステップキックの研究。原典は開けず）.  |
| `levanon1998` | Levanon J ほか (1998). Comparison of the kinematics of the full-instep and pass kicks in soccer. Med Sci Sports Exerc 30:917-927 |
| `sinclair2014` | Sinclair J ほか (2014). Three-dimensional kinematic correlates of ball velocity during maximal instep soccer kicking in males. Eur J Sport Sci 14:799-805、doi:10.1080/17461391.2014.908956 |
| `preuschl2016` | Preuschl E ほか (2016). A Kinematic Analysis of the Jumping Front-Leg Axe-Kick in Taekwondo. J Sports Sci Med 15(1):92-101 |
| `tamura2016` | Tamura A ほか (2016). Fatigue Alters Landing Shock Attenuation During a Single-Leg Vertical Drop Jump. Orthop J Sports Med 4:2325967115626412、doi:10.1177/2325967115626412 |
| `meamarbashi2010` | Meamarbashi A ほか (2010). Application of Novel Inertial Technique to Compare the Kinematics and Kinetics of the Legs in the Soccer Instep Kick. J Hum Kinet 23:3-12、doi:10.2478/v10078-010-0001-x |
| `vaningenschenau1985` | van Ingen Schenau GJ ほか (1985). The instantaneous torque-angular velocity relation in plantar flexion during jumping. Med Sci Sports Exerc 17(4):422-426、doi:10.1249/00005768-198508000-00003 |
| `duchene2026` | Duchene Y ほか (2026). Effect of Consecutive Jumping Trials on Metatarsophalangeal, Ankle, Knee and Hip Biomechanics During Take-Off and Landing. Eur J Sport Sci 26:e70217、doi:10.1002/ejsc.70217 |
| `chan2014` | Chan ZYS ほか (2014). Comparison of ankle joint kinematics of a single athlete during an ankle inversion sprain incident and normal non-injury motions. J Foot Ankle Res 7(Suppl 1):A28 |
| `fong2009` | Fong DTP ほか (2009). Biomechanics of supination ankle sprain: a case report of an accidental injury event in the laboratory. Am J Sports Med 37(4):822-827、doi:10.1177/0363546508328102 |
| `kristianslund2011` | Kristianslund E ほか (2011). Kinematics and kinetics of an accidental lateral ankle sprain. J Biomech 44:2576-2578、doi:10.1016/j.jbiomech.2011.07.014 |
| `chu2010` | Chu VWS ほか (2010). Differentiation of ankle sprain motion and common sporting motion by ankle inversion velocity. J Biomech 43(10):2035-2038、doi:10.1016/j.jbiomech.2010.03.029 |
| `flores2026` | Flores N ほか (2026). Which pronator are you? New perspectives from an unsupervised clustering approach in running. Front Sports Act Living 7:1682315、doi:10.3389/fspor.2025.1682315 |
| `digby2005` | Digby CJ ほか (2005). High-speed coupling characteristics of the foot and shank during the stance phase of running. ISB 2005（学会抄録） |
| `lysdal2022` | Lysdal FG ほか (2022). What have we learnt from quantitative case reports of acute lateral ankle sprains injuries and episodes of 'giving-way' of the ankle joint, and what shall we further investigate?. Sports Biomech 21(4)、doi:10.1080/14763141.2022.2035801 |
| `fong2012` | Fong DTP ほか (2012). Kinematics analysis of ankle inversion ligamentous sprain injuries in sports: five cases from televised tennis competitions. Am J Sports Med 40:2627-2632、doi:10.1177/0363546512458259 |
| `ha2013` | Ha SCW ほか (2013). Kinematics analysis of an ankle inversion ligamentous sprain injury case in basketball. ISBS 2013（学会抄録） |
| `pappas1985` | Pappas AM ほか (1985). Biomechanics of baseball pitching. A preliminary report. Am J Sports Med 13(4):216-222、doi:10.1177/036354658501300402 |
| `dillman1993` | Dillman CJ ほか (1993). Biomechanics of pitching with emphasis upon shoulder kinematics. J Orthop Sports Phys Ther 18(2):402-408、doi:10.2519/jospt.1993.18.2.402 |
| `fleisig1999` | Fleisig GS ほか (1999). Kinematic and kinetic comparison of baseball pitching among various levels of development. J Biomech 32(12):1371-1375、doi:10.1016/S0021-9290(99)00127-X |
| `fleisig2006` | Fleisig GS ほか (2006). Kinematics and kinetics of youth baseball pitching with standard and lightweight balls. Sports Engineering 9:155-163 |
| `fleisig2011` | Fleisig GS ほか (2011). Biomechanical comparison of baseball pitching and long-toss. J Orthop Sports Phys Ther 41(5):296-303、doi:10.2519/jospt.2011.3568 |
| `fleisig2019` | Fleisig GS ほか (2019). Baseball Pitching Biomechanics Shortly After Ulnar Collateral Ligament Repair. Orthop J Sports Med 7(8):2325967119866199、doi:10.1177/2325967119866199 |
| `fleisig2025` | Fleisig GS ほか (2025). Increases in Ball Weight and Size Decrease Elbow Varus Torque During Baseball Pitching. Orthop J Sports Med 13(9):23259671251368999、doi:10.1177/23259671251368999 |
| `berkson_imu` | Berkson E ほか (2006). IMU Arrays: The Biomechanics of Baseball Pitching. MIT Media Lab（掲載誌は未確認） |
| `gasparutto2015` | Gasparutto X ほか (2015). Shoulder joint velocity during fastball pitching in baseball. ISBS 2015, pp.907-910 |
| `woo1997` | Woo SY ほか (1997). A three-dimensional analysis of the windmill style of softball delivery for fast and change-up pitching. ISBS 1997 |
| `barrentine1998` | Barrentine SW ほか (1998). （ウインドミル投法の研究。原典は開けず）.  |
| `wagner2010` | Wagner H ほか (2010). Kinematic description of elite vs. low level players in team-handball jump throw. J Sports Sci Med 9(1):15-23 |
| `eriksson2015` | Eriksson Crommert M ほか (2015). Trunk Muscle Activation at the Initiation and Braking of Bilateral Shoulder Flexion Movements of Different Amplitudes. PLoS One 10(11):e0141777、doi:10.1371/journal.pone.0141777 |
| `kohler2024` | Köhler HP ほか (2024). How the acceleration phase influences energy flow and the resulting joint moments of the throwing shoulder in the deceleration phase of the javelin throw. Front Sports Act Living 6:1445455、doi:10.3389/fspor.2024.1445455 |
| `kaizu2018` | Kaizu Y ほか (2018). Correlation of upper limb joint load with simultaneous throwing mechanics including acceleration parameters in amateur baseball pitchers. J Phys Ther Sci 30(2):223-230、doi:10.1589/jpts.30.223 |
| `fleisig2003` | Fleisig G ほか (2003). Kinematics used by world class tennis players to produce high-velocity serves. Sports Biomech 2(1):51-64、doi:10.1080/14763140308522807 |
| `cheraghi2014` | Cheraghi M ほか (2014). Kinematics of Straight Right Punch in Boxing. Ann Appl Sport Sci 2(2):39-50 |
| `fett2021` | Fett J ほか (2021). Kinematic characteristics of the tennis serve from the ad and deuce court service positions in elite junior players. PLoS One 16(7):e0252650、doi:10.1371/journal.pone.0252650 |
| `connolly2021` | Connolly M ほか (2021). Effects of Lumbar Spine Abnormality and Serve Types on Lumbar Kinematics in Elite Adolescent Tennis Players. Sports Med Open 7(1):7、doi:10.1186/s40798-020-00295-2 |
| `marras1993` | Marras WS ほか (1993). The role of dynamic three-dimensional trunk motion in occupationally-related low back disorders. Spine 18(5):617-628、doi:10.1097/00007632-199304000-00015 |
| `marras1995` | Marras WS ほか (1995). The classification of anatomic- and symptom-based low back disorders using motion measure models. Spine 20(23):2531-2546、doi:10.1097/00007632-199512000-00013 |
| `horan2012` | Horan SA ほか (2012). （ゴルフのスイングの胸郭と骨盤の運動。題は未確認）. Sports Biomech 11:165-174 |
| `kageyama2014` | Kageyama M ほか (2014). Kinematic and Kinetic Profiles of Trunk and Lower Limbs during Baseball Pitching in Collegiate Pitchers. J Sports Sci Med 13(4):742-750 |
| `fleisig2013` | Fleisig GS ほか (2013). Trunk axial rotation in baseball pitching and batting. Sports Biomech 12(4):324-333、doi:10.1080/14763141.2013.838693 |
| `roijezon2010` | Röijezon U ほか (2010). Kinematics of fast cervical rotations in persons with chronic neck pain: a cross-sectional and reliability study. BMC Musculoskelet Disord 11:222、doi:10.1186/1471-2474-11-222 |
| `grossman1988` | Grossman GE ほか (1988). Frequency and velocity of rotational head perturbations during locomotion. Exp Brain Res 70(3):470-476、doi:10.1007/BF00247595 |
| `zangemeister1986` | Zangemeister WH ほか (1986). Active head turning and correlated cerebral potentials. Experimental and clinical aspects. Acta Otolaryngol 101(5-6):403-415、doi:10.3109/00016488609108625 |
| `forsberg2023` | Forsberg K ほか (2023). Concurrent Validity of Cervical Movement Tests Using VR Technology-Taking the Lab to the Clinic. Sensors 23(24):9864、doi:10.3390/s23249864 |
| `carriot2014` | Carriot J ほか (2014). Statistics of the vestibular input experienced during natural self-motion: implications for neural processing. J Neurosci 34(24):8347-8357、doi:10.1523/JNEUROSCI.0692-14.2014 |
| `bussone2010` | Bussone WR ほか (2010). （日常動作での頭の角速度・角加速度。題は未確認）. Biomed Sci Instrum 46:166-171 |
| `vikne2013` | Vikne H ほか (2013). Muscle activity and head kinematics in unconstrained movements in subjects with chronic neck pain; cervical motor dysfunction or low exertion motor output?. BMC Musculoskelet Disord 14:314、doi:10.1186/1471-2474-14-314 |
| `meisingset2015` | Meisingset I ほか (2015). Evidence for a general stiffening motor control pattern in neck pain: a cross sectional study. BMC Musculoskelet Disord 16:56、doi:10.1186/s12891-015-0517-2 |
| `marinsek2024` | Marinsek A ほか (2024). MmWave for Extended Reality: Open User Mobility Dataset, Characterisation, and Impact on Link Quality. arXiv:2407.02636v2 |
