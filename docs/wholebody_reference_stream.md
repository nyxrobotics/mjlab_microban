# 全身トラッキングのお手本の送り方（PC → Pi、UDP）

PC の写し（PICO の体の追跡から、その場でお手本を作る処理）が、ロボットの Pi へお手本のフレームを送る形式。
Pi 側の全身トラッキングの方策は、これを復号して参照に使う。形式を変えるときは版番号を上げ、両側を同時に直す。

符号化と復号の関数は `src/mjlab_microban/live/stream.py`（`encode_frame` / `decode_frame`）、
テストは `tests/test_live_stream.py`。

## 送り方

- 1 フレームが 1 つの UDP の包み。PC が 50 Hz で送る（20 ms ごと）。
- 宛先の既定は Pi の UDP 5556（PICO の操縦の 5555 とは別）。
- 写しが動いている間は、校正の前も、開始の前も送り続ける（Pi はこれで PC が生きていると分かる）。
- 再送も確認応答もない。失った包みは捨てる。

## 包みの中身

すべて little-endian。見出し 24 バイト、続けて本体 1312 バイト、計 1336 バイト。
長さが違う包みは捨てる。

### 見出し（24 バイト、`struct` の書式 `<4sHHHIdH`）

| 位置 | 型 | 名前 | 中身 |
|---|---|---|---|
| 0 | 4 バイト | magic | `b"MBWB"`。違えば捨てる |
| 4 | uint16 | version | 版番号。今は `1`。違えば捨てる |
| 6 | uint16 | joint_count | 関節の数。今は `21` |
| 8 | uint16 | body_count | 体の数。今は `22` |
| 10 | uint32 | seq | 連番。写しの起動時に 0、1 フレームごとに 1 増える（2^32 で 0 に戻る） |
| 14 | float64 | sent_time_s | 送った時刻（PC の Unix 時刻、秒）。遅れの計測用 |
| 22 | uint16 | flags | 状態の旗（下表） |

`joint_count` と `body_count` が下の名前の一覧の数と違えば捨てる。

### 本体（float32 × 328）

学習の HDF5 の 1 フレームを、`wholebody/data/hdf5.py` の `FRAME_KEYS` の順に並べる。

| 順 | 名前 | 形 | 単位・意味 |
|---|---|---|---|
| 1 | joint_pos | (21,) | 関節角（rad、HOME からの差ではなく絶対値） |
| 2 | joint_vel | (21,) | 関節角速度（rad/s） |
| 3 | body_pos_w | (22, 3) | 体の位置（m、世界座標） |
| 4 | body_quat_w | (22, 4) | 体の向き（単位四元数、w, x, y, z の順、世界座標） |
| 5 | body_lin_vel_w | (22, 3) | 体の速度（m/s、世界座標） |
| 6 | body_ang_vel_w | (22, 3) | 体の角速度（rad/s、世界座標） |

多次元の配列は行優先（体ごとに x, y, z を続ける）。

- 世界座標：Z が上、床が z = 0。校正したときの操縦者の正面が +X、そのときの胴（trunk）の真下が原点付近。
  学習の HDF5 と同じく、Pi は追従の開始時に参照の向き（yaw）を IMU の yaw に合わせる。
- 首の 3 関節（head, neck_roll, neck_pitch）は、PICO の頭の関節の向き（胴に対する、校正の立ち姿からの回転）を写した角度（学習の HDF5 と同じ写し）。Pi は、これを契約どおり（`wholebody/contract.py`）HMD の範囲に抑え、2.5 rad/s で追って首の目標にする。
- 速度は過去のフレームだけから作る（後退差分、50 Hz）。学習の HDF5 は中心差分（30 fps）なので、
  速度は位置より約 10 ms 遅れる。
- 各フレームは「操縦者のその時の姿勢」を写したもの。PC での補間のための遅れと通信の遅れの分だけ、
  操縦者より遅れる（遅れの実測は下の「遅れ」）。

### 関節の並び（学習の HDF5 の `joint_names`、`contract.OBSERVATION_JOINT_NAMES` と同じ）

```text
 0 head                 7 right_hip_roll      14 left_elbow
 1 neck_roll            8 right_hip_pitch     15 left_hip_yaw
 2 neck_pitch           9 right_knee          16 left_hip_roll
 3 right_shoulder_pitch 10 right_ankle_pitch  17 left_hip_pitch
 4 right_shoulder_roll  11 right_ankle_roll   18 left_knee
 5 right_elbow          12 left_shoulder_pitch 19 left_ankle_pitch
 6 right_hip_yaw        13 left_shoulder_roll 20 left_ankle_roll
```

### 体の並び（学習の HDF5 の `body_names`、robot.xml の体の順。world は含まない）

```text
 0 trunk                          11 ankle_block__configuration_left
 1 head                           12 foot
 2 roll_pitch_link                13 shoulder_2
 3 camera_cover                   14 humerus__configuration_left
 4 shoulder                       15 radius_2
 5 humerus__configuration_right   16 hip_2
 6 radius                         17 hip_block__configuration_left
 7 hip                            18 femur__configuration_left
 8 hip_block__configuration_right 19 tibia__configuration_left
 9 femur__configuration_right     20 ankle_block__configuration_right
10 tibia__configuration_right     21 foot_2
```

（robot.xml の名前のとおり、`ankle_block__configuration_left` は右脚、`foot` は右足、`foot_2` は左足、
`radius` は右の前腕、`radius_2` は左の前腕。学習の追従対象の 11 体は `env_cfg.TRACKED_BODY_NAMES`。）

## 旗（flags）

| ビット | 名前 | 意味 |
|---|---|---|
| 0 | CALIBRATED | 立ち姿の校正が済んでいる。0 の間の中身は HOME の立ち姿（原点、速度 0） |
| 1 | ACTIVE | 操縦者が開始のボタンを押し、追従してよい状態。止めると 0 に戻る |
| 2 | RESTART | 新しい参照の始まり（開始した最初のフレーム、追従中の校正のやり直しの最初のフレーム）。Pi は追従を始め直す（yaw を合わせ直し、履歴を作り直す） |
| 3 | STALE | 体の追跡が 0.25 秒以上届いていない。中身は途切れの扱いのお手本（下） |
| 4 | HELD | 今の姿勢が学習データの除外の規則に当たった（床へのめり込み、自己干渉、浮き上がりなど。過去のフレームだけで判定）。当たったフレームと、規則に当たらなくなってから 0.2 秒は、写しの姿勢を送らず、途切れの扱いのお手本を送る |
| 5 | LATE | 補間に要る入力がまだ届いていない。中身は途切れの扱いのお手本（診断用） |

途切れの扱い（STALE、HELD、LATE の間。[wholebody_live.md](wholebody_live.md) の「途切れたとき」）：お手本は途切れる直前の
速さで動き続けてから滑らかに止まり、途切れて 1 秒たつと校正した立ち姿へ向かう。入力が戻ると、写しの姿勢へ滑らかに戻る
（戻る間は旗なし）。どのフレームも、関節・胴の位置と向き・それぞれの速さが、1 刻みで学習データの 99.9% 点より大きく
変わらない。速度は送ったフレームの後退差分。

ほかのビットは 0。知らないビットは無視する。

Pi 側の扱い（推奨）：

- 追従するのは CALIBRATED と ACTIVE がともに 1 の間だけ。
- RESTART のフレームで追従を始め直す。
- 100 ms（5 フレーム）以上包みが届かなければ、通信が切れたとして扱う。
- 連番が直前より古い包み（`(seq - last) mod 2^32` が 2^31 以上）は捨てる。
- STALE、HELD、LATE の間も中身は使える参照（途切れの扱いのお手本）。どれだけ続いたら止めるかは Pi 側で決める。

## PC 側（参考）

- 体の 24 関節は、PICO のアプリから PC の `microban_teleop`（認証付き TCP）に届き、PC の中の UDP で写しへ渡る。
- 写しは、校正した立ち姿を HOME に当て、データ作りと同じ IK と制限（`wholebody/retarget/core.py`、
  `config/retarget_limits.yaml`）で 50 Hz のフレームを作る。入力が 25 Hz なら、時刻で補間して 50 Hz にする。
- 床の高さは過去 1 秒の最小、速度は後退差分、除外の規則は過去のフレームだけで判定する。

## 遅れ

PICO の評価用の録画で測った値（[wholebody_live.md](wholebody_live.md) の「突き合わせ」）。

- PC の補間の遅れ：入力 25 Hz で 40 ms、30 Hz で 33 ms（入力の間隔 + 届く時刻の揺れ）。位置と関節角の中身はこれ以上遅れない。
- 速度は後退差分なので、さらに 10 ms 遅れる。
- PC の計算：1 フレーム平均 3.5 ms、99 % 点 5.4 ms。
- これに PICO → PC と PC → Pi の通信が加わる。`sent_time_s` は PC の時刻なので、Pi と PC の時計が合っていれば通信の遅れを測れる。
