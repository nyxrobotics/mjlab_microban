# 全身トラッキングの写しをその場で動かす（PC）

PICO の体の追跡（24 関節）から、学習データと同じ写しで、その場でお手本のフレームを作り、ロボットの Pi へ 50 Hz で送る。
Pi への送り方は [wholebody_reference_stream.md](wholebody_reference_stream.md)。コードは `src/mjlab_microban/live/`。
学習の環境（mjlab、torch）は読まない（`MICROBAN_WITHOUT_TRAINING=1`。GPU も使わない）。

```text
PICO（microban_pico_client）
  │ 認証付き TCP（頭、コントローラー、体の 24 関節）
  ▼
PC: microban_teleop の bridge ──UDP 5555──▶ Pi（PICO の操縦の指令。今までどおり）
  │ UDP 5557（PC の中）: 体の 24 関節、左 X・左 Y の押下回数（live/link.py）
  ▼
PC: python -m mjlab_microban.live run ──UDP 5556──▶ Pi（お手本のフレーム、50 Hz）
```

- 写しは PC で動かす。Pi Zero 2W では Python の写しが 1 フレーム 50〜120 ms かかり、間に合わない。PC では 1 フレーム 3〜5 ms。
- 体の関節は、ヘッドセットから PC への TCP（今の bridge）で受ける。BLE で Pi に届く経路（25 Hz）を使うと、
  PICO → Pi → PC → Pi と往復になるので使わない。

## 動かす

PC で 2 つを起動する（PICO の接続と pairing は microban_teleop の README のとおり）。

```bash
# 1. bridge（microban_teleop、ブランチ tracking-wholebody）。体の関節を写しへも送る
cd ../microban_teleop
uv run microban-native-bridge --robot-host microban --wholebody-host 127.0.0.1
#    自動起動の service なら MICROBAN_WHOLEBODY_HOST=127.0.0.1 を設定する

# 2. 写し（このリポジトリ）。Pi の UDP 5556 へ送る
cd ../mjlab_microban
uv run python -m mjlab_microban.live run --robot microban
```

操作：

| 入力 | 動作 |
|---|---|
| 左 X | 校正。押してから 1 秒、まっすぐ立って静止する（腕は下ろす） |
| 左 Y | 追従の開始・停止（旗 ACTIVE。開始の最初のフレームに RESTART） |
| 写しの端末で `c` / `s` / `q` + Enter | 校正 / 開始・停止 / 終了 |

- 校正の条件（1 秒のうち、条件を満たす最も静かなフレームを使う）：全身の点の速さが脚の長さの 0.15 倍/秒未満、
  両足首の高さの差が脚の長さの 2 % 未満、背骨（骨盤 → 首の付け根）が鉛直から 10° 以内、左右の膝の曲がりの差が 5° 未満、
  上腕が真下から 35° 以内。満たさなければ、端末に理由を出し、前の校正を保つ。
- 校正した立ち姿が HOME に当たる。校正したときの正面が +X、そのときの骨盤の真下が原点。
- 写しは 2 秒ごとに状態（旗、補間の遅れ、1 フレームの計算時間、胴の位置）を端末に出す。

## ロボットなしで試す

```bash
# 操縦者の体を記録する（bridge を --wholebody-host 127.0.0.1 で動かしたまま。ロボットの電源は要らない）
uv run python -m mjlab_microban.live record --out rec.npz --seconds 300
#   最初に左 X を押して 1 秒立って静止し、左 Y で開始する（押下も記録する）

# 記録を写しに通す：Pi へ実時間で送る（--loop で繰り返す）か、ファイルに書く
uv run python -m mjlab_microban.live replay rec.npz --robot microban --loop
uv run python -m mjlab_microban.live replay rec.npz --out frames.npz

# PICO の評価用の録画（G1）を記録の形にする（立ち姿の基準つき。再生するとすぐ校正・開始する）
uv run python -m mjlab_microban.live from-clip picoteleopit_rec04 --raw data/wholebody/raw --out rec04.npz
uv run python -m mjlab_microban.live replay rec04.npz --robot microban --loop
```

`run --record rec.npz` でも、動かしながら記録できる。記録は体の 24 関節（Microban の軸、xyzw）、
ヘッドセットの時刻、PC に届いた時刻、押下回数。

## データ作りの写しとの違い

データ作り（`wholebody/retarget/core.py` の `retarget_frames`、`wholebody/data/build.py`）はクリップ全体を見る。
その場の写しは、各フレームをそれより前のフレームだけから作る。

| 項目 | データ作り | その場 |
|---|---|---|
| 世界の座標 | クリップの最初のフレームが +X 向き | 校正したフレームが +X 向き |
| 立ち姿の基準 | 録画の集まりごとに静かな立ちのフレーム | 校正の 1 秒から選んだフレーム |
| 入力の速さ | 30 fps | 25〜50 Hz を時刻で補間（位置は線形、向きは slerp）して 50 Hz |
| IK と制限 | GMR の IK、サーボの速度・加速度、股のヨー 105°、自己衝突（30 fps） | 同じ（50 Hz） |
| 床の高さ | 前後 1 秒の最小 | 過去 1 秒の最小 |
| 速度 | 中心差分 | 後退差分 |
| 除外 | 当たったフレームと前後 0.2 秒を切る | 当たったフレームと後 0.2 秒を、直前のフレームで止める（旗 HELD） |

補間の遅れは、入力の間隔の中央値と、届く時刻のばらつき（最速からの遅れの 90 % 点）の和（25 Hz で揺れがなければ 40 ms）。
体の 24 関節は pico.py の疑似骨格に置き換える（大腿 = 股、下腿 = 膝、足 = 足首、上腕 = 肩、前腕 = 肘、手 = 手首、
胴 = spine_1 の位置と spine_3 の向き、首の付け根 = neck、つま先 = foot の位置と足首の向き）。

## 突き合わせ

PICO の評価用の録画（`data/wholebody/raw/pico`、98 本、約 1 時間）をその場の写しに通し、評価用の HDF5（データ作りの写し）と
同じ時刻で比べた。立ち姿の基準はデータ作りと同じものを使い、通信の揺れはなし。HELD のフレームは差から除いた。

```bash
uv run python src/mjlab_microban/scripts/wholebody_live_compare.py --raw data/wholebody/raw --data data/wholebody --rate 25 --out live_compare_25hz.json
```

差（その場 − データ作り）。関節角は首を除く 18 関節、体は 22 体、平均 / 99 % 点：

| | 入力 25 Hz | 入力 30 Hz |
|---|---|---|
| 関節角 | 0.48° / 5.1° | 0.44° / 5.1° |
| 体の位置 | 1.7 mm / 15.3 mm | 1.6 mm / 15.1 mm |
| 胴（trunk）の位置 | 1.6 mm / 14.9 mm | 1.4 mm / 14.7 mm |
| 関節角速度 | 0.15 / 1.34 rad/s | 0.17 / 1.46 rad/s |
| 体の速度 | 0.021 / 0.129 m/s | 0.028 / 0.158 m/s |
| 体の角速度 | 0.23 / 1.61 rad/s | 0.29 / 1.90 rad/s |
| 比べたフレーム（HELD で除いた数） | 106,768（289） | 106,839（258） |

遅れ：

- 補間の遅れ（作るフレームが、届いた最新の入力より後ろにいる時間）：25 Hz で 40 ms、30 Hz で 33 ms（通信の揺れがあれば、その分のびる）。
- 補間の遅れを差し引いた中身の遅れ：位置と関節角は 0 ms（ずらして比べても差が最小になるのは 0 ms）。速度は後退差分の分 10 ms。
- 操縦者の動きからフレームを送るまで：補間の遅れ + 計算（1 フレーム平均 3.5 ms、99 % 点 5.4 ms）+ PICO → PC の通信。
  PC → Pi の UDP の分がさらに加わる。
- 校正の直後の 1 フレームは IK の最初の解き直しで 70 ms ほどかかる。`run` ではこれを裏のスレッドで済ませ、送る周期は乱さない。

除外の一致（その場の 50 Hz のフレームの数、25 Hz 入力）：データ作りが残したフレームのうち、その場が止めた（HELD）のは 434（0.24 %）。
データ作りが切ったフレームのうち 6,107 をその場も止め、8,415 は通した（データ作りは印の前の 0.2 秒と 22 フレーム未満の切れ端も切るが、
その場では先が見えない）。
