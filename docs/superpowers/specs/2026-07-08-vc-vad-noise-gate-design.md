# VC VAD ノイズゲート設計 (2026-07-08)

## 背景 / 問題

RVC は非音声入力(環境音・ブレス)からも人声風のノイズを生成する。HuBERT がノイズを
音素的特徴として解釈し、デコーダは常にフルスケールの声を出すためである。現行パイプ
ラインには 2 つの漏れ道がある:

1. **ノイズ単独チャンク** — recording の門番は発話単位の dBFS 閾値
   (`recording.silence_threshold`、既定 -40) なので、環境音やブレスだけで録音がトリ
   ガーされるとチャンク全体がノイズのまま vc worker に届く。既存の入力エンベロープ
   ダック (`apply_input_envelope`) は平均正規化された**相対**形状なので、チャンク全
   体がノイズだと gain≈1 に正規化され、フルスケールの人声風ノイズがそのまま再生さ
   れる。
2. **発話内のブレス・環境音** — "speaking" 状態に入ると発話内の全フレームが素通し
   で、録音開始前の約 0.5 秒のプリロールバッファ (環境音を含む) も先頭に付く。相対
   エンベロープで多少減衰するが、ブレスが相対的に大きいと残る。

ユーザー確認済みの症状: **両方**発生している。

## スコープ決定

- 対策は **VC パスのみ**。同じ録音チャンクが `routes_list` で transcription と vc に
  並列に流れる構成であり、transcription 側は「フィラー・感嘆詞を残す」方針のため、
  recording 側のゲート強化は行わない。
- recording 側 (`silence_threshold`) は**変更なし**。VAD ゲート稼働後もノイズトリガー
  の発話が transcription 側で問題になる場合にのみ別途検討する。

## 採用アプローチ: Silero VAD ゲート (案A)

検討した代替案:

- **案B: 純DSPエネルギーゲート** — 依存ゼロ・数十行だが、閾値がマイクゲイン依存
  (現エンベロープ設計が意図的に排除した性質の再導入) で、大きいブレスと静かな発話を
  区別できない。却下。
- **案C: ノイズ抑制前処理 (DeepFilterNet 等)** — 発話に重なった定常ノイズまで消せる
  唯一の案だが、新規依存が重く HuBERT に入る音色が変わり変換品質に影響し得る。症状
  が「発話と重なる定常ノイズ」ではないため過剰。却下。

Silero VAD (ONNX, ~2MB, MIT ライセンス, CPU で 1 チャンク数 ms) を採用。ブレスを非音
声と判定するのが定番挙動で、マイクゲインに不変。モデルファイルは snakers4/silero-vad
リポジトリの `silero_vad.onnx` (**v5**) を手動取得し config でパス指定する —
`rvc.rmvpe_model_file` と同じ運用。

## アーキテクチャ

新規モジュール **`vspeech/lib/vad.py`** (pure numpy + onnxruntime。GPU/torch 不要で
単体テスト可能):

- `create_vad_session(model_file: Path) -> InferenceSession` — RMVPE の
  `create_rmvpe_session` と同型。ただし **CPUExecutionProvider 固定** (モデルが小さ
  く、RVC と CUDA を取り合わないため)。
- `speech_probs(session, audio_16k: NDArray[np.float32]) -> NDArray` — 16kHz float32
  波形を 512 サンプル (32ms) 窓で逐次推論し、窓ごとの speech 確率列を返す。Silero
  VAD v5 の I/O (`input (1,512)` / `state (2,1,128)` / `sr` int64) を前提とし、state
  はチャンク先頭でリセットする。末尾の 512 未満の端数窓はゼロ埋めして評価する。
- `speech_gate_mask(probs, threshold, pad_ms, min_gain) -> NDArray` — 確率列 → バイ
  ナリ speech マスク → 前後 `pad_ms` の膨張 (子音の頭・語尾の食われ防止) →
  `min_gain`/1.0 のゲイン列。純粋関数。

## vc worker への組み込み (データフロー)

`vspeech/worker/vc.py` の `rvc_worker` ループで、**RVC 推論の前に** VAD を実行する:

```text
speech 受信
  → (vad_gate 有効時) 入力を 16kHz へリサンプル (CPU) → speech_probs
  → speech 比率 (確率 >= vad_threshold の窓の割合) < vad_min_speech_ratio なら:
        RVC をスキップし WorkerOutput を出さない (playback に何も流れない)
        ログ + telemetry ("vc_skip") 記録 → 次のチャンクへ
  → change_voice (従来どおり)
  → apply_input_envelope (従来どおり、有効時)
  → VAD ゲインマスクを出力サンプル軸に np.interp して乗算
```

設計上の要点:

- **スキップ判定が先** — ノイズ単独チャンクは GPU の ~0.55s を使わずに消える。並列
  チェーンの transcription には一切影響しない。
- **RVC への入力は無加工** — 入力をゼロ埋めすると HuBERT 特徴量に不連続が生じチャン
  ク境界でアーティファクトが出るため、可聴性の問題は出力側のダックで解決する。
- 相対エンベロープと違い VAD は**絶対判定**なので、「チャンク全体がノイズだと
  gain≈1 に正規化される」既存の穴を塞ぐ。
- マスクは 32ms 窓解像度 + 線形補間 (エンベロープと同じ正規化時間軸の手法) なのでク
  リックは出ない。入力 (recording レート) と出力 (target_sample_rate) の長さの違い
  は正規化 0..1 軸で吸収する。
- 16kHz へのリサンプルは **worker 側 (`vc.py`) の責務** — `vad.py` は torch 非依存を
  保つ。録音レートが 16kHz ならバイパスし、それ以外は既存の `get_resampler`
  (torchaudio, CPU) を再利用する。

## 設定 (`VcConfig` に追加)

```toml
[vc]
vad_gate = false                 # opt-in、既定は現状維持
vad_model_file = ""              # silero_vad.onnx (v5) のパス
vad_threshold = 0.5              # speech 確率の閾値
vad_min_speech_ratio = 0.1       # これ未満ならチャンクごと RVC スキップ
vad_speech_pad_ms = 100.0        # speech マスクの前後膨張
vad_min_gain = 0.0               # 非音声区間の出力ゲイン (0.0 = 完全ミュート)
```

- vc worker は既に `configs_depends_on=["vc", "rvc"]` なので reload で反映される。
  VAD セッションは worker ループ再入時に (再) 構築する。
- `config.toml.example` に上記を追記する。

## エラー処理

- `vad_gate = true` かつモデルファイルが不在/ロード失敗 → **起動時に fail loudly**
  (`check_cuda_provider` と同じ思想。黙って素通しにするとノイズが復活したことに気づ
  けない)。
- VAD 推論が実行中に例外 → そのチャンクは**ゲートなしで通す** (warning ログ)。配信
  中に声が消える方向の故障は避ける。

## テスト

- `tests/test_vad_gate.py`: `speech_gate_mask` の純粋関数テスト
  (`tests/test_vc_helpers.py` のエンベロープテストと同型) — 膨張幅、`min_gain` 適用、
  スキップ判定比率、出力長不変、端数窓の扱い。
- 実モデル依存テストは golden テスト (`VSPEECH_RVC_GOLDEN_CONFIG`) と同じ方式で
  env var `VSPEECH_VAD_MODEL` 指定時のみ実行: 無音・ホワイトノイズ → 低確率、を確認
  する最小限。
- 既存テスト (`test_vc_helpers.py`, `test_vc_telemetry.py`) は無変更で green を維持
  する (VAD は opt-in)。

## 成功基準

1. `vad_gate = true` で、ブレス・環境音だけの録音チャンクから playback に何も流れな
   い (RVC スキップがログ/telemetry で確認できる)。
2. 発話チャンク内のブレス・環境音区間が出力でミュート (既定 `vad_min_gain = 0.0`)
   され、発話部分の音質・音量は従来と変わらない。
3. transcription 側の挙動 (フィラー保持含む) に一切影響がない。
4. `vad_gate = false` (既定) では従来と完全に同一の出力。
