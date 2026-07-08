# VC VAD モデル v5→v6 移行設計 (2026-07-08)

関連: [VC VAD ノイズゲート設計](2026-07-08-vc-vad-noise-gate-design.md)（v5 での初期実装。本書はその後継モデルへの差し替え）。

## 背景 / 動機

VAD ノイズゲート (`vspeech/lib/vad.py` + `vspeech/worker/vc.py`) は Silero VAD
**v5.1.2** の ONNX モデルで実装・実機検証済みで稼働している。動機は**精度・ノイズ
除去の向上** — snakers4/silero-vad は v6 系 (最新 v6.2.1, 2026-02-24) に更新されて
おり、ブレス・環境音の非音声判定が改善していることを期待して後継モデルへ移行する。

初期実装時 (2026-07-08) の記録には「repo master は v6 で、この実装は v5 の
`state`-入力契約を前提とするため v6 は使うな」という注記があったが、これは**未検証の
防御的仮定**だった。本移行に先立ち実アーティファクトを検証した結果、この仮定は誤り
であることが判明した (下記)。

## 事前検証で確定した事実 (推測ではなく実測)

`~/.config/vstreamer/silero_vad.onnx` (現行 v5.1.2) と v6.2.1 の実モデルを
onnxruntime で直接検査・実行した結果:

1. **ONNX I/O 契約は v5.1.2 と v6.2.1 で完全一致**
   - INPUTS: `input` `[None,None]` f32 / `state` `[2,None,128]` f32 / `sr` `[]` int64
   - OUTPUTS: `output` `[None,1]` f32 / `stateN` f32
   - v5 スタイルの feed (`input`=64 context+512 window, `state`=zeros(2,1,128),
     `sr`=16000) が v6 でそのまま動作する。
2. **ノイズ判別はむしろ向上** — 同一入力での speech 確率:
   silence 0.0120 → **0.0017**、white-noise 0.0898 → **0.0105** (v6 の方が非音声を
   より確実に非音声と判定)。動機に合致。
3. **推論ラッパー・検証ロジックは無改修で v6 対応** — `create_vad_session` の
   `"state" in input_names` 検証は v5/v6 を通し v4 (h/c 入力) を弾く。正しいまま。
4. **pin 対象の同定** — v6 系の点リリース (v6.0/v6.1/v6.2/v6.2.1) はモデル本体不変。
   pin 対象は **v6.2.1** (最新リリース):
   - sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
   - サイズ 2327524 bytes (v5.1.2 と同サイズだが別モデル。v5.1.2 の sha256 は
     `2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f`)
   - このリリースタグのモデルは repo master (HEAD) と bit 一致する。

結論: 本移行は「モデルの差し替え + 表記 (docstring/コメント/設定) の更新 + 実機再
確認」に収束する。推論コードの機能変更は不要。

## スコープ

- 対象は **VC パスの VAD ゲートで使うモデルの差し替えのみ**。ゲートのアルゴリズム
  (スキップ判定・出力ダック・窓解像度マスク) は初期設計から**変更なし**。
- 検証方針は v5 と同じ**「差し替えて実機で確認」** (自動 real-model テスト +
  実機での耳確認)。定量 A/B ハーネスは作らない。
- 歴史的記録である [初期 v5 spec](2026-07-08-vc-vad-noise-gate-design.md) および
  対応する plan は**改変しない** (その作業の日付付き記録)。

## 変更内容

### 1. モデルアーティファクト (pin & 取得)

- pin: Silero VAD **v6.2.1** の
  `src/silero_vad/data/silero_vad.onnx` (sha256 `1a153a22…`, 2327524 bytes)。
- 取得 URL:
  `https://github.com/snakers4/silero-vad/raw/refs/tags/v6.2.1/src/silero_vad/data/silero_vad.onnx`
- 配置: 現行 config (`config_vc.toml`) が指す `~/.config/vstreamer/silero_vad.onnx`
  を v6 に差し替える。差し替え前に現行 v5 を `~/.config/vstreamer/silero_vad_v5.onnx`
  へ退避 (ロールバック用)。取得後に sha256 を照合する。
- モデルファイルは gitignore・非ベンダリング (`rvc.rmvpe_model_file` と同じ運用) の
  ため、これはローカル手順であり実装時に取得 + sha 照合を行う。

### 2. コード `vspeech/lib/vad.py` (表記のみ・挙動不変)

- module docstring / L19 コメント / `create_vad_session` docstring (L80/82) /
  `FileNotFoundError` メッセージ (L94-95) / `ValueError` メッセージ (L108-109) /
  `speech_probs` docstring (L117) の "v5" 記述を更新する。
- 文言は「**v5/v6 は 16kHz・512 サンプル窓・64 context・(2,1,128) state の契約を
  共有する。本実装は v6.2.1 を pin 対象とする**」という共有契約ベースに書き換える
  (将来正確かつ v4 拒否の意図も保つ)。
- `"state" in input_names` の検証ロジックは**そのまま** (v5/v6 accept, v4 reject)。

### 3. 設定

- `config.toml.example` L84-88 の VAD 節コメント "(v5)" を "(v6.2.1)" に更新し、
  取得元タグ/URL の記述を合わせる。
- `VcConfig` の VAD 既定値
  (`vad_threshold=0.5`, `vad_min_speech_ratio=0.1`, `vad_speech_pad_ms=100.0`,
  `vad_min_gain=0.0`, `vad_gate=false`) は**据置**。v6 は判定がより決定的なため v5
  値を維持し、実機で再確認して問題があるときのみ調整する。

### 4. テスト `tests/test_vad_gate.py`

- `test_real_model_silence_and_noise_score_low` は env-gated (`VSPEECH_VAD_MODEL`)
  でバージョン非依存。v6 モデルで通過する (silence/noise とも実測で閾値を大きく
  下回る)。自動ゲートとして維持し、v6 を対象とする旨のコメントを添える。
- 純粋関数テストと `test_vc_config_vad_defaults_are_off_and_sane` は既定値据置の
  ため無変更で green。

## 検証

1. **自動 real-model テスト** (v6 ファイルに対して):
   `VSPEECH_VAD_MODEL=~/.config/vstreamer/silero_vad.onnx uv run pytest tests/test_vad_gate.py -v`
   → real-model テスト含め全通過。
2. **フルゲート**: 既存の `poe check` / `poe fix` 相当
   (`uv run ruff format`・`ruff check`・`ty check`・`pytest`)。
3. **実機確認**: `vad_gate=true` + v6 モデルで、ブレス・環境音の録音チャンクが
   スキップされ (`vc_skip` ログ)、発話部分の音質・音量が従来どおりであることを
   耳で確認 (v5 検証と同手順)。

## ドキュメント & メモリ

- 本 spec を新規 dated ファイルとしてコミット。続いて writing-plans で移行 plan を
  起こす。
- プロジェクトメモリ `vc-vad-noise-gate.md` を更新: pin 対象は v6.2.1、v6 は契約
  互換 (「v6 do NOT use」注記を訂正)、sha256 を記録。

## スコープ外 (YAGNI)

- 定量 A/B ハーネス、合成音プローブ (検証は scratchpad 限定・非コミット)。
- 閾値の再チューニング (実機で問題が出た場合のみ別途)。
- v5/v6 のランタイム切替や複数モデル同時サポート。
- 8kHz 経路 (録音レートは 16kHz)。

## リスク / ロールバック

- 変更の実体はモデルファイルの差し替え。コードは版非依存 (契約共有) のため、
  実機で v6 が悪ければ退避した `silero_vad_v5.onnx` を元パスに戻すだけでロール
  バック完了 (コード変更の巻き戻し不要)。

## 成功基準

1. v6.2.1 モデル (sha256 照合済み) で `vad_gate=true` の real-model テストが通過。
2. フルゲート (ruff/ty/pytest) が green。
3. 実機で、ブレス・環境音のスキップと発話の無傷保持が v5 と同等以上。
4. `vad_gate=false` (既定) では従来と完全に同一の出力。
