# Architecture Decision Records (ADR)

このディレクトリは VStreamer Tool の**アーキテクチャ決定記録**を保持する。ADR は 3-doc モデルの中央層を担う不変記録で、決定が覆るときは書き換えず、新しい ADR で supersede する。

- **spec**（薄・不変寄り / [`../superpowers/specs/`](../superpowers/specs)）: 問題・ゴール・非ゴール・受入基準の4節だけ。実装で解決すると陳腐化しにくい記述に絞る。
- **ADR**（不変 / このディレクトリ）: 決定と却下した代替案。同期し続けず、覆すときに新 ADR で supersede する。
- **plan**（使い捨て / [`../superpowers/plans/`](../superpowers/plans)）: 手順・影響ファイル・テスト。実装後すぐ陳腐化する前提で、参照・同期しない。

## 起票のしかた

1. 採番は `docs/adr/` 直下の既存最大番号 +1（4桁ゼロ埋め）。ファイル名は `NNNN-kebab-title.md`。
2. 本文は [`template.md`](template.md) に従う（Context / Decision / **Alternatives rejected（必須）** / Consequences + Status/Date/Related）。
3. Status は起票時 `Accepted`（提案段階なら `Proposed`）。
4. この索引表に 1 行追記する。
5. 覆すときは旧 ADR の Status を `Superseded by [ADR-NNNN]` にする 1 行だけ更新し、新 ADR を起こして Related に旧 ADR を書く（本文は書き換えない）。

## 索引

| ADR | 決定 | Status | 日付 |
|----:|------|--------|------|
| [0001](0001-upgrade-voicevox-core-0-16.md) | voicevox-core を 0.16.4 へ上げ pydantic v2 移行と分離する | Accepted | 2026-06-14 |
| [0002](0002-migrate-to-pydantic-v2-native.md) | Pydantic を v2 ネイティブへ全面移行する | Accepted | 2026-06-14 |
| [0003](0003-secretstr-json-via-field-serializer.md) | SecretStr の平文 JSON シリアライズに field_serializer を使う | Accepted | 2026-06-14 |
| [0004](0004-per-destination-sender-transport.md) | sender を宛先ごと並行・永続チャネル・有界キューに再構成する | Accepted | 2026-06-14 |
| [0005](0005-true-e2e-in-process-telemetry.md) | 真 E2E を wire 伝搬で測り消費はプロセス内自己集計に限定する | Accepted | 2026-06-23 |
| [0006](0006-clock-skew-threshold-warning.md) | クロック skew は自動同期せず閾値警告で検知する | Accepted | 2026-06-23 |
| [0007](0007-per-request-jsonl-telemetry.md) | 発話ごとテレメトリを 1レコード=1段・opt-in・耐障害で JSONL 追記する | Accepted | 2026-06-23 |
| [0008](0008-code-metrics-two-lens.md) | 複雑度を lizard+complexipy の2レンズで測り radon/MI/wily/SaaS を却下する | Accepted | 2026-06-24 |
| [0009](0009-code-metrics-advisory-only.md) | code-metrics は advisory 専用としゲートしない | Accepted | 2026-06-24 |
| [0010](0010-python-health-on-demand-skill.md) | 健全性ゲートを GitHub Actions でなく手元 on-demand skill にする | Accepted | 2026-06-24 |
| [0011](0011-python-health-triage-hybrid.md) | 健全性の修正はトリアージ型ハイブリッド（ruff の2種のみ自動）にする | Accepted | 2026-06-24 |
| [0012](0012-adopt-ty-type-checker.md) | 型チェッカに ty を採用し pyright を置換する | Accepted | 2026-06-24 |
| [0013](0013-skills-at-user-level.md) | 再利用スキルをユーザーレベル ~/.claude/skills に配置する | Superseded by [0014](0014-relocate-skills-to-project.md) | 2026-06-24 |
| [0014](0014-relocate-skills-to-project.md) | スキル資産をプロジェクトレベル（.claude/skills + scripts）へ移設する | Accepted | 2026-06-25 |
| [0015](0015-startup-profiler-py-spy.md) | 起動プロファイラに py-spy を spawn+--subprocesses+--idle で使う | Accepted | 2026-06-25 |
| [0016](0016-change-voice-decompose-seeded-golden.md) | change_voice を純粋ヘルパへ分解し seeded 厳密 golden で検証する | Accepted | 2026-06-25 |
| [0017](0017-rvc-input-envelope-shape-transfer.md) | RVC 音量整合を入力平均正規化 RMS シェイプ転写で行う（マイクゲイン非依存） | Accepted | 2026-07-04 |
| [0018](0018-rvc-envelope-duck-only.md) | エンベロープゲインをダック(≤1.0)に限定し max_gain 既定を 1.0 にする | Accepted | 2026-07-04 |
| [0019](0019-vc-silero-vad-gate.md) | VC ノイズ対策に Silero VAD ゲートを採用し VC パスに限定する | Accepted | 2026-07-08 |
| [0020](0020-silero-vad-v6.md) | Silero VAD モデルを v6.2.1 へ更新する（0019 の v5 pin を更新） | Accepted | 2026-07-08 |
| [0021](0021-hubert-drop-fairseq.md) | RVC content encoder を fairseq から transformers.HubertModel へ外す | Accepted | 2026-07-09 |
| [0022](0022-hubert-onnx-runtime.md) | HuBERT content encoder を ONNX 化し fairseq/transformers を lock から撤去する | Accepted (refines [0021](0021-hubert-drop-fairseq.md)) | 2026-07-10 |
| [0023](0023-hubert-equivalence-gate.md) | HuBERT 置換の正しさを特徴量数値等価ゲートで担保する | Accepted | 2026-07-10 |
| [0024](0024-onnx-session-single-factory.md) | 全 onnxruntime セッションを単一 create_session で開き device を尊重する | Accepted | 2026-07-10 |
| [0025](0025-target-python-314-phased.md) | 3.13 を経由せず段階移行で Python 3.14 を目標にする | Accepted | 2026-07-12 |
| [0026](0026-adopt-numpy-2.md) | numpy 2 を採用する（>=2,<3） | Accepted | 2026-07-12 |
| [0027](0027-cap-onnxruntime-cuda12.md) | onnxruntime-gpu を一時 <1.27 cap で CUDA 12 を凍結する | Superseded by [0028](0028-migrate-to-cuda-13.md) | 2026-07-12 |
| [0028](0028-migrate-to-cuda-13.md) | CUDA 13 へ移行する（torch cu130 / onnxruntime 1.27 / driver R580+） | Accepted | 2026-07-12 |
| [0029](0029-audioop-lts.md) | audioop を audioop-lts へ置換する | Accepted | 2026-07-12 |
| [0030](0030-pyworld-lazy-default-rmvpe.md) | pyworld を遅延 import 化し既定 f0 抽出器を rmvpe にして rvc extra から撤去する | Accepted | 2026-07-12 |
| [0031](0031-audio-pyaudio-to-sounddevice.md) | audio extra を PyAudio から sounddevice へ移行する | Accepted | 2026-07-12 |
| [0032](0032-gui-multi-pipeline-rewrite.md) | GUI を複数 pipeline マネージャへ全面書き直しする | Accepted | 2026-07-12 |
| [0033](0033-gui-manifest-versioning.md) | GUI の version/migration を専用マニフェストに隔離し config は純粋 Config 形状を保つ | Accepted | 2026-07-12 |
| [0034](0034-gui-corrupt-file-resilience.md) | 壊れた GUI 入力に対し対象別復旧＋非破壊退避で必ず起動する | Accepted | 2026-07-12 |
| [0035](0035-bound-sender-reconnect-backoff.md) | sender の永続チャネルの再接続バックオフを有界化する（0004 を refine） | Accepted | 2026-07-15 |
| [0036](0036-whisper-resample-via-pyav.md) | whisper のリサンプルに PyAV(libswresample) を使い torchaudio/scipy を却下する | Accepted | 2026-07-16 |
| [0037](0037-transcription-vad-skip-gate.md) | Silero VAD スキップゲートを transcription パスへ拡張する（独立ワーカーを却下、0019 を拡張） | Accepted (extends 0019) | 2026-07-16 |
| [0038](0038-worker-config-preflight-fail-loud.md) | 設定不備は起動時 preflight で fail-loud に集約する（全 worker へ一般化、0019 を一般化） | Accepted | 2026-07-16 |
| [0039](0039-whisper-hosts-need-cuda12-toolkit.md) | whisper GPU ホストに CUDA 12 ツールキット（cuBLAS + cuDNN 9）を要求する（0028 を refine） | Accepted | 2026-07-16 |
| [0040](0040-subtitle-obs-backend-via-worker-type.md) | 字幕の OBS 出力を新 EventType ではなく subtitle.worker_type のバックエンドとして足す | Accepted | 2026-07-16 |
| [0041](0041-subtitle-obs-config-authority.md) | OBS バックエンドは config を表示スタイルの権威とし、OBS の構造には触れない | Accepted | 2026-07-16 |
| [0042](0042-subtitle-obs-failure-tiers.md) | OBS 接続の失敗を「観測できたものだけ即死」で階層化する（0038 を refine） | Accepted | 2026-07-16 |
| [0043](0043-obs-websocket-client-in-house.md) | obs-websocket クライアントを websockets 上に自前実装する（simpleobsws を却下） | Accepted | 2026-07-16 |
| [0044](0044-font-size-tk-points-to-obs-lfheight.md) | font_size を Tk の符号規約のまま OBS へ渡し正値だけ 96 DPI でピクセル換算する（0041 を refine） | Accepted (refines 0041) | 2026-07-17 |
| [0045](0045-gui-readiness-reuses-preflight.md) | GUI の起動前 readiness は vspeech.preflight を単一の権威として再利用する（GUI 側に必須項目マップを持たない、0038 を再利用） | Accepted | 2026-07-17 |
| [0046](0046-gui-shared-asset-paths-explicit-propagate.md) | マシン共通の素材パスは default.toml で編集し明示 propagate する（pipeline config の自己完結を保つ、0032 を維持） | Accepted | 2026-07-17 |
| [0047](0047-obs-identify-unsubscribes-from-events.md) | Identify で eventSubscriptions=0 を明示し、読まないイベントを OBS に送らせない（0043 を refine） | Accepted (refines 0043) | 2026-07-18 |
| [0048](0048-gcp-auth-channel-with-retrying-session.md) | GCP のトークン更新を retry 付き session に載せるため認証チャネルを自前で組む | Accepted | 2026-07-18 |
| [0049](0049-fcpe-baked-waveform-onnx-f0-extractor.md) | FCPE を波形入力 ONNX f0 抽出器としてスパイク先行で追加する | Proposed | 2026-07-21 |
