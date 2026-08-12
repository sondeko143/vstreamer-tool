# RVC ランタイムの torch 完全除去 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> または superpowers:executing-plans。実行の規律は implementer-led-execution に従う。

ADR: [0080](../../adr/0080-torch-free-rvc-runtime.md) (Proposed / 既定)、
[0081](../../adr/0081-ort-native-value-binding.md) (Proposed / 既定)、
[0082](../../adr/0082-rvc-resample-on-inhouse-polyphase.md) (Proposed / 既定)、
[0083](../../adr/0083-cuda-runtime-from-nvidia-wheels.md) (Proposed / 既定)

**実行順: 1 → 7 → 2 → 3 → 4 → 5 → 6。** Task 7 は Task 1 の実測（torch を外すと CUDA EP が
ロードできない）を受けて後から足したもので、plan 全体の前提を左右するため早い段階で潰す。
番号は既存 task を振り直さないために末尾のままにしてある。

Spec: [2026-08-12-rvc-torch-free-runtime-design.md](../specs/2026-08-12-rvc-torch-free-runtime-design.md)

**Goal:** `vspeech/` が torch / torchaudio を一切 import しない状態にし、`rvc` extra から
`torch` / `torchaudio` / `faiss-cpu` を削除して、VC ホストの venv から torch を消す。

**Architecture:** RVC の推論は既に onnxruntime が担っている。torch が持っている役割は
「テンソル配線」「io_binding のゼロコピー」「リサンプル」の 3 つだけなので、前 2 つを
numpy + onnxruntime ネイティブの `OrtValue` へ、3 つ目を既存の自前ポリフェーズ実装へ寄せる。
コードから torch が消えて初めて依存表から外せ、依存表から外して初めて実測の削減が出る
（`ctranslate2` がインストール済みの torch を掴むため）。

**Tech Stack:** Python 3.14 / uv / numpy 2 / onnxruntime-gpu 1.27 / pytest / ruff / ty

## Global Constraints

- Python は **3.14 のみ**（`requires-python = ">=3.14,<3.15"`）。依存操作は uv で行う。
- `uv sync --extra rvc` を単独で実行しない（他の extra が外れる）。**`uv sync --all-extras` を使う。**
  ad-hoc な Python 実行も `uv run --all-extras` を使う。
- 新規の `#` コメントと docstring は**英語**で書く。ユーザーが読む文字列（ログ・例外メッセージ・
  `config.py` の `description=`・argparse/click の help）は**日本語のまま**。
  触ったファイルに日本語コメントが残っていたら、そのファイルのコメントは同じ変更で英訳する。
- import は 1 行 1 つ（ruff `force-single-line = true`）。
- pydantic v2 API のみ。v1 API（`parse_obj` / `.dict()` / `root_validator` / `json_encoders` 等）を使わない。
- GPU 対応 onnxruntime セッションは `vspeech/lib/onnx_session.py` の `create_session` からのみ開く。
  `vspeech/lib/vad.py`（Silero VAD、CPUExecutionProvider 固定）だけが意図的な例外。
  **ファクトリを二重化しない。**
- `vspeech/` は `fairseq` / `transformers` / `pydantic_settings` を import しない。本 plan 完了後は
  `torch` / `torchaudio` も同じ扱いになる。
- 検証コマンドの成否は**パイプを通さずに終了コードで判定する**。`cmd | tail -1 && echo $?` の
  `$?` は `tail` の結果であって `cmd` の結果ではない。pytest は完全な node ID で指定する。
- 性能・メモリの主張には **N（試行回数）を明示する**。1 回の実行は測定ではない。
- 実機資産（この開発機に存在する）:
  - 発話単位 VC の設定: `~/.config/vstreamer/config_vc.toml`
  - ストリーミング VC 送信側の設定: `~/.config/vstreamer/config_stream_producer.toml`
  - GPU: index 0 = RTX 4060 Laptop、index 1 = RTX 5060 Ti（driver 610.88）

## Implementer Authority

この plan が拘束するのは 3 つだけ: **公開契約**（各 task の「契約」欄の名前・型・方向）、
**Global Constraints の逐語値**、**各 task の受入基準**。

それ以外 — 内部設計、関数・ファイルの分割、命名、アルゴリズム、テストの設計と粒度、
エラー処理の形、依存の使い方 — はすべて実装者が決める。plan に書かれていない実装を
選んだことは逸脱ではない。

この plan が参照する ADR のうち、実装者を拘束するのは **効力: 制約** のものだけ。**既定** と
**便宜** は「なぜ今こうなっているか」の記録であって、守る義務はない。より良い方法があれば
変えてよい。変えたら report の逸脱1行に書く。

plan の記述より良い方法を見つけたら、良い方を採る。plan は使い捨てなので書き換えない。
その選択が adr-writing の基準に当たるならトリガ2 で ADR を起票する。

停止して人間に確認するのは、上の 3 つのいずれかを**変える必要がある**と判断したときだけ。

---

### Task 1: 変更前ベースラインの採取と比較方式の確定

**目的:** torch 版の出力・レイテンシ・常駐/起動を実機で採取し、以降の A/B 比較で使う
判定方式を実測に基づいて確定する。

**範囲:** `scripts/`、`.gitignore`。`vspeech/` は変更しない。

**契約**
- Produces: ストリーミング変換のベースライン成果物（入力ブロック列・seed・出力ブロック列を含む）と、
  それを現行実装と突き合わせる比較エントリポイント。設定ファイルのパスは既存 golden と同じく
  **環境変数で外から渡す**（リポジトリに機材固有パスを置かない）。
- Produces: 「比較方式レポート」— 判定に使う方式（bit 一致か、相関/SNR か）と、後者の場合の閾値。
  Task 3 はこのレポートの方式で判定される。
- Consumes: なし。

**受入基準**
- [ ] 同一入力・同一 seed で**別プロセスとして 2 回**採取した Stream VC の出力が bit 一致するか
      どうかが測定され、結果（一致 / 不一致なら最大差分・相関・SNR）が記録されている。
- [ ] bit 一致しない場合、Task 3 の判定に使う方式と閾値が、この自己ノイズの実測値に基づいて
      決められ、根拠とともに記録されている。
- [ ] 変更前の Stream VC の 1 推論あたりレイテンシ p50 / p95 が **N≥200 tick** で記録されている。
- [ ] 変更前の VC パイプラインの常駐メモリと起動時間が記録されている。測定手順も記録されている
      （Task 6 が torch 非導入で同じ手順を再実行するため）。
- [ ] ベースライン成果物（バイナリ）は gitignore され、リポジトリに入らない。

**検証**
- `uv run --all-extras pytest` の終了コードが 0。
- 採取コマンドを実機で実行し、上記 4 種の記録が生成されること。
- `uv run poe check` の終了コードが 0（既知の受容済み指摘を除く）。

**コミット単位:** 採取・比較ハーネスの追加で 1 コミット。ベースライン成果物自体はコミットしない。

---

### Task 2: RVC 経路のリサンプルを自前ポリフェーズへ一本化（ADR-0082）

**目的:** torchaudio への依存を変換経路から取り除く。

**範囲:** `vspeech/lib/rvc.py`、`vspeech/worker/vc.py`、対応するテスト。

**契約**
- Consumes: `vspeech/lib/resample.py` の `PolyphaseResampler` / `make_resampler`（既存、変更しない）。
- Produces: `vspeech/lib/rvc.py` から `get_resampler` が無くなる。発話単位 VC と VC ワーカーの
  VAD 前処理は、どちらも `vspeech/lib/resample.py` の実装を通る。
- Produces: `change_voice` の公開シグネチャは変更しない（`voice_sample_rate` 引数を含め現状維持）。

**受入基準**
- [ ] `vspeech/` のどのモジュールも `torchaudio` を import しない。
- [ ] 入力サンプルレートが 16000 のとき、リサンプルを行わない（変換前と同じく素通し）。
- [ ] 入力サンプルレートが 16000 以外のとき、`change_voice` は変換前と同じ長さの出力を返す。
- [ ] VC ワーカーの VAD 前処理が、rvc extra 非導入の環境でも 16kHz 入力に対して動作する
      （変換前から保たれている性質を壊さない）。
- [ ] 既存の numeric golden を実機で実行した結果（相関・SNR の実測値、PASS / FAIL）が記録されている。
      FAIL の場合は閾値を緩めず、実測値を記録した上で golden を再取得し、その旨を記録する。

**検証**
- `uv run --all-extras pytest` の終了コードが 0。
- `VSPEECH_RVC_GOLDEN_CONFIG=~/.config/vstreamer/config_vc.toml` を設定して
  `uv run --all-extras pytest tests/test_change_voice_golden.py::test_change_voice_matches_seeded_golden`
  を実行し、終了コードと相関・SNR の実測値を記録する。
- `uv run ruff format .` / `uv run ruff check .` / `uv run ty check` の終了コードが 0。

**コミット単位:** リサンプラ差し替えで 1 コミット。golden 再取得が必要になった場合は別コミット。

---

### Task 3: 変換経路を numpy + onnxruntime ネイティブへ移す（ADR-0081）

**目的:** テンソル配線と io_binding から torch を取り除き、`vspeech/` を torch-free にする。

**範囲:** `vspeech/lib/rvc.py`、`vspeech/lib/pitch_extract.py`、`vspeech/lib/stream_vc.py`、
`vspeech/worker/vc.py`、`vspeech/stream_vc/runner.py`、対応するテスト。

**契約**
- Consumes: Task 1 の比較方式レポートとベースライン成果物。
- Consumes: `vspeech/lib/cuda_util.py` の `Device`（既存、変更しない）。
- Produces: `pitch_extract` および `pitch_extract_rmvpe` / `pitch_extract_fcpe` の `audio` 引数が
  numpy 配列を受け取る。torch の `Tensor` は受け取らない。
- Produces: `vspeech/lib/rvc.py` の `change_voice` / `load_hubert_model` の公開シグネチャは
  変更しない（`device: Device` を受け取り、`NDArray[np.int16]` を返す）。
- Produces: `vspeech/lib/rvc.py` から `_torch_device` が無くなる。
- Produces: `StreamingVc` の `__init__` 引数と `process_block(block) -> NDArray[np.int16]` /
  `warmup()` / `emit_delay_samples` の公開契約は変更しない。
- Produces: `next_context` / `crossfade_weights` / `overlap_add` / `sola_offset` の公開契約は変更しない。

**受入基準**
- [ ] `vspeech/` のどのモジュールも `torch` を import しない。
- [ ] `vspeech.stream_vc.consumer` / `udp` / `jitter` / `wire` を import しても torch が
      ロードされない（ADR-0055 の性質を維持）。
- [ ] `vspeech.lib.cuda_driver` / `cuda_util` / `onnx_session` を import しても torch が
      ロードされない（ADR-0078 の性質を維持）。
- [ ] CUDA デバイスで Stream VC の 1 ブロック変換出力が、Task 1 のベースラインに対して
      Task 1 が確定した方式で合格する。実測値が記録されている。
- [ ] CPU デバイス（`device.type != "cuda"`）でも変換が動作する。
- [ ] Stream VC の 1 推論あたりレイテンシ p50 が、Task 1 のベースラインの **+5% 以内**。
      p50 / p95 の実測値が **N≥200 tick** で記録されている。+5% を超えた場合は
      ADR-0081 が留保している device 側バッファ再利用を足して再測定し、その結果も記録する。
- [ ] f0 抽出器 `rmvpe` と `fcpe` の両方で変換が動作する。
- [ ] 設定スキーマに破壊的変更がない（既存の `~/.config/vstreamer/config_vc.toml` と
      `config_stream_producer.toml` がそのまま読める）。

**検証**
- `uv run --all-extras pytest` の終了コードが 0。
- `uv run --all-extras pytest tests/test_forbidden_imports.py` の終了コードが 0。
- Task 1 の比較エントリポイントを実機で実行し、判定結果と実測値を記録する。
- Task 1 のレイテンシ採取を同じ N で再実行し、p50 / p95 を記録する。
- `uv run ruff format .` / `uv run ruff check .` / `uv run ty check` の終了コードが 0。

**コミット単位:** 変換経路の numpy 化で 1 コミット。レイテンシ基準を満たすための追加最適化が
必要になった場合は別コミット。

---

### Task 4: テストの torch 依存整理と構造ガードの拡張

**目的:** ランタイム側テストから torch を落とし、torch がランタイムへ戻らないことを構造的に固定する。

**範囲:** `tests/`。

**契約**
- Consumes: Task 3 が確定した `pitch_extract` / `StreamingVc` の公開契約。
- Produces: `tests/test_forbidden_imports.py` の `FORBIDDEN` に `torch` と `torchaudio` が含まれる。
- Produces: torch を必要とするテストは、オフラインの ONNX 生成ツールを対象とするものだけになり、
  torch 非導入環境では skip される（fail しない）。

**受入基準**
- [ ] `FORBIDDEN` に `torch` / `torchaudio` が含まれ、`vspeech/` 全体を対象とするガードが緑になる。
- [ ] ランタイムの挙動を検証するテスト（変換経路・f0 抽出・ストリーミング）が torch を import しない。
- [ ] オフライン ONNX 生成ツールのテストは、torch が無い環境で skip され、
      テスト実行全体の終了コードを 0 のままにする。
- [ ] 変換経路の numeric golden テストが、torch 非導入環境でも収集時にエラーにならない。
- [ ] テスト総数が Task 3 完了時点から減っていない（skip は減少に数えるので、
      skip になった件数と理由を記録する）。

**検証**
- `uv run --all-extras pytest` の終了コードが 0。
- torch を import 不能にした状態でのテスト実行（手段は実装者が決める）で終了コードが 0。
- `uv run ruff check .` / `uv run ty check` の終了コードが 0。

**コミット単位:** テスト整理と構造ガード拡張で 1 コミット。

---

### Task 5: 依存の削除とオフラインツールのオーバーレイ退避（ADR-0080）

**目的:** `rvc` extra から `torch` / `torchaudio` / `faiss-cpu` を外し、venv から torch を消す。

**範囲:** `pyproject.toml`、`uv.lock`、`poe_tasks.toml`、`CLAUDE.md`。

**契約**
- Consumes: Task 3 / Task 4 の完了（コードとテストが torch-free であること）。
- Consumes: Task 7 の完了（CUDA ランタイムの供給元が torch 以外に確立していること）。
- Produces: `rvc` extra が `torch` / `torchaudio` / `faiss-cpu` を宣言しない。
- Produces: `poe export-hubert-onnx` と `poe export-fcpe-onnx` が、プロジェクト環境に torch が
  無い状態でも実行できる（`uv run --with` のオーバーレイから torch を供給する）。
- Produces: `CLAUDE.md` の記述が実態と一致する（torch の位置づけ、rvc extra の内容、
  オフラインツールの実行方法）。

**受入基準**
- [ ] `uv sync --all-extras` した venv に `torch` / `torchaudio` / `faiss-cpu` が含まれない。
- [ ] `uv lock --check` の終了コードが 0。
- [ ] `python -m vspeech --config ~/.config/vstreamer/config_vc.toml` が起動し、
      プロセス内で torch がロードされない。
- [ ] `python -m vspeech --config ~/.config/vstreamer/config_stream_producer.toml` が起動し、
      プロセス内で torch がロードされない。
- [ ] `poe export-fcpe-onnx` がオーバーレイ経由で完走し、生成物が変更前と同等であることが
      確認されている（同等性の判定方法も記録する）。
- [ ] `poe export-hubert-onnx` の実行手段がオーバーレイに移っており、実行して完走するか、
      実行できない場合はその理由（GPU / 資産の要件）が記録されている。
- [ ] `uv audit` の結果が変更前より悪化していない。

**検証**
- `uv sync --all-extras` の後、venv のパッケージ一覧に上記 3 つが無いことを確認する。
- `uv lock --check` の終了コードが 0。
- 上記 2 つの設定でパイプラインを起動し、起動ログと torch 非ロードを確認する。
- `uv run --all-extras pytest` の終了コードが 0。

**コミット単位:** 依存削除で 1 コミット、poe タスクのオーバーレイ退避で 1 コミット、
`CLAUDE.md` 更新で 1 コミット。

---

### Task 6: 実機検証・実測値の記録・ADR の昇格

**目的:** spec の受入基準すべてに実測の裏づけを与え、決定層を確定させる。

**範囲:** `docs/adr/`、`docs/adr/README.md`。

**契約**
- Consumes: Task 1〜5 および Task 7 の記録（比較結果・レイテンシ・golden・依存状態）。
- Produces: ADR-0080 / 0081 / 0082 / 0083 の Status が、実装の結果に応じて `Accepted` へ昇格するか、
  覆った場合は supersede される。

**受入基準**
- [ ] torch 非導入の VC パイプラインについて、常駐メモリと起動時間が Task 1 と**同じ手順**で
      測定され、削減量が記録されている。
- [ ] spec の受入基準 9 項目すべてについて、満たしたことの根拠（実測値または検証コマンドの
      終了コード）が対応づけて記録されている。満たせなかった項目があれば、その項目と理由が
      明示されている。
- [ ] ADR-0080 / 0081 / 0082 / 0083 の Status が `Proposed` のまま残っていない。
- [ ] ADR-0083 の 2 つの追記が Context / Consequences 本文へ畳み込まれている（Accepted になると
      本文は不変になるため、訂正を末尾に積んだまま昇格させない）。
- [ ] `docs/adr/README.md` の索引の Status 列が各 ADR と一致している。
- [ ] 実機での聴感確認が必要な項目（発話単位 VC の音質、ストリーミング VC の音質）が、
      ユーザーに渡せる形で明示されている。

**検証**
- `uv run --all-extras pytest` の終了コードが 0。
- `uv run poe check` の終了コードが 0（既知の受容済み指摘を除く。除いたものは列挙する）。
- 記録した実測値と ADR の記述を突き合わせ、乖離がないことを確認する。

**コミット単位:** ADR の Status 昇格と索引更新で 1 コミット。

---

### Task 7: CUDA ランタイムの供給元を torch から nvidia wheel へ移す（ADR-0083）

**目的:** torch を外しても onnxruntime の CUDA 実行プロバイダが立つ状態を、依存を削る前に成立させる。

**実行順:** Task 1 の直後（Task 2 より前）。この task が成立しなければ plan 全体の前提が崩れるため、
先に潰す。番号だけが末尾にある。

**範囲:** `pyproject.toml`、`uv.lock`、`vspeech/lib/onnx_session.py`、対応するテスト。

**契約**
- Consumes: `vspeech/lib/onnx_session.py` の `create_session`（既存の単一ファクトリ。
  [ADR-0024](../../adr/0024-onnx-session-single-factory.md) の「ファクトリを二重化しない」を維持する）。
- Consumes: `vspeech/worker/vc.py` と `vspeech/stream_vc/runner.py` が既に呼んでいる
  `check_cuda_provider`（既存、fail loud の担い手。変更しない）。
- Produces: `rvc` extra が CUDA ランタイム（cuBLAS / cuDNN ほか onnxruntime の CUDA EP が要求するもの）を
  供給する nvidia wheel を宣言する。
- Produces: GPU 対応セッションを開く経路が CUDA ライブラリのロードを保証する。保証する場所は 1 箇所だけ。

**受入基準**
- [ ] torch を import 不能にした状態で、実機の RVC デコーダ / HuBERT / f0 の各 ONNX 資産に対して
      GPU セッションが開き、プロバイダ一覧に `CUDAExecutionProvider` が含まれる。
- [ ] torch が存在する環境でも同じ経路が `CUDAExecutionProvider` を返し、CUDA DLL の二重ロードが
      起きない。
- [ ] CUDA ライブラリが供給されない状態では起動時に fail loud する。CPU 実行へ黙って落ちない。
- [ ] `uv lock --check` の終了コードが 0。
- [ ] `uv audit` の結果が変更前より悪化していない。
- [ ] nvidia wheel 追加による venv のディスク増加量と、パイプラインの常駐メモリの変化が
      実測で記録されている（Task 1 と同じ `vc-footprint` 手順を使う）。

**検証**
- torch を import 不能にした子プロセスで GPU セッションを開き、プロバイダ一覧を確認する
  （成否は終了コードで判定する）。
- `uv run --all-extras pytest` の終了コードが 0。
- `uv lock --check` の終了コードが 0。
- `uv run ruff format .` / `uv run ruff check .` / `uv run ty check` の終了コードが 0。
- `uv run poe vc-footprint --config ~/.config/vstreamer/config_vc.toml --runs 3 --settle 10` を
  Task 1 と同じ手順で実行し、常駐メモリを記録する。

**コミット単位:** nvidia wheel の追加と、CUDA ライブラリ読み込みの保証で 1 コミット。
