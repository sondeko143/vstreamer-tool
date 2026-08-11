# デバイス解決層の torch 非依存化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> または superpowers:executing-plans。実行の規律は implementer-led-execution に従う。

ADR: [0078](../../adr/0078-torch-free-device-resolution.md) (Proposed / 既定)、
[0079](../../adr/0079-fp16-by-compute-capability.md) (Proposed / 既定)

**Goal:** GPU デバイスの解決と ONNX の CUDA プロバイダ判定から torch を外し、音声認識パイプラインが torch を読み込まずに動くようにする。

**Architecture:** デバイスを表す値型を torch から自前のものに置き換え、GPU の列挙を CUDA Driver API に移す。torch を使い続ける変換経路（RVC）は、その入口で自前の値型を `torch.device` に変換する。これにより torch の import が変換経路の内側だけに閉じる。

**Tech Stack:** Python 3.14 / ctypes（`nvcuda.dll`）/ onnxruntime-gpu / pytest

## Global Constraints

- 依存パッケージを追加しない。`pyproject.toml` の依存宣言に新しいパッケージを足さない。
- コメントと docstring は英語で書く。ログメッセージ・例外メッセージ・`config.py` の `description=` は日本語のままにする（ADR-0064）。
- 編集したファイルに日本語のコメント・docstring が残っている場合、同じ変更で英語に直す。
- import は 1 行 1 つ（ruff `force-single-line = true`）。
- `InferenceSession` を構築してよいのは `vspeech/lib/onnx_session.py` と `vspeech/lib/vad.py` の 2 ファイルだけ（ADR-0024、`tests/test_onnx_session.py` が強制）。
- Silero VAD は `CPUExecutionProvider` 固定でデバイスを受け取らない（ADR-0024 の明示的例外）。この性質を変えない。
- `gpu_id` と `gpu_name` の設定項目名・意味・優先順位（`gpu_id` が先に評価される）を変えない。
- コミットするドキュメント・コードに、環境固有の絶対パスや LAN の IP を書かない（gitleaks pre-commit が拒否する）。

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

### Task 1: CUDA Driver API による GPU 列挙

**目的:** torch を使わずに、CUDA の ordinal 空間でデバイス数・名前・compute capability を得る。

**範囲:** `vspeech/lib/`、`tests/`

**契約**
- Produces: CUDA デバイス 1 台分の情報を表す型。ordinal（`int`）、名前（`str`）、compute capability の major と minor（`int`）を読み取れること。
- Produces: 全 CUDA デバイスを ordinal 昇順で返す列挙関数。CUDA ドライバが無い環境、ロードに失敗した環境、デバイスが 0 台の環境では、例外を送出せず空の列を返す。

**受入基準**
- [ ] CUDA ドライバが利用できない環境で、列挙が例外を送出せず空の結果を返す。
- [ ] 列挙が GPU コンテキストを生成しない（VRAM を確保しない）。
- [ ] 列挙結果の順序が CUDA の ordinal 昇順である。
- [ ] このモジュールを import しても torch がロードされない。
- [ ] ドライバのロード失敗が、原因の分かる日本語のログとして 1 度だけ残る。

**検証**
```sh
uv run ty check
uv run pytest tests/
```
どちらも終了コード 0 であること。実 GPU のあるホストでは、列挙結果の名前と ordinal が `nvidia-smi` の表示と一致すること。

**コミット単位:** 列挙層とそのテストで 1 コミット。

---

### Task 2: デバイス値型と解決ポリシーの torch 非依存化

**目的:** `get_device` と fp16 可否判定から torch を外し、fp16 可否を compute capability で決める（ADR-0079）。

**範囲:** `vspeech/lib/cuda_util.py`、`vspeech/lib/rvc.py`（`half_precision_available` の移設元）、`tests/`

**契約**
- Consumes: Task 1 の列挙関数と列挙結果の型。
- Produces: デバイスを表す不変の値型。属性 `type`（`"cuda"` または `"cpu"`）と `index`（`int | None`）を持つ。
- Produces: `get_device(gpu_id: int | None, gpu_name: str) -> tuple[<デバイス値型>, str]`。第 2 要素はデバイス名。
- Produces: `half_precision_available(device: <デバイス値型>) -> bool`。`vspeech/lib/cuda_util` から公開する。
- Produces: compute capability から fp16 可否を返す純関数。引数は major と minor。

**受入基準**
- [ ] `gpu_id` が指定されていればそれが優先され、指定が無く `gpu_name` があれば名前の部分一致で解決される。両方無ければ CPU に解決される。この優先順位が変更前と同一である。
- [ ] CUDA デバイスが 1 台も見えない環境で、`gpu_id` や `gpu_name` が指定されていても CPU に解決され、起動が失敗しない。
- [ ] `gpu_name` が 1 台も一致しない場合の解決結果が、変更前と同一である。
- [ ] fp16 可否が compute capability 7.0 以上と 6.0 で可、6.1 と 6.2 と 6.0 未満で不可になる。
- [ ] fp16 可否の純関数が、実 GPU を持たない環境で major/minor の境界値ごとに検証される。検証対象に 5.2 / 6.0 / 6.1 / 6.2 / 7.0 / 7.5 / 8.9 / 12.0 を含む。
- [ ] `type` が `"cpu"` のデバイスに対して fp16 可否が不可を返す。
- [ ] `vspeech/lib/cuda_util` を import しても torch がロードされない。
- [ ] 解決したデバイス番号とデバイス名が、変更前と同じ内容でログに残る。

**検証**
```sh
uv run ty check
uv run pytest tests/
```
どちらも終了コード 0 であること。終了コードはパイプを通さず直接確認する。

**コミット単位:** 値型・解決ポリシー・fp16 判定とそのテストで 1 コミット。

---

### Task 3: ONNX セッションの device 型とプロバイダ判定

**目的:** `create_session` の引数型を自前のデバイス値型にし、CUDA EP を要求するかを onnxruntime の申告に基づいて決める。

**範囲:** `vspeech/lib/onnx_session.py`、`CLAUDE.md`、`tests/`

**契約**
- Consumes: Task 2 のデバイス値型。
- Produces: `create_session(model_file: Path, device: <デバイス値型>, log_severity: int | None = None) -> InferenceSession`。引数名と順序を変更前から変えない。

**受入基準**
- [ ] デバイスが `"cuda"` でも、実行時に利用可能な実行プロバイダの一覧に CUDA が無ければ CUDA を要求しない。
- [ ] デバイスが `"cpu"` のとき CUDA を要求しない。
- [ ] CUDA を要求する場合の `device_id` が、`index` が `None` のとき 0 になる。
- [ ] `InferenceSession` の構築箇所が `vspeech/lib/onnx_session.py` と `vspeech/lib/vad.py` の 2 ファイルのみである。
- [ ] `vspeech/lib/onnx_session` を import しても torch がロードされない。
- [ ] CLAUDE.md の onnx_session 不変条件の記述が、`torch.device` を前提としない現在の契約に更新されている。

**検証**
```sh
uv run ty check
uv run pytest tests/test_onnx_session.py
```

**コミット単位:** セッション生成と CLAUDE.md 更新で 1 コミット。

---

### Task 4: 変換経路の境界変換

**目的:** RVC とストリーミング VC が自前のデバイス値型を受け取り、内部で `torch.device` に変換して従来どおり動くようにする。

**範囲:** `vspeech/lib/rvc.py`、`vspeech/lib/stream_vc.py`、`vspeech/lib/pitch_extract.py`、`vspeech/worker/vc.py`、`vspeech/stream_vc/runner.py`、`tests/`

**契約**
- Consumes: Task 2 のデバイス値型と `half_precision_available`、Task 3 の `create_session`。
- Produces: 変換経路の公開関数（`change_voice`、`load_hubert_model`、ストリーミング VC の構築）が、`torch.device` ではなく自前のデバイス値型を受け取る。

**受入基準**
- [ ] 変換経路が解決する fp16 可否の値が、変更前と同一である。
- [ ] 既存の再現可能な RVC 基準出力と、変更後の変換結果が一致する。
- [ ] ストリーミング VC の 1 ブロックあたりの変換結果が、変更前と一致する。
- [ ] `gpu_id` / `gpu_name` から解決されたデバイスで、変換が CUDA 実行プロバイダ上で走ることが従来どおり検出される。
- [ ] 音声変換ワーカーとストリーミング VC のデバイス解決ログの内容が、変更前と同一である。

**検証**
```sh
uv run ty check
uv run pytest tests/test_rvc_helpers.py tests/test_stream_vc.py tests/test_pitch_extract.py
```
RVC 基準出力の照合は GPU と rvc extra を要する。実行方法は既存の基準出力テストに従う。

**コミット単位:** 変換経路の境界変換で 1 コミット。

---

### Task 5: 回帰ガードと依存宣言

**目的:** torch がデバイス解決層に戻ってこないことをテストで固定し、音声認識の依存宣言から torch を外す。

**範囲:** `tests/test_forbidden_imports.py`、`pyproject.toml`、`uv.lock`

**契約**
- Consumes: Task 2 と Task 3 の公開契約。

**受入基準**
- [ ] デバイス解決と ONNX セッション生成に関わるモジュールを import しても torch がロードされないことが、汚染されない別プロセスで検証される。
- [ ] 音声認識の extra の依存宣言に torch が含まれない。
- [ ] ロックファイルが依存宣言と整合する。
- [ ] 音声変換の extra の依存宣言は変更されない。
- [ ] 依存パッケージが追加されていない。

**検証**
```sh
uv run pytest tests/test_forbidden_imports.py
uv lock --check
uv run pytest
```
最後の全件実行は、既存の緑を壊していないことの確認。

**コミット単位:** 回帰ガードと依存宣言で 1 コミット。

---

### Task 6: 実機検証

**目的:** spec の数値目標と、移行によるデバイス解決結果の不変を実機で確認する。

**範囲:** 実行のみ。コード変更はこの task では行わない。

**契約**
- Consumes: Task 1〜5 の成果すべて。

**受入基準**
- [ ] 音声認識パイプラインのプロセスに torch の DLL がロードされていない。
- [ ] 同パイプラインの常駐メモリが、変更前より 400MB 以上小さい。
- [ ] 同パイプラインの起動が、ワーカー開始のログ出力までの時間で変更前より 2 秒以上短い。
- [ ] 音声認識・音声変換・ストリーミング VC の各ホストで、起動ログのデバイス解決結果（番号と名前）が変更前と一致する。
- [ ] 音声認識が実マイク入力で従来どおり文字起こしできる。
- [ ] 音声変換の出力を実際に聴いて、変更前と差がない。

**検証**
実行中プロセスのロード済みモジュール一覧と常駐メモリを取得し、変更前の値と比較する。起動時間はログのタイムスタンプ差で測る。基準となる変更前の値は Task 1 着手前に同じ方法で取得しておく。

**コミット単位:** コミットなし。結果を報告する。

---

## self-review

spec の受入基準と task の対応:

| spec の受入基準 | 対応 task |
|---|---|
| torch がロードされない / メモリ 400MB 減 / 起動 2 秒短縮 | Task 6 |
| GPU 名・番号の解決結果が前後で同一 | Task 2、Task 6 |
| CUDA 不在時に CPU フォールバック | Task 1、Task 2 |
| fp16 が CC で決まり製品名に影響されない | Task 2 |
| fp16 判定が GPU 無し環境で境界値検証される | Task 2 |
| CUDA EP の要求が利用可能プロバイダに基づく | Task 3 |
| RVC の変換結果が前後で一致 | Task 4 |
| fp16 可否の解決値が前後で同一 | Task 4 |
| 音声認識だけの依存で起動し文字起こしできる | Task 5、Task 6 |
| デバイス層 import で torch が入らない（別プロセス検証） | Task 5 |
| 依存パッケージが追加されない | Task 5 |
