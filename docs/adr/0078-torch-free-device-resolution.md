# 0078. デバイス解決と ONNX プロバイダ判定を torch から切り離す（0024 を refine）

- Status: Accepted
- Date: 2026-08-11
- 効力: 既定
- Related: refines [ADR-0024](0024-onnx-session-single-factory.md); spec [2026-08-11-torch-free-device-layer-design.md](../superpowers/specs/2026-08-11-torch-free-device-layer-design.md); [ADR-0028](0028-migrate-to-cuda-13.md), [ADR-0039](0039-whisper-hosts-need-cuda12-toolkit.md)（ドライバ要件）; [ADR-0079](0079-fp16-by-compute-capability.md)（同じ列挙経路に乗る判定）

## Context

[ADR-0024](0024-onnx-session-single-factory.md) は onnxruntime セッションを単一ファクトリから開き、呼び出し側が渡した device を尊重することを決めた。この不変条件は正しく、維持する。ただし副作用として、device の型が `torch.device` であること、および CUDA 可否の判定が `torch.cuda.is_available()` であることが、torch をデバイスに触る全ワーカーの依存に広げた。

音声認識（whisper）の推論は ctranslate2 が行い、torch を必要としない。それにもかかわらず音声認識パイプラインは torch を読み込む。使っているのは解決結果のうち**デバイス番号ひとつだけ**で、デバイス名はログ出力にしか使われていない。

この整数 1 個の代償を実測した。

- torch の import 単体で常駐メモリ **+476.7MB**、起動時間 **+3.17 秒**（torchaudio は追加 +3.2MB / +0.19 秒）。
- 稼働中の音声認識パイプライン（常駐 1343MB）では、ctranslate2 が使う CUDA 12 版 cuBLAS（145.8MB）と torch が持ち込む CUDA 13 版 cuBLAS（133.0MB）が**二重にロード**されている。加えて torch 由来のプロファイラ DLL が 87.1MB、torch 本体の DLL 群が 30.7MB。

常駐パイプラインが複数あるため、この重複は台数分だけ効く。

ただし torch を引いているのはデバイス解決だけではない。**`ctranslate2` 自身が
`try: import torch / except ImportError` の形で torch を任意依存として掴む**
(`ctranslate2/specs/model_spec.py` の `torch_is_available`)。したがってデバイス層を
torch-free にしても、**torch が環境に導入されている限り音声認識プロセスは torch を読み込む**。
削減が実現するのは torch を導入しないホストに限られる。実測 (torch を遮断して
`import faster_whisper` + デバイス層):

| 環境 | torch | 起動 | 常駐 |
|---|---|---|---|
| 全 extra を入れた単一 venv | ロードされる | 3.66s | 527.1MB |
| torch 非導入 | されない | 0.82s | 63.0MB |

変更前は `whisper` extra が torch を宣言していたため、`uv sync --extra whisper` は必ず
torch を導入した。**torch 非導入の音声認識ホストは、この決定によって初めて成立する。**

## Decision

デバイスを表す値を `torch.device` から、`type` と `index` だけを持つ自前の値型に置き換える。属性名は torch に合わせ、変換経路（RVC）は境界で `torch.device` に変換する。これにより torch の import を変換経路の内側に閉じる。

GPU の列挙（デバイス数・名前・compute capability）は CUDA Driver API（`nvcuda.dll`）を ctypes で呼ぶ。依存パッケージは増えない（ドライバ同梱で、R580+ は [ADR-0028](0028-migrate-to-cuda-13.md) / [ADR-0039](0039-whisper-hosts-need-cuda12-toolkit.md) により既に必須要件）。`cuInit` はコンテキストを生成しないため VRAM を消費しない。

`create_session` が CUDA 実行プロバイダを要求するかどうかは、onnxruntime 自身が報告する利用可能プロバイダ一覧に基づいて決める。

[ADR-0024](0024-onnx-session-single-factory.md) の「ファクトリを二重化しない」「呼び出し側の device を尊重する」という不変条件は維持する。変えるのは device の**型**と可否判定の**出所**だけである。

## Alternatives rejected

- **NVML（`nvidia-ml-py` または `nvml.dll` を直接）** — 解決結果は最終的に ctranslate2 の `device_index` と ORT CUDA EP の `device_id` に渡る。これらが解釈するのは CUDA の ordinal 空間であって、NVML の列挙順ではない。両者が一致する保証はなく、複数 GPU 機で静かに別の GPU を掴む余地が生まれる。CUDA Driver API は runtime API と同じ `CUDA_DEVICE_ORDER` を読むため ordinal 空間が揃う。`nvidia-ml-py` は加えて依存が増える。
- **音声認識だけ `gpu_id` を直接渡す最小修正** — メモリと起動時間の削減効果は同じだが、`gpu_name` による GPU 選択が機能後退する（実際に `gpu_name` で GPU を指定して運用している設定がある）。結節点も残るため、次に `create_session` を使うワーカーが増えた時点で同じ問題が再発する。
- **変換経路（RVC）も含めて一気に torch フリーにする** — 効果は大きい（常駐 2249MB のパイプラインにも効く）が、HuBERT ONNX → f0 → デコーダを GPU 上で繋ぐ dlpack ゼロコピーの置き換えを伴い、ストリーミングの遅延予算に直結する。切り分けて別の決定とする。
- **現状維持** — 常駐パイプラインの台数だけ重複コストが掛かり続ける。

## Consequences

音声認識だけを導入したホストで torch が消え、常駐メモリと起動時間が減り、cuBLAS の二重ロードが解消する（実測 464MB / 2.84 秒）。

**逆に、全 extra を単一環境に導入する構成では何も減らない。** 変換経路（RVC）が torch を必要とする以上 torch は導入され、`ctranslate2` がそれを掴むためである。この構成で削減を得るには、音声認識を別環境で動かす運用（役割ごとに extra を絞る）が要る。本 ADR はその前提条件を作るだけで、運用の変更までは含まない。

CUDA EP の可否判定が ORT 自身の申告になることで、CLAUDE.md が戒める「`torch.cuda.is_available()` だけで EP を決める」形が構造的に取れなくなる。

ordinal の意味は `CUDA_DEVICE_ORDER` に依存する。開発機（RTX 4060 / RTX 5060 Ti の 2 枚差し）では Driver API と torch の解決結果が ordinal・名前とも一致することを確認済みだが、これはホストごとに変わりうる。移行時は各ホストの起動ログで解決結果の同一性を確認する必要がある。

`nvcuda.dll` がロードできない環境では CPU にフォールバックする。現行の `torch.cuda.is_available()` が False のときと同じ経路である。
