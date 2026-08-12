# 0080. 変換経路も含めてランタイム全体を torch から切り離し、rvc extra から torch/torchaudio/faiss-cpu を外す（0078 の続き）

- Status: Accepted
- Date: 2026-08-12
- 効力: 既定
- Related: extends [ADR-0078](0078-torch-free-device-resolution.md); spec [2026-08-12-rvc-torch-free-runtime-design.md](../superpowers/specs/2026-08-12-rvc-torch-free-runtime-design.md); [ADR-0081](0081-ort-native-value-binding.md)（推論境界）, [ADR-0082](0082-rvc-resample-on-inhouse-polyphase.md)（信号処理境界）, [ADR-0069](0069-torch-213-and-terminal-torchaudio.md)（外す対象のピン）, [ADR-0072](0072-stream-vc-lookahead.md)（下記のとおり、その測定値の再現性を失わせる）

## Context

[ADR-0078](0078-torch-free-device-resolution.md) は「変換経路（RVC）も含めて一気に torch フリーにする」を却下案として明示的に切り分け、別の決定に回した。本 ADR がその決定である。

0078 は同時に、削減が成立する条件も記録している — **`ctranslate2` が `try: import torch` の形で torch を任意依存として掴むため、torch が環境に導入されている限りプロセスは torch を読み込む**。この機構は変わっていない（再実測: `import ctranslate2` が 5.39 秒、完了時点で `torch in sys.modules == True`）。`ctranslate2` はコア依存なので、これは VC ホストにも等しく効く。つまり `rvc` extra が `torch` / `torchaudio` を宣言し続ける限り、変換経路のコードから torch を消しても削減はゼロである。

その上で、変換経路の torch が実際に何をしているかを調べた。

- RVC の推論そのものは onnxruntime が行う。torch が担っているのはテンソル配線、io_binding のゼロコピー、リサンプルの 3 つだけである。
- ゼロコピーは**すでに毎 tick 破れている**。f0 抽出（rmvpe / fcpe）は波形を `.detach().cpu().numpy()` して host 側で `session.run` しており、GPU 常駐は成立していない。torch のゼロコピーが実際に守っているのは HuBERT 特徴量 1 本（約 50KB/tick）で、これは 40–70ms の推論に対して 2 桁小さい転送である。
- リサンプルは [ADR-0073](0073-device-boundary-inhouse-polyphase-resampler.md) で導入した自前ポリフェーズ実装（numpy のみ）と重複している。ストリーミング経路は capture 側で既に自前実装が 16kHz へ落としており、torchaudio を通らない。

`faiss-cpu` も `rvc` extra にあるが、repo 内のどの `.py` からも import されておらず、対応する設定項目（index / index_rate 相当）も存在しない。

## Decision

`vspeech/` パッケージ全体が `torch` / `torchaudio` を import しない状態にし、`rvc` extra から `torch` / `torchaudio` / `faiss-cpu` を削除する。

置き換えの中身は 2 つの ADR に分ける — ONNX セッション間の受け渡しは [ADR-0081](0081-ort-native-value-binding.md)、リサンプルは [ADR-0082](0082-rvc-resample-on-inhouse-polyphase.md)。

オフラインの ONNX 生成タスク（`poe export-hubert-onnx` / `poe export-fcpe-onnx`）は torch を必要とし続けるので、`uv run --with torch` のオーバーレイへ退避する。`poe convert-hubert` が既に取っている形と同じで、ランタイム依存には戻さない。

この不変条件は `tests/test_forbidden_imports.py` の `FORBIDDEN` に `torch` / `torchaudio` を加えて恒久化する。同ファイルの AST 走査は既に `vspeech/**/*.py` 全体を対象にしているため、ガードの追加コストはゼロに近い。

## Alternatives rejected

- **ストリーミング経路（Stream VC）だけを torch-free にする** — 発話単位の `change_voice` が torch を残す以上 `rvc` extra から torch を外せず、`ctranslate2` が掴むため削減は実測ゼロになる。得られるのは境界が 1 つ増えることだけで、torch を積んだまま numpy と torch の 2 実装を保守することになる。
- **torch を遅延 import に留める** — 同じ理由で無意味。`ctranslate2` はコア依存であり、インストールされている torch を遅延 import が防ぐことはできない。
- **オフラインツールのために `dev` グループへ torch を置く** — `[tool.uv] default-groups = "all"` なので venv に torch が残り、この決定の目的そのものが消える。
- **`faiss-cpu` は将来の index 機能のために温存する** — import も設定項目も無い依存を「いつか使うかも」で保持すると、`uv audit` の対象面積と同期のたびの 16MB を払い続けることになる。実際に index 機能を足すときに 1 行戻せばよい。
- **現状維持** — VC ホストの常駐メモリと起動時間が torch のぶん重いままになる。0078 が whisper ホストについて解いた問題が、VC ホストで手つかずのまま残る。

## Consequences

VC ホストの venv から torch が消える。削減量の実測（[ADR-0078](0078-torch-free-device-resolution.md) と同じ量 — readiness marker までの起動時間と、trampoline を除いた実プロセスの常駐 working set。変更前は N=3、変更後は N=6 の中央値）:

| 設定 | 起動時間 | 常駐 WS |
|---|---|---|
| `config_vc.toml`（発話単位 VC、rmvpe） | 20.33s → **5.56s**（-14.77s） | 2332.1MB → **2146.7MB**（-185.4MB） |
| `config_stream_producer.toml`（Stream VC、fcpe） | 18.84s → **5.88s**（-12.96s） | 2246.2MB → **1825.0MB**（-421.2MB） |

起動時間の削減は 0078 が torch の import 単体について測った 3.17 秒より大きい。主因は上の Context が挙げた `ctranslate2` で、torch が venv から消えると `import ctranslate2` は 5.39 秒から **0.39 秒**になる。常駐の削減量が 2 設定で揃わない理由は特定していない。

**オフラインで ONNX 資産を作るときだけ、オーバーレイ経由で torch が降ってくる。** 初回は数百 MB のダウンロード待ちが増える。`export-hubert-onnx` は fp16 グラフを CUDA 上で export するため、この経路には引き続き GPU と CUDA 対応 torch が要る。

`dio` / `harvest`（pyworld）を選ぶ経路は元から任意依存で、この決定では変えない。

torch を必要とするテストはオフラインツール用のものだけになる。それらは `importorskip` で守り、ランタイム側のテストからは torch を落とす。

**[ADR-0072](0072-stream-vc-lookahead.md) が記録した lookahead の測定値は、この変更後のツールでは再現しない。** `scripts/stream_vc_lookahead_eval.py` の対数メルは torchaudio の `MelSpectrogram` を使っていたが、torchaudio を落とすため numpy 実装に置き換わった。両者は数値的に互換ではなく、実測でビンあたり平均 |Δ| が 16kHz で 0.49dB、**48kHz で 2.09dB**、そこから導かれる `spectral_distance` が 48kHz で平均 +0.30 / p95 +0.26 ずれる。48kHz が実際に使われるレートである。0072 は p95 の差 0.78dB・平均の差 0.28dB を根拠に `lookahead_ms=40` を選んでおり、このずれはその効果量の 25〜35%、平均の差にいたっては全量を超える。

1 回の実行の中では比較は依然として整合しているので、**lookahead=40 という決定自体は覆らない**。覆るのは 0072 の表の再現性だけである。0072 は Accepted なので本文は書き換えない — 数値を採り直す必要が生じたときは、新しいツールで測り直した表を新しい ADR に置くこと。

`ctranslate2` と torch の関係（0078 で記録した機構）は変わらない。全 extra を単一 venv に入れる開発機では、他に torch を引くものが無くなるため今回はじめて torch が消える — ただし将来 torch を引く依存が 1 つでも戻れば、コード側が torch-free でも常駐は元に戻る。守っているのは構造ガードではなく依存表であり、`test_forbidden_imports.py` はコード側しか見ていない。
