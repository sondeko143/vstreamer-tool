# 0049. FCPE を波形入力 ONNX f0 抽出器としてスパイク先行で追加する

- Status: Proposed
- Date: 2026-07-21
- Related: spec [`../superpowers/specs/2026-07-21-fcpe-f0-extractor-design.md`](../superpowers/specs/2026-07-21-fcpe-f0-extractor-design.md); [ADR-0030](0030-pyworld-lazy-default-rmvpe.md)（f0 既定 rmvpe）, [ADR-0022](0022-hubert-onnx-runtime.md)（hubert オフライン ONNX 化）, [ADR-0024](0024-onnx-session-single-factory.md)（create_session 一元化）, [ADR-0045](0045-gui-readiness-reuses-preflight.md)（preflight 再利用）

## Context

f0 抽出は実時間 VC 経路のレイテンシに乗る処理で、既定の rmvpe は精度に優れる一方で大きめの ONNX を GPU 実行する。FCPE はより小型で高速なピッチ推定器で、rmvpe と同じ「16k 波形入力 → f0」契約に載る見込みがある。ただし制約が2つ、リスクが2つある。制約: (1) 本家 torchfcpe は公式 ONNX export を持たず torch 実装のみ、(2) このパイプラインの f0 抽出器は ONNX・波形入力前提で統一されている（rmvpe.onnx が mel とデコードを graph に焼き込み、waveform+threshold→f0 を返す）。リスク: (A) mel/STFT を含む FCPE が `torch.onnx.export` で通るか未証明、(B) rmvpe は既に ONNX・GPU で動くため「ONNX 化した FCPE が ONNX 化した rmvpe よりこの環境で速い」ことが未証明（＝本件の狙いそのもの）。

## Decision

FCPE を rmvpe と同じ契約（16k 波形 + threshold → f0(Hz)、mel・LynxNet・local_argmax デコードを graph に焼き込む）の**第二の波形入力 ONNX f0 抽出器**として追加する。ただし本実装の前に**スパイクで go/no-go を確認する**: 最小 export ＋ 同一条件での rmvpe.onnx とのレイテンシ実測で、①export 成功、②有意に高速（既定バー: median で ≥30% 高速）、③f0 が torch FCPE と概ね一致、を満たしたときだけ本実装へ進む。ONNX 資産は既存 hubert と同様にオフライン生成（poe タスク＋`uv run --with` overlay、torch 参照との golden 数値等価で自己検証）し gitignore する。torchfcpe は runtime 依存に入れず overlay 専用。threshold(0.006)・f0 レンジ・デコード方式は生成時に固定。rmvpe は既定のままで FCPE はオプトイン。

## Alternatives rejected

- **torch 版 FCPE をそのまま実行（ONNX 化しない）** — このパイプラインの f0 抽出器は ONNX・波形入力契約で統一されており、torch 経路を持ち込むと create_session 一元化（[ADR-0024](0024-onnx-session-single-factory.md)）と既存実行系から外れ、rmvpe との比較・置換も揃わない。
- **mel 入力の部分 graph（mel/デコードを Python 側に残す）** — rmvpe.onnx の全焼き込み方式と非対称になり pitch 抽出経路に別扱いコードが増える。lj1995 の mel 入力 rmvpe.onnx がこのプロジェクトで動かなかった前例もあり、波形入力・焼き込みに揃える。
- **事前生成した fcpe.onnx を同梱/ホスト** — 素性の確かな配布物が存在せず（唯一のコミュニティ版は未マージ）、FCPE/torchfcpe 本体と bundled 重みの再配布条件も未確認。hubert と同じくオフライン生成＋gitignore が安全側。ライセンスが許せば将来オプションとして別途決める。
- **一括ビルド（スパイク無し）** — リスク A/B がどちらも未証明で、両方外すと config/worker/preflight/test の周辺実装が丸ごと無駄になる。安価なスパイクで先に潰す。
- **rmvpe を FCPE で置換して既定にする** — 精度は rmvpe が上とされ FCPE は「速いが精度は落ちる」。既定を動かすと既存ユーザの品質が黙って下がる。オプトインに留める。

## Consequences

- f0 抽出器が2つの波形入力 ONNX（rmvpe/fcpe）を持ち、create_session 一元化と preflight/GUI readiness（[ADR-0045](0045-gui-readiness-reuses-preflight.md)）にそのまま乗る。必須アセット検証は preflight に一箇所足すだけで GUI が追従する。
- オフライン生成ツールが hubert（convert/export-hubert）に続く2本目になり、`uv run --with` overlay 方式の踏襲で torchfcpe を runtime/uv.lock から隔離できる。
- スパイクが no-go（export 不可 or 速くない）なら本実装に進まず、この ADR は Accepted に昇格させず Deprecated とし、否定結果を残す。
- ≥30% バーは初期値。スパイク結果次第で見直す（妥当な有意差があれば緩める判断もあり得る）。
- 実装がスパイクで裏づけられたら Status を Accepted に昇格する。
