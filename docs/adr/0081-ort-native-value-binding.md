# 0081. ONNX セッション間の受け渡しを onnxruntime ネイティブの OrtValue に統一する（torch の dlpack ゼロコピーを捨てる）

- Status: Proposed
- Date: 2026-08-12
- 効力: 既定
- Related: [ADR-0080](0080-torch-free-rvc-runtime.md)（この決定の動機）; refines [ADR-0024](0024-onnx-session-single-factory.md)（セッション生成は単一ファクトリのまま）; [ADR-0053](0053-streaming-vc-fixed-block-crossfade.md)（shape 固定の由来）

## Context

現状の変換経路は CUDA デバイス上で、torch テンソルの `data_ptr()` を `io_binding.bind_input` に渡し、出力を dlpack で torch に戻している。狙いは HuBERT → （特徴量の 2 倍アップサンプル）→ デコーダという 2 つの ONNX セッションを GPU 上で繋ぐことだった。

この設計が実際に守っているものを測ると、根拠が薄い。

- **f0 抽出が既に host を経由している。** rmvpe / fcpe は波形を `.detach().cpu().numpy()` して `session.run` する。GPU 常駐は毎 tick 破れており、ゼロコピーが守っているのは HuBERT 特徴量 1 本だけである。
- **データ量が小さい。** ストリーミングの既定形状（context 500ms + block 160ms）で、波形が約 42KB、特徴量が約 50KB、デコーダ出力が約 105KB。40–70ms の推論に対して PCIe 転送は 2 桁小さい。
- **2 倍アップサンプルは GPU で行う理由がない。** `interpolate(scale_factor=2)` の nearest は要素の複製そのもので、numpy の `repeat` と定義が一致する。
- **dlpack 経路には既に広い except があり、失敗すると numpy コピーへ黙って落ちる。** 警告は出るが、ゼロコピーが保証ではなく最良ケースであることを設計自身が認めている。

これらに対して払っている代償が torch の常駐 476.7MB / 起動 3.17 秒である（[ADR-0078](0078-torch-free-device-resolution.md) の実測）。

## Decision

torch の `data_ptr` / dlpack を使わず、onnxruntime 自身の `OrtValue` と `bind_ortvalue_input` / `bind_output(..., "cuda")` で入出力を束ねる。セッション間で受け渡す値は numpy を経由し、2 倍アップサンプルは host 側で行う。

**device 側バッファの再利用（`OrtValue.update_inplace`）は最初から入れない。** ストリーミングは shape が固定なので再利用の余地はあるが、まず素直に置き換えて測定し、spec のレイテンシ基準（1 推論あたり p50 が変更前 +5% 以内）を満たさなかった場合にのみ足す。

[ADR-0024](0024-onnx-session-single-factory.md) の「`create_session` を二重化しない」「呼び出し側の device を尊重する」は維持する。変えるのはセッションへ値を渡す方法だけである。

## Alternatives rejected

- **dlpack のゼロコピーを維持するために torch を残す** — 守れるのは特徴量 1 本（約 50KB/tick、推論時間比で 0.3% 未満の見積もり）であり、その対価が 476.7MB / 3.17 秒になる。釣り合わない。
- **2 倍アップサンプルを HuBERT の ONNX グラフに焼き込んで再 export する** — GPU 常駐を保ったまま torch を外せるが、既存の `hubert_fp32.onnx` / `hubert_fp16.onnx` / `mapping.json` が全ホストで無効になり、再 export を強制する。得られるのは上と同じ 0.3% で、資産の作り直しに見合わない。
- **io_binding をやめて素の `session.run` に統一する** — 実装は最も単純になる。ただし入出力の device 配置の判断を全面的に ORT に委ねることになり、CUDA EP のまま host バッファを渡す形が残る。まず OrtValue で明示的に組み、測定で差が出ないことが分かってからならこちらへ単純化してよい（その場合は本 ADR を supersede する）。
- **shape 固定を活かして最初から `update_inplace` でバッファを使い回す** — おそらく速いが、それは測定ではなく予想である。基準を満たすかどうかを先に測り、必要になってから足す方が、要らない状態を抱え込まずに済む。

## Consequences

HuBERT 出力が毎 tick host を経由する。これは新しい往復ではなく、f0 抽出が既に通っている往復への合流である。それでもレイテンシへの影響はゼロではないので、spec の受入基準として p50 / p95 を両版で測る。

`_ort_output_to_torch` の広い except（dlpack 失敗時の numpy フォールバック）が不要になる。「速い経路と遅い経路のどちらを通ったか分からない」状態が構造的に消える。

fp16 の扱いは numpy 側へ移る。torch の `.half()` と numpy の `astype(np.float16)` はどちらも最近接偶数丸めなので、変換結果は一致するはずである — これは spec の bit 一致基準で確認される。

CPU 実行経路（`device.type != "cuda"`）は元から numpy で `session.run` しており、この決定で CUDA 経路がそちらに寄る。2 本あった実装が実質 1 本になる。
