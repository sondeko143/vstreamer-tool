# Follow-ups

レビューで見つかったが、そのブランチのスコープ外として意図的に見送った項目。
**捨てたのではなく、繰り延べたもの。** 該当箇所を触るときに拾うこと。

各項目は「なぜ今直さなかったか」を必ず書く。理由の無い繰り延べは、ただの見落としと区別がつかない。

---

## RVC HuBERT の ONNX 化（spec ②、branch `feat/rvc-hubert-onnx`、2026-07-10）

設計書: [docs/superpowers/specs/2026-07-10-rvc-hubert-onnx-design.md](superpowers/specs/2026-07-10-rvc-hubert-onnx-design.md)

### ✅ 解決済み

**`create_rmvpe_session` が `device` を無視する** — 解決（`create_session` と 1 実装に統合）

`cuda.is_available()` だけを見て `CUDAExecutionProvider` を挿し、呼び出し側の device を無視していた。
`create_session` 側は spec ② の `882b21f` で直したが、20 行の重複だったこちらは取り残されていた。

**同じ形でもう一度書けば、次も同じようにドリフトする。** 実装を
[`vspeech/lib/onnx_session.py`](../vspeech/lib/onnx_session.py) に一本化し、RVC decoder / HuBERT /
RMVPE の 3 経路が同じ関数を通るようにした。`tests/test_onnx_session.py` が EP 選択を
変異テストで固定し、「両者が同一の関数を指していること」自体も構造ゲートとして検査する。

**EP テストが `provider_options` / `device_id` を見ていない** — 解決（同上）

`device.index if device.index is not None else 0` の `None` 漏れ対策にテストが無かった。
`tests/test_onnx_session.py::test_a_bare_cuda_device_yields_device_id_zero` が固定する
（`device_id` を `device.index` に戻すと落ちることを確認済み）。

### テストの穴（退行が静かに通る）

**`get_device` の優先順位が未固定** — [`vspeech/lib/cuda_util.py`](../vspeech/lib/cuda_util.py)

`f8486c3` で `gpu_id = 0` を実デバイスとして扱うようにした副作用で、`gpu_id` と `gpu_name` を
**両方**設定したときの勝者が `gpu_name` → `gpu_id` に静かに入れ替わった。意図した変更に便乗した
2 つ目の挙動変化で、テストが固定していない。実運用 config はどちらか一方しか設定しないため実害は無い。

**`_NoDlpack` スタブが意図しない経路で fallback に入る** — [`tests/test_rvc_helpers.py`](../tests/test_rvc_helpers.py)

`_ortvalue` も `to_dlpack` も持たないため、内側の `except AttributeError` が想定していない
`AttributeError` で外側の `except Exception` に落ちる。numpy fallback には到達するが、
「dlpack が無いので優雅に迂回した」のか「dlpack が壊れている」のかを区別できない。

### 堅牢性（既知の失敗モードは無い）

**`_ort_output_to_torch` の広い `except Exception`** — [`vspeech/lib/rvc.py`](../vspeech/lib/rvc.py)

本物の dlpack バグを飲み込んで黙って numpy 経由に落ちうる。リファクタ前の `infer()` の
インライン実装と挙動は同一だが、共有ヘルパになったので影響範囲が広がった。
少なくとも fallback 時に `logger.warning` を出すべき。

> **前科あり。** spec ② では `export_graph` の同型の `except Exception` が `UnicodeEncodeError`
> （torch.onnx が `✅` を cp1252 の stdout に書こうとして落ちる）を「dynamo 失敗」と誤認し、
> 黙って legacy exporter に落ちていた。広い except は静かに劣った経路を選ぶ。

**`export_hubert_onnx.main()` の非アトミックな資産公開** — [`scripts/export_hubert_onnx.py`](../scripts/export_hubert_onnx.py)

`shutil.move` を 2 回独立に呼ぶ。fp32 が成功して fp16 が失敗すると、新しい fp32 と古い fp16 と
旧 `mapping.json` が同居する。次のロードで `parse_output_names` が `ValueError` を投げるので
静かな破損ではないが、アトミックな公開ではない。同一 FS の rename なので現実的な確率は低い。

**`mapping.json` の `exporter` が fp32 側しか記録しない** — [`scripts/export_hubert_onnx.py`](../scripts/export_hubert_onnx.py)

fp16 の `export_graph` の戻り値を捨てている。fp16 だけ legacy にフォールバックしても記録に残らない。
来歴のみで runtime は読まない。フォールバック自体は traceback を出すので気づける。

### 道具側の癖（コードを歪めない）

**`ty` の "Code is unreachable" ヒント** — `vspeech/lib/rvc.py` の
`device.index if device.index is not None else 0` 4 箇所

`ty` が `torch.device.index` を non-optional に narrowing するために出る。`ty check` は exit 0。
正しいコードを道具に合わせて歪めないこと。
