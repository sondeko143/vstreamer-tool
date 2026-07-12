# Follow-ups

レビューで見つかったが、そのブランチのスコープ外として意図的に見送った項目。
**捨てたのではなく、繰り延べたもの。** 該当箇所を触るときに拾うこと。

各項目は「なぜ今直さなかったか」を必ず書く。理由の無い繰り延べは、ただの見落としと区別がつかない。

---

## CUDA 13 化（フェーズ2、branch `feat/cuda13-on-312`、2026-07-12）— ctranslate2 が CUDA 12 に取り残される

**whisper ホストは CUDA 12.x の cuBLAS + cuDNN 9 が必要。** `ctranslate2` 4.8.x（faster-whisper が使う）は
**CUDA 12 専用**で、推論時に `cublas64_12.dll` を要求する。**CUDA 13 ビルドは存在しない**
（faster-whisper [#1431](https://github.com/SYSTRAN/faster-whisper/issues/1431) 未対応）。

CUDA 12 時代（torch `+cu128`）は torch 同梱の `torch/lib/cublas64_12.dll` が供給していたが、フェーズ2の
`+cu130` 化で torch が持つのは `cublas64_13.dll` になり、供給元が消えた:
`whisper warmup failed: Library cublas64_12.dll is not found`。

**今回の対処（2026-07-12、実機）**: whisper ホストに **CUDA Toolkit 12.8** を入れて cuBLAS を供給し、動作確認済み。
CUDA 13.2 ドライバは CUDA 12 アプリも走らせるので、torch/onnxruntime(CUDA13) と ctranslate2(CUDA12) が共存する。

**今回直さなかった理由**: 実機に toolkit を入れれば直り、コード変更ゼロで済んだ。恒久策は 2 案あるがどちらも
検証コスト（実機 whisper 推論）が要るのでスコープ外にした:

1. whisper extra に `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` を足し、whisper worker で `os.add_dll_directory`
   してそれらの `bin` を DLL 探索に載せる。CUDA Toolkit のインストールを不要にできるが、Windows の DLL 探索が
   面倒で壊れやすい（faster-whisper [#1276](https://github.com/SYSTRAN/faster-whisper/issues/1276) の沼）。
2. **CUDA 13 化そのものを巻き戻す**（torch `cu130→cu128`・onnxruntime `1.27→<1.27`、cp314 は維持）。全部
   CUDA 12 に揃い whisper は無改修で動く。CUDA 13 の唯一の利点（onnxruntime を 1.27+ に保つ）は runtime 中立
   なので損失は小さい。

**教訓**: フェーズ2の「混在 CUDA」検証を `ctranslate2.get_cuda_device_count()`（デバイス数照会だけで cuBLAS を
ロードしない）で済ませたのが漏れ。実際の whisper 推論を 1 回回していれば、コミット前に気づけた。

**デプロイ時の必須事項**: whisper を GPU で回すホストには **CUDA Toolkit 12.x（cuBLAS + cuDNN 9）** を入れること
（R580+ ドライバとは別に）。vc/RVC 専用ホスト（torch + onnxruntime のみ）は CUDA 13 のドライバだけでよい。

## sounddevice 移行（フェーズ3、branch `feat/python-314`、2026-07-12）— レビュー指摘の繰り延べ

PyAudio → sounddevice 移行の `/code-review high` で出た軽微な指摘のうち、今回スコープ外にしたもの。
実機（録音・再生）は検証済みで、いずれも固定デバイス構成では無害。overflow ログ（recording）と
host-API フィルタの `is not None` 化（audio.py）はレビュー後に修正済み。

### playback: ストリーム再構築時に PortAudio のデバイス一覧を再取得しない — [`playback.py`](../vspeech/worker/playback.py)

旧 PyAudio 版は `update_stream_if_changed` で `PyAudio()` を作り直して PortAudio を再初期化し、
デバイスを再列挙していた。sounddevice 版は `sd.query_devices()`（初期化時のキャッシュ）を見るだけで、
起動後に抜き差し／再接続されたデバイスを拾えない。**今回直さなかった理由**: ランタイムのデバイス
変更という縁ケースで、固定デバイス（`Line 4` 等）の実運用では起きない。再列挙には
`sd._terminate()` / `sd._initialize()`（private API）が要り、副作用の検証コストが見合わない。

### `pyaudio_recording_worker` / `pyaudio_playback_worker` の名前が実装と食い違う — recording.py / playback.py

sounddevice 化したのに関数名に `pyaudio_` が残っている。**今回直さなかった理由**: 純粋な rename で
`create_*_task` の呼び出し箇所に波及するだけ。機能価値ゼロ。次にこのファイルを触るとき一緒に直す。

## Python 3.12 化（spec ③、branch `feat/python-312`、2026-07-10）

### 次の昇格（3.13）の障害

**障害は 2 つある。** かつてここには「唯一のブロッカーは `audioop`」と書いていたが誤りだった
（2026-07-12 に PyPI の wheel を直接確認して訂正）。`audioop`（標準ライブラリの除去）と、
`numpy>=1.23,<2` の上限（cp313 wheel の不在）の両方を外さないと 3.13 では動かない。

#### 障害 1: `audioop`

**`audioop` は PEP 594 で 3.13 から標準ライブラリを外れる** — 3.12 では `DeprecationWarning` が出るだけ。

使用は 3 箇所・2 関数だけで、いずれも整数 PCM の `bytes` を直接扱う:

| 箇所 | 呼び出し |
|---|---|
| [`vspeech/worker/playback.py:127`](../vspeech/worker/playback.py) | `audioop.mul` — 再生音量のスケール |
| [`vspeech/worker/vc.py:261`](../vspeech/worker/vc.py) | `mul` — RVC 出力へのゲイン適用 |
| [`vspeech/worker/recording.py:66`](../vspeech/worker/recording.py) | `audioop.rms` — dBFS 算出（録音の開始/停止判定） |

**今回直さなかった理由**: spec ③ のスコープは 3.12 への昇格であり、3.13 ではない。`audioop` は 3.12 では
正常に動く。ここで numpy へ書き換えると、録音の閾値判定（`get_dbfs`）と vc のゲイン経路という
**実機でしか検証できない 2 経路**を、3.12 昇格の検証と同時に動かすことになる。切り分け不能な変更は混ぜない。

拾うときの選択肢は 2 つ:
1. `audioop-lts`（PyPI の 3.13+ 向け後方移植）を依存に足す。最小コスト。
2. numpy で書き直す。`rms` は `sqrt(mean(x**2))`、`mul` は乗算 + int16 クランプ。
   **`mul` のクランプ挙動に注意** — `vc.py` は既に `change_voice` の int16 キャストで
   オーバーシュートの wrap（クリック音）を踏んでいる（`feat/rvc-input-envelope` の教訓）。
   `audioop.mul` は飽和させる。numpy の素の astype は wrap する。ここを取り違えると無音の劣化になる。

   `audioop-lts`（選択肢 1）を推す。`_audioop.c` の本物の C ポートなので `import audioop` がそのまま通り、
   3 箇所は無変更で、`mul` の飽和セマンティクスが構造的に保存される（上の wrap 罠を踏まずに済む）。
   宣言は `audioop-lts ; python_version >= '3.13'` — このパッケージ自身の `Requires-Python` が
   `>=3.13` なので、マーカー無しだと現行 3.12 の解決が壊れる。

#### 障害 2: `numpy>=1.23,<2` の上限

**この上限を外さないと 3.13 に上げられない。** `uv.lock` は numpy 1.26.4 に解決されるが、
**1.26.4 に cp313 wheel は無い**（cp39〜cp312 と pp39 のみ）。しかも numpy 側の `Requires-Python` は
上限なしの `>=3.9` なので、リゾルバはこれを候補から外さず、3.13 では sdist ビルドを試みて失敗する。
つまり 3.13 は `<2` の撤廃を**強制**する。

**今回直さなかった理由**: この `<2` は既知の非互換ではなく予防的な上限
（[docs/superpowers/specs/2026-07-09-rvc-hubert-fairseq-free-design.md](superpowers/specs/2026-07-09-rvc-hubert-fairseq-free-design.md) が
「onnxruntime-gpu / faiss-cpu / pyworld の numpy 2.x 対応が未確認」として後続 spec 送りにした）。
spec ③ のスコープは 3.12 であり、numpy 2 への移行は RVC・録音の実機経路を巻き込むので別ブランチに切る。

拾うときの見通し: この上限は**自動的に解ける**。numpy 1.26 に cp313 が無い以上、cp313 wheel を
配っているライブラリ（pyworld 0.3.5 / faiss-cpu 1.14.3 / ctranslate2 4.8.x はいずれも配布済み）は
**必然的に numpy 2 でビルドされている**。自コードも NEP 50 安全 — int16 演算はすべて
`astype(float)` → 明示 `np.clip` → キャストの形（[`vspeech/worker/vc.py:139`](../vspeech/worker/vc.py),
[`vspeech/lib/rvc.py:410`](../vspeech/lib/rvc.py)）で、弱スカラーの値ベース昇格に依存していない。

> **なお 3.13 自体に上げる旨みは無い**（incremental GC は 3.13.0 final 直前に revert、free-threading は
> 3.14 まで experimental で、そもそも GPU/バッファ律速のこのパイプラインでは無意味）。上げるなら
> サポートが 1 年長い **3.14** を直接狙う。有効化作業は同一。3.14 の唯一の穴は `pyworld`（cp314 wheel 無し
> → sdist ビルド。ただし実機は `f0_extractor_type="rmvpe"` なので依存から外す手もある）。

### Docker イメージ（未検証。この環境に docker が無い）

**イメージのビルドと起動は一度も実行できていない。** 下記の修正はすべて静的な確認と、Windows から
`--os linux` を指定してダウンローダを実際に走らせた結果に基づく。**初回ビルドは必ず人間が確認すること。**

### 解決済み（この branch 内で対処）

**`voicevox` extra が win32 限定で、Docker には voicevox-core が入っていなかった** — 解決

`voicevox = ["voicevox-core ; sys_platform == 'win32'"]` だったため、`uv export --extra voicevox` が
生成する `requirements-pod.txt` から voicevox-core が丸ごと落ちていた。Linux イメージで
`import voicevox_core` すれば `ImportError` になる。0.14 時代は Linux の core をダウンローダが
供給していたのでこれで正しかったが、0.16.4 への上げ（PR #1）で Linux も wheel 配布に変わった。
`marker` 付きの `[tool.uv.sources]` リストで win_amd64 / manylinux_2_34_x86_64 を出し分ける。

**Dockerfile が voicevox 0.14.3 のダウンローダを叩いていた** — 解決（0.16.4 へ）

`RUN cp voicevox_core/*.so.* ./` も 0.14 の遺物（当時は core の `.so` を CWD に置く必要があった）。
0.16 の wheel は自前の拡張を同梱するので不要。代わりに必要なのは ONNX Runtime・OpenJTalk 辞書・
`.vvm` の 3 点で、これらは wheel に入っていないのでダウンローダで取る。

**`RUN ./download` は非対話ビルドで落ちる** — 解決（`yes y | ./download`）

0.16 のダウンローダは `onnxruntime` と `models` の 2 つについて利用規約への同意を対話で要求する。
`--yes` 相当のフラグは無い。stdin が EOF だと `Error: unexpected end of file` で終わる。
（EULA 表示時に `ERROR something went wrong with the pager` も出るが、終了コードは 0 で続行する。）

**`onnxruntime_path` を焼き込まない** — `LD_LIBRARY_PATH` で解決

Linux 版が置くのは `libvoicevox_onnxruntime.so.1.17.3` というバージョン付きの実体のみで、
symlink は作られない。`Onnxruntime.load_once()` はこの versioned filename を dlopen するので、
`LD_LIBRARY_PATH=/app/voicevox/onnxruntime/lib` を通せばバージョンを Dockerfile に書かずに済む。
ダウンローダの出力先を `./voicevox` にしてあるのは、`VoicevoxConfig` の既定パス
(`./voicevox/dict/open_jtalk_dic_utf_8-1.11`, `./voicevox/models/vvms`) にそのまま一致させるため。

**`requirements-pod.txt` が spec ② 以降 stale だった** — 解決（`make` で再生成）

`uv.lock` は numpy 1.26.4 なのに `requirements-pod.txt` は 1.23.5 のままで、Docker イメージだけが
古い numpy を入れていた。②で lock を更新したとき `make` を回していなかった。
**`pyproject.toml` / `uv.lock` を触ったら `make` も回すこと**（Makefile はそう書いてあるが、CI が無いので
誰も強制しない）。イメージは `python:3.12-slim` へ上げた — `requirements-pod.txt` は
`requires-python = ">=3.12,<3.13"` で解決した lock の投影なので、3.10 に流し込むのは元々不整合だった。

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
