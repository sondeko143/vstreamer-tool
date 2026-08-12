# 0083. CUDA ランタイム（cuBLAS / cuDNN）の供給元を torch から nvidia wheel へ移す

- Status: Accepted
- Date: 2026-08-12
- 効力: 既定
- Related: [ADR-0080](0080-torch-free-rvc-runtime.md)（この決定を必要にした変更）; refines [ADR-0024](0024-onnx-session-single-factory.md)（呼び出し場所が単一ファクトリであること）; [ADR-0039](0039-whisper-hosts-need-cuda12-toolkit.md)（同型の代替案をかつて却下した記録）; [ADR-0028](0028-migrate-to-cuda-13.md)

## Context

[ADR-0080](0080-torch-free-rvc-runtime.md) の実装に入って初めて分かったことがある。**`rvc` extra から torch を外すと、CUDA ランタイムの供給元も一緒に消える。**

`onnxruntime-gpu` の wheel は CUDA ランタイムを同梱していない。この venv で `onnxruntime_providers_cuda.dll` が必要とする `cublasLt64_13.dll` / `cublas64_13.dll` / `cudnn64_9.dll` などを供給しているのは `.venv/Lib/site-packages/torch/lib/` であり、`import torch` がそのディレクトリを DLL 検索パスへ載せている。venv に `nvidia-*` の pip パッケージは 1 つも入っていない。

実測（Task 1）: `import torch` を外した状態で CUDA セッションを開こうとすると

```
Error loading "...onnxruntime_providers_cuda.dll" which depends on "cublasLt64_13.dll" which is missing.
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*
providers: ['CPUExecutionProvider']
```

となり CPU へ落ちる。同じコードの先頭に `import torch` を足すと `CUDAExecutionProvider` が現れる。

つまり ADR-0080 は、コードを numpy 化するだけでは完結しない。**供給元を決めない限り依存表から torch を外せない。**

調査で分かった 4 点が選択を決める。

- **`onnxruntime` は nvidia wheel を読み込む機構を自前で持っている。** `onnxruntime/__init__.py` の `preload_dlls()` が nvidia wheel の DLL を `ctypes.CDLL` で明示ロードする。ただし **import 時に自動では呼ばれない**（公開関数として提供されているだけ）。また torch が既に CUDA/cuDNN をロード済みで、かつその torch の CUDA メジャーが onnxruntime のそれと一致する場合は、何もせず戻る（不一致なら警告を出して自分でロードする）。
- **ただし引数なしで呼んでも足りない。** onnxruntime 1.27.0 の `_get_nvidia_dll_paths` は `site-packages/nvidia/{cublas,cufft,cuda_runtime,cudnn}/bin/` という **CUDA 12 世代までの配置**をハードコードしているが、CUDA 13 世代の wheel は配置を変えている。接尾辞を落とした `nvidia-cublas` / `nvidia-cufft` / `nvidia-cuda-runtime` は **`nvidia/cu13/bin/x86_64/` の 1 箇所**に全 CUDA ライブラリを入れ、旧名のまま残った `nvidia-cudnn-cu13` だけが `nvidia/cudnn/bin/` という旧来の形を保っている。したがって引数なしの `preload_dlls()` では cuDNN しか見つからず、cuBLAS / cuFFT / cudart は見落とされる（実測: preload を無効化すると 4 セッションすべてが `CPUExecutionProvider` に落ちた）。`preload_dlls` は与えられた 1 ディレクトリの直下しか見ないため、**CUDA 側と cuDNN 側で別のディレクトリを渡す 2 回の呼び出しが要る**。
- **`onnxruntime-gpu[cuda,cudnn]` はそのままでは解決できない。** NVIDIA は CUDA 13 世代で wheel 名から `-cu13` 接尾辞を落としており、onnxruntime-gpu 1.27.0 が pin している `nvidia-cuda-nvrtc-cu13~=13.0` は PyPI 上ではスタブ（0.0.1）しか存在しない。実体は `nvidia-cublas==13.6.0.2` / `nvidia-cuda-nvrtc==13.3.33` で、cuDNN だけが旧名のまま `nvidia-cudnn-cu13==9.24.0.43` である。**extra には乗れず、こちらで名指しして pin する必要がある。**
- **常駐メモリの削減は失われない。** cuBLAS / cuDNN は現状でも onnxruntime が読み込んでいる。供給元が torch から nvidia wheel に変わるだけで、ロードされる DLL は同じである。消えるのは torch 自身（`torch_cuda.dll` 390.6MB / `torch_cpu.dll` 291.1MB / `torch_python.dll` 19.1MB とその Python モジュール群）だけである。

## Decision

CUDA ランタイムを nvidia の pip wheel から供給する。必要な wheel を `rvc` extra に**こちらで名指しして** pin し、`onnxruntime-gpu[cuda,cudnn]` の extra には依存しない（上記のとおり解決不能なため）。

pin するのは `nvidia-cublas` / `nvidia-cudnn-cu13` / `nvidia-cufft` / `nvidia-cuda-runtime` の 4 つ（`nvidia-cuda-nvrtc` と `nvidia-nvjitlink` は前 2 つの推移的依存として入る）。前 3 つは `onnxruntime_providers_cuda.dll` の import table に載る＝欠けると DLL の load 自体が失敗する。`nvidia-cuda-runtime` だけは import table に無いが `preload_dlls` の探索対象なので、欠けると起動ごとに偽の失敗メッセージが出る（2.5MB なので入れる）。上限は DLL 名に焼かれた CUDA メジャー版（`cublas64_**13**` / `cufft64_**12**` / `cudnn64_**9**`）で切る。

`onnxruntime.preload_dlls()` の呼び出しは、GPU 対応セッションを開く唯一のファクトリである `vspeech/lib/onnx_session.py` の `create_session` の側に置く。[ADR-0024](0024-onnx-session-single-factory.md) が「ファクトリを二重化しない」と決めた場所がそのまま「CUDA ライブラリの読み込みを保証する唯一の場所」になる。呼び出しはプロセスにつき 1 回で足りる。**呼び出しの形は `directory=` を明示した 2 回**（CUDA 側と cuDNN 側）で、渡すディレクトリはライブラリ本体（`cublasLt64_{メジャー}.dll` / `cudnn64_*.dll`）を実際に探して決める。配置をこちらに焼き込まないので、上流が配置を戻しても、arch サブディレクトリ名が変わっても壊れない。実測コストは 7〜11ms（プロセスにつき 1 回）。

`preload_dlls()` は失敗しても print するだけで例外を投げないが、既存の `check_cuda_provider()` がセッション生成後に実際のプロバイダ一覧を見て fail loud するため、CPU への暗黙のフォールバックは起きない。この二段構えを維持する。

## Alternatives rejected

- **VC ホストに CUDA 13 ツールキット + cuDNN 9（CUDA 13 ビルド）の導入を要求する**（[ADR-0039](0039-whisper-hosts-need-cuda12-toolkit.md) が whisper ホストに課した形の拡張） — pip の重さは増えないが、プロビジョニング手順が 1 つ増え、DLL の解決が PATH 依存になる。この開発機がその脆さの実例で、`CUDA/v13.3` は `bin` / `lib` が空で cuBLAS を持たず、PATH が通っている cuDNN 9.20 は CUDA 12.9 ビルドの方である（CUDA 13.2 ビルドも同梱されているが PATH には無い）。venv に閉じる方が再現性が高い。
- **`onnxruntime-gpu[cuda,cudnn]` の extra に乗る** — 上流が意図した形であり最も望ましいが、1.27.0 の metadata が旧命名（`nvidia-cuda-nvrtc-cu13~=13.0`）を pin しているため PyPI の現状では解決できない。上流がこれを修正したら、こちらの名指し pin を捨てて extra に戻す（その時点で本 ADR を supersede する）。
- **torch を残す** — [ADR-0080](0080-torch-free-rvc-runtime.md) が成立しなくなる。
- **`preload_dlls()` をワーカーごとに呼ぶ** — セッションを開く経路が複数あるかのような形になり、[ADR-0024](0024-onnx-session-single-factory.md) が潰した「ファクトリの二重化」を呼び出し側で再現してしまう。

## Consequences

[ADR-0039](0039-whisper-hosts-need-cuda12-toolkit.md) が「Windows の DLL 探索が壊れやすい」を理由にこの型を却下したのは、**faster-whisper / ctranslate2 の探索**についてであって onnxruntime についてではない。onnxruntime は自前の `preload_dlls()` を持つため、その却下理由はここには当てはまらない。0039 の決定（whisper GPU ホストは CUDA 12 ツールキットを要求する）はそのまま有効で、本 ADR は変更しない。

venv のディスク使用量は増える。実測で **+1481.2MB**（3710.5MB → 5191.7MB）で、増加分はほぼ全量（1480.9MB）が `site-packages/nvidia/` である。これは本決定単体のコストで、[ADR-0080](0080-torch-free-rvc-runtime.md) が torch / torchaudio を外したあとの venv 全体は 2336.1MB になる。

**常駐メモリは増えない。** 同一プロセスで同じ 4 セッションを開いて供給元だけを変えると、nvidia wheel 供給（torch を import 不能化）が 2146.2MB、torch 供給が 2292.2MB（各 N=3 の中央値）で、供給元を移すこと自体はメモリを増やさず、むしろ torch 本体の分だけ減る。

torch と nvidia wheel が同居する環境（全 extra を入れた開発機、オフラインの ONNX 生成ツールを使う場合）で torch が先に import されていれば、`preload_dlls()` はそれを検出して何もせず戻る。二重ロードは起きない（ロード済みモジュールのフルパスを列挙し、同じ DLL 名が 2 つのディレクトリから載っている組が 1 つも無いことで確認した）。

`directory=` を明示する副産物として、**torch が入っているが import されていない環境でも供給元が nvidia wheel に固定される**（`directory=None` だと onnxruntime は互換 torch の `lib` へ迂回する）。torch 除去の前後で経路が同じになるので、除去前に取った実測がそのまま除去後の実測になる。

pin した nvidia wheel のバージョン追随がこちらの持ち物になる。onnxruntime を上げるときは、その build が要求する CUDA メジャーバージョンと cuDNN のメジャーバージョンを確認する必要がある（`onnxruntime.print_debug_info()` が両方を報告する）。**この確認を lock が代行してはくれない** — onnxruntime-gpu の nvidia 依存は `[cuda]` / `[cudnn]` extra 側にあり、本プロジェクトはその extra を要求していないので、解決器はこちらの pin と ORT の要求メジャーを突き合わせない。実際のガードは `tests/test_onnx_session.py::test_each_preload_is_pointed_at_a_directory_that_has_what_it_wants` で、ORT が要求する DLL 名が実在しなくなった時点で赤くなる。

罠がもう 1 つある: **`nvidia-cublas` は win_amd64 wheel を出さない版がある**（13.6.1.10 など。13.6.0.2 にはある）。上限を CUDA メジャーで切ってあるので解決自体は通るが、Windows で wheel が無い版を掴もうとすると lock 時に失敗する。

### 供給元を移すと同じ ONNX グラフの推論結果が bit 一致しなくなる

cuBLAS / cuDNN のビルドが違えばカーネル選択が違い、fp16 の丸めが変わるためで、欠陥ではない。ただし [ADR-0080](0080-torch-free-rvc-runtime.md) が定めた「変更前と bit 一致」の判定にはそのまま影響するので、実測値を残す。

計測は [ADR-0081](0081-ort-native-value-binding.md) の実装時に、同一プロセス・同一入力・同一 seed で行った。torch を import すると `preload_dlls` が何もせず torch の `lib` が供給元になるので、その有無で 2 条件を作った。

- **1 tick（Stream VC、context 500ms + block 160ms）:** HuBERT 特徴量 corr 0.999998 / SNR 53.02dB、デコーダ出力 corr 0.999977 / SNR 43.43dB / lag 0 / 振幅スペクトル corr 0.999997。fcpe の pitchf も SNR 94.81dB でわずかに動く（f0_coarse は整数なので完全一致）。**ADR-0081 が予想した「fp16 の丸め」のクラスそのもの**（`scripts/hubert_metrics.py` が記録している 39.52dB と同種）。
- **発話単位（1 秒、`config_vc.toml`、rmvpe）:** 既存 golden に対し corr 0.999756 / SNR 33.12dB / max\|diff\| 750。RMS 8285.7 → 8283.5、ピーク 22976 で不変、振幅スペクトル corr 0.99996、最良ラグ 0。**区間 SNR は先頭 53dB から末尾 28dB へ単調に低下する** — NSF の位相アキュムレータ（f0 の cumsum）に丸め差が蓄積する形で、時間とともに位相がずれていくことを示す。内容は同じで位相だけが動く。
- **200 ブロックの Stream VC:** 通しの corr 0.72 / SNR 2.53dB まで悪化するが、これは SOLA が拡大している。ブロックごとに ±5ms（＝ `sola_search_ms` そのもの）のラグを許して測り直すと corr の中央値 0.9999、0.99 を下回るのは 200 中 5 ブロックのみ。最良ラグは 150 ブロックで非ゼロ、範囲は -5.00〜+3.02ms で探索窓に収まる。ブロックごとの RMS 比は中央値 0.99999（0.987〜1.011）。**`sola_offset` の argmax が別の（等価な）整列を選んだ結果であって、劣化ではない。**
- 供給元を固定すれば**プロセスを跨いでも bit 一致する**（nvidia wheel 供給で N=200 × 独立 4 回、いずれも max\|diff\| = 0）。非決定になったわけではない。

帰結として:

1. **`tests/assets/rvc_golden/` の成果物は torch 供給時代のものなので、torch を依存から外したあとに採り直す必要がある。** 特に `change_voice_golden.npz` に対する現状のスコアは 33.12dB で、`scripts/hubert_metrics.py` の `SNR_MIN_DB = 35.0` をわずかに下回る。閾値を緩めるのではなく golden を採り直すこと（閾値は fp16 の丸めの実測から来ている値で、供給元の差はそれと同種だが同一ではない）。
2. torch を残したまま numpy 版と torch 版を比較したい場合は、**先に `import torch` してから** ハーネスを走らせれば供給元が揃い、比較が実装だけの A/B になる。
3. 出力波形が変わる以上、**ブランチのマージ前に実機の聴感確認が要る。** 上の数値は「内容は同じで位相が動く」を支持するが、位相の話は数値だけで閉じない。
