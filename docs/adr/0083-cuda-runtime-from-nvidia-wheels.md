# 0083. CUDA ランタイム（cuBLAS / cuDNN）の供給元を torch から nvidia wheel へ移す

- Status: Proposed
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

調査で分かった 3 点が選択を決める。

- **`onnxruntime` は nvidia wheel を読み込む機構を自前で持っている。** `onnxruntime/__init__.py` の `preload_dlls()` が `site-packages/nvidia/{cublas,cufft,cuda_runtime,cudnn}/bin/` から DLL を `ctypes.CDLL` で明示ロードする。ただし **import 時に自動では呼ばれない**（公開関数として提供されているだけ）。また torch が既に CUDA/cuDNN をロード済みなら何もせず戻る。
- **`onnxruntime-gpu[cuda,cudnn]` はそのままでは解決できない。** NVIDIA は CUDA 13 世代で wheel 名から `-cu13` 接尾辞を落としており、onnxruntime-gpu 1.27.0 が pin している `nvidia-cuda-nvrtc-cu13~=13.0` は PyPI 上ではスタブ（0.0.1）しか存在しない。実体は `nvidia-cublas==13.6.0.2` / `nvidia-cuda-nvrtc==13.3.33` で、cuDNN だけが旧名のまま `nvidia-cudnn-cu13==9.24.0.43` である。**extra には乗れず、こちらで名指しして pin する必要がある。**
- **常駐メモリの削減は失われない。** cuBLAS / cuDNN は現状でも onnxruntime が読み込んでいる。供給元が torch から nvidia wheel に変わるだけで、ロードされる DLL は同じである。消えるのは torch 自身（`torch_cuda.dll` 390.6MB / `torch_cpu.dll` 291.1MB / `torch_python.dll` 19.1MB とその Python モジュール群）だけである。

## Decision

CUDA ランタイムを nvidia の pip wheel から供給する。必要な wheel を `rvc` extra に**こちらで名指しして** pin し、`onnxruntime-gpu[cuda,cudnn]` の extra には依存しない（上記のとおり解決不能なため）。

`onnxruntime.preload_dlls()` の呼び出しは、GPU 対応セッションを開く唯一のファクトリである `vspeech/lib/onnx_session.py` の `create_session` の側に置く。[ADR-0024](0024-onnx-session-single-factory.md) が「ファクトリを二重化しない」と決めた場所がそのまま「CUDA ライブラリの読み込みを保証する唯一の場所」になる。呼び出しはプロセスにつき 1 回で足りる。

`preload_dlls()` は失敗しても print するだけで例外を投げないが、既存の `check_cuda_provider()` がセッション生成後に実際のプロバイダ一覧を見て fail loud するため、CPU への暗黙のフォールバックは起きない。この二段構えを維持する。

## Alternatives rejected

- **VC ホストに CUDA 13 ツールキット + cuDNN 9（CUDA 13 ビルド）の導入を要求する**（[ADR-0039](0039-whisper-hosts-need-cuda12-toolkit.md) が whisper ホストに課した形の拡張） — pip の重さは増えないが、プロビジョニング手順が 1 つ増え、DLL の解決が PATH 依存になる。この開発機がその脆さの実例で、`CUDA/v13.3` は `bin` / `lib` が空で cuBLAS を持たず、PATH が通っている cuDNN 9.20 は CUDA 12.9 ビルドの方である（CUDA 13.2 ビルドも同梱されているが PATH には無い）。venv に閉じる方が再現性が高い。
- **`onnxruntime-gpu[cuda,cudnn]` の extra に乗る** — 上流が意図した形であり最も望ましいが、1.27.0 の metadata が旧命名（`nvidia-cuda-nvrtc-cu13~=13.0`）を pin しているため PyPI の現状では解決できない。上流がこれを修正したら、こちらの名指し pin を捨てて extra に戻す（その時点で本 ADR を supersede する）。
- **torch を残す** — [ADR-0080](0080-torch-free-rvc-runtime.md) が成立しなくなる。
- **`preload_dlls()` をワーカーごとに呼ぶ** — セッションを開く経路が複数あるかのような形になり、[ADR-0024](0024-onnx-session-single-factory.md) が潰した「ファクトリの二重化」を呼び出し側で再現してしまう。

## Consequences

[ADR-0039](0039-whisper-hosts-need-cuda12-toolkit.md) が「Windows の DLL 探索が壊れやすい」を理由にこの型を却下したのは、**faster-whisper / ctranslate2 の探索**についてであって onnxruntime についてではない。onnxruntime は自前の `preload_dlls()` を持つため、その却下理由はここには当てはまらない。0039 の決定（whisper GPU ホストは CUDA 12 ツールキットを要求する）はそのまま有効で、本 ADR は変更しない。

venv のディスク使用量は増える（nvidia wheel 群）。**常駐メモリは増えない**見込みだが、これは予想なので実測で確認する。

torch と nvidia wheel が同居する環境（全 extra を入れた開発機、オフラインの ONNX 生成ツールを使う場合）では `preload_dlls()` は torch のロードを検出して何もせず戻る。二重ロードは起きない。

pin した nvidia wheel のバージョン追随がこちらの持ち物になる。onnxruntime を上げるときは、その build が要求する CUDA メジャーバージョンと cuDNN のメジャーバージョンを確認する必要がある（`onnxruntime.print_debug_info()` が両方を報告する）。

## 追記（実装で判明した訂正）

上の Context は「`preload_dlls()` が `site-packages/nvidia/{cublas,cufft,cuda_runtime,cudnn}/bin/` から読む」と書いているが、**その配置は CUDA 12 世代までのものだった。** 実装して初めて分かった:

- **CUDA 13 世代の wheel は配置も変えている。** 接尾辞を落とした `nvidia-cublas` / `nvidia-cufft` / `nvidia-cuda-runtime` は、コンポーネントごとのディレクトリではなく **`nvidia/cu13/bin/x86_64/` の 1 箇所**に全 CUDA ライブラリを入れる。旧名のまま残った `nvidia-cudnn-cu13` だけが `nvidia/cudnn/bin/` という旧来の形を保っている。
- onnxruntime 1.27.0 の `_get_nvidia_dll_paths` は旧配置を**ハードコードしている**ため、`preload_dlls()` を引数なしで呼んでも cuBLAS / cuFFT / cudart を見つけられない（cuDNN だけ見つかる）。実測でも、preload を無効化すると 4 セッションすべてが `CPUExecutionProvider` に落ちた。
- したがって呼び出しは **`directory=` を明示した 2 回**になる。`preload_dlls` は与えられた 1 ディレクトリの直下しか見ないので、CUDA 側と cuDNN 側で別のディレクトリを渡す必要がある。ディレクトリはライブラリ本体（`cublasLt64_*.dll` / `cudnn64_*.dll`）を探して決めるので、上流が配置を戻しても壊れない。
- 副産物として、**torch が入っているが import されていない**環境でも供給元が nvidia wheel に固定される（`directory=None` だと onnxruntime は torch の `lib` へ迂回する）。torch 除去の前後で経路が同じになるので、ここで取った実測がそのまま除去後の予測になる。

pin したのは `nvidia-cublas` / `nvidia-cudnn-cu13` / `nvidia-cufft` / `nvidia-cuda-runtime` の 4 つ（`nvidia-cuda-nvrtc` と `nvidia-nvjitlink` は前 2 つの推移的依存として入る）。上 3 つは `onnxruntime_providers_cuda.dll` の import table に載る＝欠けると DLL の load 自体が失敗する。`nvidia-cuda-runtime` だけは import table に無いが `preload_dlls` の対象なので、欠けると起動ごとに偽の失敗メッセージが出る（2.5MB なので入れた）。

罠がもう 1 つある: **`nvidia-cublas` は win_amd64 wheel を出さない版がある**（13.6.1.10 など。13.6.0.2 にはある）。上限を CUDA メジャーで切ってあるので解決自体は通るが、Windows で wheel が無い版を掴もうとすると lock 時に失敗する。

実測（venv のディスクは +1481.2MB / 3710.5MB → 5191.7MB）: 常駐メモリは増えない、という予想は正しかった。同一プロセスで同じ 4 セッションを開いて比較すると、nvidia wheel 供給（torch を import 不能化）が 2146.2MB、torch 供給が 2292.2MB（各 N=3 の中央値）で、**供給元を移すこと自体はメモリを増やさず、むしろ torch 本体の分だけ減る。**
