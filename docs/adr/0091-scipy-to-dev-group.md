# 0091. scipy を rvc extra から dev グループへ移す

- Status: Accepted
- Date: 2026-08-13
- 効力: 既定
- Related: [ADR-0080](0080-torch-free-rvc-runtime.md)（この判断を保留にした ADR）; [ADR-0082](0082-rvc-resample-on-inhouse-polyphase.md)（scipy の最後の実質用途を消した）; [ADR-0070](0070-f0-voiced-run-median-filter.md)（テスト側で残った唯一の用途）; [ADR-0085](0085-gate-runtime-weight-on-outcome.md)（外して安全だと言える根拠）

## Context

`scipy` は `rvc` extra に入っていたが、[ADR-0082](0082-rvc-resample-on-inhouse-polyphase.md) がリサンプルを自前ポリフェーズへ一本化した時点で、ランタイムからの用途が無くなっていた。[ADR-0080](0080-torch-free-rvc-runtime.md) はこれに気付いていたが、torch/torchaudio/faiss-cpu を外すという自分の決定とは別件だとして保留し、`pyproject.toml` の当該行に「次にここを触る人が、rvc extra から落として dev へ移すかを決めること」と書き残していた。本 ADR がその決定である。

外して安全だと言える根拠を 4 つとも確認した。

- **`vspeech/` / `cli/` / `scripts/` からの import は 0 件。** 実 import は `tests/lib/test_pitch_extract.py` の 1 行だけで、他のソース中の "scipy" はすべて「使っていない」と書いた散文かコメントである。
- **lock 上で scipy を要求しているのは `voicerecog` 自身のみ。** 推移依存で入ってくる経路が無いので、要求を外せば解決結果から本当に消える。
- **重さゲートの 11 測定パスはどれも scipy を読み込まない**（[ADR-0085](0085-gate-runtime-weight-on-outcome.md) の `tests/runtime_startup_baseline.json`）。ランタイム経路に載っていないことが、名指しの grep ではなく計測で裏づけられている。
- サイズは wheel 35.6MiB / 展開 **100MB**。[ADR-0036](0036-whisper-resample-via-pyav.md) と [ADR-0073](0073-device-boundary-inhouse-polyphase-resampler.md) が「120MB を足すのは割に合わない」として scipy を却下してきたのと同じ重さが、rvc 側には残っていた。

テスト側の用途は消せない性質のものである。`tests/lib/test_pitch_extract.py` は `median_filter_f0` が**素朴な `medfilt` と違う**ことを主張するために `signal.medfilt` を独立参照として使う（[ADR-0070](0070-f0-voiced-run-median-filter.md) の決定そのもの — 有声区間単位で掛けるので、ゼロ詰めで走る素朴版とは結果が変わる）。ここを自作の素朴 medfilt で置き換えると、自分のコードを自分のコードと比べることになりテストの意味が失われる。

## Decision

**`scipy` を `rvc` extra から外し、`[dependency-groups] dev` へ移す。** バージョン式（`>=1.10.1,<2`）はそのまま持ち越す。

`[tool.uv] default-groups = "all"` なので、`uv sync` 系はこれまでどおり scipy を入れる — 開発・テスト環境の見た目は変わらない。変わるのは `--no-dev` を付けたランタイム install で、そこから 100MB が消える。

## Alternatives rejected

- **`rvc` extra に置いたままにする** — 保留の理由（別の決定だから）は ADR-0080 が閉じた時点で消えている。ランタイムが使わない 100MB を、GPU 変換を回すためだけのマシンが運び続ける理由が無い。
- **scipy を完全に落とし、`medfilt` を自前実装に置き換える** — 依存は消えるが、テストが「独立した参照実装との差」を主張できなくなる。ADR-0070 の主張は素朴な medfilt との**違い**そのものなので、比較対象を自作にするとトートロジーになる。100MB は dev グループなら払う価値がある。
- **テストごと消す** — 論外。ADR-0070 の決定を守っている唯一のテストである。
- **`test` のような専用グループを新設する** — 既存の dev グループが既にテスト依存（pytest 一式・onnx）の置き場になっており、グループを増やすと `uv sync` の指定が増えるだけで区別の実益が無い。

## Consequences

- ランタイムの `rvc` extra が **100MB 軽くなる**。[ADR-0083](0083-cuda-runtime-from-nvidia-wheels.md) の nvidia wheel 群（展開 1,481MiB）に比べれば小さいが、こちらは**誰も使っていない** 100MB である。
- 重さゲートの baseline は**変わらない**。scipy はどの測定パスにも載っていなかったので、再記録は不要（実際 `test_runtime_footprint.py` は緑のまま）。
- `--no-dev` でランタイムだけを入れた環境では `tests/lib/test_pitch_extract.py::test_median_filter_f0_does_not_pull_voiced_run_edges_toward_unvoiced` が `ImportError` で落ちる。テストを走らせる環境に dev グループを入れないという運用は元から成り立たないので、ゲートは足さない。
- 将来 `vspeech/` 側で scipy が要るとなったら、この移動を戻すのではなく **extra へ足す判断を新しく起票する**。[ADR-0036](0036-whisper-resample-via-pyav.md) と [ADR-0073](0073-device-boundary-inhouse-polyphase-resampler.md) が同じサイズを理由に却下しているので、そのときは両者との突合が要る。
