# 0069. torch を 2.13.0 へ上げ、道連れで torchaudio を終端版 2.11.0 に固定する

- Status: Accepted
- Date: 2026-08-09
- Related: [0028](0028-migrate-to-cuda-13.md), [0036](0036-whisper-resample-via-pyav.md), [0039](0039-whisper-hosts-need-cuda12-toolkit.md)

## Context

`uv audit` の唯一の残件が torch の GHSA-rrmf-rvhw-rf47（CVE-2025-3000）だった。深刻度 LOW・ローカルのみ・`torch.jit.script` 経由で、本 repo は `torch.jit` を一切呼ばないため**到達不能として accept**してきた。その修正版が **2.13.0** として出た（OSV の fixed 範囲は 2.13.0 のみ。2.11 / 2.12.1 では直らない）。実リスクは無いが、audit に恒久的な既知 1 件が居座ると「0 件が正常」という運用ができない。

上げるにあたって効いてくる制約を実測で洗った。

- **torchaudio 2.10 の wheel metadata は `Requires-Dist: torch ==2.10.0` をハードピンしている**。torch だけを 2.13 にすると uv が unsatisfiable を返し、解決できない。torchaudio の同時昇格は選択ではなく強制。
- **torchaudio は 2.11.0（2026-03-23）が最終リリース**で、torch 2.12 / 2.13 のリリース列に対応版が出ていない。ただし 2.11 は torch ピンを持たず（`Requires-Dist` が 1 行も無い）、上流が「compatible with **future versions of torch**」と明記した前方互換版として出されている。
- torch `2.13.0+cu130` の cp314 win_amd64 wheel は存在する。`Requires-Dist` は 2.10 と同一集合で、追加の推移依存は無い。
- 同梱 CUDA DLL の集合が 2.10 と**完全一致**する（`cublas64_13` / `cudnn64_9` / `cudart64_13` …）。
- 2.11 / 2.12 / 2.13 の Backwards Incompatible Changes は、この repo が使う torch API 面（`device` / `cuda.get_device_properties` / `from_numpy` / `inference_mode` / `utils.dlpack` / `functional.pad`・`interpolate` / `clamp`）に一つも触れていない。

実機（RTX 4060 + 5060 Ti）の隔離環境で、torch 2.13 + torchaudio 2.11 の import と `Resample`（CPU / CUDA）、および torch 2.13 + onnxruntime-gpu 1.27 の**同一プロセス内 CUDA EP 同居**が動くことを確認した。

## Decision

`torch` を `2.10.0` → **`2.13.0`**（whisper / rvc の両 extra）、`torchaudio` を `2.10.0` → **`2.11.0`** に上げる。index は `pytorch-cu130` のまま動かさない。

torchaudio 2.11 は「たまたま最新」ではなく**上流が更新を止めた終端版**として固定する。以後 torch を上げるたびに torchaudio の対応版を探すことはしない。

## Alternatives rejected

- **2.11 / 2.12.1 に留める** — torchaudio との版整合（2.11 同士）は取れるが、CVE の fixed は 2.13.0 のみなので目的を果たさない。
- **torch だけ 2.13、torchaudio は 2.10 据え置き** — 解決不能。torchaudio 2.10 が `torch==2.10.0` をハードピンしており、uv が unsatisfiable を返す。
- **cu132 index へ移す** — torch 2.12 以降は cu132 ビルドもあるが、CUDA 13.2 ランタイムはより新しいドライバを要求する。cu130 で足り、[ADR-0028](0028-migrate-to-cuda-13.md) の「全 GPU ホスト R580+」という前提を動かさずに済むので採らない。
- **先に torchaudio を削除してから torch だけ上げる** — 削除は resampler の実装を差し替える作業であり、RVC の出力数値を変えうる（scipy `resample_poly` への置換は seeded golden で corr 0.976 / SNR 12.2 dB と FAIL する。リサンプラ出力同士は 52 dB 差なのに RVC チェーンが約 40 dB 増幅する）。CVE 潰しと混ぜると切り分けが効かない。削除は別 ADR で扱う。
- **CVE を accept したまま据え置く** — 実リスクは無いが、audit の既知 1 件が恒久化し、新規検出との区別にコストが乗り続ける。修正版が存在する以上、据え置く理由が無い。

## Consequences

- `uv audit` が 0 件になる。以後 audit に出た行は本物の新規として扱える。
- 依存グラフは 88 パッケージ中 **torch / torchaudio の 2 行だけ**動く。推移依存の変化は無い。wheel は 1.87 GB → 1.92 GB（+48 MB）。
- [ADR-0028](0028-migrate-to-cuda-13.md)（CUDA 13 / ドライバ R580+）と [ADR-0039](0039-whisper-hosts-need-cuda12-toolkit.md)（whisper ホストは CUDA 12 ツールキットを別途要する）は**不変**。同梱 DLL 集合が変わらないため、ホスト側の前提条件は動かない。
- torchaudio には今後上流の更新が来ない。torch を上げ続ける限り「上流が前方互換だと言っている」一点に寄りかかる状態が残る。これを断つ手は torchaudio の削除（`vspeech/lib/rvc.py` の `get_resampler` を pure-torch へ自前移植する）であり、[ADR-0036](0036-whisper-resample-via-pyav.md) が明文化した「VC/RVC パスは torchaudio」という住み分けを refine する形で別途起票する。
- RVC の seeded golden は bit-exact ではなく tolerance 判定（corr ≥ 0.999 / SNR ≥ 35 dB）なので、マイナー 3 版跨ぎでカーネル選択が変われば動きうる。**実機 GPU での golden 再実行を本 ADR の昇格条件とする**（耳確認は別途）。
