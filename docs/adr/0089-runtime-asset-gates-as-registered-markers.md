# 0089. GPU・実行時資産に依存するテストのゲートを登録済みマーカーで宣言する

- Status: Accepted
- Date: 2026-08-13
- 効力: 既定
- Related: [ADR-0088](0088-mirror-test-layout-on-package-tree.md)（同じ棚卸しから出た姉妹決定）

## Context

通常実行の 16 skip は、すべて GPU か実行時資産に依存するテストである。ゲートは各ファイルがその場で書いた `pytest.mark.skipif(...)` で、条件は 7 つの環境変数に散っている:

`VSPEECH_RVC_GOLDEN_CONFIG` / `VSPEECH_HUBERT_ASSET_DIR` / `VSPEECH_HUBERT_GOLDEN_DIR` / `VSPEECH_FCPE_ONNX` / `VSPEECH_VAD_MODEL` / `VSPEECH_STREAM_VC_CONFIG` / `VSPEECH_VVOX_ASSETS`

問題は 3 つある。

- **登録済みマーカーは `voicevox_e2e` 1 つしかない**。残りは無名のブール式なので、`pytest --markers` にも `-m` にも出てこない。「GPU のあるマシンで GPU 依存テストだけ回す」ができず、ファイルを名指しするしかない。
- **何を用意すれば skip が解けるのかが、生きたドキュメントのどこにも書かれていない**。この 7 変数が説明されているのは `docs/superpowers/plans/` と `.superpowers/sdd/` のレポートだけで、どちらも CLAUDE.md 自身が「実装後に陳腐化する使い捨て層」と宣言している場所である。
- **条件の書き方が毎回違う**。`pytestmark` でファイル全体を止めるもの、モジュールスコープの `_gpu_gate` 変数を各テストに貼るもの、デコレータに直書きするものが混在し、CUDA 判定の `_cuda_available()` は 3 ファイルにコピーされている。skip 理由の文面も英語と日本語が混ざっている。

新しい GPU 依存テストを足す人は、既存のどれかを見つけてコピーするしかない。3 つ目の `_cuda_available()` はそうやって増えた。

## Decision

**資産・環境への依存を、`pyproject.toml` に登録した `requires_*` マーカーで宣言する。** 条件の評価は `tests/conftest.py` の `pytest_runtest_setup` 1 箇所に集約し、テスト側は要求を名前で述べるだけにする。

マーカーは直交する最小単位で切り、必要なら重ねる:

| マーカー | 満たす条件 |
|---|---|
| `requires_cuda` | CUDA デバイスが 1 台以上（`list_cuda_devices()`） |
| `requires_rvc_config` | `$VSPEECH_RVC_GOLDEN_CONFIG` が実在する TOML を指す |
| `requires_rvc_golden` | `tests/assets/rvc_golden/change_voice_golden.npz` が存在する |
| `requires_hubert_assets` | `$VSPEECH_HUBERT_ASSET_DIR` と `$VSPEECH_HUBERT_GOLDEN_DIR/hubert_golden.npz` |
| `requires_hubert_fp16_golden` | `$VSPEECH_HUBERT_GOLDEN_DIR/hubert_golden_fp16.npz` |
| `requires_fcpe_onnx` | `$VSPEECH_FCPE_ONNX` が実在する onnx を指す |
| `requires_vad_model` | `$VSPEECH_VAD_MODEL` が実在するモデルを指す |
| `requires_stream_vc_config` | `$VSPEECH_STREAM_VC_CONFIG` が実在する実機 config を指す |
| `requires_torch` | `torch` が import 可能（オフラインツールの overlay 環境のみ） |
| `voicevox_e2e` | VOICEVOX 実行時資産。既存どおり `addopts` で既定除外 |

登録済みなので `uv run pytest --markers` が**そのまま生きたインデックスになる**。skip 理由は 1 箇所で生成し、満たしていない環境変数の名前と、それを用意する手段（`poe` タスクがあるものはタスク名、無いものは「machine-specific」）を必ず含める。

## Alternatives rejected

- **`skipif` のまま、条件だけ conftest のヘルパ関数に切り出す** — `_cuda_available()` の重複は消えるが、`--markers` にも `-m` にも出てこない点は変わらない。「この環境で何が走らなかったか」「GPU マシンで何を回せばいいか」が答えられないままで、これが棚卸しで見つかった問題の本体である。
- **`--run-gpu` のような CLI フラグを足す** — 明示的に opt-in する形は CI 向きだが、このプロジェクトの GPU テストは「資産がある開発機では黙って走ってほしい」もので、毎回フラグを付けるのは実態に合わない。マーカー + 自動判定なら、資産がある機械では自動で走り、`-m` で明示選択もできる。
- **マーカーを使わず、資産の有無だけで自動判定する（現状の実質的な挙動）** — 宣言が無いので、テストが「何を必要としているか」をコードから読み取るしかない。マーカーは実行時の判定だけでなく、テストの要求仕様の記述でもある。
- **`requires_gpu_rvc` のような複合マーカー 1 つにまとめる** — 貼るのは楽だが、CUDA が無いのか config が無いのかを skip 理由が区別できなくなる。実際に必要な組み合わせは「CUDA + config」「CUDA + fp16 golden」「config だけ」と異なるので、直交させて重ねるほうが理由が正確になる。

## Consequences

- `uv run pytest --markers` が資産要求の一覧になり、使い捨ての plan を掘らなくても「skip を解くには何が要るか」が分かる。
- `uv run pytest -m requires_cuda` で GPU 依存分だけを回せる。GPU 機での確認手順が「ファイル名を並べる」から「マーカーを選ぶ」に変わる。
- ゲートの評価が import 時から setup 時に移る。ファイル全体を落とすために `pytestmark` を使っていた 2 ファイルは、モジュールレベルで資産を読んでいないことを確認したうえで移した。`tests/scripts/test_export_hubert_onnx.py` のように torch をテスト関数の中で import している形は、そのまま `requires_torch` に置き換わる。
- 新しいゲートを足すときは、テストにマーカーを貼る前に `pyproject.toml` への登録が要る。`--strict-markers` は入れていないので登録漏れは警告どまりだが、登録しないと `--markers` に出ないので目的を果たさない。
- 文言の使い分けは既存の慣習をそのまま引き継ぐ。**skip 理由は英語**（1 箇所に集約。8 件中 7 件が既に英語で、assertion メッセージに近い診断出力）、**`pyproject.toml` のマーカー説明は日本語**（既存の `voicevox_e2e` と同じで、`--markers` に出る help テキストは argparse/click の `help=` と同種）。[ADR-0064](0064-code-comments-in-english.md) の線引きに沿っている。cp932 コンソールで `pytest --markers` が落ちないことは実測で確認した。
