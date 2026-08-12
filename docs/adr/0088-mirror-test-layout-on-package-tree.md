# 0088. テストの配置をプロダクションのパッケージ木に写し、共有フィクスチャを conftest に集める

- Status: Accepted
- Date: 2026-08-13
- 効力: 既定
- Related: [ADR-0089](0089-runtime-asset-gates-as-registered-markers.md)（同じ棚卸しから出た姉妹決定）; [ADR-0064](0064-code-comments-in-english.md)（docstring の言語）

## Context

`tests/` はフラットなまま 3 か月で 26 → 60 → 15 本と増え、92 ファイル / 974 テスト / 20,664 行に達した。棚卸しで測った結果、「どのテストが何を守っているか」も「どのモジュールがどれだけ厚いか」も、ディレクトリを見ても `pytest -q` の出力を見ても分からない状態になっている。

- **役割が書かれていない**: モジュール docstring があるのは 28/92 (30%)。付いているのは 8 月に追加された device-rate / cuda / onnx / footprint 群で、規約は事実上存在するのに明文化されていないため 7 月の 60 本には効いていない。
- **名前が対象を指していない**: 56/92 (61%) が同名のソースモジュールを持たない。実害のある衝突が 3 組ある。`test_stream_vc.py` は `vspeech/lib/stream_vc.py` を、`test_stream_vc_*.py` 19 本は**別パッケージ** `vspeech/stream_vc/` をテストしている。`test_config_stream_vc.py` と `test_stream_vc_config.py` はアナグラム同然の名前で同じ対象。`test_subtitle_dispatch.py` は 4 テスト中 3 つが config のラウンドトリップである。
- **重複がすでに 3 件発生している**。いずれも 7 月のバースト中:
  1. `test_subtitle_dispatch.py::test_obs_password_survives_a_toml_round_trip` (`37ef5b6`, 03:14) は **17 分後**の `test_config_secret.py::test_every_secret_str_field_survives_export_to_toml` (`c0a588d`, 03:31) に完全に包含された。後者はスキーマを再帰探索して SecretStr フィールドを機械的に列挙し、`subtitle.obs.password` を実際に拾う。狭いほうが生き残ったのは「secret」で探す人が開かない名前のファイルにいたからである。
  2. `test_stream_vc_config.py::test_stream_vc_defaults` (07-23) と `test_config_stream_vc.py::test_stream_vc_defaults_are_local_in_process` (07-24) が、どちらも `transport_type is in_process` の既定を主張している。
  3. `test_stream_vc_preflight.py::test_stream_vc_disabled_no_problems` と `test_preflight.py::test_disabled_recording_playback_stream_vc_skip_rate_checks` が同じ `collect_problems` の同じ側面を見ており、`_fields` ヘルパも両方にコピーされている。
- **conftest が 1 つも無い**ため、フィクスチャがコピーで増えている。25 個のヘルパ/クラス名が 2 ファイル以上で定義されており（`enabled_telemetry` ×5、`_open_log` ×4、`opened_streams` ×4、`_FakeStream` ×3 ほか）、`telemetry.reset()` / `configure()` のグローバル状態リセット儀式は 11 ファイルに 52 箇所ある。新規ファイルで書き忘れればテスト順序依存のバグになる。
- **厚さが極端に偏っていて、それが見えない**。`lib/obs_ws.py` は 131 statement に対しテスト 3,534 行 (99.2%)、`worker/subtitle_obs.py` は 145 に対し 2,217 行 (99.3%)。一方で最大モジュールの `worker/transcription.py` は 285 statement で 57.2%、`worker/tts.py` 46.6%、`main.py` 47.2%、`stream_vc/subsystem.py` 32.5%。`worker/receiver.py`（常時起動する gRPC サーバ）と `lib/ami.py` はどのテストからも import されていない。
- `scripts/` を対象とするテストの置き場も割れている。8 本が `tests/` に、4 本が `scripts/tests/` にあり、振り分ける規則が存在しない。

放置すると同じことが起き続ける。重複 3 件はすべて「既存のテストを見つけられなかった」ことが原因で、探せなかった理由はファイル名が対象を指していないことにある。

## Decision

**テストの配置をプロダクションのパッケージ木の鏡像にし、ファイル名の先頭を対象モジュール名にする。**

```
tests/
  conftest.py          共有フィクスチャ + 資産ゲート（プロジェクト全体）
  pcm.py               フィクスチャにできない純関数ビルダ（モジュールレベルから呼ぶため）
  test_<mod>.py        vspeech/<mod>.py（main / preflight / logger / exceptions / shared_context）
  test_runtime_footprint.py   例外: 特定モジュールを持たない横断ゲート
  cli/                 cli/*
  config/              vspeech/config.py（節ごとに 1 ファイル）
  lib/                 vspeech/lib/*
  stream_vc/           vspeech/stream_vc/*
  worker/              vspeech/worker/*
  scripts/             scripts/*（旧 scripts/tests/ を吸収）
  assets/
```

規則は 3 つだけ:

1. **ディレクトリ = ソースのパッケージ**。`vspeech/lib/audio.py` のテストは `tests/lib/` に置く。どのパッケージにも属さない横断ゲート（`test_runtime_footprint.py`）だけが `tests/` 直下に残り、その旨を docstring に書く。
2. **ファイル名は `test_<module>[_<aspect>].py`**。`<module>` は対象ソースモジュールの名前そのもの。1 モジュールに複数の観点があるときだけ `_<aspect>` を足す（`test_audio_rate.py` / `test_audio_resolver.py` / `test_audio_stream_close.py`）。これで `ls tests/lib/test_audio*` が厚さを直接見せる。
3. **各テストディレクトリに `__init__.py` を置く**。`tests/worker/test_playback.py` と `tests/stream_vc/test_playback.py` のように鏡像化すると basename が衝突するため、pytest の既定 import mode では必須になる。

共有フィクスチャは `tests/conftest.py` に集める。第一の対象は**グローバルシングルトンのリセット**で、`telemetry` は autouse フィクスチャで毎テスト初期化し、52 箇所の手書きリセットと書き忘れの余地を消す。

sounddevice のスタブは**フィクスチャそのものではなくファクトリ**として置く（`stub_device_table` / `record_opened_streams`）。コピーされていたのは検索と差し替えの手順であって、デバイステーブルとスタブのストリームクラスは入力側と出力側で本当に違うからで、そこは呼び出し側に残す。`_FakeStream` / `_FakeDevice` / `_OpenedStream` を 1 つに畳むと、各テストが「開かれたストリーム」について主張している内容（書き込まれたフレーム・close 順・start 失敗）まで混ざる。

モジュールレベルから呼ばれる純関数ビルダ（`i16` / `sine` / `peak_frequency`）はフィクスチャにできないので `tests/pcm.py` に置く。1 ディレクトリでしか使わないフィクスチャはそのディレクトリの `conftest.py` に置いてよい。

包含が確認された重複 3 件はこの移設に合わせて削除し、対象が同じで分かれていたファイルは統合する。

## Alternatives rejected

- **フラットのまま docstring 規約だけ足す** — 役割は読めるようになるが、探す前にファイルを開く必要がある点は変わらない。実際に起きた重複 3 件は「開かなかった」のではなく「候補として目に入らなかった」ことによる。`ls` で候補が絞れないかぎり同じことが起きる。厚さの偏りも見えないままになる。
- **`importmode = "importlib"` にして `__init__.py` を置かない** — basename 衝突は同じく解消し、ファイルは増えない。しかし import mode はスイート全体の挙動を変える設定で、pytest 本体の挙動に依存する面が広い。`__init__.py` はどの import mode でも同じ結果になり、既に `scripts/tests/__init__.py` が存在して同じ形になっている。テスト配置の整理と pytest の import 機構の変更を同時にやる理由がない。
- **共有ヘルパを全部 `tests/helpers.py` のような通常モジュールに置く** — conftest と違い明示 import が要り、autouse フィクスチャ（このリファクタで最も効く部分、telemetry リセット）が書けない。pytest のフィクスチャ探索は conftest を前提にしており、そこから外れる利点がない。逆に、フィクスチャの意味を必要としない純関数まで conftest に押し込む理由も無いので、そちらは `tests/pcm.py` に分けてある。
- **機能単位（`tests/rvc/` `tests/subtitle/` …）で切る** — 読み物としては自然だが、境界がプロダクションのどこにも対応しないので「新しいテストをどこに置くか」が再び判断になる。パッケージ木への写像は機械的で、判断の余地がないことが要点である。
- **1 モジュール 1 ファイルを厳格に強制する** — `test_obs_ws.py` は 1,317 行あり、これを 1 ファイルに閉じ込め続ける理由はない。観点ごとの分割は許し、名前の先頭を揃えることで `ls` 上のまとまりを保つほうが安い。

## Consequences

- `ls tests/lib/` が `vspeech/lib/` の目次になり、厚さの偏りが一目で分かる。逆に、テストが 1 本も無いモジュール（`lib/ami.py`、`worker/receiver.py`）はディレクトリの穴として見える。
- 新しいテストの置き場が判断ではなく写像になる。「対象モジュールのパスを `tests/` 以下に写す」だけで決まり、そこに既存ファイルがあれば重複候補として必ず目に入る。
- node ID が全部変わる。`uv run pytest tests/test_event_chains.py` のような手元のコマンドや、ドキュメント中のパスは一度に書き換わる。CLAUDE.md が名指ししているパスも同じ commit で直す。
- 空の `__init__.py` が 6 個増える（`tests/` 自身のぶんは `scripts/tests/__init__.py` の移動で賄われる）。pytest のモジュール名解決のためだけに存在する。
- 統合したファイルでは、同じ対象を見ていた 2 群のテストが 1 ファイルに並ぶ。並べた結果さらに重複が見つかることがあるが、それは分かれていた間は見つからなかったものである。
- この決定は配置だけを扱う。**どのモジュールが薄いか**（`worker/transcription.py` 57.2% ほか）は可視化されるだけで解消しない。厚みを足すかどうかは別の判断として残る。
- **既存 ADR の本文にある旧パスは直さない。** ADR は不変記録で、`tests/test_change_voice_golden.py` は当時の正しい名前である。生きているドキュメント（CLAUDE.md / `docs/secret-scanning.md` / `.gitleaks.toml` / コード中のコメント）だけを新しいパスに直し、旧パスから新パスへの対応はこの ADR が担う。
