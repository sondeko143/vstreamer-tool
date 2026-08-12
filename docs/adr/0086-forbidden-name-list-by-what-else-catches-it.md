# 0086. 名前による import 禁止を「他のゲートが捉えないもの」だけに絞り、理由は ADR 層に置く

- Status: Accepted
- Date: 2026-08-12
- 効力: 既定
- Related: [ADR-0085](0085-gate-runtime-weight-on-outcome.md)（成果ゲート。本 ADR はその棚卸し側）; [ADR-0084](0084-dependency-table-torch-gate.md)（依存表ガード。維持）; [ADR-0080](0080-torch-free-rvc-runtime.md)（`FORBIDDEN` に torch/torchaudio を足すと決めた ADR。**不変条件は変わらず、それを固定する仕組みだけが依存表ガードへ移る**）; [ADR-0022](0022-hubert-onnx-runtime.md)（transformers の理由の所在）, [ADR-0021](0021-hubert-drop-fairseq.md), [ADR-0066](0066-config-input-file-only.md)（pydantic-settings の理由の所在）; spec [2026-08-12-outcome-based-runtime-gate-design.md](../superpowers/specs/2026-08-12-outcome-based-runtime-gate-design.md)

## Context

`tests/test_forbidden_imports.py` の `FORBIDDEN` は `fairseq` / `transformers` /
`pydantic_settings` / `torch` / `torchaudio` の 5 名を `vspeech/` の import から締め出していた。
5 件は 2 年かけて**3 種類の別々の理由**（重さの防止・バージョン下限の保護・監査対象面積の抑制）で
積み上がったのに、執行する仕組みは 1 つで、理由は docstring の散文に現在形で書かれていた。
**その理由が今も成立するか、そして本当にこのゲートが捉えているのかは、一度も確かめられて
いなかった。**

実測すると、両方について食い違いが出た（すべて本リポジトリ、Python 3.14.5 / win32）。

**記録された理由の側。**

- `fairseq` の理由「requires-python を上げる際の唯一の障害（upstream は 0.12.2 で凍結・
  アーカイブ済み）」は**成立しない**。本リポジトリで `uv add --no-sync fairseq` は終了コード 0
  で通り、0.12.2 を解決する（104 パッケージ）。追加できないどころか普通に入る。
- `transformers` の理由「uv.lock に載るだけで `uv audit` に 3 件の勧告が入る」は
  **バージョン依存で、安定した根拠ではない**。同じリポジトリで `uv add --no-sync transformers`
  は 4.57.6 を選び、`uv audit --frozen` は**終了コード 1・勧告 7 件**を返す。一方
  `uv add --no-sync 'transformers>=5'` は 5.15.0 を選び、`uv audit --frozen` は
  **終了コード 0・勧告 0 件**を返す。禁止の根拠が、解決がどちらへ転ぶかと勧告データベースの
  その日の状態で反転する。
- `pydantic_settings` の重さの数値は**成立する**。N=10 の子プロセス測定で、単体 import が
  **+240 modules / +18.75 MiB**、実際の起動経路（`import vspeech.main` との差分）が
  **+31 modules / +1.51 MiB**。[ADR-0066](0066-config-input-file-only.md) の記録（32 modules /
  約 1.6 MB）を再現している。ただし ADR-0066 自身が「これ単独では動機にならない」と書いており、
  決定を支えていたのは重さではなく「対価のない経路」の方である。

**どのゲートが捉えるのかの側。** 実際に混入させて確かめた。

`vspeech/` の 55 モジュールのうち、pytest の収集が終わった時点で `sys.modules` に載るのは
**52 件**である（実測）。残る 3 件は `vspeech/__main__.py` / `vspeech/lib/voicevox.py` /
`vspeech/worker/subtitle.py` で、後ろの 2 件はテストが**実行時に** import する
（`tests/test_voicevox_lib.py:94` の `importlib.import_module`、
`tests/test_subtitle_dispatch.py:40`）。この 3 分類すべてで混入させた。

- 収集時に載る 52 件の側。`vspeech/lib/ami.py` に `import torch` を 1 行入れると、pytest は
  **収集段階で終了コード 2** になる（`tests/test_transcription_helpers.py` と
  `tests/test_transcription_worker.py` が `ModuleNotFoundError: No module named 'torch'` で
  collection error）。**名前ゲートは走りさえしない。**
- 実行時に import される 2 件の側。`vspeech/lib/voicevox.py` と `vspeech/worker/subtitle.py`
  の両方に同じ 1 行を入れると **1 failed + 3 errors**（`test_subtitle_dispatch.py` 1 件と
  `test_voicevox_lib.py` 3 件、いずれも `ModuleNotFoundError`）。どれも skip されない。
- `vspeech/__main__.py` の側。同じ 1 行で **6 件**が落ちる。当時まだ載っていた名前ゲート 1 件と、
  `tests/test_main.py` のエントリポイント smoke 5 件である。
- 逆に `vspeech/lib/rvc.py` に `from transformers import HubertModel`、`vspeech/worker/vc.py`
  （遅延ロードされる worker）に `import pydantic_settings` を入れ、両者をインストールした状態で
  フルスイートを回すと、**落ちるのは名前ゲートの 2 件だけ**（2 failed, 1175 passed）。依存表
  ガードも成果ゲートも緑のままだった。

つまり **`torch` にとって名前ゲートは重複の信号**であり、`transformers` と
`pydantic_settings` にとっては**唯一の信号**である。同じ 1 つのリストに載っていたせいで、
この差は見えなかった。

## Decision

`FORBIDDEN` に載せる基準を「**その名前が戻ってきたときに、他のすべてのゲートが緑のままか**」
1 つに定める。緑のままなら載せる。どれかが赤くなるなら載せない。

結果として残るのは `transformers` と `pydantic_settings` の 2 件で、`fairseq` / `torch` /
`torchaudio` は外す。**[ADR-0080](0080-torch-free-rvc-runtime.md) の不変条件（`vspeech/` が
torch / torchaudio を import しない）は変わらない。** 変わるのはそれを固定する仕組みで、
名前ゲートから依存表ガードと（インストールされていない以上、必ず落ちる）スイート自身へ移る。
外した 3 件を代わりに守るものは次のとおり。

- `torch` / `torchaudio` — [ADR-0084](0084-dependency-table-torch-gate.md) の依存表ガード。
  pyproject 側は 3 つの異なるテーブル（`dependencies` / `optional-dependencies.rvc` の URL 指定 /
  `dependency-groups`）と綴り違い（`Torch` / `faiss_cpu`）で、uv.lock 側は宣言なしの純推移
  ケースで、それぞれ混入させて 6 パラメトリゼーションすべてが発火することを確認した。
- `fairseq` — 同じ依存表ガードが 1 辺先で受け持つ。`uv add --no-sync fairseq` は torch 2.13.0 と
  torchaudio 2.11.0 を uv.lock へ引き込み、解決済み集合の検査が 2 件落ちる（このとき
  `fairseq` の名前ゲートは緑のままだった）。

**理由はテストに書かない。** 各エントリは理由を持つ ADR への**パス**だけを持ち、ゲートの失敗
メッセージがそのパスを印字する。パスが腐ると失敗メッセージの価値が消えるので、パスの実在と
「その文書が実際にその名前に言及していること」をテストで固定する。理由の本文が 2 箇所に
なるのを避けるためであり、今回失効を招いたのはまさにその転記だった。

## Alternatives rejected

- **5 件をそのまま維持する** — 実測で 2 件の理由が現状を説明しておらず、3 件は他のゲートと
  重複していた。放置すれば「もう理由のない規則」を執行し続ける状態が続く。今回それが
  1 度は起きたことが確認された以上、次も気づかれない。
- **5 件すべて外す** — `transformers` と `pydantic_settings` は、インストールしても依存表
  ガードが緑（transformers 4.57.6 / 5.15.0 のどちらも torch を引かない、pydantic-settings 2.15.0
  も同様）で、遅延ロードされる worker から import すれば成果ゲートにも映らない。実測で
  「名前ゲートだけが落ちる」ことを確認しており、外せば無防備になる。
- **成果ゲート（[ADR-0085](0085-gate-runtime-weight-on-outcome.md)）に吸収させる** —
  `pydantic_settings` については起動経路上の locus なら捉える（716 → 747 modules、予算 732 を
  超過）。しかし (a) 遅延 worker の locus は原理的に射程外で、(b) 捉えたときの案内が
  「意図した変更なら基準を採り直せ」であり、**意図して外したものに対しては誤った指示**になる。
  常駐メモリ側は 58.81 → 60.32 MiB で予算 64.0 MiB に届かず、そもそも捉えない。
- **理由を pyproject.toml のコメントへ転記する** — 本ブランチで一度起票して撤回した案
  （commit 7450c90）。理由は既に ADR にあり、写せば 2 箇所保守になる。理由が要るのはゲートが
  赤くなった瞬間であって、配達先はテストの失敗メッセージである。
- **`uv audit` の結果を禁止の根拠に使い続ける** — 上記のとおり同一リポジトリで解決先により
  終了コード 1 と 0 の両方になる。ゲートの根拠としては不安定すぎる。監査は
  `poe audit` が独立に回しており、そちらの仕事である。

## Consequences

名前リストは「重さの代理」であることをやめ、「**他のどのゲートも捉えない決定境界**」だけを
持つ。エントリが 5 件から 2 件に減り、残る 2 件はどちらも、外れれば実際に無防備になることが
混入実験で示されている。

失敗メッセージから理由へ 1 ホップで到達できる。理由の本文はテストにも設定にも存在しないので、
[ADR-0022](0022-hubert-onnx-runtime.md) / [ADR-0066](0066-config-input-file-only.md) を
supersede すれば、テスト側を触らずに理由が更新される。

**各エントリは ADR を 2 本持つ。** 1 本目は「ランタイムから外すと決めた ADR」、2 本目は常に
本 ADR である。理由が 2 つに割れているためで、片方だけでは読者を誤らせる。とくに ADR-0022 は
自身の根拠を「lock に載る勧告」と述べており、それは本 ADR が**バージョン依存**と実測した当の
ものである。1 本目だけを指すと、読者は**このプロジェクトがもう依拠していない根拠**に着地し、
「その根拠はもう無いのだから解禁してよい」と読める。2 本目が「他のどのゲートも捉えない」という
現に効いている理由を運ぶ。

外した 3 件について、代わりに守るゲートと残余リスクを
`tests/test_forbidden_imports.py` の `FORBIDDEN` の直下にコメントとして残した。残余は 3 つある。

**(1) 手で venv に入れた torch は [ADR-0084](0084-dependency-table-torch-gate.md) が意図的に
見ていない**（オフラインの `uv run --with` オーバーレイが正当に torch を持つため）ので、その
状態の venv でだけ `vspeech/` の `import torch` が緑になる。

**(2) torch への辺は古い版まで遡ると切れる。** `fairseq<0.12` は 0.11.1 に解決してなお torch を
引くが、`fairseq==0.6.2` は torch を要求しない（本リポジトリで実測: lock 86 → 92 パッケージ、
torch なし、`tests/test_forbidden_imports.py` は**どのゲートも発火せず終了コード 0**）ので、
明示的にピン止めすれば依存表ガードをすり抜ける（ただし 2019 年のリリースで HuBERT を含まず、
このリポジトリのオフライン変換器が指せる対象ではない）。

**(3) 依存表ガードを「もっと硬く」しようとすると、fairseq の保護が消える。** `[tool.uv]` に
`constraint-dependencies = ["torch<0"]` を置くのは自然な強化に見えるが、そうすると**バージョン
指定なしの `uv add fairseq` が終了コード 0 で通り、torch を要求しない 0.6.2 へ黙って後退する**
（本リポジトリで実測: fairseq 0.6.2 / torch なし / 92 パッケージ、
`tests/test_forbidden_imports.py` は**どのゲートも発火せず終了コード 0**）。
[ADR-0085](0085-gate-runtime-weight-on-outcome.md) が `constraint-dependencies` を ban として
却下したのと同じ挙動が、ここでは**既存の保護を無効化する**形で出る。**誰もゲートに触らないまま
fairseq の守りが消えるので、この強化を入れるなら fairseq を `FORBIDDEN` へ戻すこと。**

この基準は将来 `FORBIDDEN` に名前を足すときの判定でもある。**足す前に、その名前が戻ってきた
ときに他のゲートが緑のままかを実際に混入させて確かめること。** 緑にならないなら、そのゲートに
任せる方が保守対象が 1 つ少ない。
