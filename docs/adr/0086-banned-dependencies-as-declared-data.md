# 0086. 残す禁止依存を、理由と失効条件つきの宣言データにする

- Status: Proposed
- Date: 2026-08-12
- 効力: 既定
- Related: [ADR-0085](0085-gate-runtime-weight-on-outcome.md)（重さの分は成果ゲートへ移る）; refines [ADR-0084](0084-dependency-table-torch-gate.md)（依存表ガードの読み先を変える）; spec [2026-08-12-outcome-based-runtime-gate-design.md](../superpowers/specs/2026-08-12-outcome-based-runtime-gate-design.md)

## Context

[ADR-0085](0085-gate-runtime-weight-on-outcome.md) が重さの分を成果ゲートへ移したあとも、成果では捉えられない禁止は残る。`fairseq` はバージョン下限の障害、`transformers` は監査対象面積といったもので、どちらも「そのパッケージ固有の事情」であり、どんな測定でも捕まらない。

問題はその**理由の置き場**である。現在は `tests/test_forbidden_imports.py` の docstring に散文で書かれており、

- 依存を追加する人はテストの docstring を読まない。依存表（`pyproject.toml`）とゲートが別ファイルに分かれているので、ゲートの存在に気づく契機がない。
- **いつその根拠が切れるかが書かれていない。** 実際 `transformers` の根拠（`uv audit` の勧告 3 件）は失効しており、実測では現在 `uv audit` が終了コード 0 を返す。誰も気づかないまま執行が続いていた。

cargo-deny の `[bans]` は理由を第一級のデータとして持つが、Python にその相当物は無く、uv の `constraint-dependencies` は ban として機能しない（[ADR-0085](0085-gate-runtime-weight-on-outcome.md) の却下案に実測）。

## Decision

残す禁止依存を `pyproject.toml` の**宣言データ**として持つ。各エントリは最低限、対象・**理由**・**失効条件**（何が起きたらこの禁止を見直すか）を持つ。テストはこの表を読むだけにし、名前と理由をテスト側に二重に書かない。

置き場を `pyproject.toml` にするのは、**依存を触る人が必ず開くファイルだから**である。禁止の一覧が依存の宣言と同じ画面にあることが、この決定の目的の半分を占める。

失効条件を持たせたことの帰結として、**エントリの棚卸しは実測で行う**。理由が成立しているかを確かめられない禁止は、残す根拠がない。

## Alternatives rejected

- **docstring の散文のまま置く** — 失効条件を書く場所が構造として無く、実際に失効を見逃した（`transformers`）。依存を触る人の視界にも入らない。
- **テストのソースに Python の dict として置く** — 宣言データにはなるが、依存の宣言から離れたまま。依存を追加する人がゲートの存在に気づく契機は増えない。
- **cargo-deny 相当の外部ツールを導入する** — Python に定着したものが無い（[ADR-0085](0085-gate-runtime-weight-on-outcome.md)）。
- **`[tool.uv] constraint-dependencies` に理由をコメントで添える** — 依存表の中に置ける点は望ましいが、ban として機能しない。推移依存では禁止を報告せず古いバージョンへ後退する（実測: `uv add fairseq` が torch を要求しない 0.6.2 を選んで成功）。機能しない機構に理由を添えても、理由ごと機能しない。
- **禁止をすべて廃止し、成果ゲートと `uv audit` だけに任せる** — 成果では捉えられない禁止が実在する。`fairseq` のバージョン下限障害はどんな測定でも出ず、`uv audit` は勧告が修正されれば緑になる（実測でそうなっている）。

## Consequences

禁止エントリの追加が「テストを編集する」から「依存表にデータを 1 行足す」に変わる。理由と失効条件が必須になるので、理由を言語化できない禁止は追加できない。

**失効条件は自動では検証されない。** 書いてあるだけでは、次に誰かが読むまで失効に気づかない。これは現状（書く場所すらない）より良いが、自動検証には至っていない。`transformers` のように `uv audit` で機械的に確かめられるものは、失効条件をそのコマンドで書ける。

`pyproject.toml` に本プロジェクト固有のテーブルが 1 つ増える。uv も他のツールも読まない領域なので、解決や配布への影響は無い。
