# 0045. GUI の起動前 readiness は vspeech.preflight を単一の権威として再利用する（GUI 側に必須項目マップを持たない）

- Status: Proposed
- Date: 2026-07-17
- Related: spec [2026-07-16-gui-run-readiness-design.md](../superpowers/specs/2026-07-16-gui-run-readiness-design.md); [ADR-0038](0038-worker-config-preflight-fail-loud.md)（起動時 preflight の fail-loud、本 ADR が再利用する権威）; [ADR-0032](0032-gui-multi-pipeline-rewrite.md)（gui → vspeech の一方向依存）

## Context

GUI から pipeline を起動しても、必須の素材・資格情報が欠けていると起動後に worker がモデル／デバイス読込で落ちる。ユーザーは Start → 即死 → 下部の生ログを読む、の往復を強いられる。この「起動成功に何が要るか」は config の pydantic validation では出ない — ほぼ全フィールドが既定値を持つため `model_validate` は通ってしまう。

一方 [ADR-0038](0038-worker-config-preflight-fail-loud.md) は既に vspeech 側へ集中 preflight（層A）を確立している。enable 済み worker だけを対象に、必須フィールド・参照ファイル／ディレクトリの存在・デバイス発見可否・依存の有無を実リソース取得なしで安価に検査し、検出した全問題を `ConfigProblem(worker, message)` のリストとして 1 つの `ConfigError` に集約する。ACP の必須 4 項目、GCP の key.json 実在、voicevox の辞書／モデル、rvc の model／hubert／rmvpe（`f0_extractor_type` 条件付き）、デバイス解決、`routes_list` の妥当性は、すべて既にそこにある。

つまり「何が必須か」の権威は既に存在する。GUI に同じ知識をもう一組持てば worker / preflight / GUI の三重持ちになる。[ADR-0038](0038-worker-config-preflight-fail-loud.md) は自身の Consequences で既に 2 箇所（preflight と worker）を更新する保守コストを警告しており、3 箇所目は drift を確実にする。

## Decision

GUI の起動前 readiness は **`vspeech.preflight` を単一の権威として再利用**し、GUI 側に必須項目の宣言マップを持たない。[ADR-0032](0032-gui-multi-pipeline-rewrite.md) が定めた gui → vspeech の一方向依存にそのまま乗る。

これを可能にするため vspeech 側へ 2 つの**加算的**変更を行う（既存の起動時挙動・ログ出力は不変）:

- 問題収集を `collect_problems(config) -> list[ConfigProblem]` として `preflight()` から分離する。`preflight()` はそれを呼び非空なら `ConfigError` を送出する薄い wrapper に留める。GUI は例外を制御フローに使わず `collect_problems` を呼ぶ。
- `ConfigProblem` に**省略可能な構造化フィールド** `field`（例 `"rvc.model_file"`）を足す。GUI が問題からその設定箇所へ移動するために要る。

readiness は問題（✗）の集合として得られ、enable 済みで問題ゼロの worker を ✓ として描く。これにより GUI が起動前に見せる判定は、起動時に preflight が fail-loud する判定と**構成上同一**になり、必須リソースが増えても preflight を更新すれば GUI は自動追随する。

## Alternatives rejected

- **GUI 側に worker × backend の宣言的 requirements マップを新設する** — 同じドメイン知識の 3 つ目の複製になり、[ADR-0038](0038-worker-config-preflight-fail-loud.md) が既に警告する 2 箇所更新コストを 3 箇所へ悪化させる。preflight と GUI がズレると「GUI は緑なのに起動して落ちる／その逆」という最悪の形で顕在化する。
- **`ConfigProblem` のメッセージ文字列から欄名を正規表現で抽出する** — メッセージ文面を変えた瞬間に黙って壊れる。構造化情報を文字列から復元するのは drift。だから `field` を構造として足す。
- **preflight を GUI から subprocess で走らせて結果を読む** — 同一プロセスで import できる（[ADR-0032](0032-gui-multi-pipeline-rewrite.md) の gui → vspeech 依存）のに、起動コストとシリアライズ層を足すだけ。
- **pydantic の validation を必須化して readiness の根拠に使う** — [ADR-0038](0038-worker-config-preflight-fail-loud.md) が既に却下済み（環境 I/O を config ロードに混ぜると、config を読むだけの経路でも走る）。加えて既定値を消して必須化すると、その worker を使わない pipeline の config まで読めなくなる（`Config` は常に全 worker 分のフィールドを持つ）。

## Consequences

- GUI の緑／赤が起動時 preflight と一致することが構成上保証され、必須リソース追加時に GUI を追随させる作業が消える。
- vspeech の `ConfigProblem` は GUI が読む公開契約になる（境界の拡大）。`field` の付与は加算的で、既存の整形ログ出力は不変。
- readiness の粒度は preflight の粒度に縛られる。層B の失敗（モデル実ロード、GPU 確保、VOICEROID2 の常駐、OBS 接続）は原理的に起動前には出せず、起動後の失敗バナーで拾うほかない。
- preflight はデバイス列挙などの I/O を伴うため、GUI はフォーム変更ごとの無制限呼び出しを避ける必要がある。
- audio extra 未導入の環境では preflight のデバイス検査が import 段で失敗しうる。GUI は readiness の評価失敗そのもので落ちてはならない。
