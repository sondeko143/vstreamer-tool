# 0084. torch 不在を pyproject と uv.lock に対するテストで守る

- Status: Superseded by [ADR-0087](0087-weight-guarded-by-measurement-not-named-bans.md)
- Date: 2026-08-12
- 効力: 制約
- Related: [ADR-0080](0080-torch-free-rvc-runtime.md)（この決定が塞ぐ穴を Consequences で自認していた ADR）; [ADR-0078](0078-torch-free-device-resolution.md)（ctranslate2 と torch の関係を記録）

## Context

[ADR-0080](0080-torch-free-rvc-runtime.md) の成果（起動 20.33s → 5.56s、常駐 -185MB / -421MB、venv 5191.7MB → 2336.1MB）は、**コードが torch を import しないこと**ではなく **venv に torch が入っていないこと**に依存している。`ctranslate2`（コア依存）が `try: import torch` で掴むため、1 行も import していなくても、入っていれば全パイプラインが +476.7MB / +3.17s を払う。

ところが既存のガードはコードしか見ていなかった。`tests/test_forbidden_imports.py` の AST ゲートは `vspeech/` の import 文を走査するだけで、依存表には触れない。ADR-0080 自身の Consequences がこれを自認している —「守っているのは構造ガードではなく依存表であり、`test_forbidden_imports.py` はコード側しか見ていない」。

したがって `uv add torch` 1 回、あるいは既存依存が torch への推移辺を 1 本生やしただけで、**既存のゲートが全部緑のまま**削減が消える。ブランチの利益そのものが無防備だった。

## Decision

`torch` / `torchaudio` / `faiss-cpu` が**解決後の依存集合に現れたらテストを落とす**。突き合わせ先は 2 つで、両方ともリポジトリ内のファイルである。

- **`pyproject.toml`** — `project.dependencies` / 全 `optional-dependencies` / 全 `dependency-groups` を PEP 508 として読み、宣言された配布名を PEP 503 正規化して照合する。`torch @ https://...whl` のような URL 指定も、`Torch` / `faiss_cpu` のような綴り違いも同じ 1 つの名前に落ちる。
- **`uv.lock`** — `[[package]]` の名前を同じ規則で照合する。lock は本プロジェクト唯一の環境（`sys_platform == 'win32'`）に対する解決後の集合なので、**どのテーブルも宣言していない推移辺がここに現れる**。宣言側だけでは捕まらない経路を捕まえるのはこちら。

ガード自身が空回りしていないことも同じファイルで固定する（宣言のみのケース・推移のみのケース・類似名 `torchvision` の偽陽性）。置き場所は `tests/test_forbidden_imports.py` — 「torch を戻さない」という 1 つの不変条件が 1 ファイルに閉じ、[ADR-0024](0024-onnx-session-single-factory.md) が別の文脈で潰した「ガードの二重化」を作らない。

## Alternatives rejected

- **インストール済み環境（`importlib.metadata`）を見る** — 最も直接的だが、テストが走る環境の状態に依存する。この環境は `uv run --with` のオーバーレイでオフラインツールを動かすことがあり、そのときは torch が正当に存在する。CI 上での再現性も無い。「入っているか」ではなく「入れることになっているか」を問うのが正しい。
- **`uv lock --check` に任せる** — lock と pyproject の同期しか見ない。両方に torch が入っていれば緑のままで、この件について何も言わない。
- **AST ゲートを拡張する（`import torch` を全ツリーで禁じる）** — 因果が逆。問題は import ではなくインストールであり、オフラインツール（`scripts/export_hubert_onnx.py` など）は torch を import して**よい**。
- **何もせず ADR-0080 の Consequences の注意書きに委ねる** — 実際に守るのは人間の記憶になる。削減の全額が 1 コマンドで消えるのに、その 1 コマンドが緑で通る。
- **`torchvision` など torch 系全部を一括禁止する** — 現に問題を起こす 3 つ（依存表に載っていた `torch` / `torchaudio` / `faiss-cpu`）に限る。将来必要になったら足す方が、根拠のない禁止リストを育てるより安い。

## Consequences

ADR-0080 の削減が、記憶ではなくテストで守られる。`uv add torch` は次の `pytest` で赤くなり、メッセージが理由（ctranslate2 が掴む・477MB と 3.2s）を述べる。

lock 側の照合があるので、**推移辺で入ってきた場合も捕まる**。ただし赤くなるのは lock を取り直したあとであり、`uv lock` を走らせずに `pyproject.toml` を手で編集した段階では宣言側の照合が受け持つ。2 つある理由がこれで、どちらか一方では穴が残る。

このゲートは**依存を戻す判断そのものを禁じてはいない**。torch を再び入れる決定をするなら、本 ADR を supersede したうえで `FORBIDDEN_DISTRIBUTIONS` から外すこと。テストを個別に skip したり除外したりして通すのは、ADR-0080 の受入根拠を黙って捨てることになる。

`faiss-cpu` は import 名ではなく配布名として載せている（このリポジトリのどの `.py` も import していない）。「import されないもの」を import ゲートで守ることはできず、それが依存表を見るこのゲートが要る理由の 2 つめでもある。
