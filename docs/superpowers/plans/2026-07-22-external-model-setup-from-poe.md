# 外部モデル導入を poe 起点で辿れるようにする Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Silero VAD / rmvpe / FCPE / HuBERT の導入方法を `uv run poe` のタスク一覧と `poe <task> --help` から具体的に辿れるようにする。

**Architecture:** ダウンロードはせず案内のみ。(1) 純 stdlib の案内スクリプト `scripts/setup_models.py` を新設し `poe models` から表示、(2) HuBERT の 2 タスクに FCPE 式 `--help` epilog を付与、(3) `config.toml.example` の rmvpe/Silero VAD 箇所から `poe models` へ導線を張る。FCPE (ADR-0049) の discoverability を他 3 つに揃える。

**Tech Stack:** Python 3.14 / argparse / poethepoet (`poe_tasks.toml`) / pytest。

**ADR: none** — 可逆な doc/DX プラミングで契約変更なし。却下した代替 (自動 fetch タスク) は保留中の GUI 自動取得フィーチャの領域で、そちらが独自の ADR を持つ。

## Global Constraints

- 自動ダウンロード/取得機構は作らない (curl/fetch を行うタスクを増やさない)。案内のみ。
- `THIRD_PARTY_NOTICES.md` がライセンスの単一情報源。案内はそこへ辿らせるだけで再掲しない。
- 日本語 stdout の Windows 文字化け対策: `if isinstance(sys.stdout, io.TextIOWrapper): sys.stdout.reconfigure(encoding="utf-8")`（`export_fcpe_onnx.py:43` と同型）。shell-echo は使わない。
- import は 1 行 1 つ (ruff `force-single-line`)。ruff format / ruff check / ty check が緑であること。
- 入手元 (verbatim): Silero VAD = `snakers4/silero-vad` の `silero_vad.onnx` (v6.2.1, MIT) / rmvpe = `wok000/weights_gpl` の `rmvpe_20231006.onnx` (GPL-3.0) / ContentVec = `hubert_base.pt` (RVC 配布, origin `auspicious3000/contentvec`, MIT) / FCPE = `uv run poe export-fcpe-onnx`。
- config キー (verbatim): `[vc] vad_model_file` / `[transcription] vad_model_file` / `[rvc] rmvpe_model_file` / `[rvc] fcpe_model_file` / `[rvc] hubert_model_file` / `[rvc] f0_extractor_type`。
- poe タスクは `poe_tasks.toml` の `[tool.poe.tasks]` に定義し、`uv run poe` で一覧に出る。

## File Structure

- `scripts/setup_models.py` (新規) — 案内テキストを組み立て `build_guide() -> str` で返し、`main()` で print する。torch 等を import しない純 stdlib。責務は「外部モデルの入手元 + config キーの表示」だけ。
- `scripts/tests/test_setup_models.py` (新規) — `build_guide()` の本文をアサート。
- `poe_tasks.toml` (変更) — `models` タスクを追加。
- `config.toml.example` (変更) — rmvpe / Silero VAD 箇所に `uv run poe models` 導線を追記。
- `scripts/convert_hubert.py` (変更, `main()` の argparse) — `--help` epilog 付与。
- `scripts/export_hubert_onnx.py` (変更, `main()` の argparse) — `--help` epilog 付与。

---

### Task 1: `poe models` 案内タスク

**Files:**
- Create: `scripts/setup_models.py`
- Test: `scripts/tests/test_setup_models.py`
- Modify: `poe_tasks.toml`（`export-fcpe-onnx` 行の直後にタスク追加）

**Interfaces:**
- Produces: `scripts.setup_models.build_guide() -> str`（案内テキスト全文を返す。テストと `main()` が使う）、`scripts.setup_models.main() -> None`。

- [ ] **Step 1: 失敗するテストを書く**

Create `scripts/tests/test_setup_models.py`:

```python
from scripts import setup_models


def test_guide_lists_download_only_models_with_sources_and_keys():
    guide = setup_models.build_guide()
    # Silero VAD (入手元 + config キー)
    assert "snakers4/silero-vad" in guide
    assert "vad_model_file" in guide
    # rmvpe (入手元 + GPL + NOTICES 導線 + config キー)
    assert "wok000/weights_gpl" in guide
    assert "GPL-3.0" in guide
    assert "THIRD_PARTY_NOTICES" in guide
    assert "rmvpe_model_file" in guide


def test_guide_points_derived_models_to_their_own_poe_tasks():
    guide = setup_models.build_guide()
    # FCPE / HuBERT は各自の poe タスク --help が詳細の持ち主
    assert "poe export-fcpe-onnx" in guide
    assert "fcpe_model_file" in guide
    assert "poe convert-hubert" in guide
    assert "poe export-hubert-onnx" in guide
    assert "hubert_model_file" in guide
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest scripts/tests/test_setup_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.setup_models'`（またはコレクションエラー）。

- [ ] **Step 3: 案内スクリプトを実装**

Create `scripts/setup_models.py`:

```python
"""外部モデル (Silero VAD / rmvpe / FCPE / HuBERT) の導入方法を表示する。

`uv run poe models` から起動する案内専用スクリプト。ダウンロードはせず、入手元と設定する
config キーだけを示す。ライセンス詳細は THIRD_PARTY_NOTICES.md が単一情報源で、ここはそこへ
辿らせる。純 stdlib のみ (torch 等を import しない) なのでプロジェクト環境でそのまま実行・
import できる。
"""

import io
import sys

# 日本語を Windows の cp932/cp1252 stdout でも壊さない (プロジェクト頻出の encoding 対策)。
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

GUIDE = """\
外部モデルの導入ガイド (このリポジトリはモデルを同梱しません)
==============================================================

RVC / VC / transcription が使う外部モデルは各自取得し、config のパスに設定します。
ライセンス詳細は THIRD_PARTY_NOTICES.md を参照してください。

■ Silero VAD  (MIT)
  用途 : VAD ノイズゲート
  入手 : snakers4/silero-vad の silero_vad.onnx (v6.2.1)
         https://github.com/snakers4/silero-vad
  設定 : [vc]            vad_model_file = "~/.config/vstreamer/silero_vad.onnx"
         [transcription] vad_model_file = "~/.config/vstreamer/silero_vad.onnx"

■ rmvpe  (GPL-3.0 — 詳細は THIRD_PARTY_NOTICES.md の第3節)
  用途 : f0 抽出 (既定)
  入手 : wok000/weights_gpl の rmvpe_20231006.onnx
         https://huggingface.co/wok000/weights_gpl
  設定 : [rvc] f0_extractor_type = "rmvpe"
              rmvpe_model_file   = "~/.config/vstreamer/rmvpe.onnx"

■ FCPE  (rmvpe より高速・低精度; 任意)
  用途 : f0 抽出 (rmvpe の代替)
  入手 : uv run poe export-fcpe-onnx --output ~/.config/vstreamer/fcpe.onnx
         (手動ダウンロード不要。詳細は `uv run poe export-fcpe-onnx --help`)
  設定 : [rvc] f0_extractor_type = "fcpe"
              fcpe_model_file    = "~/.config/vstreamer/fcpe.onnx"

■ HuBERT / ContentVec  (MIT)
  用途 : RVC content encoder
  入手 : hubert_base.pt (RVC が配布する ContentVec; origin auspicious3000/contentvec) を
         2 段変換して ONNX 資産化する。手順は各タスクの --help を参照:
           uv run poe convert-hubert     --help
           uv run poe export-hubert-onnx --help
  設定 : [rvc] hubert_model_file = "<変換で出力した資産ディレクトリ>"

RVC 声モデル (rvc.model_file) は利用者が用意します (この案内の対象外)。
"""


def build_guide() -> str:
    """案内テキストを返す (テストが本文をアサートできるよう関数化)。"""
    return GUIDE


def main() -> None:
    print(build_guide())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest scripts/tests/test_setup_models.py -v`
Expected: PASS（2 tests passed）。

- [ ] **Step 5: `poe models` タスクを追加**

`poe_tasks.toml` の `export-fcpe-onnx = { ... }` 行（52 行目）の直後に追記:

```toml
# 外部モデル (Silero VAD / rmvpe / FCPE / HuBERT) の入手方法と設定キーを表示するだけの案内
# タスク。ダウンロードはしない (rmvpe は GPL、URL 陳腐化、保留中の GUI 自動取得フィーチャを
# 侵さないため案内に留める)。ライセンス詳細は THIRD_PARTY_NOTICES.md。
models = { cmd = "python scripts/setup_models.py", help = "外部モデル (Silero VAD / rmvpe / FCPE / HuBERT) の入手方法と config キーを表示" }
```

- [ ] **Step 6: タスクが一覧に出て実行できることを確認**

Run: `uv run poe models`
Expected: 上記ガイドが文字化けせず表示される（`snakers4/silero-vad` / `wok000/weights_gpl` / 各 config キーを含む）。
Run: `uv run poe --help`（またはタスク一覧）
Expected: `models` タスクが help 文つきで並ぶ。

- [ ] **Step 7: ruff / ty を通す**

Run: `uv run ruff format scripts/setup_models.py scripts/tests/test_setup_models.py && uv run ruff check scripts/setup_models.py scripts/tests/test_setup_models.py && uv run ty check scripts/setup_models.py`
Expected: フォーマット済み・lint 0・ty 0 diagnostics。

- [ ] **Step 8: コミット**

```bash
git add scripts/setup_models.py scripts/tests/test_setup_models.py poe_tasks.toml
git commit -m "feat(models): poe models で外部モデルの入手方法と config キーを案内"
```

---

### Task 2: `config.toml.example` から `poe models` への導線

**Files:**
- Modify: `config.toml.example`（rmvpe 箇所と Silero VAD 箇所）

**Interfaces:**
- Consumes: Task 1 で追加した `poe models` タスク名。

- [ ] **Step 1: rmvpe 箇所に導線を追記**

`config.toml.example` の rmvpe 設定（現状 180–181 行）を置換:

Old:
```toml
# rmvpe の重みファイル（f0_extractor_type = "rmvpe" のとき必要）
rmvpe_model_file = "./rmvpe.onnx"
```

New:
```toml
# rmvpe の重みファイル（f0_extractor_type = "rmvpe" のとき必要）
# 入手方法: uv run poe models（wok000/weights_gpl, GPL-3.0）
rmvpe_model_file = "./rmvpe.onnx"
```

- [ ] **Step 2: Silero VAD (vc) 箇所に導線を追記**

`config.toml.example` の Silero VAD 説明（現状 122–123 行）を置換:

Old:
```toml
# snakers4/silero-vad リポジトリの silero_vad.onnx (v6.2.1) を取得してパスを指定する。
# vad_model_file = "./silero_vad.onnx"
```

New:
```toml
# snakers4/silero-vad リポジトリの silero_vad.onnx (v6.2.1) を取得してパスを指定する。
# 入手方法の一覧: uv run poe models
# vad_model_file = "./silero_vad.onnx"
```

- [ ] **Step 3: 導線が入ったことを確認**

Run: `uv run rg "uv run poe models" config.toml.example`
Expected: 2 行ヒット（rmvpe 箇所・Silero VAD 箇所）。

- [ ] **Step 4: コミット**

```bash
git add config.toml.example
git commit -m "docs(config): rmvpe / Silero VAD の入手導線を poe models へ張る"
```

---

### Task 3: HuBERT タスクの `--help` epilog

**Files:**
- Modify: `scripts/convert_hubert.py:220`（`main()` の `argparse.ArgumentParser()`）
- Modify: `scripts/export_hubert_onnx.py:247`（`main()` の `argparse.ArgumentParser()`）

**Interfaces:**
- Consumes: なし（FCPE `export_fcpe_onnx.py:159-174` の epilog パターンを踏襲）。

注記: 両スクリプトはモジュール先頭で `torch` / `transformers` / `fairseq` を import するため、ベース環境では import できず `--help` を自動テストできない（FCPE epilog も同理由でテスト無し）。検証は差分の目視 + ruff、実表示は overlay 環境での手動 `--help`。

- [ ] **Step 1: `convert_hubert.py` に epilog を付ける**

`scripts/convert_hubert.py` の `main()` 冒頭を置換:

Old:
```python
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="hubert_base.pt")
```

New:
```python
    parser = argparse.ArgumentParser(
        description="fairseq ContentVec (hubert_base.pt) を transformers HubertModel 資産へ変換する (offline, 1/2 段目)。",
        epilog=(
            "HuBERT (ContentVec) 資産は 2 段のオフライン変換で用意する。\n"
            "入力 hubert_base.pt は RVC が配布する ContentVec (MIT, origin auspicious3000/contentvec)。\n"
            "\n"
            "手順:\n"
            "  1. uv run poe convert-hubert \\\n"
            "         --input  ~/.config/vstreamer/hubert_base.pt \\\n"
            "         --output ./hubert_contentvec \\\n"
            "         --golden ./hubert_golden\n"
            "  2. uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden\n"
            "  3. config の [rvc] に設定:\n"
            '       hubert_model_file = "./hubert_contentvec"   # 資産ディレクトリ (ファイルではない)\n'
            "\n"
            "ライセンスは THIRD_PARTY_NOTICES.md を参照。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="hubert_base.pt")
```

- [ ] **Step 2: `export_hubert_onnx.py` に epilog を付ける**

`scripts/export_hubert_onnx.py` の `main()` の argparse を置換:

Old:
```python
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path, help="hubert_contentvec/")
```

New:
```python
    parser = argparse.ArgumentParser(
        description="transformers HubertModel 資産 (hubert_contentvec/) を ONNX へ export する (offline, 2/2 段目)。",
        epilog=(
            "これは HuBERT 資産づくりの 2 段目。先に `uv run poe convert-hubert` で\n"
            "hubert_base.pt を資産ディレクトリへ変換しておくこと (詳細は convert-hubert --help)。\n"
            "\n"
            "手順:\n"
            "  uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden\n"
            "  -> <asset>/hubert_fp32.onnx + hubert_fp16.onnx + mapping.json を書き出す\n"
            "\n"
            "config の [rvc] は資産ディレクトリを指す:\n"
            '  hubert_model_file = "./hubert_contentvec"\n'
            "\n"
            "ライセンスは THIRD_PARTY_NOTICES.md を参照。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--asset", required=True, type=Path, help="hubert_contentvec/")
```

- [ ] **Step 3: 差分を確認**

Run: `uv run rg -n "hubert_model_file|auspicious3000|THIRD_PARTY_NOTICES|RawDescriptionHelpFormatter" scripts/convert_hubert.py scripts/export_hubert_onnx.py`
Expected: 両ファイルに epilog（入手元・config キー・NOTICES 導線）と `RawDescriptionHelpFormatter` が入っている。

- [ ] **Step 4: ruff / ty を通す**

Run: `uv run ruff format scripts/convert_hubert.py scripts/export_hubert_onnx.py && uv run ruff check scripts/convert_hubert.py scripts/export_hubert_onnx.py && uv run ty check scripts/convert_hubert.py scripts/export_hubert_onnx.py`
Expected: フォーマット済み・lint 0・ty は既存の accepted `# ty: ignore` 以外に新規 diagnostics 0（epilog は文字列追加のみ）。

- [ ] **Step 5: コミット**

```bash
git add scripts/convert_hubert.py scripts/export_hubert_onnx.py
git commit -m "docs(hubert): convert/export タスクの --help を自己完結化 (入手元+config キー)"
```

---

## 最終確認

- [ ] **Step: 健全性ゲート**

Run: `uv run --all-extras poe check`
Expected: 既存の accepted 例外（audit の torch CVE 等）以外は緑。新規テスト `test_setup_models.py` を含め pytest PASS。

## Self-Review

**Spec coverage:**
- 受入基準①「hubert 2 タスクの `--help` 自己完結（入手元・2 段手順・`rvc.hubert_model_file`）」→ Task 3。
- 受入基準②「`uv run poe` に案内タスク 1 つ、実行で Silero VAD・rmvpe の入手元＋キー表示、FCPE/HuBERT は各自タスクを指す」→ Task 1（`build_guide` の内容 + `models` タスク）。
- 受入基準③「rmvpe が GPL-3.0 で詳細が `THIRD_PARTY_NOTICES.md` にある旨を辿れる」→ Task 1（ガイド本文に "GPL-3.0" と "THIRD_PARTY_NOTICES" を含む、テスト済み）。
- 受入基準④「`config.toml.example` の rmvpe/Silero VAD から導線が辿れる」→ Task 2。
- 非ゴール「自動 DL 機構を作らない / VAD 既定 ON 化しない / モデル同梱しない / ライセンス表記刷新しない」→ 追加タスクなし（`models` は表示のみ、config 既定は不変）。

**Placeholder scan:** TBD/TODO・「適切に」等の曖昧表現なし。全コードブロックは実内容。

**Type consistency:** `build_guide() -> str` は Task 1 で定義しテスト・`main()` が使用。`GUIDE` 定数名一致。argparse epilog は両 hubert タスクとも `argparse.RawDescriptionHelpFormatter` を使用（FCPE と同一 API）。
