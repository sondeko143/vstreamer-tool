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
