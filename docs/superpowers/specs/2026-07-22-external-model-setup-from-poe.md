# 外部モデルの導入を poe 起点で分かるようにする

## 問題

外部モデル (Silero VAD / rmvpe / FCPE / HuBERT) は本リポジトリに同梱されず、利用者が各自取得して
config のパスに設定する必要がある。だが「導入方法へ辿り着く導線」が不均一である。

- **FCPE** は `poe export-fcpe-onnx --help` が入手 (自動生成) から config キー設定までを自己完結で案内する (ADR-0049)。
- **HuBERT** は poe タスク (`convert-hubert` / `export-hubert-onnx`) はあるが、`--help` が素の argparse で、hubert_base.pt の入手元も 2 段変換の手順も設定キーも示さない。
- **rmvpe** と **Silero VAD** は poe タスク自体が存在しない。rmvpe の入手元は利用者向けの導線 (config 例・エラー文言) に一切書かれておらず、`THIRD_PARTY_NOTICES.md` のライセンス表でしか辿れない。

結果、「poe を起点に導入方法が具体的に分かる」体験が FCPE 以外で成立していない。

## ゴール

4 つの外部モデルすべてについて、**入手方法**と**設定する config キー**が、利用者が実際に見る場所から
具体的に分かる: FCPE / HuBERT は各自の `poe <task> --help`、Silero VAD / rmvpe は `config.toml.example`
の該当設定箇所。

## 非ゴール

- モデルの自動ダウンロード/取得機構 (curl/fetch を行うタスク) は作らない。rmvpe の GPL や URL 陳腐化の
  懸念を避け、保留中の GUI 自動取得フィーチャの領域を侵さないため。
- VAD の既定 ON 化はしない。
- モデルファイル自体の同梱・再配布はしない (現状維持)。
- ライセンス表記の刷新はしない (`THIRD_PARTY_NOTICES.md` が単一情報源のまま)。案内はそこへ辿らせるだけ。

## 受入基準

- `poe convert-hubert --help` と `poe export-hubert-onnx --help` が、hubert_base.pt の入手元・
  2 段変換の手順・設定する config キー (`rvc.hubert_model_file`) を自己完結で示す。
- `config.toml.example` の各外部モデルの設定箇所に、入手元と設定方法が具体的に記載される:
  Silero VAD (`vad_model_file`) / rmvpe (`rmvpe_model_file`) / FCPE (`fcpe_model_file`) /
  HuBERT (`hubert_model_file`)。
- 上記のうち rmvpe が GPL-3.0 である旨と、詳細が `THIRD_PARTY_NOTICES.md` にある旨が
  `config.toml.example` から辿れる。
