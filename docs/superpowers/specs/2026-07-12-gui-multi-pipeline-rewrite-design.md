# GUI 全面書き直し: マルチ pipeline 管理 (フェーズ1) 設計

## 問題

現行 GUI は単一 `Config` を全 worker タブで編集し、単一の vspeech サブプロセスだけを起動 / 停止する構造になっている。複数の pipeline を並行して管理できず、起動のたびに config ファイルパスの指定が要る。さらに、壊れた設定ファイルや将来の設定スキーマ変更に耐える仕組みが無い。

## ゴール

- GUI から起動する場合、固定のユーザーデフォルトのプロファイルから常に設定を読み、config ファイルパス引数を不要にする。
- 複数の pipeline を独立に起動 / 停止できる。
- 各 pipeline の設定を GUI で個別に編集・保存・管理できる。
- 設定スキーマ変更（フィールド追加 / リネーム）と壊れたファイルの load に耐える migration 機構を持つ。
- 新規 pipeline がプリセットから最初から配線済み・実行可能である。

## 非ゴール

- subtitle / translation worker のフォーム対応（生 TOML で編集可能なまま）。
- 5 worker（recording / playback / vc / transcription / tts）以外のバックエンド詳細のフォーム化。
- pause / resume / reload の実行中コントロール（フェーズ2）。
- リモート（マルチマシン）配線の GUI 補助（フェーズ2）。

## 受入基準

- [ ] config パス引数なしで GUI を起動でき、固定のユーザーデフォルト場所からプロファイルが読まれる。
- [ ] 複数の pipeline を同時に起動でき、各々が独立したプロセスとして個別に停止できる。
- [ ] 各 pipeline の設定を GUI で個別に編集・保存でき、次回起動時にその設定で動作する。
- [ ] 新規 pipeline をプリセットから作成すると、追加設定なしでそのまま起動して動作する。
- [ ] 同時に走る pipeline 同士でポートが衝突しない。割当てから起動までの間に他プロセスが握った競合も検出し、起動を止める。
- [ ] 壊れた pipeline 設定・雛形・マニフェストのいずれがあっても GUI は必ず起動し、壊れた原本は失われず退避される。
- [ ] 旧バージョンの設定ファイルを読み込んでも起動でき、現行バージョンへ収束する。

---

- 決定根拠: [ADR-0032](../../adr/0032-gui-multi-pipeline-rewrite.md) , [ADR-0033](../../adr/0033-gui-manifest-versioning.md) , [ADR-0034](../../adr/0034-gui-corrupt-file-resilience.md)
- 実装計画: [plan](../plans/2026-07-12-gui-multi-pipeline-rewrite.md)
