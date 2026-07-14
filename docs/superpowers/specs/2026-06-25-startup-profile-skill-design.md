# Startup-profile skill — design

## 問題

`python -m vspeech` の起動が、最初のワーカーがログを出すまで数秒スタールすることがある。原因はワーカー起動時の同期的なブロッキング待ち（DNS/socket/TLS・SMB・subprocess）に散在しており、一つずつ潰すのは非効率。起動時間の内訳を再現可能に捕捉・解析したい。

## ゴール

最小 config で vspeech をサンプリング profiler 下に起動し、解析可能なフレームグラフを捕捉して、起動ホットスポットをランク付け・分類（blocking-io / import / compute / idle）する on-demand プロジェクト skill。イベントループ idle はヘッドラインから除外する。観測のみでアプリ source は一切触らない。

## 非ゴール

- 継続監視・回帰検知・CI ゲーティング。
- アプリ内プロファイリング計装。
- 表面化した認証ルックアップ自体の修正（別途フォローアップの決定）。

## 受入基準

- アプリ source を一切変更せずに起動を捕捉できる。
- 最小 config（GPU/extras 不要）で走り、任意の config も指定できる。
- 捕捉が解析可能なフレームグラフとして保存され、視覚ツールにも読み込める。
- 実行後にオーファンプロセス（gRPC サーバ等）が残らない。
- 各ホットスポットが blocking-io / import / compute / idle に分類される。
- idle をヘッドライン集計から除外した active 時間が示される。
- ブロッキング待ち（DNS/socket/subprocess）が捕捉される。
- 機械可読（JSON）出力も選べる。

---

- 決定根拠: [ADR-0015](../../adr/0015-startup-profiler-py-spy.md) , [ADR-0014](../../adr/0014-relocate-skills-to-project.md)
- 実装計画: なし（この spec に対のプランは無い）
