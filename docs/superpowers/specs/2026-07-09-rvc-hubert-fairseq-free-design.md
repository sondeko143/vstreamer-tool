# RVC HuBERT の fairseq-free 化 設計書

## 問題

RVC のボイスチェンジは content encoder として fairseq に依存しており、これが Python バージョン引き上げの唯一にして最大の障害になっている。上流の公式 fairseq は 2022 年のリリースで凍結され、リポジトリは archived（read-only）となって、新しい Python への対応修正は永久に来ない。このため本 repo は自前ビルドの cp311 固定 wheel をピンしており、この 1 本が torch / torchaudio の cp311 URL ピンと numpy の旧い上限を連鎖的に縛っている。3 つのピンはすべて fairseq 由来であり、fairseq が残る限り結び目は解けない。

## ゴール

- RVC の content encoder を fairseq 非依存にする。
- 置換の正しさを、HuBERT 特徴量の数値等価で機械的に担保する。
- 変換に必要な fairseq は runtime から隔離し、一度きりのオフライン工程に押し込める。
- numpy 上限を、後段作業を阻害しない範囲まで緩和する。

## 非ゴール

- content encoder の ONNX 化（後続 spec ②）。
- Python バージョンの引き上げそのもの（後続 spec ③）。本 spec は現行バージョンのまま完結する。
- torch / torchaudio の除去。
- RVC 推論本体・f0 抽出・VAD ゲートの変更。
- faiss index 周りの変更。
- 設定の後方互換 alias（個人利用・gitignore のためクリーン入れ替え）。

## 受入基準

- [ ] RVC を有効化した環境の依存ツリーに fairseq が含まれない。
- [ ] runtime のソースに fairseq への依存（import）が 1 件も存在しない。
- [ ] 新実装の HuBERT 特徴量が、旧実装を基準として v1（層 9 + final_proj）と v2（層 12）の両方で cosine ≥ 0.9999 かつ max-abs ≤ 1e-4 に収まる。
- [ ] ボイスチェンジ出力音声が、再ベースラインした golden に対し許容誤差内である（相関・SNR のしきい値は決定根拠 ADR に従う）。
- [ ] numpy 上限が緩和され、RVC / whisper を有効化した環境が解決でき、それらの経路が import できる。
- [ ] 設定項目「変換済み資産のありか」を指す新しい意味で、ボイスチェンジが起動・動作する。
- [ ] プロジェクトの健全性ゲート一式が green（既知の受容済み例外を除く）。
- [ ] 実機での耳チェックが良好（自動ゲートには含めない最終確認）。

---

- 決定根拠: [ADR-0021](../../adr/0021-hubert-drop-fairseq.md) , [ADR-0023](../../adr/0023-hubert-equivalence-gate.md)
- 実装計画: [plan](../plans/2026-07-09-rvc-hubert-fairseq-free.md)
