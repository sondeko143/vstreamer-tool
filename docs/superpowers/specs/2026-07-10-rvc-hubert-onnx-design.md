# RVC HuBERT の ONNX 化 設計書（spec ②）

## 問題

前 spec で RVC の content encoder は fairseq 非依存になったが、runtime に transformers が残った。ロックファイルは universal であり、extra に宣言されているだけのパッケージもロックに残って全体が監査対象になる。そのため transformers の既知脆弱性がロックに居座り続け、さらに変換専用の依存が持つ numpy の旧い上限が universal 解決を古い numpy に固定していた。これは次段（Python 引き上げ）の最後の障害そのものである。「extra に隔離する」だけでは不十分で、オフラインツールの依存を依存宣言から完全に外さない限り解けない。

## ゴール

- runtime 依存を縮小し、content encoder を onnxruntime のセッションだけで動かす。runtime から transformers / safetensors を撤去する。
- fairseq / transformers を依存宣言とロックの両方から完全に取り除く。変換・export の依存はロックに載せないオフライン工程で供給する。
- 特徴量の等価性（fp32）と fp16 の振る舞いを、いずれも観測可能なしきい値で担保する。
- 呼び出し側が指定した実行デバイスが尊重されるようにする（実装中に発見した既存バグの修正）。
- numpy の下限を、後段作業を阻害しない最新版まで繰り上げる。

## 非ゴール

- Python バージョンの引き上げそのもの（後続 spec ③）。本 spec は現行バージョンのまま完結する。
- torch / torchaudio の除去。
- RVC 推論本体・f0 抽出・VAD ゲートの変更。
- RMVPE 側の同型デバイスバグの修正。
- fairseq からの再変換手段の削除（再生成できる状態を残す）。
- 設定の後方互換 alias。設定項目の意味は不変で、設定例・GUI は無改修。

## 受入基準

- [ ] ロックファイルに fairseq と transformers のエントリが存在しない。
- [ ] runtime のソースに fairseq / transformers への依存（import）が 1 件も存在しない。
- [ ] 脆弱性監査の報告が、受容済みの 1 件のみになる。
- [ ] ロックの numpy が、意図した新しい下限へ繰り上がっている。
- [ ] fp32 の content encoder 出力が、旧 fairseq 基準に対し v1・v2 の両方で cosine ≥ 0.9999 かつ max-abs ≤ 1e-4 に収まる。
- [ ] fp16 の content encoder 出力が、置き換え対象である従来 fp16 実装を参照として cosine ≥ 0.9999 かつ max-abs ≤ 5e-2 に収まる（デバイス依存のため該当環境のみで検査）。
- [ ] ボイスチェンジ出力音声が、再ベースラインしない既存 golden に対し許容誤差内である（相関・SNR のしきい値は決定根拠 ADR に従う）。
- [ ] 実行デバイスに CPU を指定した場合に CPU で走る。
- [ ] 全 extra を有効化した状態でプロジェクトの健全性ゲート一式が green（既知の受容済み例外を除く）。
- [ ] 実機での耳チェックが良好（自動ゲートには含めない最終確認）。

---

- 決定根拠: [ADR-0022](../../adr/0022-hubert-onnx-runtime.md) , [ADR-0023](../../adr/0023-hubert-equivalence-gate.md) , [ADR-0024](../../adr/0024-onnx-session-single-factory.md)
- 実装計画: [plan](../plans/2026-07-10-rvc-hubert-onnx.md)
