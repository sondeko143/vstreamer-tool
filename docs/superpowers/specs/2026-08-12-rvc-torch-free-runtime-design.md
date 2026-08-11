# RVC ランタイムの torch 完全除去

## 問題

コア依存の `ctranslate2` は、venv に torch が入っていれば import 時にそれを道連れにする（実測: `import ctranslate2` 5.39s、その時点で `torch in sys.modules == True`）。したがって torch が **インストールされている限り**、vspeech のどのプロセスも torch のロード代（実測 +476.7MB / +3.17s）を払う。

ADR-0078 で whisper 経路のコードからは torch を外したが、`rvc` extra が `torch` / `torchaudio` を要求し続けているため、VC を動かすホストではその削減がまったく効かない。

一方で RVC の推論は onnxruntime が担っており、torch が実際に行っているのはテンソル配線・io_binding のゼロコピー・リサンプルだけである。そのゼロコピーも、f0 抽出が毎 tick 波形を host へ降ろしている時点で既に破れており、残る利得は特徴量 1 本（〜50KB/tick、対 40–70ms の推論）にとどまる。リサンプルは ADR-0073 で numpy 実装が入っており、torchaudio はその重複である。

`faiss-cpu` も `rvc` extra にあるが、repo 内のどの `.py` からも import されておらず、対応する設定項目も存在しない。

## ゴール

- vspeech ランタイム全体が torch / torchaudio を import しない。
- `rvc` extra から `torch` / `torchaudio` / `faiss-cpu` を削除し、VC ホストの venv に torch が存在しない状態にする。
- 音質・レイテンシ・設定の互換性を回帰させない。

## 非ゴール

- whisper / transcription 経路（ADR-0078 で対応済み）。
- オフラインの ONNX 生成ツールから torch 依存をなくすこと。実行環境をランタイムから切り離すのみ。
- ADR-0058（static-shape CUDA graph）の再開。
- RVC の音質そのものの改善。

## 受入基準

- [ ] 全 extra を同期した venv に `torch` / `torchaudio` / `faiss-cpu` が含まれない。
- [ ] VC を有効にした設定で pipeline が起動し、プロセス内で torch がロードされない。
- [ ] Stream VC・発話単位 VC・VAD ゲート・f0 抽出（rmvpe / fcpe）が、変更前と同じ設定ファイルのまま動作する。設定スキーマの破壊的変更がない。
- [ ] Stream VC の 1 ブロック変換出力が、同一入力・同一 seed で変更前実装と int16 bit 一致する。
- [ ] 発話単位 VC の出力が既存の numeric golden の判定（相関・SNR）を満たす。満たさない場合は、実測値を記録した上で golden を再取得し、実機での聴感確認を経ている。
- [ ] Stream VC の 1 推論あたりレイテンシの p50 が、変更前の測定値の +5% を超えて悪化しない。同一機材・同一設定・N≥200 tick で測定した p50 / p95 が両版について記録されている。
- [ ] VC ホストにおける常駐メモリと起動時間の削減量が、実測値として記録されている。
- [ ] HuBERT / FCPE の ONNX 生成タスクが、torch を持たないプロジェクト環境からでも実行できる。
- [ ] vspeech ランタイムのどのモジュールも torch / torchaudio を import しないことが、構造的なテストで恒久的に守られている。
