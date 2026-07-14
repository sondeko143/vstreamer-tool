# Python 3.14 移行ロードマップ 設計書

## 問題

`requires-python` は `>=3.12,<3.13`。3.12 は bugfix フェーズが終了しており security-only は 2028-10 まで（PEP 693）。サポート窓を延ばしたいが、numpy `<2` cap・CUDA 12.8・onnxruntime-gpu の cap・pyworld / audioop 等の cp314 ギャップが絡み合っている。これらはいずれも GPU / 音声の実機経路に触れ CI では検証できず、同時に動かすと退行の切り分けが不能になる。

## ゴール

- Python 3.14 に到達し、サポート窓を延長する。
- 移行過程で GPU / 音声の退行を各変数単位で切り分け可能にする（各段階の実機検証が通ってから次へ進む）。

## 非ゴール

- runtime 性能の改善（本ワークロードでは 3.14 でも利得はほぼゼロ。これは保守性のための移行である）。
- 3.13 を経由すること。
- 実機検証を伴わない一括アップグレード。

## 受入基準

3 フェーズを「1 段階 = 1 変数」で進め、各段階が実機検証で green になってから次へ進む。

- [ ] フェーズ①（numpy 2 化・Python は 3.12 のまま）: 全 extra の依存解決が通り、numpy が 2.x に解決される。実機の GPU / 音声経路に退行が無い。
- [ ] フェーズ②（CUDA 13 化・Python は 3.12 のまま）: 全 extra の依存解決が通り、実機で GPU 経路（whisper / RVC ボイチェン）が動作し退行が無い。
- [ ] フェーズ③（Python 3.14 化）: 3.14 で全 extra の依存解決が通り、健全性ゲートが green、エントリポイントが起動し、実機で退行が無い。
- [ ] 各フェーズは前段の実機検証が green になってから着手される。

---

- 決定根拠: [ADR-0025](../../adr/0025-target-python-314-phased.md) , [ADR-0026](../../adr/0026-adopt-numpy-2.md) , [ADR-0027](../../adr/0027-cap-onnxruntime-cuda12.md) , [ADR-0028](../../adr/0028-migrate-to-cuda-13.md) , [ADR-0029](../../adr/0029-audioop-lts.md) , [ADR-0030](../../adr/0030-pyworld-lazy-default-rmvpe.md) , [ADR-0031](../../adr/0031-audio-pyaudio-to-sounddevice.md)
- 実装計画: なし
