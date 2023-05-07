# VStreamer Tool

AmiVoice Cloud Platform API, VOICEROID2, Google translation API v3 などと連携してそういう配信をするときに使う  
以下の機能を実装

1. 録音
2. 文字起こし
    - [AmiVoice Cloud Platform API](https://acp.amivoice.com/)
    - [Google Cloud Speech-to-Text API](https://cloud.google.com/speech-to-text?hl=ja)
    - [Whisper](https://github.com/openai/whisper)
3. 音声合成
    - VOICEROID2
    - [VOICEVOX](https://github.com/VOICEVOX/voicevox_core)
4. 翻訳 (Google translation API v3)
5. 字幕
6. ボイスチェンジャー
    - RVC モデル
    - [VC Client](https://github.com/w-okada/voice-changer) を参考にさせていただいております

## 設定

poetry で依存パッケージをインストール

```sh
# 文字起こし/字幕生成のみ
poetry install
# openai/whisper を文字起こしに使用する場合
poetry install -E whisper
# 録音/再生 (portaudio が必要)
poetry install -E audio
# openai whisper
poetry install -E whisper
# 音声合成
poetry install -E vroid2 # VOICEROID2
poetry install -E voicevox # Voicevox
# RVC
poetry install -E rvc
# gui (ttk)
poetry install -E gui
```

設定項目は `config.toml.example` や `vspeech/config.py` を参照してください。ごめんなさい。

VOICEROID2 を使う場合は 64bit 版エディターがダウンロードされている必要があります。

VOICEVOX を使う場合は指定したディレクトリ (デフォルト: `./voicevox_core`) に  と Open JTalk から配布されている辞書ファイルを配置してください。(onnxruntime も必要?)

詳細は [voicevox_core_python_api](https://github.com/VOICEVOX/voicevox_core/tree/0.14.1/crates/voicevox_core_python_api) の環境構築を確認してください。

whisper, RVC は対応するバージョンの cuda がインストールされている必要があります。

## 実行

```sh
poetry run python -m vspeech --config ./config.toml
```

GUI

```sh
poetry run python -m vspeech.gui -c config.toml
```
