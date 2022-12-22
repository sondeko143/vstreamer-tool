# VStreamer Tool

AmiVoice Cloud Platform API, VOICEROID2, Google translation API v3 と連携してそういう配信をするときに使う  
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

## 設定

pipenv で依存パッケージをインストール

```sh
pipenv sync
```

設定項目は以下の help コマンドから確認できます

```sh
pipenv run python -m vspeech --help
```

VOICEROID2 を使う場合は 64bit 版エディターがダウンロードされている必要があります。

VOICEVOX を使う場合は指定したディレクトリ (デフォルト: `./voicevox_core`) にビルド済みのコアライブラリと Open JTalk から配布されている辞書ファイルを配置してください。

詳細は <https://github.com/VOICEVOX/voicevox_core> の環境構築を確認してください。

## 実行

```sh
pipenv run python -m vspeech
```

GUI

```sh
pipenv run python -m vspeech.gui
```
