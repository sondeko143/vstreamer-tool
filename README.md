# VStreamer Tool

AmiVoice Cloud Platform API, VOICEROID2, Google translation API v3 などと連携してそういう配信をするときに使う  
以下の機能を実装

1. 録音
2. 文字起こし
    - [AmiVoice Cloud Platform API](https://acp.amivoice.com/)
    - [Google Cloud Speech-to-Text API](https://cloud.google.com/speech-to-text?hl=ja)
    - [Whisper](https://github.com/SYSTRAN/faster-whisper) (faster-whisper)
3. 音声合成
    - VOICEROID2
    - [VOICEVOX](https://github.com/VOICEVOX/voicevox_core)
4. 翻訳 (Google translation API v3)
5. 字幕
6. ボイスチェンジャー
    - RVC モデル
    - [VC Client](https://github.com/w-okada/voice-changer) を参考にさせていただいております

## 設定

Python 3.12 が必要です (`>=3.12,<3.13`)。uv で依存パッケージをインストールします。

```sh
# 全部入り。迷ったらこれ
uv sync --all-extras

# 一部だけ入れる場合は、必要な extra を 1 つのコマンドに並べる
uv sync --extra audio --extra whisper
```

**`uv sync` は指定した extras の集合に環境を合わせます。**あとから
`uv sync --extra rvc` と単独で叩くと、それ以外の extras (`voicevox`, `whisper`,
`audio`, `gui` …) は**アンインストールされます**。機能を「足す」つもりで実行すると壊れます。

| extra | 内容 |
| --- | --- |
| (なし) | 文字起こし / 翻訳 / 字幕 |
| `audio` | 録音・再生 (portaudio が必要) |
| `whisper` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) による文字起こし |
| `vroid2` | VOICEROID2 音声合成 |
| `voicevox` | VOICEVOX 音声合成 |
| `rvc` | RVC ボイスチェンジャー |
| `mozc` | AmiVoice の結果をかな漢字変換する (`transcription.transliterate_with_mozc`) |
| `gui` | ttkbootstrap の GUI |

設定項目は `config.toml.example` や `vspeech/config.py` を参照してください。ごめんなさい。

VOICEROID2 を使う場合は 64bit 版エディターがダウンロードされている必要があります。

VOICEVOX を使う場合、`voicevox-core` の wheel には **ONNX Runtime・Open JTalk 辞書・
音声モデル (`.vvm`) が含まれていません**。[VOICEVOX のダウンローダ](https://github.com/VOICEVOX/voicevox_core/releases/tag/0.16.4)
で取得し、`voicevox.openjtalk_dir` / `model_dir` / `onnxruntime_path` を指してください。
既定値は `./voicevox/dict/open_jtalk_dic_utf_8-1.11` と `./voicevox/models/vvms` です。

```sh
curl -sSfL https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/download-windows-x64.exe -o download-voicevox.exe
./download-voicevox.exe -o ./voicevox --exclude c-api --devices cuda
```

(テストスイート用の資産は `uv run poe voicevox-assets` が `tests/assets/voicevox` へ取得します。)

VOICEVOX は whisper / RVC が使う `onnxruntime-gpu` とは**別ビルド**の
`voicevox_onnxruntime` を読みます。正しい方が読まれるよう `onnxruntime_path` は明示してください。

whisper, RVC は CUDA 12.8 (`torch 2.10.0+cu128`) がインストールされている必要があります。

## 実行

```sh
uv run python -m vspeech --config ./config.toml
```

GUI

```sh
uv run python -m gui -c config.toml
```
