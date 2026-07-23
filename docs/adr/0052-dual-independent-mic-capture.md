# 0052. ストリーミングと発話系でマイクを二重独立キャプチャする

- Status: Accepted
- Date: 2026-07-22
- Related: [spec](../superpowers/specs/2026-07-22-rvc-streaming-vc-split-machine-design.md), [0050](0050-streaming-vc-separate-subsystem.md), [0031](0031-audio-pyaudio-to-sounddevice.md)

## Context

ストリーミング VC(連続ブロック)と発話系(無音区切り発話→文字起こし)は、同じマイク音声を必要とする。発話系の recording worker は無音検出ループとキャプチャが密結合しており、これを共有 fan-out へ作り替えると発話系に手が入り「発話系を維持」という要件のリスクになる。ユーザは入力(Input)worker が複数になっても構わないとしている。

## Decision

ストリーミング用に独立した入力キャプチャを追加し、発話系の recording は無改変のまま別途マイクを開く(二重独立キャプチャ)。同一デバイスの二重 open が OS で排他になる環境向けに、「単一キャプチャ → 両系統へ fan-out」を明示的なフォールバックとして設計に残す。既定は二重独立キャプチャとする。

## Alternatives rejected

- **単一キャプチャ fan-out を既定にする** — サンプル同一・単一 open という利点はあるが、既存 recording のキャプチャ/無音検出ループに手を入れることになり、発話系維持のリスクが上がる。排他デバイス環境向けのフォールバックとしては保持する。

## Consequences

発話系を無改変で残せる(維持要件に安全)。反面、排他モード WASAPI や一部の仮想オーディオケーブルでは同一デバイスの二重 open が失敗し得る。その環境では二重独立キャプチャは成立せず、fan-out を**実装する**ことが必要になる — フォールバックはここに設計として残しているだけで、コードとしては未実装であり、切り替えスイッチは存在しない(`vspeech/stream_vc/capture.py` の docstring にも未実装と明記してある)。二系統は独立した VAD/タイミングになる(サンプル完全同一は保証しない)。
