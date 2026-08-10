# 0071. デバイスのネイティブレートを WASAPI カウンターパートから自動解決し、設定で上書きできるようにする

- Status: Proposed
- Date: 2026-08-10
- Related: [spec](../superpowers/specs/2026-08-10-device-sample-rate-in-process-design.md), [0070](0070-device-boundary-inhouse-polyphase-resampler.md)（本 ADR はその前提を満たす）, [0038](0038-worker-config-preflight-fail-loud.md)

## Context

[0070](0070-device-boundary-inhouse-polyphase-resampler.md) はデバイスをネイティブレートで開くと決めた。そのためには真のレートを知る必要があるが、PortAudio の報告値は当てにならない。

- **PortAudio は MME / DirectSound / WDM-KS のデバイスに対し `default_samplerate` を 44100 固定で返す。** 実測: この機体の MME 36 デバイスと DirectSound 36 デバイスは、48kHz で動作中の Voicemeeter / Realtek エンドポイントも含めて全部 44100 と答える。
- **MME はプローブにも答えない。** `check_input_settings` は 16000 / 44100 / 48000 のどれを渡しても OK を返す（実測）。候補レートを試して当てる方法は成立しない。
- **WASAPI のデバイスだけは `default_samplerate` が実際のミックスフォーマットを返す。** 同じ Realtek スピーカーが WASAPI では 48000、MME では 44100 と出る。
- **MME のデバイス名は WASAPI のデバイス名を 31 文字で切り詰めたものである。** 実測: `Microphone (2- Aukey-PC-LM1E Au` ⊂ `Microphone (2- Aukey-PC-LM1E Audio)`。この前方一致で MME/DirectSound のデバイスから WASAPI 側の同一エンドポイントを引ける。

## Decision

デバイスを開くレートを、次の順で解決する。

1. 設定の明示値（`stream_vc.input_device_rate` / `stream_vc.output_device_rate` / `recording.input_device_rate` / `playback.output_device_rate`。いずれも `int | None`、`None` = 自動）。
2. 対象が WASAPI のデバイスなら、その `default_samplerate`。
3. それ以外のホスト API なら、WASAPI 側で**デバイス名が前方一致する**同方向（入力/出力）のデバイス群を集め、そのミックスレートが**一意に定まればそれ**を採る。

一意に定まらなければ preflight で fail-loud にし、どの設定キーを書けばよいかをメッセージに載せる（[0038](0038-worker-config-preflight-fail-loud.md)）。解決したレートと**どの経路で解決したか**は、デバイスを open するときにログへ出す。

旧挙動（OS 任せ）へ戻すための専用フラグは作らない。その境界を流れる音声のレートと同じ値を明示指定すれば変換が素通しになり、それが旧挙動と同じになる。

## Alternatives rejected

- **`default_samplerate` をそのまま使う** — MME/DirectSound で 44100 固定の嘘を返すため、48kHz のエンドポイントでは 48000→44100 の OS リサンプルが残る。[0070](0070-device-boundary-inhouse-polyphase-resampler.md) の目的（OS の変換段を通らない）を、最も使われているホスト API で達成できない。
- **`check_*_settings` でレート候補をプローブする** — MME はどのレートでも OK を返すので判別できない。WASAPI では効くが、WASAPI は `default_samplerate` が既に正しいのでプローブが要らない。効かせたい相手にだけ効かない。
- **設定を必須にする（未指定なら preflight エラー）** — ヒューリスティックを背負わずに済むが、実測でこの機体の MME/DirectSound 45 デバイス中 43 が一意に解決できる。解けるものまで含めて全ユーザーに Windows のサウンド設定を見ながら 4 項目書かせるのは、誤設定の総量をむしろ増やす。自動を既定にし、解けないときだけ明示させる。
- **「OS 任せに戻す」真偽フラグを別に持つ** — レートを明示指定すれば同じ結果になるので、設定面を 1 つ増やす価値がない。
- **アドレス帳のように デバイス名 → レート の対応表をプロジェクトに持つ** — 環境ごとに違うものをリポジトリに置くことになり、[0038](0038-worker-config-preflight-fail-loud.md) の「設定は設定ファイルに、問題は起動時に」という方針から外れる。

## Consequences

- 設定を何も書かなくても動く。解決できないのは「Microsoft サウンド マッパー」「プライマリ サウンド ドライバー」のような疑似デバイスだけで（実測でこの 2 つのみ）、その場合は起動時に落ちてどのキーを書けばよいかが出る。
- **名前の前方一致というヒューリスティックを 1 つ背負う。** 同名デバイスが複数あってもミックスレートが揃っていれば解決でき、食い違えば fail-loud に倒れる。黙って間違ったレートを選ぶ経路は作らない。
- **WDM-KS は前方一致で解けない。** デバイス名の付け方が WASAPI と違う（`Line 1 (Virtual Cable 1)` vs `Line 1 (Virtual Audio Cable)`）。WDM-KS を使うなら明示指定が要る。WDM-KS を既定にする予定は無いので許容する。
- ログに解決経路が残るので、「なぜこのレートで開いたか」を後から追える。デバイス構成を変えたときの切り分けが効く。
- 設定項目が 4 つ増える。すべて `None` 既定なので、既存の config ファイルはそのまま読める。
