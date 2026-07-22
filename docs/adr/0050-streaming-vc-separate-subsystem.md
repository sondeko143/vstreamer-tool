# 0050. ストリーミング VC を Command/routing の外の専用サブシステムとして分離する

- Status: Proposed
- Date: 2026-07-22
- Related: [spec](../superpowers/specs/2026-07-22-rvc-streaming-vc-split-machine-design.md), [0004](0004-per-destination-sender-transport.md), [0051](0051-stream-transport-swappable-tiered.md), [0053](0053-streaming-vc-fixed-block-crossfade.md), [0054](0054-stream-vc-config-section.md)

## Context

既存のイベント/ルーティングモデル(`Command`→`WorkerInput`→`WorkerOutput`)は自己完結な離散リクエストで、ストリーム識別子・順序番号・ブロック間状態を持たない。低遅延の連続 VC は、固定ブロックを途切れなく処理し、ブロック間で文脈/クロスフェード状態を保持し、順序・ジッタ・欠落を扱う必要がある。この要件を既存 routing に載せると、発話単位前提の routing/sender/receiver を streaming 都合で歪める。一方、発話単位の字幕/翻訳(録音→文字起こし→翻訳→字幕)は現状の routing のまま維持したい。

## Decision

ストリーミング VC を既存の Command/routing モデルの**外**に置く専用サブシステムとして実装する。発話系は既存 routing のまま無改変で並走させる。両系統の接点はマイク入力のみとする。

## Alternatives rejected

- **既存 routing に載せる(案A: 入力ストリーム worker → vc worker → 既存 unary sender で流用)** — 発話単位の離散モデルに seq/ブロック間状態/ジッタを後付けすることになる。`_send` の例外握り潰し(`sender.py`)による無音穴や単一 sender の順序依存など、streaming に不利な性質をそのまま抱え、低遅延を実測で追い込みにくい。routing を streaming 都合で歪めるコストが、transport/routing 再利用で得る節約を上回る。

## Consequences

発話系を無傷で残せるため「発話単位の字幕/翻訳を維持」という要件に最も安全。streaming 側は連番・バックプレッシャ・ジッタ吸収を自前で設計できる。反面、sender/receiver/routing の再利用は限定的になり新規コードが増える。設定も別系統になる([0054](0054-stream-vc-config-section.md))。マシン間トランスポートは差し替え層として別途決める([0051](0051-stream-transport-swappable-tiered.md))。
