# 0041. OBS バックエンドは config を表示スタイルの権威とし、OBS の構造には触れない

- Status: Proposed
- Date: 2026-07-16
- Related: spec [2026-07-16-subtitle-obs-websocket-backend-design.md](../superpowers/specs/2026-07-16-subtitle-obs-websocket-backend-design.md); [ADR-0040](0040-subtitle-obs-backend-via-worker-type.md)（配線）

## Context

OBS の `Text (GDI+)` ソースは、テキストだけでなく表示スタイルもすべて設定として持つ（`font{face,size,flags}` / `color` / `opacity` / `outline` / `outline_size` / `outline_color` / `outline_opacity` / `align` / `valign` / `bk_color` / `bk_opacity` / `extents` / `extents_cx` / `extents_cy` / `extents_wrap`）。obs-websocket の `SetInputSettings` は汎用（obs_data をそのまま流す）なので、これらは vspeech から押し込める。

つまり「テキストだけ送って見た目は OBS の UI で設定する」ことも「config を権威にして全部 push する」こともできる。前者は実装が最小で OBS ユーザーには自然だが、`font_family` / `font_size` / `font_style` / `font_color` / `outline_color` / `anchor` / `margin` / `window_width` / `window_height` / `bg_color` が OBS バックエンドでは死にキーになる。

死にキーは黙って効かない。`config.toml.example` は全設定を文書化しており、subtitle は GUI にフォームが無く生 TOML で編集する（[ADR-0032](0032-gui-multi-pipeline-rewrite.md) の非ゴール）。つまり利用者は `font_size` を手で書いて、何も起きない理由をどこからも知らされない。設定を変えて何も起きないのは、設定が無いことより悪い。

加えて [ADR-0040](0040-subtitle-obs-backend-via-worker-type.md) は OBS を「同一イベントの別バックエンド」として位置づけた。`worker_type` を切り替えたら見た目が変わる、というのはその位置づけと整合しない。transcription の ACP↔GCP↔WHISPER や tts の VR2↔VOICEVOX は、バックエンドを変えても config の意味が変わらない。

別の軸として、「ソースが無ければ vspeech が作る」という選択肢もある。`CreateInput` はシーン名を要求するため、これはユーザーのシーンコレクションを書き換えることを意味する。

## Decision

**config を表示スタイルの唯一の権威とする。** OBS バックエンドは接続確立時と reload 時に、テキストに加えてスタイル一式を `SetInputSettings` で push する。tk 版の設定キーはすべて生きたまま、同じ意味を保つ:

| config | text_gdiplus |
|---|---|
| `font_family` / `font_size` / `font_style` | `font: {face, size, flags}`（`"bold"` → `flags` の bold ビット、それ以外 → 0。tk の `weight` 判定と同一規則）|
| `font_color` | `color` + `opacity: 100` |
| `outline_color` | `outline: true` / `outline_size: 1` / `outline_color` / `outline_opacity: 100` |
| `anchor` の e/w 成分 | `align`（tk の `justify_val` と同一規則: e→right / w→left / それ以外→center）|
| `anchor` の n/s 成分 | `valign`（n→top / s→bottom / それ以外→center）|
| `window_width` / `window_height` − `margin`×2 | `extents: true` / `extents_cx` / `extents_cy` / `extents_wrap: true` |
| `bg_color` | `bk_color` / `bk_opacity`（`systemTransparent` → `bk_opacity: 0`）|

**同時に、vspeech は OBS の構造には一切触れない。** シーン、input の存在、シーン内での配置・変形はユーザーが所有する。vspeech が所有するのは、ユーザーが作った input の設定値だけ。

## Alternatives rejected

- **OBS を権威にする（`text` だけ push）** — 実装は最小で OBS への結合も最小だが、config の 10 キーが OBS バックエンドで黙って死にキーになる（`config.toml.example` には載ったまま、生 TOML で編集でき、何も起きない）。`worker_type` を TK↔OBS で切り替えると見た目が変わってしまい、[ADR-0040](0040-subtitle-obs-backend-via-worker-type.md) が「同一イベントの別実装」として置いた位置づけと矛盾する。spec のゴール「config が唯一の権威であり続ける」に正面から反する。
- **段階導入（まず `text` だけ、スタイルは後続ブランチ）** — 実機で `SetInputSettings` が通ることを先に確かめられるので安全側だが、その中間状態では GUI に効かない項目が並ぶ期間が生まれる。実機確認は実装の最初のステップとして行えば足りる（後述）。
- **ソースが無ければ `CreateInput` で自動生成する** — `CreateInput` はシーン名を要求するため、ユーザーのシーンコレクションを書き換えることになる。どのシーンに入れるかを config で持たせても、結局サイズと位置はユーザーが手で調整する必要があり、「自動生成」は最後まで自動にならない。所有権の境界（構造＝ユーザー / 設定値＝vspeech）が曖昧になり、シーンを壊す事故の余地だけが残る。
- **`bg_color` を OBS バックエンドでは無視する** — 緑背景（`#00ff00`）は tk ウィンドウをクロマキーで抜くための手段であって、OBS の Text ソースには不要に見える。しかし無視すると `bg_color` だけが死にキーとして残り、権威の一貫性が崩れる。`bk_color` / `bk_opacity` へ素直に写せば、`systemTransparent` は透過に、`#00ff00` は緑の箱になり、tk と同じ挙動を保てる。

## Consequences

- 死にキーがゼロになる。`worker_type` を TK↔OBS で切り替えても見た目が変わらない。
- config の reload が OBS 上の表示に追従する。このため OBS バックエンドは `context.need_reload` / `reset_need_reload()` を使う。subtitle は現在 `configs_depends_on` を登録しながら `need_reload` を誰も読まない唯一のワーカー（毎フレーム `context.config` を直読みして代替している）だが、OBS バックエンドは他の全ワーカーと同じ機構に乗る。tk バックエンドの毎フレーム直読みは無改変のまま残す。
- OBS 側で手動変更したスタイルは、次の接続確立時と reload 時に上書きされる。これは「config が権威」の直接の帰結であり、意図した挙動。
- `SetInputSettings` に渡す値の形式（特に `color` の int が `0x00BBGGRR` か `0xAABBGGRR` か、`font.flags` のビット定義）は OBS のバージョンに依存する実装詳細である。ここが本設計で最も外れやすい箇所なので、実装の最初のステップで実機の OBS に対して確認し、マッピング層に閉じ込める。
- `Text (GDI+)` は Windows 専用プラグインである。Linux / macOS の OBS は `Text (FreeType 2)` で設定キーが異なる（`from_file` / `text_file` / `color1` / `word_wrap` 等）。本 ADR は Windows の `text_gdiplus` のみを対象とする（spec の非ゴール）。パイプライン側がヘッドレスになることと、OBS が動くホストが Windows であることは両立する。
