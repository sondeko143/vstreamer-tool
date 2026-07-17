# 0044. font_size を Tk の符号規約のまま OBS へ渡し、正値だけ 96 DPI でピクセルへ換算する（0041 を refine）

- Status: Accepted (refines [ADR-0041](0041-subtitle-obs-config-authority.md))
- Date: 2026-07-17
- Related: [ADR-0041](0041-subtitle-obs-config-authority.md)（config が表示スタイルの権威。本 ADR はその対応表の `font_size` 行だけを差し替える）; [ADR-0040](0040-subtitle-obs-backend-via-worker-type.md)（TK/OBS は同一イベントの別バックエンド）

## Context

[ADR-0041](0041-subtitle-obs-config-authority.md) の対応表は `font_size` を `font: {size}` へ**素通し**すると定めた。実機で確かめたところ、これは同じ `font_size` で TK と OBS の見た目が大きく食い違う（利用者の報告: 「同じフォントサイズでも TK と比べて OBS は大分小さく表示される」）。ADR-0041 の中心的な約束「`worker_type` を TK↔OBS で切り替えても見た目が変わらない」が破れていた。

**測定**（Meiryo UI bold、Windows GDI の `CreateFontIndirectW` + `GetTextMetricsW` を両経路に対して直接叩いた。画面 DPI 96）:

| config `font_size = 24` | 送られる値 | `lfHeight` | セル高 (`tmHeight`) | em 高 |
|---|---|---|---|---|
| TK | — | **-32** | 41px | 32px |
| OBS（素通し） | 24 | **+24** | 24px | 19px |

**OBS は TK の 59%** で描画されていた。原因は 2 つの効果の積である:

1. **単位**: Tk の正の `size` は**ポイント**（Tcl が画面 DPI でピクセルへ換算する。24pt = 32px @96DPI）。OBS は `obs-text.cpp` で `int font_size = obs_data_get_int(font_obj, "size"); ... lf.lfHeight = face_size;` と**論理単位（≒ピクセル）へ素通し**し、DPI 換算を一切しない。
2. **基準**: `LOGFONT.lfHeight` は符号で意味が変わる。**負なら文字の em 高、正ならセル高**（内部レディング込み）。Tk は負を渡し、OBS の素通しは正になる。よって OBS の 24 は「内部レディング込み 24px」で、字はさらに小さい。

ここで効く事実が 1 つある: **Tk の `size` の符号規約は `LOGFONT.lfHeight` と同一**である（正=ポイント、負=ピクセル＝em 高）。実測で `Font(size=24)` と `Font(size=-32)` は同一のメトリクス（cell 41 / ascent 34）を返す。

## Decision

`lib/obs_text_settings.build_text_settings` は `font_size` を **Tk の符号規約のまま** OBS へ渡す。正値のときだけポイント→ピクセル換算し、**負にして**送る:

- `font_size > 0`（ポイント）→ `font.size = -round(font_size * 96 / 72)`
- `font_size < 0`（既にピクセル＝em 高）→ `font.size = font_size` （そのまま）

OBS は値を検証・クランプせず `lfHeight` へ素通しするため、負値はそのまま「em 高」として効き、**TK と同一のフォントインスタンス**になる。**実機で確認済み**（利用者報告: 「-値をいれたら tk と OBS 同じ大きさになりました」）。

DPI は **96 固定**とする。実行時の DPI は使わない。

## Alternatives rejected

- **素通しのまま（ADR-0041 の当初の決定）** — 実測 59%。ADR-0041 の「切り替えても見た目が変わらない」を破る。
- **正のピクセル値へ換算する（`round(font_size * 96/72)` = 32）** — 単位の差は消えるが基準の差が残る。OBS の正値は**セル高**なので、セル 32px 対 TK のセル 41px で **78%**。59% よりましなだけで一致しない。一致させるには内部レディングを足した値（Meiryo UI では 41）が要るが、それはフォント固有で、算出には GDI が要る。
- **フォント固有のレディング係数を掛ける（`* 96/72 * 1.28`）** — Meiryo UI 専用のマジックナンバーになる。他のフォントで外れる。
- **実行時の画面 DPI で換算する** — ヘッドレス（Linux / Docker）では画面が無く取得できない。加えて**必要なのは OBS ホストの DPI** であって vspeech ホストのものではなく、クロスホスト構成では原理的に知りようがない。そもそも OBS はキャンバス（固定解像度）へ描くので DPI 非依存であり、実行時 DPI を掛けるのは二重に誤りになる。
- **`font_size` は TK と OBS で単位が違うと文書化して利用者に再調整させる** — 正直ではあるが、ADR-0041 の中心的な約束と、ADR-0040 が置いた「同一イベントの別バックエンド」という位置づけの両方を放棄する。transcription の ACP↔GCP↔WHISPER や tts の VR2↔VOICEVOX は、バックエンドを変えても config の意味が変わらない。
- **`font.size` に正値 41（＝実測で一致する値）を送る** — 実機では一致するが、41 という数字はフォントと DPI に依存し、ヘッドレスでは算出できない。負値なら `font_size * 96/72` で機械的に出せる。

## Consequences

- 同じ `font_size` で TK と OBS が同じ大きさになる。ADR-0041 の約束が `font_size` についても成立する。
- **既存の負値設定が壊れない。** `font_size = -32`（Tk ではピクセル指定）を素通しするのが要で、これを一律に `-round(size * 96/72)` へ通すと `+43` になり、OBS 側が「セル高 43px」と解釈して TK（em 32px）とずれる。符号で分岐する理由はここにある。
- **OBS の GUI にはフォントサイズが負値で表示される。** vspeech が設定値を push したソースを OBS の UI で開くと `-32` のような値が見える。ADR-0041 により config が権威で、OBS 側の手動変更はどのみち上書きされるため、実害は「見慣れない値が見える」ことに留まる。
- **DPI 96 以外の画面では TK と OBS が一致しない。** これは避けられない: TK はウィンドウを DPI スケールして描き、それを OBS がウィンドウキャプチャするので TK 側の見た目は画面 DPI に比例するが、OBS の Text ソースはキャンバスへ直接描くので DPI に依存しない。両者が一致しうるのは 96 DPI のときだけである。96 を選ぶのは、それが OBS 自身の描画が前提とする値だからである。
- OBS が将来 `font.size` を検証・クランプするようになれば、この経路は壊れる。そのときは正値＋内部レディング補正へ移るか、本 ADR を supersede すること。**この決定はソース読解ではなく実機測定に依拠している**（`obs-text.cpp` の読解から色の並びを推定して外した前例が ADR-0041 にある）。
