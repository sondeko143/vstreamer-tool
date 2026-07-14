# 0017. RVC 音量整合を入力平均正規化 RMS シェイプ転写で行う

- Status: Accepted
- Date: 2026-07-04
- Related: spec [2026-07-04-rvc-input-envelope-design.md](../superpowers/specs/2026-07-04-rvc-input-envelope-design.md); [ADR-0018](0018-rvc-envelope-duck-only.md)（ダック限定の後続修正）

## Context

RVC ボイスチェンジ後の出力に入力音声の音量変化（エンベロープ）を反映させる機能が、「入力の抑揚を忠実に写す」という目的に対して欠陥を抱えていた。旧実装は (1) `audioop.rms()` が既に振幅を返すのに `sqrt` を掛けて小さい値を持ち上げダイナミクスを平坦化（圧縮）し、(2) 入力の *絶対* 正規化振幅をそのまま乗じるため出力レベルがマイクゲイン／声量の絶対値に丸ごと連動し RVC 出力自身のエンベロープを無視し、(3) 約 5ms 窓ごとに定数ゲインを階段状に飛ばして zipper noise を生み、(4) 窓が 80 サンプルと短く RMS 推定がノイジーで、(5) `min_volume=0.1` により入力無音でも出力の 10% が漏れていた。「形を写す」のではなく「マイクゲインに連動して減衰させる」実装になっていた。目的は入力の *相対的な抑揚の形* を、RVC 出力本来のレベルを保ちつつ滑らかに、マイクゲインの絶対値に左右されず転写することである。

## Decision

入力音量整合を **正規化シェイプ転写** で実装する。入力 RMS エンベロープを自身の平均で正規化して相対抑揚（平均 1）を取り出し、出力サンプル軸へ線形補間したうえで `gain = shape_in ** strength` として出力サンプルに乗算する。出力側での除算（÷shape_out）は行わない。`shape_in` の平均は 1 なので出力の平均レベルは RVC 出力のまま保存され、入力を自身の平均で正規化しているためマイクゲインの絶対値に非依存となる。設定は後方互換 alias を持たないクリーン入れ替えとし、旧 `min_volume`/`max_volume`/`volume_adjust_window` を削除、`envelope_strength`/`min_gain`/`max_gain`/`volume_adjust_window_ms` を新設する。

## Alternatives rejected

- **旧実装（`sqrt` ＋絶対正規化振幅の乗算）** — ダイナミクスを平坦化し、出力レベルをマイクゲインに連動させる。目的の逆。
- **RVC `change_rms` 流の入出力両側正規化 `gain = shape_in / shape_out`** — 無音区間で `shape_out→0` によりゲインが飽和し RVC のノイズフロアを最大 ×4 増幅、さらに `÷shape_out` が出力音量と逆相関して compressor 的圧縮（gain vs |output| 相関 −0.81）を生む。数値再現で棄却。
- **後方互換 alias を残す** — 個人所有・gitignore の config に不要な複雑性。

## Consequences

「入力の抑揚を出力に掛ける（overlay）」という素直な意味に一致し、静音フレームのノイズを持ち上げず、出力自身のダイナミクスも圧縮しない。config は破壊的変更となるが影響は自分の config のみ。`max_gain` の既定値と方向（ブースト可否）は後続の [ADR-0018](0018-rvc-envelope-duck-only.md) で改めて決定する。
