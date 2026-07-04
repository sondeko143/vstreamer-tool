# RVC 入力音声エンベロープ転写の再設計 設計書

- 日付: 2026-07-04
- ステータス: 承認済み（実装計画へ）
- ブランチ: `feat/rvc-input-envelope`（vstreamer-tool）
- 対象: `vspeech/worker/vc.py` の `adjust_output_vol_to_input_voice` 機能（[vc.py:115-174](../../../vspeech/worker/vc.py)）

## 1. 背景と目的

RVC ボイスチェンジ後の出力に、入力音声の音量変化（エンベロープ）を反映させる機能。現行実装は「動く」が、**入力の音量変化を忠実に写す** という目的に対して次の欠陥を持つ:

1. **`sqrt` の誤用**（[vc.py:120](../../../vspeech/worker/vc.py)）: `audioop.rms()` は既に **振幅 (RMS)** を返すため `rms/max` は線形な振幅比 [0,1]。そこへ `sqrt` を掛けると小さい値ほど持ち上がり（0.01→0.1）、**ダイナミクスを平坦化（圧縮）** してしまう＝目的の逆。power↔amplitude / 知覚ラウドネスの取り違えと推測。
2. **絶対ゲインを乗じている**（[vc.py:168](../../../vspeech/worker/vc.py)）: RVC 出力に入力の *絶対* 正規化振幅（≈0.1〜1.0）をそのまま乗算。出力レベルがマイクゲイン／声量の絶対値に丸ごと連動し、RVC 出力自身のエンベロープを無視する。「形を写す」ではなく「減衰させる」に近い。
3. **ゲインが階段状・平滑化なし**（[vc.py:163-172](../../../vspeech/worker/vc.py)）: 約 5ms 窓（16kHz で 80 サンプル）ごとに定数ゲイン、境界でハードに飛ぶ。補間もクロスフェードも無く、振幅変調アーティファクト（zipper noise）が乗る。
4. **窓が短すぎる（80 サンプル）**: RMS 推定がノイジーで、上記の階段と相まって悪化。
5. **`min_volume=0.1` の下限**: 入力無音でも RVC 出力の 10% が漏れる。
6. **`audioop` が非推奨**（Python 3.13 で削除）。numpy（rvc extra 必須依存）へ移せば正しさ・平滑性・速度を同時解決できる。`output_data += bytes` の O(n²) 連結も解消。

**目的**: 入力の *相対的な抑揚の形* を、RVC 出力本来のレベルを保ちながら滑らかに転写する。マイクゲインの絶対値に左右されない、頑健な実装にする。

方針は Approach A（**正規化シェイプ転写**）。**入力**の RMS エンベロープを自身の平均で正規化して「相対的な抑揚の形（平均 1）」を取り出し、サンプル解像度へ線形補間して出力に掛ける。

> **修正1 (2026-07-04, 実音声テスト後)**: 初版は RVC `change_rms` に倣い出力側エンベロープでも割る `gain = shape_in / shape_out` としていたが、実機で **(1) 無音区間にノイズ**（出力が静かなフレームで `shape_out→0` によりゲインが `max_gain` へ飽和し RVC のノイズフロアを最大 4× 増幅）、**(2) `max_gain>1` で compressor 的な音**（`÷shape_out` が出力の音量に逆相関してゲインを与える = ダイナミクス圧縮、gain vs |output| 相関 −0.81）が発生。根本原因は `÷shape_out` の除算。**出力側の除算を廃止し `gain = shape_in ** strength` に修正**（数値再現で無音ノイズ ×1.0・相関 +0.00 を確認）。
>
> **修正2 (2026-07-04, 同上)**: 修正1 後、**声の大きい部分で音割れ（クリップ）**。RVC 出力は既に int16 フルスケール近く（ピーク ±30000 等）なので、`gain = shape_in` が `max_gain=4.0` まで大きい部分を持ち上げると int16 を超えてハードクリップ（数値再現で 1864 サンプルが ±32767 に張り付き）。フルスケール信号は原理的にブースト不可。**ゲインは「ダック」＝ 1.0 以下** に限る（`|out*gain| ≤ |out| ≤ 32767` で常にクリップ無し）。`max_gain` 既定を **4.0 → 1.0** に変更（大きい入力の瞬間は RVC のレベルのまま、静かな瞬間だけ下げる）。`max_gain>1` は出力にヘッドルームがある場合のみ（クリップ自己責任）。

## 2. スコープ / 非目標

- **スコープ**: `adjust_output_vol_to_input_voice` の音量転写ロジックの再実装（新しい純関数ヘルパへ切り出し）、`VcConfig` の設定入れ替え、`config.toml.example` の `[vc]` 更新、ユニットテスト追加。
- **非目標（YAGNI）**:
  - `change_voice`（RVC 推論本体）・`rvc_worker` の warmup/推論経路の変更。
  - `input_boost` の `audioop.mul`（[vc.py:138](../../../vspeech/worker/vc.py)）の numpy 化 — エンベロープとは別関心。今回は据え置き（`audioop` import はこの用途で残る）。
  - マルチチャンネル対応（RVC 出力は mono 固定）。
  - 設定の後方互換 alias — **クリーン入れ替え**（config は個人利用・gitignore のため破壊的変更可）。

## 3. 設計

### 3.1 新ヘルパ（純関数・テスト対象）

`vspeech/worker/vc.py` にモジュール関数として追加（GPU/モデル不要でユニットテスト可能）:

```python
def apply_input_envelope(
    output_i16: NDArray[np.int16],  # RVC 出力（mono, int16）
    input_pcm: bytes,               # 入力 PCM（input_boost 適用前）
    input_sample_width: int,        # 入力 1 サンプルのバイト幅
    input_rate: int,                # 入力サンプルレート（window_ms→サンプル換算に使用）
    window_ms: float,               # RMS フレーム長（ミリ秒）
    strength: float,                # 0=無効 .. 1=フル転写
    min_gain: float,                # ゲイン下限クランプ
    max_gain: float,                # ゲイン上限クランプ
) -> NDArray[np.int16]:
```

**処理（修正後の実装）:**
1. 入力 bytes を `input_sample_width` に応じた dtype で読み、float へ。絶対スケールは後段の平均正規化で相殺するため正規化定数は不要。
2. 入力を **フレーム数 N** に比例分割し各フレームの RMS を計算 → `rms_in[N]`。N は入力サンプル数と `window_ms`（入力レート基準のフレーム長）から決定（`N = max(1, floor(input_samples / frame_len))`）。**出力側のフレーム RMS は計算しない**（除算を廃止したため）。
3. `rms_in` を自身の平均で正規化した相対抑揚を **出力サンプル長 `out_len` へ線形補間**（`np.interp`）→ 滑らかな `shape_in`（平均 1）。
4. ゲイン算出:
   ```
   shape_in = interp(rms_in / mean(rms_in))     # 入力の相対抑揚（平均 1）
   gain     = shape_in ** strength
   gain     = clip(gain, min_gain, max_gain)
   ```
5. `out_f * gain` を int16 レンジ [-32768, 32767] にクリップして返す。

**なぜこれで良いか**: `shape_in` の平均は 1 なので `out * shape_in` の平均レベルは RVC 出力のまま（レベル保存）。入力を自身の平均で正規化しているのでマイクゲインの絶対値に非依存。出力側の抑揚は割らないので、静かな出力フレームを増幅せず（ノイズを持ち上げず）、出力自身のダイナミクスも圧縮しない。「入力の音量変化を出力に *掛ける*（overlay）」という素直な意味に一致する。

### 3.2 エッジケース

- `output_i16` が空、または `input_pcm` が空 → RVC 出力をそのまま返す。
- `mean(rms_in) < eps`（入力が実質無音）→ 抑揚が定義できないため RVC 出力をそのまま返す（0 に潰さない）。
- `strength <= 0` → ゲインが全域 1.0 = RVC 出力と恒等（出力長・値ともに不変）。
- 出力長は常に不変（`len(output_i16)` を保つ）。
- `input_sample_width` が INT16 以外でも動く（dtype をバイト幅から選択）。ただし主対象は INT16。

### 3.3 設定（`VcConfig`、クリーン入れ替え）

`vspeech/config.py`:

```python
class VcConfig(BaseModel):
    enable: bool = False
    adjust_output_vol_to_input_voice: bool = True
    envelope_strength: float = 1.0        # 新規: 0..1（抑揚転写の深さ）
    min_gain: float = 0.1                 # 旧 min_volume。ゲイン下限（無音時のダック下限）
    max_gain: float = 1.0                 # ダック上限。>1 はフルスケール出力をクリップ（修正2）
    volume_adjust_window_ms: float = 25.0 # 旧 volume_adjust_window(160 バイト) を置換
```

- 旧 `min_volume` / `max_volume` / `volume_adjust_window` は **削除**。
- **`max_gain` を >1 にする理由（重要）**: 旧 `max_volume=1.0` のままだと「平均より大きいフレーム」を持ち上げられず、抑揚が片側（ダッキング）のみになる。相対シェイプを対称に転写するため上限は 1 超が必須。既定 4.0。
- `config.toml.example` の `[vc]` セクションを新フィールドに合わせて更新。

### 3.4 ワーカー統合

[vc.py:115-174](../../../vspeech/worker/vc.py) を書き換え:
- `input_vols` の事前計算ブロック（L115-134）を削除。
- `change_voice` 呼び出し後の `input_vols` 適用ループ（L156-174, `raw_frames`/`chunks`/`audioop.mul`）を削除。
- 代わりに:
  ```python
  if vc_config.adjust_output_vol_to_input_voice:
      audio = apply_input_envelope(
          audio, speech.sound.data, input_sample_width,
          input_rate=speech.sound.rate,
          window_ms=vc_config.volume_adjust_window_ms,
          strength=vc_config.envelope_strength,
          min_gain=vc_config.min_gain, max_gain=vc_config.max_gain,
      )
  output_data = audio.tobytes()
  ```
- エンベロープ経路から不要になった import / ヘルパを整理: `audioop.rms`・`sqrt`・（この用途の）`chunks`・`floor`。`audioop`（`mul` = input_boost 用）は残る。`chunks` が他で未使用なら削除。

## 4. テスト（`tests/test_vc_helpers.py` に追加。GPU/RVC 不要）

`apply_input_envelope` を純関数として特性テスト:

- **一定振幅入力** → ゲインが全域 ≈1、出力レベルが RVC 出力とほぼ一致（レベル保存）。
- **ランプ入力**（時間方向に振幅増加）→ ゲインが時間方向に単調増加。
- **平滑性** → 隣接サンプルのゲイン差の最大が閾値以下（階段でない＝補間が効いている）。
- **無音入力** → 出力が RVC 出力と不変（パススルー）。
- **`strength=0`** → 出力が RVC 出力と一致（恒等）。
- **出力長** → 入力の RVC 出力長と常に一致。

いずれも実 numpy のみで完結し、既存の `tests/test_vc_helpers.py`（`asyncio_mode="auto"`、モデル不要）に同居できる。

## 5. 検証

- `uv run pytest tests/test_vc_helpers.py` で新テストが green。
- `uv run ruff format . && uv run ruff check . && uv run ty check` を通す。
- 既存 `tests/test_event_chains.py` 等に影響がないこと（ルーティング非変更）。
- 実機の音声確認は RVC extra + GPU が要るため任意。数値挙動はユニットテストで担保。
