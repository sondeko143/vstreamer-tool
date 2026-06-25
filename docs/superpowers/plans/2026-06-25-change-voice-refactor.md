# `change_voice` 純粋ヘルパ分解リファクタ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vspeech/lib/rvc.py:207` の `change_voice`（dep 129・12引数）を振る舞い保存のまま7つのヘルパへ分解し、def-use 結合を下げて各ステージを単体検証可能にする。

**Architecture:** `change_voice` を「純粋ヘルパ5＋モデル結合ヘルパ2＋薄いオーケストレータ」に分割。シグネチャと唯一の呼び出し元 `vspeech/worker/vc.py` は不変。純粋ヘルパは pytest 特性テスト（CPU）で、全パイプラインは数値ゴールデンテスト（実機 GPU）で担保する。

**Tech Stack:** Python 3.11 / uv / pytest（asyncio_mode=auto）/ torch+torchaudio cu128 / onnxruntime-gpu / fairseq / ruff / ty / poethepoet。

## Global Constraints

- Python **3.11 のみ**（`>=3.11,<3.12`）。`TaskGroup`/`except*` を下げない。
- 依存追加なし。RVC 系の実行・テストは **必ず `uv run --extra rvc`**（base env に torch/onnxruntime-gpu/fairseq 無し）。
- imports は **1行1つ**（ruff `force-single-line=true`、自動ソート）。型チェックは **ty**。
- Pydantic v2 のみ（v1 API 禁止）。
- `change_voice` の **数値挙動・dtype・演算順序・スライス意味は厳密保存**。`vspeech/worker/vc.py` は無改変。
- 検証は全工程この環境で実施（CUDA 有効・モデル3点存在・RVC worker の TOML config、quality=0）。config パスは環境変数 `VSPEECH_RVC_GOLDEN_CONFIG` 経由で渡し、リポジトリにマシン固有パスを残さない。
- コミットは feature ブランチ `refactor/change-voice-helpers` 上。

---

## Execution Update (2026-06-25): 確率的モデル → シード固定の厳密ゴールデン

Task 1 の捕捉中に判明: RVC シンセサイザは内部乱数を持ち `change_voice` 出力は**設計上 run-to-run で非決定的**（CUDA/CPU とも mean≈3.6%）。当初の「自己ノイズ床を許容差にした近似一致」は不成立。実測で次が判明し、**シード固定の厳密一致**へ強化した:

- 実行直前に `torch.manual_seed(s)` + `torch.cuda.manual_seed_all(s)` + `onnxruntime.set_seed(s)` → 出力 **bit-exact**（self-noise 0）。
- `infer` **前**の演算（feats/pitch/pitchf/p_len/sid）は無シードでも run 間 max=0。確率的なのは未改変の `infer` のみ。

これに伴う **Task 1 / Task 3 の確定仕様**（下の各タスク本文中の旧コードはこの節で上書き）:

- 捕捉スクリプトに `seed_all(seed=0)` を追加し、実行直前にシード→`change_voice`（保険で2回実行し bit-exact を assert）。npz は `voice_frames` / `voice_sample_rate` / `seed` / `output` を保存（self_noise 系フィールドは廃止）。
- ゴールデンテストは保存 `seed` で同一にシードしてからリファクタ後 `change_voice` を実行し、`max|Δ| ≤ 1` の**厳密一致**を assert（許容差ベースは廃止）。実コードは Task 3 Step 1 の更新版を参照。

> 実コードの source of truth は `scripts/capture_change_voice_golden.py` と `tests/test_change_voice_golden.py`。本節がタスク本文の旧コードに優先する。

---

### Task 1: ゴールデン捕捉インフラ＋リファクタ前キャプチャ

**Files:**
- Create: `scripts/capture_change_voice_golden.py`
- Modify: `.gitignore`（末尾に1行追加）
- Output（gitignore 下）: `tests/assets/rvc_golden/change_voice_golden.npz`

**Interfaces:**
- Produces（Task 3 が consume）:
  - `build_rvc_runtime(config_path: pathlib.Path) -> dict[str, Any]` — keys: `rvc_config, device, half_available, hubert_model, session, rmvpe_session, target_sample_rate, emb_output_layer, use_final_proj, f0_enabled`
  - `run_change_voice(rt: dict[str, Any], voice_frames: bytes, voice_sample_rate: int) -> NDArray[np.int16]`
  - `make_fixed_input(voice_sample_rate: int = 48000, seconds: float = 1.0) -> bytes`
  - module constant `GOLDEN_NPZ: pathlib.Path`

> このタスクは production コードを一切変更しない（`change_voice` はリファクタ前の現状を捕捉する）。Task 2 より**前に**完了させること。

- [ ] **Step 1: `.gitignore` に golden 出力先を追加**

`.gitignore` の `/tests/assets/voicevox/` 行の直後に追記:

```
/tests/assets/rvc_golden/
```

- [ ] **Step 2: 捕捉スクリプトを作成**

Create `scripts/capture_change_voice_golden.py`:

```python
"""Capture a numeric golden for change_voice BEFORE refactoring.

Run on the pre-refactor code with the real RVC model + GPU:
    uv run --extra rvc python scripts/capture_change_voice_golden.py \
        --config path/to/your-rvc-config.toml

Rebuilds the device/hubert/RVC-session/rmvpe-session exactly as rvc_worker does,
runs change_voice twice on a fixed deterministic input, and writes
tests/assets/rvc_golden/change_voice_golden.npz (gitignored): the input, the
run-1 output, and the run-to-run self-noise (the GPU-nondeterminism floor the
golden test tolerates).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "assets" / "rvc_golden"
GOLDEN_NPZ = GOLDEN_DIR / "change_voice_golden.npz"


def make_fixed_input(voice_sample_rate: int = 48000, seconds: float = 1.0) -> bytes:
    """A deterministic 220 Hz sine as int16 PCM bytes (no RNG -> reproducible)."""
    n = int(voice_sample_rate * seconds)
    t = np.arange(n, dtype=np.float64) / voice_sample_rate
    wave = (0.3 * np.sin(2 * np.pi * 220.0 * t) * 32767.0).astype(np.int16)
    return wave.tobytes()


def build_rvc_runtime(config_path: Path) -> dict[str, Any]:
    """Reconstruct the rvc_worker runtime (device + models + metadata)."""
    from vspeech.config import Config
    from vspeech.config import F0ExtractorType
    from vspeech.lib.cuda_util import get_device
    from vspeech.lib.pitch_extract import create_rmvpe_session
    from vspeech.lib.rvc import create_session
    from vspeech.lib.rvc import half_precision_available
    from vspeech.lib.rvc import load_hubert_model

    with open(config_path, "rb") as f:
        config = Config.read_config_from_file(f)
    rvc_config = config.rvc

    device, _ = get_device(rvc_config.gpu_id, rvc_config.gpu_name)
    half_available = half_precision_available(id=device.index)
    hubert_model = load_hubert_model(
        file_name=rvc_config.hubert_model_file, device=device, is_half=half_available
    )
    session = create_session(rvc_config.model_file, gpu_id=device.index)
    if rvc_config.f0_extractor_type == F0ExtractorType.rmvpe:
        rmvpe_session = create_rmvpe_session(rvc_config.rmvpe_model_file, device.index)
    else:
        rmvpe_session = None
    modelmeta: Any = session.get_modelmeta()
    metadata: dict[str, Any] = json.loads(modelmeta.custom_metadata_map["metadata"])
    return {
        "rvc_config": rvc_config,
        "device": device,
        "half_available": half_available,
        "hubert_model": hubert_model,
        "session": session,
        "rmvpe_session": rmvpe_session,
        "target_sample_rate": metadata["samplingRate"],
        "f0_enabled": metadata["f0"],
        "emb_output_layer": metadata.get("embOutputLayer", 9),
        "use_final_proj": metadata.get("useFinalProj", True),
    }


def run_change_voice(
    rt: dict[str, Any], voice_frames: bytes, voice_sample_rate: int
) -> NDArray[np.int16]:
    from vspeech.lib.rvc import change_voice

    return change_voice(
        voice_frames=voice_frames,
        half_available=rt["half_available"],
        rvc_config=rt["rvc_config"],
        voice_sample_rate=voice_sample_rate,
        target_sample_rate=rt["target_sample_rate"],
        device=rt["device"],
        emb_output_layer=rt["emb_output_layer"],
        use_final_proj=rt["use_final_proj"],
        hubert_model=rt["hubert_model"],
        session=rt["session"],
        f0_enabled=rt["f0_enabled"],
        rmvpe_session=rt["rmvpe_session"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    rt = build_rvc_runtime(args.config)
    voice_frames = make_fixed_input(args.sample_rate, args.seconds)

    out1 = run_change_voice(rt, voice_frames, args.sample_rate)
    out2 = run_change_voice(rt, voice_frames, args.sample_rate)

    n = min(out1.shape[0], out2.shape[0])
    diff = np.abs(out1[:n].astype(np.int32) - out2[:n].astype(np.int32))
    self_max = int(diff.max()) if n else 0
    self_mean = float(diff.mean()) if n else 0.0
    print(
        f"output_len={out1.shape[0]} len2={out2.shape[0]} "
        f"self_noise max={self_max} mean={self_mean:.4f}"
    )

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        GOLDEN_NPZ,
        voice_frames=np.frombuffer(voice_frames, dtype=np.int16),
        voice_sample_rate=np.int64(args.sample_rate),
        output=out1,
        self_noise_max=np.int64(self_max),
        self_noise_mean=np.float64(self_mean),
    )
    print(f"wrote {GOLDEN_NPZ}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 捕捉スクリプトを実機実行（リファクタ前コード）**

Run（warmup の torch.compile/onnx グラフ構築で数分かかり得る。十分なタイムアウトで）:

```
uv run --extra rvc python scripts/capture_change_voice_golden.py --config path/to/your-rvc-config.toml
```

Expected: `output_len=... self_noise max=<小さい整数> mean=<小数>` と `wrote .../change_voice_golden.npz` が表示され、npz が生成される。`self_noise max` の値を控える（Task 3 の許容差設定に使う）。

- [ ] **Step 4: ruff/ty でスクリプトを整える**

Run:

```
uv run ruff format scripts/capture_change_voice_golden.py
uv run ruff check scripts/capture_change_voice_golden.py
uv run ty check scripts/capture_change_voice_golden.py
```

Expected: いずれも pass（または ruff が自動整形）。

- [ ] **Step 5: コミット（npz は gitignore 済みで含まれない）**

```bash
git add scripts/capture_change_voice_golden.py .gitignore
git commit -m "test(rvc): add change_voice numeric-golden capture (pre-refactor baseline)"
```

---

### Task 2: `change_voice` を純粋＋モデルヘルパへ分解

**Files:**
- Create: `tests/test_rvc_helpers.py`
- Modify: `vspeech/lib/rvc.py`（`change_voice` を書き換え、直前に7ヘルパを追加。L207–337 を置換）

**Interfaces:**
- Consumes: 既存 `extract_features` / `get_resampler` / `pitch_extract` / `infer`（シグネチャ不変）。
- Produces（module-private、Task 3 はゴールデンで全体を見るのみ）:
  - `_pad_input_to_block(voice_frames: bytes) -> np.ndarray`
  - `_quality_padding(audio: torch.Tensor, rvc_config: RvcConfig, voice_sample_rate: int, target_sample_rate: int) -> tuple[torch.Tensor, int]`
  - `_extract_hubert_feats(hubert_model: HubertModel, audio_pad: torch.Tensor, device: torch.device, half_available: bool, emb_output_layer: int, use_final_proj: bool) -> torch.Tensor`
  - `_select_pitch(audio_pad: torch.Tensor, rvc_config: RvcConfig, f0_enabled: bool, p_len: int, device: torch.device, rmvpe_session: InferenceSession | None) -> tuple[torch.Tensor | None, torch.Tensor | None]`
  - `_is_model_half(session: InferenceSession) -> bool`
  - `_align_pitch_to_feats(pitch: torch.Tensor | None, pitchf: torch.Tensor | None, feats_len: int) -> tuple[torch.Tensor | None, torch.Tensor | None]`
  - `_postprocess(audio1: torch.Tensor, t_pad_tgt: int) -> NDArray[np.int16]`

- [ ] **Step 1: 純粋ヘルパの特性テストを書く（失敗する）**

Create `tests/test_rvc_helpers.py`:

```python
from typing import cast

import numpy as np
import torch
from onnxruntime import InferenceSession
from torch.nn import functional

from vspeech.config import RvcConfig
from vspeech.config import RvcQuality
from vspeech.lib.rvc import _align_pitch_to_feats
from vspeech.lib.rvc import _is_model_half
from vspeech.lib.rvc import _pad_input_to_block
from vspeech.lib.rvc import _postprocess
from vspeech.lib.rvc import _quality_padding
from vspeech.lib.rvc import _select_pitch


def test_pad_input_to_block_rounds_up_to_128_and_left_pads():
    raw = np.arange(1, 201, dtype=np.int16)  # 200 -> next multiple of 128 is 256
    out = _pad_input_to_block(raw.tobytes())
    assert out.shape[0] == 256
    np.testing.assert_allclose(out[-200:], raw.astype(np.float32) / 32768.0, rtol=1e-6)
    np.testing.assert_array_equal(out[:56], np.zeros(56))


def test_pad_input_to_block_already_aligned_no_pad():
    raw = np.ones(128, dtype=np.int16)
    out = _pad_input_to_block(raw.tobytes())
    assert out.shape[0] == 128
    np.testing.assert_allclose(out, np.ones(128, dtype=np.float32) / 32768.0, rtol=1e-6)


def test_quality_padding_zero_is_noop():
    audio = torch.arange(10, dtype=torch.float32).view(1, -1)
    cfg = RvcConfig(quality=RvcQuality.zero)
    audio_pad, t_pad_tgt = _quality_padding(audio, cfg, 16000, 40000)
    assert t_pad_tgt == 0
    assert audio_pad.shape == (10,)
    np.testing.assert_array_equal(audio_pad.numpy(), audio.squeeze(0).numpy())


def test_quality_padding_positive_reflects():
    audio = torch.arange(10, dtype=torch.float32).view(1, -1)
    cfg = RvcConfig(quality=RvcQuality.one)
    vsr, tsr = 16000, 40000
    audio_pad, t_pad_tgt = _quality_padding(audio, cfg, vsr, tsr)
    sec = (1 * (10 - 1)) / vsr  # repeat=1
    assert t_pad_tgt == round(tsr * sec)
    expected = functional.pad(audio, (9, 9), mode="reflect").squeeze(0)
    np.testing.assert_array_equal(audio_pad.numpy(), expected.numpy())
    assert audio_pad.shape[0] == 10 + 2 * 9


class _FakeInput:
    def __init__(self, type_str: str):
        self.type = type_str


class _FakeSession:
    def __init__(self, type_str: str):
        self._inputs = [_FakeInput(type_str)]

    def get_inputs(self):
        return self._inputs


def test_is_model_half_float_is_false():
    session = cast(InferenceSession, _FakeSession("tensor(float)"))
    assert _is_model_half(session) is False


def test_is_model_half_float16_is_true():
    session = cast(InferenceSession, _FakeSession("tensor(float16)"))
    assert _is_model_half(session) is True


def test_align_pitch_to_feats_trims_tail():
    pitch = torch.arange(10).view(1, -1)
    pitchf = torch.arange(10).view(1, -1).float()
    p, pf = _align_pitch_to_feats(pitch, pitchf, 4)
    assert p is not None and pf is not None
    np.testing.assert_array_equal(p.numpy(), np.array([[6, 7, 8, 9]]))
    np.testing.assert_array_equal(pf.numpy(), np.array([[6, 7, 8, 9]], dtype=np.float32))


def test_align_pitch_to_feats_none_passthrough():
    assert _align_pitch_to_feats(None, None, 4) == (None, None)


def test_postprocess_no_trim_when_zero():
    audio1 = torch.arange(6, dtype=torch.int16)
    out = _postprocess(audio1, 0)
    np.testing.assert_array_equal(out, np.arange(6, dtype=np.int16))


def test_postprocess_trims_both_ends():
    audio1 = torch.arange(10, dtype=torch.int16)
    out = _postprocess(audio1, 2)
    np.testing.assert_array_equal(out, np.arange(10, dtype=np.int16)[2:-2])


def test_select_pitch_disabled_returns_none():
    audio_pad = torch.zeros(16000, dtype=torch.float32)
    result = _select_pitch(
        audio_pad=audio_pad,
        rvc_config=RvcConfig(),
        f0_enabled=False,
        p_len=10,
        device=torch.device("cpu"),
        rmvpe_session=None,
    )
    assert result == (None, None)
```

- [ ] **Step 2: テストを実行し、失敗を確認**

Run:

```
uv run --extra rvc python -m pytest tests/test_rvc_helpers.py -v
```

Expected: ImportError（`_pad_input_to_block` 等が存在しない）で collection/実行が FAIL。

- [ ] **Step 3: 7ヘルパを追加し `change_voice` を薄いオーケストレータへ書き換え**

`vspeech/lib/rvc.py` の現 `change_voice`（L207–337）を、以下の **7ヘルパ＋新 change_voice** で置換する（`load_hubert_model` の直後に挿入）:

```python
def _pad_input_to_block(voice_frames: bytes) -> np.ndarray:
    input_sound = np.frombuffer(voice_frames, dtype="int16")
    input_size = input_sound.shape[0]
    if input_size % 128 != 0:
        input_size = input_size + (128 - (input_size % 128))
    audio = input_sound.astype(np.float32) / 32768.0
    if audio.shape[0] < input_size:
        audio = np.concatenate([np.zeros([input_size]), audio])
    return audio


def _quality_padding(
    audio: torch.Tensor,
    rvc_config: RvcConfig,
    voice_sample_rate: int,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int]:
    repeat = rvc_config.quality.value
    quality_padding_sec = (repeat * (audio.shape[1] - 1)) / voice_sample_rate
    t_pad = round(voice_sample_rate * quality_padding_sec)
    t_pad_tgt = round(target_sample_rate * quality_padding_sec)
    audio_pad = functional.pad(audio, (t_pad, t_pad), mode="reflect").squeeze(0)
    return audio_pad, t_pad_tgt


def _extract_hubert_feats(
    hubert_model: HubertModel,
    audio_pad: torch.Tensor,
    device: torch.device,
    half_available: bool,
    emb_output_layer: int,
    use_final_proj: bool,
) -> torch.Tensor:
    feats = audio_pad
    if half_available:
        feats = feats.half()
    else:
        feats = feats.float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()  # nosec B101 - internal shape invariant
    feats = feats.view(1, -1)
    feats = extract_features(
        model=hubert_model,
        feats=feats,
        dev=device,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )
    return cast(
        torch.Tensor,
        functional.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1),
    )


def _select_pitch(
    audio_pad: torch.Tensor,
    rvc_config: RvcConfig,
    f0_enabled: bool,
    p_len: int,
    device: torch.device,
    rmvpe_session: InferenceSession | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not f0_enabled:
        return None, None
    pitch, pitchf = pitch_extract(
        audio_pad,
        rvc_config.f0_up_key,
        16000,
        rvc_config.window,
        f0_extractor=rvc_config.f0_extractor_type,
        rmvpe_session=rmvpe_session,
        silence_front=0,
    )
    pitch = pitch[:p_len]
    pitchf = pitchf[:p_len]
    pitch_t = torch.tensor(pitch, device=device).unsqueeze(0).long()
    pitchf_t = torch.tensor(pitchf, device=device, dtype=torch.float).unsqueeze(0)
    return pitch_t, pitchf_t


def _is_model_half(session: InferenceSession) -> bool:
    return session.get_inputs()[0].type != "tensor(float)"


def _align_pitch_to_feats(
    pitch: torch.Tensor | None,
    pitchf: torch.Tensor | None,
    feats_len: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if pitch is not None and pitchf is not None:
        return pitch[:, -feats_len:], pitchf[:, -feats_len:]
    return pitch, pitchf


def _postprocess(audio1: torch.Tensor, t_pad_tgt: int) -> NDArray[np.int16]:
    if t_pad_tgt != 0:
        audio1 = audio1[t_pad_tgt : -1 * t_pad_tgt]
    return audio1.detach().cpu().numpy()


def change_voice(
    voice_frames: bytes,
    half_available: bool,
    rvc_config: RvcConfig,
    voice_sample_rate: int,
    target_sample_rate: int,
    device: torch.device,
    emb_output_layer: int,
    use_final_proj: bool,
    hubert_model: HubertModel,
    session: InferenceSession,
    f0_enabled: bool,
    rmvpe_session: InferenceSession | None,
) -> NDArray[np.int16]:
    vc_start_time = time.time()
    audio_np = _pad_input_to_block(voice_frames)
    audio = torch.from_numpy(audio_np).to(device=device, dtype=torch.float32)

    resampler = get_resampler(voice_sample_rate, 16000, device)
    audio = resampler(audio).unsqueeze(0)

    audio_pad, t_pad_tgt = _quality_padding(
        audio, rvc_config, voice_sample_rate, target_sample_rate
    )
    sid = torch.tensor(0, device=device).unsqueeze(0).long()

    feats = _extract_hubert_feats(
        hubert_model=hubert_model,
        audio_pad=audio_pad,
        device=device,
        half_available=half_available,
        emb_output_layer=emb_output_layer,
        use_final_proj=use_final_proj,
    )

    p_len = audio_pad.shape[0] // rvc_config.window
    if feats.shape[1] < p_len:
        p_len = feats.shape[1]
    pitch, pitchf = _select_pitch(
        audio_pad=audio_pad,
        rvc_config=rvc_config,
        f0_enabled=f0_enabled,
        p_len=p_len,
        device=device,
        rmvpe_session=rmvpe_session,
    )

    vc_end_time = time.time()
    logger.info(
        "rvc: pitch size adjusted: elapsed time: %s", vc_end_time - vc_start_time
    )

    is_model_half = _is_model_half(session)
    feats_len = feats.shape[1]
    pitch, pitchf = _align_pitch_to_feats(pitch, pitchf, feats_len)
    p_len_tensor = torch.tensor([feats_len], device=device).long()

    with torch.inference_mode():
        audio1 = (
            infer(
                session=session,
                is_half=is_model_half,
                feats=feats,
                pitch_length=p_len_tensor,
                pitch=pitch,
                pitchf=pitchf,
                sid=sid,
            )[0]
            * 32767.5
        ).data.to(dtype=torch.int16)

    del feats, p_len_tensor
    vc_end_time = time.time()
    logger.info("rvc: inferred: elapsed time: %s", vc_end_time - vc_start_time)

    result = _postprocess(audio1, t_pad_tgt)
    del pitch, pitchf, sid
    return result
```

> 振る舞い保存メモ: (1) L289 の未使用 `p_len_tensor` dead store を削除（L307 相当のみ残す）。(2) `sid = 0; sid = torch.tensor(...)` を1行へ統合（値は同一）。(3) `f0_up_key` ローカルは廃し `_select_pitch` 内で `rvc_config.f0_up_key` を直接参照。(4) `_is_model_half` は `!= "tensor(float)"` で元の if/else と等価。(5) `_postprocess` の `audio1[t_pad_tgt : -1 * t_pad_tgt]` は元の `offset:end` と同一スライス。その他の演算・dtype・順序は不変。

- [ ] **Step 4: 純粋ヘルパテストを実行し、pass を確認**

Run:

```
uv run --extra rvc python -m pytest tests/test_rvc_helpers.py -v
```

Expected: 全テスト PASS。

- [ ] **Step 5: ruff / ty を pass させる**

Run:

```
uv run ruff format vspeech/lib/rvc.py tests/test_rvc_helpers.py
uv run ruff check vspeech/lib/rvc.py tests/test_rvc_helpers.py
uv run ty check vspeech/lib/rvc.py tests/test_rvc_helpers.py
```

Expected: clean（`change_voice` リファクタで新規 ty 診断を増やさない。既存の周辺診断には触れない）。

- [ ] **Step 6: コミット**

```bash
git add vspeech/lib/rvc.py tests/test_rvc_helpers.py
git commit -m "refactor(rvc): decompose change_voice into pure + model helpers"
```

---

### Task 3: 数値ゴールデン等価テスト＋受け入れ

**Files:**
- Create: `tests/test_change_voice_golden.py`

**Interfaces:**
- Consumes: Task 1 の `build_rvc_runtime` / `run_change_voice` と `GOLDEN_NPZ`（`scripts/` を `sys.path` に追加して import）。Task 2 のリファクタ後 `change_voice`。

- [ ] **Step 1: ゴールデンテストを作成**

Create `tests/test_change_voice_golden.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

GOLDEN_NPZ = REPO_ROOT / "tests" / "assets" / "rvc_golden" / "change_voice_golden.npz"
GOLDEN_CONFIG = Path(os.environ["VSPEECH_RVC_GOLDEN_CONFIG"])  # via env var, not in repo

pytestmark = pytest.mark.skipif(
    not GOLDEN_NPZ.exists()
    or not torch.cuda.is_available()
    or not GOLDEN_CONFIG.exists(),
    reason="golden npz / CUDA / $VSPEECH_RVC_GOLDEN_CONFIG not available",
)


def test_change_voice_matches_golden():
    from capture_change_voice_golden import build_rvc_runtime
    from capture_change_voice_golden import run_change_voice

    data = np.load(GOLDEN_NPZ)
    voice_frames = data["voice_frames"].astype(np.int16).tobytes()
    voice_sample_rate = int(data["voice_sample_rate"])
    golden = data["output"]
    self_max = int(data["self_noise_max"])
    self_mean = float(data["self_noise_mean"])

    rt = build_rvc_runtime(DEFAULT_CONFIG)
    out = run_change_voice(rt, voice_frames, voice_sample_rate)

    assert out.shape == golden.shape, f"length changed: {out.shape} vs {golden.shape}"
    diff = np.abs(out.astype(np.int32) - golden.astype(np.int32))
    # Tolerate GPU-kernel nondeterminism (self-noise floor), reject real drift.
    atol_max = max(8, 4 * self_max)
    atol_mean = max(1.0, 4 * self_mean)
    assert diff.max() <= atol_max, f"max diff {diff.max()} > {atol_max}"
    assert diff.mean() <= atol_mean, f"mean diff {diff.mean()} > {atol_mean}"
```

> 許容差 `atol_max`/`atol_mean` の定数（8 / 4×）は初期値。Step 2 実行で `diff.max()`/`diff.mean()` が捕捉した self-noise を僅かに超える程度（=非決定性のみ）であることを確認し、もし正当な非決定性で落ちるなら定数を観測値ベースに微調整する（挙動破壊なら多数サンプルが大きくずれて落ちる＝検知が目的）。

- [ ] **Step 2: ゴールデンテストを実機実行**

Run:

```
uv run --extra rvc python -m pytest tests/test_change_voice_golden.py -v -s
```

Expected: PASS。`-s` で `diff.max()`/`diff.mean()` を目視し、self-noise 床と同水準であることを確認（挙動不変の数値的確証）。落ちた場合は systematic-debugging で原因（どのヘルパ抽出か）を切り分ける。

- [ ] **Step 3: フルチェック（poe check）**

Run:

```
uv run --extra rvc poe check
```

Expected: ruff/ty/pytest が green（GPU/モデル非依存テストは常時、ゴールデン/ヘルパは extra 同期下で実行）。`poe check` の構成は `pyproject.toml` の `[tool.poe.tasks]` 準拠。

- [ ] **Step 4: dep 低下の確認（受け入れメトリクス）**

Run:

```
uv run poe metrics --top 30
```

Expected: `change_voice` の dep が大幅低下（目安 ~20–30）、`both-high` から外れる。新ヘルパは個別に低 dep。出力を控える（before 129 → after を記録）。

- [ ] **Step 5: ゴールデンテストをコミット**

```bash
git add tests/test_change_voice_golden.py
git commit -m "test(rvc): golden equivalence test for refactored change_voice"
```

---

## Self-Review

**1. Spec coverage（spec 各節 → タスク対応）:**
- §3 分解設計（7ヘルパ）→ Task 2 Step 3。✓
- §3.1 latent 2件（dead store / sid）→ Task 2 Step 3 の振る舞い保存メモ。✓
- §4.1 純粋ヘルパ・ユニットテスト（quality>0 含む）→ Task 2 Step 1（`test_quality_padding_positive_reflects` 他）。✓
- §4.2 数値ゴールデン（捕捉→リファクタ→テスト）→ Task 1（捕捉）/ Task 2（リファクタ）/ Task 3（テスト）。✓
- §5 リスク（GPU 非決定性・device 選択・warmup・dep 検証）→ Task 1 self-noise 捕捉 / `get_device` 複製 / 長時間タイムアウト / Task 3 Step 4。✓
- §6 受け入れ基準 → Task 3 Step 2–4。✓
- 制約「vc.py 不変」→ どのタスクも `vspeech/worker/vc.py` を Modify に含めない。✓

**2. Placeholder scan:** TBD/TODO/「適切に処理」等なし。全ステップに実コード/実コマンド/期待出力あり。許容差定数は実値（8 / 4×）で、調整は実行時の検証手順として明記（プレースホルダではない）。✓

**3. Type consistency:** ヘルパ名・引数名・戻り値型は §Interfaces と Task 2 Step 3 実装、Task 1/3 の呼び出しで一致（`build_rvc_runtime`/`run_change_voice`/`_pad_input_to_block`/`_quality_padding`/`_extract_hubert_feats`/`_select_pitch`/`_is_model_half`/`_align_pitch_to_feats`/`_postprocess`）。`change_voice` のシグネチャは現状と同一。✓
