"""FCPE onnx が torch 参照実装と f0 で一致することの golden テスト。

資産 (fcpe.onnx) が無い通常実行では skip する。実行するには:

    uv run poe export-fcpe-onnx --output ./fcpe.onnx
    VSPEECH_FCPE_ONNX=./fcpe.onnx uv run --extra rvc --with torchfcpe pytest tests/test_fcpe_onnx.py -v

参照は torchfcpe の forward (= __call__、export したもの) と比較する。model.infer(...) は
f0_min による無声マスク等の後処理を足すので、export した forward とは別物である点に注意。
"""

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest

_ASSET = os.environ.get("VSPEECH_FCPE_ONNX")


@pytest.mark.skipif(
    not _ASSET,
    reason="fcpe.onnx が無い (uv run poe export-fcpe-onnx で生成し VSPEECH_FCPE_ONNX で指す)",
)
def test_fcpe_onnx_matches_torch():
    import torch
    import torchfcpe  # ty: ignore[unresolved-import]  # overlay 専用 (--with torchfcpe)

    from vspeech.lib.onnx_session import create_session

    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    wav = (0.6 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    bundled = torchfcpe.spawn_bundled_infer_model(torch.device("cpu")).eval()
    with torch.no_grad():
        ref_t = bundled(torch.from_numpy(wav).unsqueeze(0), sr, "local_argmax", 0.006)
    ref = ref_t.squeeze(-1).squeeze(0).cpu().numpy()

    assert _ASSET is not None
    sess = create_session(Path(_ASSET), torch.device("cpu"))
    got_raw = cast(np.ndarray, sess.run(None, {"waveform": wav[None, :]})[0])
    got = got_raw.squeeze(-1).squeeze(0)

    m = min(len(got), len(ref))
    voiced = ref[:m] > 1.0
    assert voiced.sum() > 0
    # 220Hz トーンなので有声フレームは ~220Hz を返すはず
    np.testing.assert_allclose(np.median(ref[:m][voiced]), 220.0, rtol=0.05)
    np.testing.assert_allclose(got[:m][voiced], ref[:m][voiced], rtol=0.02, atol=1.0)
