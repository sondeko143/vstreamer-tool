"""Golden test that the FCPE onnx matches the torch reference implementation on f0.

Skipped on a normal run, where the asset (fcpe.onnx) is absent. To run it:

    uv run poe export-fcpe-onnx --output ./fcpe.onnx
    VSPEECH_FCPE_ONNX=./fcpe.onnx uv run --all-extras --with torch --with torchfcpe pytest tests/test_fcpe_onnx.py -v

torch comes from the overlay: it left the dependency table with ADR-0080, and `--extra rvc`
on its own would also deselect the other extras.

The reference compared against is torchfcpe's forward (= __call__, the thing that was
exported). Note that model.infer(...) adds post-processing such as the unvoiced mask from
f0_min and is therefore a different thing from the exported forward.
"""

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from vspeech.lib.cuda_util import Device

_ASSET = os.environ.get("VSPEECH_FCPE_ONNX")


@pytest.mark.skipif(
    not _ASSET,
    reason="fcpe.onnx が無い (uv run poe export-fcpe-onnx で生成し VSPEECH_FCPE_ONNX で指す)",
)
def test_fcpe_onnx_matches_torch():
    import torch  # ty: ignore[unresolved-import]  # overlay only (--with torch)
    import torchfcpe  # ty: ignore[unresolved-import]  # overlay only (--with torchfcpe)

    from vspeech.lib.onnx_session import create_session

    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    wav = (0.6 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    bundled = torchfcpe.spawn_bundled_infer_model(torch.device("cpu")).eval()
    with torch.no_grad():
        ref_t = bundled(torch.from_numpy(wav).unsqueeze(0), sr, "local_argmax", 0.006)
    ref = ref_t.squeeze(-1).squeeze(0).cpu().numpy()

    assert _ASSET is not None
    sess = create_session(Path(_ASSET), Device("cpu"))
    got_raw = cast(np.ndarray, sess.run(None, {"waveform": wav[None, :]})[0])
    got = got_raw.squeeze(-1).squeeze(0)

    m = min(len(got), len(ref))
    voiced = ref[:m] > 1.0
    assert voiced.sum() > 0
    # It is a 220Hz tone, so voiced frames should come back at about 220Hz
    np.testing.assert_allclose(np.median(ref[:m][voiced]), 220.0, rtol=0.05)
    np.testing.assert_allclose(got[:m][voiced], ref[:m][voiced], rtol=0.02, atol=1.0)


@pytest.mark.skipif(
    not _ASSET,
    reason="fcpe.onnx が無い (uv run poe export-fcpe-onnx で生成し VSPEECH_FCPE_ONNX で指す)",
)
def test_fcpe_onnx_generalizes_over_length_and_zeros_unvoiced():
    """The trace is fixed at N=16000, but this checks that it matches torch at several
    lengths (generalizes over N) and fabricates no pitch over unvoiced spans (every frame
    equals torch)."""
    import torch  # ty: ignore[unresolved-import]  # overlay only (--with torch)
    import torchfcpe  # ty: ignore[unresolved-import]  # overlay only (--with torchfcpe)

    from vspeech.lib.onnx_session import create_session

    assert _ASSET is not None
    sr = 16000
    bundled = torchfcpe.spawn_bundled_infer_model(torch.device("cpu")).eval()
    sess = create_session(Path(_ASSET), Device("cpu"))

    # Check the voiced tone matches at several lengths (including a non-multiple of the
    # hop and one near the minimum length)
    for n in (8000, 12345, 24000, 433):
        t = np.arange(n, dtype=np.float32) / sr
        wav = (0.6 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        with torch.no_grad():
            ref = (
                bundled(torch.from_numpy(wav).unsqueeze(0), sr, "local_argmax", 0.006)
                .squeeze(-1)
                .squeeze(0)
                .cpu()
                .numpy()
            )
        got_raw = cast(np.ndarray, sess.run(None, {"waveform": wav[None, :]})[0])
        got = got_raw.squeeze(-1).squeeze(0)
        m = min(len(got), len(ref))
        voiced = ref[:m] > 1.0
        if voiced.sum() > 0:
            np.testing.assert_allclose(
                got[:m][voiced], ref[:m][voiced], rtol=0.02, atol=1.0
            )

    # The unvoiced span (silence in the second half): FCPE's forward emits NaN (0/0) on
    # unvoiced frames. The onnx collapses that to 0 inside the graph (the same contract as
    # rmvpe), so it matches torch (NaN->0) on every frame and fabricates no pitch where
    # there is none.
    from vspeech.lib.pitch_extract import pitch_extract_fcpe

    t = np.arange(sr, dtype=np.float32) / sr
    wav = (0.6 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    wav[sr // 2 :] = 0.0
    with torch.no_grad():
        ref = (
            bundled(torch.from_numpy(wav).unsqueeze(0), sr, "local_argmax", 0.006)
            .squeeze(-1)
            .squeeze(0)
            .cpu()
            .numpy()
        )
    ref = np.nan_to_num(ref, nan=0.0)  # torch's forward emits NaN where unvoiced
    got_raw = cast(np.ndarray, sess.run(None, {"waveform": wav[None, :]})[0])
    got = got_raw.squeeze(-1).squeeze(0)
    # the baked-in nan_to_num leaves the onnx with no NaN
    assert not np.isnan(got).any()
    m = min(len(got), len(ref))
    np.testing.assert_allclose(got[:m], ref[:m], atol=1.0)

    # The runtime helper also leaves no NaN and brings the silent span to about 0
    f0 = pitch_extract_fcpe(wav, sess)
    assert not np.isnan(f0).any()
    silent = f0[len(f0) * 6 // 10 :]
    assert float(np.max(np.abs(silent))) < 5.0
