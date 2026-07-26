"""Export torchfcpe's bundled FCPE to a "waveform-input" ONNX.

A **one-shot** offline step, not part of the runtime. Its dependencies (torchfcpe / onnx /
onnxscript) are supplied by the poe task's `uv run --with` (they are not in
pyproject/uv.lock).

    uv run poe export-fcpe-onnx --output ~/.config/vstreamer/fcpe.onnx

Launch it as **`python -m scripts.export_fcpe_onnx`**, not
`python scripts/export_fcpe_onnx.py`.

The contract of the fcpe.onnx it writes:
  input   waveform  (1, N)  float32  16kHz mono
  output  f0        (1, T, 1)  Hz     unvoiced frames are 0
  threshold / sample_rate(16000) / decoder_mode("local_argmax") are baked in at export
  time (not made variable at runtime = a non-goal of ADR-0049).

Why the export needs to be special (ADR-0049 / what the spike found):
  * The new dynamo exporter cannot handle the data-dependent branch inside wav2mel
    (.item() on wav.min()<-1).
  * The legacy tracer (dynamo=False) cannot handle it either, because of the complex type
    of torch.stft(return_complex=True).
  * -> So MelModule's STFT is reproduced exactly with conv1d (cos/sin DFT basis + hann
    window, center=False), matching the original torch.stft to f0 max_rel ~1e-6, and
    exported with the legacy tracer at opset17.
  * output_proj's weight_norm parametrization is stripped before the export.

This script asserts (1) that the conv-STFT matches the original torch.stft and (2) that
the onnx matches torch; if either fails, no asset is written.
"""

import argparse
import io
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as Fnn

# torch.onnx's verbose output and warnings contain non-ASCII (emoji) and crash on a
# Windows cp1252 stdout. Pin it to UTF-8 (the encoding guard this project keeps needing).
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

THRESHOLD = 0.006
SR = 16000
DECODER = "local_argmax"
OPSET = 17
# Verification tolerance (the spike measured ~1e-6; leave plenty of margin)
REL_TOL = 1e-3


def _patched_mel_call(
    self, y, key_shift=0, speed=1, center=False, no_cache_window=False
):
    """Replace MelModule.__call__'s inference path (center=False, key_shift=0) with a
    conv1d-DFT.

    torch.stft(return_complex=True) does not go into ONNX, so the magnitude spectrum is
    reproduced exactly with cos/sin kernels that include the hann window. Assumes
    win_size==n_fft (the bundled default is 1024==1024).
    """
    n_fft = self.n_fft
    win_size = self.win_size
    hop = self.hop_length
    clip_val = self.clip_val
    # The cos/sin kernels have length n_fft. torch.stft centre-zero-pads when
    # win_length<n_fft, but equal lengths are assumed here (the bundled default is
    # 1024==1024). Fail loudly if they differ.
    assert win_size == n_fft, (
        f"conv-STFT replacement assumes win_size==n_fft (got {win_size} != {n_fft})"
    )
    y = y.squeeze(-1)  # (B, N)
    window = torch.hann_window(win_size, device=y.device, dtype=y.dtype)
    pad_left = (win_size - hop) // 2
    pad_right = max((win_size - hop + 1) // 2, win_size - y.size(-1) - pad_left)
    mode = "reflect" if pad_right < y.size(-1) else "constant"
    y = Fnn.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)  # (B, L)

    n_freq = n_fft // 2 + 1
    n = torch.arange(n_fft, device=y.device, dtype=y.dtype)
    k = torch.arange(n_freq, device=y.device, dtype=y.dtype)
    angle = (2.0 * math.pi / n_fft) * k.unsqueeze(1) * n.unsqueeze(0)  # (n_freq, n_fft)
    cos_k = (window.unsqueeze(0) * torch.cos(angle)).unsqueeze(1)  # (n_freq,1,n_fft)
    sin_k = (window.unsqueeze(0) * torch.sin(angle)).unsqueeze(1)
    yb = y.unsqueeze(1)  # (B,1,L)
    real = Fnn.conv1d(yb, cos_k, stride=hop)  # (B, n_freq, T)
    imag = -Fnn.conv1d(yb, sin_k, stride=hop)
    spec = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-9)  # (B, n_freq, T)

    from torchfcpe import mel_extractor as ME

    if self.out_stft:
        spec = spec[:, :512, :]
    else:
        spec = torch.matmul(self.mel_basis, spec)  # (B, n_mels, T)
    spec = ME.dynamic_range_compression_torch(spec, clip_val=clip_val)
    spec = spec.transpose(-1, -2)  # (B, T, n_mels)
    return spec


class FcpeWave(torch.nn.Module):
    """The export wrapper that takes a waveform (1, N) and returns f0 (1, T, 1) in Hz."""

    def __init__(self, bundled):
        super().__init__()
        self.bundled = bundled

    def forward(self, waveform):
        # FCPE's forward returns NaN (0/0) from the threshold mask on a fully unvoiced
        # frame. rmvpe.onnx makes unvoiced 0, so collapse it to 0 inside the graph to
        # match that contract and keep NaN from leaking into RVC's NSF.
        f0 = self.bundled(waveform, SR, DECODER, THRESHOLD)
        return torch.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)


# The waveform lengths to verify. They include a non-multiple of the hop (12345) and the
# minimum length the baked-in reflect-pad requires (FLOOR). The graph is traced at
# N=16000, but N is variable through dynamic_axes. Actually pushing several lengths
# through here confirms every time that the trace did not bake N in (i.e. it generalizes).
# The minimum sample count the reflect-pad(432) requires (the same value as the runtime's
# FCPE_MIN_SAMPLES)
FLOOR = 433
VERIFY_LENGTHS = (16000, 24000, 12345, 8000, FLOOR)
ABS_TOL_HZ = 1.0  # absolute difference over all frames, unvoiced included (Hz)


def _tone(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    x = 0.6 * np.sin(2 * np.pi * 220.0 * t)  # 220Hz -> voiced
    x += 0.02 * np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return x[None, :]


def _voicing_signal() -> np.ndarray:
    # 220Hz for the first half, silence for the second. This exercises the threshold
    # voicing branch (unvoiced frame = 0) and lets an unmasked comparison verify that the
    # onnx does not fabricate pitch over the unvoiced span.
    n = 16000
    t = np.arange(n, dtype=np.float32) / SR
    x = (0.6 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    x[n // 2 :] = 0.0
    return x[None, :]


def _f0(t: torch.Tensor) -> np.ndarray:
    return t.detach().squeeze(-1).squeeze(0).cpu().numpy()


def _max_rel(a: np.ndarray, b: np.ndarray, voiced: np.ndarray) -> float:
    if int(voiced.sum()) == 0:
        return 0.0
    return float(np.max(np.abs(a[voiced] - b[voiced]) / np.maximum(b[voiced], 1e-6)))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    m = min(len(a), len(b))
    # Collapse NaN to 0 before comparing. Taking the max with NaN still present makes
    # NaN>tol False and lets the difference slip through (unvoiced frames can be NaN).
    diff = np.abs(np.nan_to_num(a[:m]) - np.nan_to_num(b[:m]))
    return float(np.max(diff)) if diff.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export bundled FCPE to a waveform-input ONNX.",
        epilog=(
            "FCPE モデルは手動ダウンロード不要。このタスクは uv run --with で torchfcpe を\n"
            "入れ、bundled の FCPE 重みを自動取得して onnx を生成する。\n"
            "\n"
            "手順:\n"
            "  1. uv run poe export-fcpe-onnx --output ~/.config/vstreamer/fcpe.onnx\n"
            "  2. config の [rvc] に設定:\n"
            '       f0_extractor_type = "fcpe"\n'
            '       fcpe_model_file   = "~/.config/vstreamer/fcpe.onnx"\n'
            "\n"
            "FCPE は rmvpe より高速だが精度は落ちる (ADR-0049)。既定は rmvpe のまま。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="出力 fcpe.onnx のパス"
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="torch 参照 f0 を保存するディレクトリ (任意)",
    )
    args = parser.parse_args()

    # deferred import, overlay only (pyproject's ty override tolerates the unresolved
    # import)
    import torchfcpe
    from torchfcpe import mel_extractor as ME

    bundled = torchfcpe.spawn_bundled_infer_model(torch.device("cpu")).eval()
    wrap = FcpeWave(bundled).eval()

    waves: dict[object, np.ndarray] = {n: _tone(n) for n in VERIFY_LENGTHS}
    waves["voicing"] = _voicing_signal()

    # (1) Save the f0 of the original torch.stft path for each length
    with torch.no_grad():
        ref_orig = {k: _f0(wrap(torch.from_numpy(w))) for k, w in waves.items()}

    # Strip the weight_norm parametrization (it is unstable under export)
    import torch.nn.utils.parametrize as P

    for _n, mod in bundled.named_modules():
        if getattr(mod, "parametrizations", None):
            for pname in list(mod.parametrizations.keys()):
                P.remove_parametrizations(mod, pname, leave_parametrized=True)

    # (2) Swap the STFT for the conv1d-DFT and confirm it matches the original torch.stft
    # at every length
    ME.MelModule.__call__ = _patched_mel_call
    with torch.no_grad():
        ref_conv = {k: _f0(wrap(torch.from_numpy(w))) for k, w in waves.items()}

    for k in waves:
        m = min(len(ref_orig[k]), len(ref_conv[k]))
        voiced = ref_orig[k][:m] > 1.0
        rc = _max_rel(ref_conv[k][:m], ref_orig[k][:m], voiced)
        ac = _max_abs(ref_conv[k], ref_orig[k])
        if rc > REL_TOL or ac > ABS_TOL_HZ:
            raise SystemExit(
                f"conv-STFT != torch.stft (N={k}, max_rel={rc:.3g}, max_abs={ac:.3g})"
            )
    print(f"[conv-STFT vs torch.stft] OK over {list(waves)}")

    # The export traces at N=16000 -> verifies every length -> moves to --output only on
    # success. tmp is created in the same parent directory as --output (a move across a
    # different drive/mount raises WinError 17, so it goes in out.parent rather than
    # tempfile's TEMP).
    import onnxruntime as ort

    out = args.output.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".onnx", dir=str(out.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.onnx.export(
            wrap,
            (torch.from_numpy(waves[16000]),),
            str(tmp),
            input_names=["waveform"],
            output_names=["f0"],
            dynamic_axes={"waveform": {1: "N"}, "f0": {1: "T"}},
            opset_version=OPSET,
            dynamo=False,
        )
        sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        for k, w in waves.items():
            got_raw = cast(np.ndarray, sess.run(None, {"waveform": w})[0])
            got = np.atleast_1d(got_raw.squeeze(-1).squeeze(0))
            # The export itself guarantees that the in-graph nan_to_num is effective
            # (_max_abs applies nan_to_num to both sides and would miss a NaN difference,
            # so reject it directly here).
            if bool(np.isnan(got).any()):
                raise SystemExit(f"onnx が NaN を出力しました (N={k})")
            if len(got) != len(ref_conv[k]):
                raise SystemExit(
                    f"onnx frame 数が torch と不一致 (N={k}: {len(got)} != {len(ref_conv[k])})"
                )
            m = min(len(got), len(ref_conv[k]))
            voiced = ref_conv[k][:m] > 1.0
            rel = _max_rel(got[:m], ref_conv[k][:m], voiced)
            # Unmasked: also checks that the onnx does not fabricate pitch on unvoiced
            # frames
            ab = _max_abs(got, ref_conv[k])
            if rel > REL_TOL or ab > ABS_TOL_HZ:
                raise SystemExit(
                    f"onnx != torch (N={k}, max_rel={rel:.3g}, max_abs={ab:.3g})"
                )
        tone = ref_conv[16000]
        med = float(np.median(tone[tone > 1.0]))
        print(f"[onnx vs conv-torch] OK over {list(waves)} median_f0={med:.1f}Hz")
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    print(f"OK -> {out}")

    if args.golden is not None:
        args.golden.expanduser().mkdir(parents=True, exist_ok=True)
        np.savez(args.golden.expanduser() / "fcpe_golden.npz", f0=ref_conv[16000])
        print(f"golden f0 saved -> {args.golden}/fcpe_golden.npz")


if __name__ == "__main__":
    main()
