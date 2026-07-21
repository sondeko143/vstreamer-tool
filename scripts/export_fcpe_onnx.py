"""torchfcpe の bundled FCPE を「波形入力」ONNX へ export する。

**一度きり**のオフライン処理。runtime には含めない。依存 (torchfcpe / onnx /
onnxscript) は poe task の `uv run --with` が供給する (pyproject/uv.lock には載せない)。

    uv run poe export-fcpe-onnx --output ~/.config/vstreamer/fcpe.onnx

`python scripts/export_fcpe_onnx.py` ではなく **`python -m scripts.export_fcpe_onnx`** で
起動すること。

出力 fcpe.onnx の契約:
  入力  waveform  (1, N)  float32  16kHz mono
  出力  f0        (1, T, 1)  Hz     無声フレームは 0
  threshold / sample_rate(16000) / decoder_mode("local_argmax") は export 時に焼き込む
  (runtime では可変化しない = ADR-0049 の非ゴール)。

なぜ特殊な export が要るか (ADR-0049 / スパイクの知見):
  * 新 dynamo exporter は wav2mel 内のデータ依存分岐 (wav.min()<-1 を .item()) で不可。
  * legacy tracer (dynamo=False) も torch.stft(return_complex=True) の複素型で不可。
  * → MelModule の STFT を conv1d(cos/sin DFT 基底 + hann 窓, center=False)で厳密再現し
    (元 torch.stft と f0 max_rel ~1e-6)、legacy tracer + opset17 で export する。
  * output_proj の weight_norm パラメトリゼーションは export 前に剥がす。

このスクリプト自身が (1) conv-STFT が元 torch.stft と一致 (2) onnx が torch と一致 を
アサートし、通らなければ資産を書き出さない。
"""

import argparse
import io
import math
import sys
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as Fnn

# torch.onnx の verbose 出力や警告に非 ASCII (絵文字) が混じり、Windows の cp1252
# stdout でクラッシュする。UTF-8 に固定する (プロジェクト頻出の encoding 対策)。
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

THRESHOLD = 0.006
SR = 16000
DECODER = "local_argmax"
OPSET = 17
# 検証許容 (スパイク実測は ~1e-6; 十分な余裕を持たせる)
REL_TOL = 1e-3


def _patched_mel_call(
    self, y, key_shift=0, speed=1, center=False, no_cache_window=False
):
    """MelModule.__call__ の inference 経路 (center=False, key_shift=0) を conv1d-DFT で置換。

    torch.stft(return_complex=True) は ONNX に載らないので、hann 窓込みの cos/sin カーネルで
    magnitude スペクトルを厳密再現する。win_size==n_fft の前提 (bundled 既定 1024==1024)。
    """
    n_fft = self.n_fft
    win_size = self.win_size
    hop = self.hop_length
    clip_val = self.clip_val
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
    """波形 (1, N) を受けて f0 (1, T, 1) Hz を返す export 用ラッパ。"""

    def __init__(self, bundled):
        super().__init__()
        self.bundled = bundled

    def forward(self, waveform):
        return self.bundled(waveform, SR, DECODER, THRESHOLD)


def _test_wave() -> np.ndarray:
    t = np.arange(SR, dtype=np.float32) / SR
    x = 0.6 * np.sin(2 * np.pi * 220.0 * t)  # 220Hz -> voiced
    x += 0.02 * np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    return x[None, :]


def _f0(t: torch.Tensor) -> np.ndarray:
    return t.detach().squeeze(-1).squeeze(0).cpu().numpy()


def _max_rel(a: np.ndarray, b: np.ndarray, voiced: np.ndarray) -> float:
    return float(np.max(np.abs(a[voiced] - b[voiced]) / np.maximum(b[voiced], 1e-6)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export bundled FCPE to a waveform-input ONNX."
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

    import torchfcpe  # overlay 専用の遅延 import (pyproject の ty override で未解決 import を許容)
    from torchfcpe import mel_extractor as ME

    bundled = torchfcpe.spawn_bundled_infer_model(torch.device("cpu")).eval()
    wrap = FcpeWave(bundled).eval()
    wav = torch.from_numpy(_test_wave())

    # (1) 元 torch.stft 経路の f0 を保存
    with torch.no_grad():
        ref_orig = _f0(wrap(wav))

    # weight_norm パラメトリゼーションを剥がす (export で不安定)
    import torch.nn.utils.parametrize as P

    for _n, mod in bundled.named_modules():
        if getattr(mod, "parametrizations", None):
            for pname in list(mod.parametrizations.keys()):
                P.remove_parametrizations(mod, pname, leave_parametrized=True)

    # (2) STFT を conv1d-DFT に差し替え
    ME.MelModule.__call__ = _patched_mel_call
    with torch.no_grad():
        ref_conv = _f0(wrap(wav))

    m = min(len(ref_orig), len(ref_conv))
    voiced = ref_orig[:m] > 1.0
    rel_conv = _max_rel(ref_conv[:m], ref_orig[:m], voiced)
    print(
        f"[conv-STFT vs torch.stft] voiced={int(voiced.sum())} max_rel={rel_conv:.3g}"
    )
    if rel_conv > REL_TOL:
        raise SystemExit(
            f"conv-STFT が元 torch.stft と一致しません (max_rel={rel_conv:.3g} > {REL_TOL})"
        )

    # export -> 一時ファイル -> 検証 -> 成功時のみ --output へ移動
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "fcpe.onnx"
        torch.onnx.export(
            wrap,
            (wav,),
            str(tmp),
            input_names=["waveform"],
            output_names=["f0"],
            dynamic_axes={"waveform": {1: "N"}, "f0": {1: "T"}},
            opset_version=OPSET,
            dynamo=False,
        )

        import onnxruntime as ort

        sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        got_raw = cast(np.ndarray, sess.run(None, {"waveform": _test_wave()})[0])
        got = got_raw.squeeze(-1).squeeze(0)
        m2 = min(len(got), len(ref_conv))
        v2 = ref_conv[:m2] > 1.0
        rel_onnx = _max_rel(got[:m2], ref_conv[:m2], v2)
        print(
            f"[onnx vs conv-torch] max_rel={rel_onnx:.3g} median_f0={np.median(ref_conv[:m2][v2]):.1f}Hz"
        )
        if rel_onnx > REL_TOL:
            raise SystemExit(
                f"onnx が torch と一致しません (max_rel={rel_onnx:.3g} > {REL_TOL})"
            )

        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(tmp).replace(args.output.expanduser())

    print(f"OK -> {args.output}")

    if args.golden is not None:
        args.golden.expanduser().mkdir(parents=True, exist_ok=True)
        np.savez(args.golden.expanduser() / "fcpe_golden.npz", f0=ref_conv)
        print(f"golden f0 saved -> {args.golden}/fcpe_golden.npz")


if __name__ == "__main__":
    main()
