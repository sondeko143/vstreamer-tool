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
import os
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
    # cos/sin カーネルは長さ n_fft。torch.stft は win_length<n_fft を中央ゼロ埋めするが
    # ここでは等長を前提にする (bundled 既定 1024==1024)。違えば loud に失敗させる。
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
    """波形 (1, N) を受けて f0 (1, T, 1) Hz を返す export 用ラッパ。"""

    def __init__(self, bundled):
        super().__init__()
        self.bundled = bundled

    def forward(self, waveform):
        # FCPE の forward は完全無声フレームで threshold マスク由来の NaN (0/0) を返す。
        # rmvpe.onnx は無声を 0 にするので、契約を揃え NaN が RVC の NSF に漏れないよう
        # graph 内で 0 に潰す。
        f0 = self.bundled(waveform, SR, DECODER, THRESHOLD)
        return torch.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)


# 検証する波形長。非 hop 倍数 (12345) と、焼き込み reflect-pad が要求する最小長 (FLOOR) を
# 含める。グラフは N=16000 でトレースするが dynamic_axes で N は可変。ここで実際に複数長を
# 通し、トレースが N を焼き込んでいない (= 一般化する) ことを毎回確認する。
FLOOR = 433  # reflect-pad(432) が要求する最小サンプル数 (runtime の FCPE_MIN_SAMPLES と同値)
VERIFY_LENGTHS = (16000, 24000, 12345, 8000, FLOOR)
ABS_TOL_HZ = 1.0  # 無声フレームを含む全フレームの絶対差 (Hz)


def _tone(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    x = 0.6 * np.sin(2 * np.pi * 220.0 * t)  # 220Hz -> voiced
    x += 0.02 * np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return x[None, :]


def _voicing_signal() -> np.ndarray:
    # 前半 220Hz / 後半 無音。threshold voicing 分岐 (無声フレーム=0) を通し、
    # onnx が無声区間に pitch を捏造しないことを非マスク比較で検証するため。
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
    # NaN を 0 に潰してから比較する。NaN のまま max を取ると NaN>tol が False になり
    # 差分を見逃す (無声フレームは NaN になりうる)。
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

    import torchfcpe  # overlay 専用の遅延 import (pyproject の ty override で未解決 import を許容)
    from torchfcpe import mel_extractor as ME

    bundled = torchfcpe.spawn_bundled_infer_model(torch.device("cpu")).eval()
    wrap = FcpeWave(bundled).eval()

    waves: dict[object, np.ndarray] = {n: _tone(n) for n in VERIFY_LENGTHS}
    waves["voicing"] = _voicing_signal()

    # (1) 元 torch.stft 経路の f0 を各長で保存
    with torch.no_grad():
        ref_orig = {k: _f0(wrap(torch.from_numpy(w))) for k, w in waves.items()}

    # weight_norm パラメトリゼーションを剥がす (export で不安定)
    import torch.nn.utils.parametrize as P

    for _n, mod in bundled.named_modules():
        if getattr(mod, "parametrizations", None):
            for pname in list(mod.parametrizations.keys()):
                P.remove_parametrizations(mod, pname, leave_parametrized=True)

    # (2) STFT を conv1d-DFT に差し替え、各長で元 torch.stft と一致することを確認
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

    # export は N=16000 でトレース -> 全長を検証 -> 成功時のみ --output へ move。
    # tmp は --output と同じ親ディレクトリに作る (別ドライブ/mount への move は
    # WinError 17 になるため tempfile の TEMP ではなく out.parent に置く)。
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
            # graph 内の nan_to_num が効いていることを export 自身が保証する
            # (_max_abs は両辺 nan_to_num するので NaN 差分を見逃す。ここで直接弾く)。
            if bool(np.isnan(got).any()):
                raise SystemExit(f"onnx が NaN を出力しました (N={k})")
            if len(got) != len(ref_conv[k]):
                raise SystemExit(
                    f"onnx frame 数が torch と不一致 (N={k}: {len(got)} != {len(ref_conv[k])})"
                )
            m = min(len(got), len(ref_conv[k]))
            voiced = ref_conv[k][:m] > 1.0
            rel = _max_rel(got[:m], ref_conv[k][:m], voiced)
            # 非マスク: onnx が無声フレームに pitch を捏造しないことも見る
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
