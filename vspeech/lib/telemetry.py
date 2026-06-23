import json
import os
from contextlib import contextmanager
from datetime import datetime
from math import ceil
from math import floor
from pathlib import Path
from time import perf_counter
from time import time

from vspeech.logger import logger


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (q / 100.0) * (len(sorted_vals) - 1)
    lo = floor(rank)
    hi = ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "count": float(len(s)),
        "p50": _percentile(s, 50.0),
        "p95": _percentile(s, 95.0),
        "max": s[-1],
        "mean": sum(s) / len(s),
    }


class Telemetry:
    def __init__(self) -> None:
        self.enabled: bool = False
        self.max_samples: int = 5000
        self._durations: dict[str, list[float]] = {}
        self._e2e: list[float] = []
        self._jsonl = None

    def configure(
        self, enabled: bool, max_samples: int, jsonl_path: str = ""
    ) -> None:
        self.enabled = enabled
        self.max_samples = max_samples
        self._close_jsonl()
        if enabled and jsonl_path:
            try:
                resolved = datetime.now().strftime(jsonl_path.replace("%%", "%"))
                path = Path(resolved)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._jsonl = open(path, "a", encoding="utf-8")
            except OSError as e:
                logger.warning(
                    "telemetry jsonl disabled (cannot open %s): %s", jsonl_path, e
                )
                self._jsonl = None

    def _close_jsonl(self) -> None:
        if self._jsonl is not None:
            try:
                self._jsonl.close()
            except OSError:
                pass
            self._jsonl = None

    def reset(self) -> None:
        self._durations = {}
        self._e2e = []
        self._close_jsonl()

    def _append(self, buf: list[float], seconds: float) -> None:
        buf.append(seconds)
        if len(buf) > self.max_samples:
            del buf[0]

    def record(self, stage: str, seconds: float, trace_id: str = "") -> None:
        if not self.enabled:
            return
        self._append(self._durations.setdefault(stage, []), seconds)
        self._emit_jsonl(stage, seconds, trace_id)

    def record_e2e(self, seconds: float, trace_id: str = "") -> None:
        if not self.enabled:
            return
        self._append(self._e2e, seconds)
        self._emit_jsonl("e2e", seconds, trace_id)

    @contextmanager
    def timer(self, stage: str, trace_id: str = ""):
        if not self.enabled:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            self.record(stage, perf_counter() - start, trace_id)

    def _emit_jsonl(self, stage: str, seconds: float, trace_id: str) -> None:
        if self._jsonl is None:
            return
        record = {
            "ts": time(),
            "trace_id": trace_id,
            "stage": stage,
            "dur_s": seconds,
            "pid": os.getpid(),
        }
        try:
            self._jsonl.write(json.dumps(record) + "\n")
            self._jsonl.flush()
        except OSError as e:
            logger.warning("telemetry jsonl write failed, disabling: %s", e)
            self._close_jsonl()

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage, samples in self._durations.items():
            if samples:
                out[stage] = _stats(samples)
        if self._e2e:
            out["e2e"] = _stats(self._e2e)
        return out

    def log_summary(self) -> None:
        s = self.summary()
        if not s:
            return
        logger.info("=== telemetry summary (seconds) ===")
        for stage, m in s.items():
            logger.info(
                "%-14s n=%-5d p50=%.3f p95=%.3f max=%.3f mean=%.3f",
                stage,
                int(m["count"]),
                m["p50"],
                m["p95"],
                m["max"],
                m["mean"],
            )


telemetry = Telemetry()
