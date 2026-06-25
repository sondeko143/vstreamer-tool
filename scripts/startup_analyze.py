"""Analyze a py-spy speedscope profile of vspeech startup.

Reads a speedscope JSON file (produced by ``startup_capture.py`` /
``py-spy record -f speedscope``), aggregates per-frame self/inclusive time,
and classifies frames into buckets so the actionable startup-latency causes
(blocking network/DNS/TLS I/O, SMB shares) stand out from import and compute
cost. Pure stdlib; advisory only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

# Importlib machinery + module-body execution -> "import" (cold-cache / AV cost).
IMPORT_NAMES = {
    "_find_and_load",
    "_find_and_load_unlocked",
    "_load_unlocked",
    "exec_module",
    "_call_with_frames_removed",
    "_bootstrap",
    "_handle_fromlist",
    "import_module",
    "<module>",
}
# Network/DNS/TLS calls + google-auth's metadata probe -> the actionable stalls.
BLOCKING_NAME_TOKENS = (
    "getaddrinfo",
    "create_connection",
    "do_handshake",
    "google_auth_default",
    "get_service_account_info",
    "_metadata",
)
# Heavy compute / model-loading libraries.
COMPUTE_TOKENS = (
    "torch",
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "numpy",
    "fairseq",
    "pyworld",
)
# asyncio event-loop waiting -> "idle". Excluded from the active-time headline so
# slack at the tail of a fixed sampling window does not dilute the real stalls.
IDLE_NAME_TOKENS = (
    "GetQueuedCompletionStatus",
    "epoll_wait",
    "kqueue_control",
    "_poll",
)
IDLE_FILE_TOKENS = (
    "selectors.py",
    "windows_events.py",
    "proactor_events.py",
    "selector_events.py",
)
# A background thread parked at its run/lock base in threading.py is waiting, not
# doing startup work (e.g. grpc's polling threads under --idle sampling).
THREADING_IDLE_NAMES = {
    "run",
    "_bootstrap",
    "_bootstrap_inner",
    "_wait_for_tstate_lock",
    "wait",
    "acquire",
}
# Waiting on / launching an external process (e.g. google-auth shelling out to
# `gcloud` for credentials) is an actionable startup stall, like network I/O.
SUBPROCESS_WAIT_NAMES = {"communicate", "_communicate", "_execute_child", "wait"}
BUCKETS = ("blocking-io", "import", "compute", "idle", "other")


@dataclass
class FrameStat:
    name: str
    file: str | None
    line: int | None
    self_weight: float
    inclusive_weight: float
    bucket: str


def classify_frame(name: str, file: str | None) -> str:
    n = name or ""
    f = file or ""
    fl = f.lower().replace("\\", "/")
    base = fl.rsplit("/", 1)[-1]
    # Import (module loading / module-body execution) takes precedence: importing
    # google.auth or torch should read as import cost, not blocking/compute.
    if n in IMPORT_NAMES or "importlib" in fl or f.startswith("<frozen importlib"):
        return "import"
    if any(tok in n for tok in IDLE_NAME_TOKENS) or any(
        tok in fl for tok in IDLE_FILE_TOKENS
    ):
        return "idle"
    if base == "threading.py" and n in THREADING_IDLE_NAMES:
        return "idle"
    if any(tok in n for tok in BLOCKING_NAME_TOKENS):
        return "blocking-io"
    if base == "subprocess.py" and n in SUBPROCESS_WAIT_NAMES:
        return "blocking-io"
    if f.startswith("\\\\") or fl.startswith("//"):  # SMB / UNC path
        return "blocking-io"
    if "google/auth" in fl:
        return "blocking-io"
    if any(tok in fl or tok in n for tok in COMPUTE_TOKENS):
        return "compute"
    return "other"


def compute_frame_stats(doc: dict) -> tuple[list[FrameStat], float, str]:
    frames = doc.get("shared", {}).get("frames", [])
    profiles = doc.get("profiles", [])
    unit = profiles[0].get("unit", "unknown") if profiles else "unknown"

    self_w = [0.0] * len(frames)
    incl_w = [0.0] * len(frames)
    total = 0.0
    for prof in profiles:
        samples = prof.get("samples") or []
        weights = prof.get("weights") or []
        for stack, w in zip(samples, weights):
            w = float(w)
            total += w
            if stack:
                self_w[stack[-1]] += w
            for idx in set(stack):  # inclusive: once per sample even if recursive
                incl_w[idx] += w

    stats = [
        FrameStat(
            name=str(fr.get("name", "?")),
            file=fr.get("file"),
            line=fr.get("line"),
            self_weight=self_w[i],
            inclusive_weight=incl_w[i],
            bucket=classify_frame(str(fr.get("name", "")), fr.get("file")),
        )
        for i, fr in enumerate(frames)
    ]
    return stats, total, unit


def bucket_totals(stats: list[FrameStat]) -> dict[str, float]:
    totals = {b: 0.0 for b in BUCKETS}
    for s in stats:
        totals[s.bucket] = totals.get(s.bucket, 0.0) + s.self_weight
    return totals


def rank_by_self(stats: list[FrameStat]) -> list[FrameStat]:
    return sorted(stats, key=lambda s: s.self_weight, reverse=True)


def _pct(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole else 0.0


def _loc(s: FrameStat) -> str:
    if not s.file:
        return s.name
    return f"{s.file}:{s.line}" if s.line is not None else s.file


def render_summary(stats: list[FrameStat], total: float, unit: str, top: int) -> str:
    totals = bucket_totals(stats)
    idle_w = totals["idle"]
    active = total - idle_w
    lines = [
        "",
        "=== startup-profile: where startup time goes ===",
        f"total sampled: {total:.3f} {unit}  (active {active:.3f}, idle {idle_w:.3f})",
        "",
        "by bucket (self time, % of active):",
    ]
    for b in BUCKETS:
        w = totals[b]
        ref = total if b == "idle" else active
        suffix = " of total" if b == "idle" else ""
        lines.append(f"  {b:<12} {w:>9.3f} {unit}  ({_pct(w, ref):5.1f}%{suffix})")

    lines.append("")
    lines.append(f"top {top} startup frames (self time, idle excluded):")
    lines.append(f"{'act%':>6} {'self':>9} {'bucket':<12} frame (file:line)")
    shown = [s for s in rank_by_self(stats) if s.self_weight > 0 and s.bucket != "idle"]
    for s in shown[:top]:
        lines.append(
            f"{_pct(s.self_weight, active):6.1f} {s.self_weight:9.3f} "
            f"{s.bucket:<12} {s.name} ({_loc(s)})"
        )

    lines.append("")
    blocking = totals["blocking-io"]
    if active > 0 and blocking > 0:
        lines.append(
            f"blocking-io is {_pct(blocking, active):.1f}% of active startup "
            f"(excl. {_pct(idle_w, total):.1f}% idle) - network/DNS/TLS, SMB, or "
            "subprocess waits are the first thing to remove."
        )
    return "\n".join(lines)


def stats_to_json(stats: list[FrameStat], total: float, unit: str) -> str:
    ranked = rank_by_self(stats)
    buckets = bucket_totals(stats)
    payload = {
        "total": total,
        "active": total - buckets["idle"],
        "unit": unit,
        "buckets": buckets,
        "frames": [
            {
                **asdict(s),
                "self_pct": _pct(s.self_weight, total),
                "inclusive_pct": _pct(s.inclusive_weight, total),
            }
            for s in ranked
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="startup-analyze")
    parser.add_argument(
        "--input", "-i", required=True, help="speedscope JSON from startup_capture"
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    doc = json.loads(Path(args.input).read_text(encoding="utf-8"))
    stats, total, unit = compute_frame_stats(doc)

    if args.json:
        print(stats_to_json(stats, total, unit))
    else:
        print(render_summary(stats, total, unit, args.top))
    return 0  # advisory: never gate


if __name__ == "__main__":
    raise SystemExit(main())
