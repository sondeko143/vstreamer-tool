# Startup-profile skill — design

Date: 2026-06-25
Status: implemented (branch `feat/startup-profiling`). This doc records the
as-built design; an "As-built findings" section at the end notes where the
implementation diverged from the original plan.

## Problem

Launching the pipeline (`python -m vspeech --config …`) sometimes stalls for
several seconds before the first worker logs that it started. Investigation
(systematic debugging) traced this to **synchronous credential lookups** inside
`google-auth` at worker startup:

- `sender` called `get_id_token_credentials()` before logging "sender worker
  started"; with no GCP credentials configured it constructed
  `CeIdTokenCredentials(...)`, which synchronously hit `metadata.google.internal`.
  On a non-GCE host that blocks ~3–15 s (`getaddrinfo` failure × `retry_count=5`
  + backoff; measured 14.6 s) before raising `TransportError` (caught → `None`).
  Being synchronous in the coroutine, it stalled the whole event loop. **Fixed**
  in the same branch by gating on `config.use_ce_credentials` (separate commit).
- `translation_worker_google` calls `get_credentials()` at worker startup
  (translation.py:76, before "translation worker [google] started"); with empty
  `[gcp]` creds it falls to `google_auth_default()`. The real config
  (`config_transc_tts.toml`) has `translation.enable = true` with empty
  `[gcp.service_account_info]`, so this path is live. On *this* host
  `google_auth_default()` does **not** hit the metadata server — it finds the
  Google Cloud SDK and shells out to a `gcloud config get project` subprocess
  (~2 s). The profiler surfaced this; see As-built findings.

Rather than hunt these one at a time, we want a **repeatable way to capture and
analyze where startup time goes** — especially time lost to blocking waits
(DNS/socket/TLS, SMB, subprocess) versus imports versus model loading.

## Goal

An on-demand **project skill** that:
1. Launches `vspeech` under a sampling profiler with a minimal config and
   captures an analyzable flamegraph (speedscope JSON).
2. Analyzes that flamegraph to rank startup hotspots and classify them
   (blocking-io / import / compute / idle), so the actionable startup-latency
   causes surface immediately and event-loop idle is excluded from the headline.

Observation only — the skill never edits `vspeech` source (same philosophy as
`code-metrics`).

## Approach (as built)

- **Profiler: py-spy, app unmodified.** External sampling profiler run via
  `uvx py-spy` (0.4.2). py-spy surfaces blocking calls as the Python frames that
  called them and emits speedscope JSON, both human-viewable (speedscope.app) and
  machine-analyzable. No app instrumentation, matching the "never edit source"
  rule. (Rejected: viztracer + an in-app exit hook — touches app code; cProfile
  — weak for async/blocking in flamegraph form; pyinstrument — in-process and
  works, but emits the `evented` speedscope variant the analyzer doesn't parse.)
  - py-spy is driven with **`--subprocesses`** and **`--idle`** in **spawn** mode
    (`py-spy record … -- <python> -m vspeech …`). The reasons are environmental
    and are detailed in As-built findings.
- **Layout: project skill (current convention).** Executable code lives in the
  project's `scripts/`, wired as `[tool.poe.tasks]`; the skill (SKILL.md +
  references) lives in `.claude/skills/<name>/`, checked into the repo. Mirrors
  the `code-metrics` skill (`scripts/metrics.py` + `.claude/skills/code-metrics/`).

## Components

### Layout
```text
scripts/
  startup_capture.py              # the "task": py-spy spawn run (py-spy reaps the tree)
  startup_analyze.py              # speedscope analyzer (stdlib only, unit-tested)
  fixtures/minimal_startup.toml   # bundled minimal config
  tests/test_startup_analyze.py   # analyzer unit tests (synthetic speedscope)
  tests/test_startup_capture.py   # capture helper unit tests (command building)
.claude/skills/startup-profile/
  SKILL.md
  references/speedscope-schema.md
pyproject.toml [tool.poe.tasks]:
  startup-profile = { cmd = "python scripts/startup_capture.py" }
  startup-analyze = { cmd = "python scripts/startup_analyze.py" }
.gitignore: + .startup-profiles/  # capture artifacts (profile JSON + log)
```

### 1. `fixtures/minimal_startup.toml`
Strips machine-specific noise while still exercising the suspect startup paths:
- `transcription.enable = false` — excludes the WHISPER GPU model load, which
  would dominate startup and needs CUDA/extras.
- `translation.enable = true` with empty `[gcp]` creds — runs the
  `get_credentials` → `google_auth_default()` credential lookup at worker start
  (confirmed translation.py:76). `google.cloud.translate` is importable in the
  base venv (verified), so no extras needed.
- `telemetry.enable = false`, `log_file = ""` — removes the SMB log/telemetry
  shares; a throwaway `listen_port`. Other workers default to `enable = false`.

`startup_capture.py --config <path>` profiles any config (e.g. the real one) too;
the bundled file is just the default.

### 2. `scripts/startup_capture.py` (the "task")
- Resolves the venv python (`.venv/Scripts/python.exe` on win32) and config
  (default: the bundled fixture).
- Runs, in **spawn** mode, `uvx py-spy record -f speedscope -o <out> --rate R
  --duration N --subprocesses --idle -- <venv-python> -m vspeech --config <cfg>`,
  redirecting py-spy + vspeech output to `<out>.log` (so the app's own
  "… worker started" lines can be cross-referenced) and printing the tail after.
- **No process management:** py-spy spawns and reaps the child tree itself; after
  `--duration` (default 30 s) it writes the speedscope file and exits, leaving no
  orphan gRPC server (verified — port free, zero leftover python processes). This
  is simpler than the originally-planned `--pid` attach + explicit tree-kill,
  which was abandoned (see As-built findings).
- Writes artifacts to `./.startup-profiles/` (gitignored) and prints the analyze
  command + total wall time.
- Helper functions (`venv_python`, `build_pyspy_cmd`, `build_vspeech_cmd`,
  `default_config`) are pure and unit-tested.

### 3. `scripts/startup_analyze.py` (analyzer, stdlib only)
- Parses the `sampled` speedscope JSON (`shared.frames`, `profiles[].samples` +
  `weights`), aggregates per-frame **self** and **inclusive** time across all
  profiles/threads.
- Classifies each frame (precedence import → idle → blocking-io → compute →
  other):
  - **blocking-io** — `getaddrinfo`/`create_connection`/`do_handshake`,
    `google_auth_default`/`get_service_account_info`/`_metadata`, `subprocess.py`
    `communicate`/`_execute_child`/`wait`, `\\host\share` UNC paths, `google/auth`
    files → the actionable startup-latency causes.
  - **import** — `<module>`/importlib machinery, heavy deps (torch, grpc, …).
  - **compute** — model loading (torch/onnxruntime/ctranslate2/…).
  - **idle** — asyncio event-loop wait (selectors/proactor) and parked threads
    (`threading.py` run/lock waits).
- Reports **active = total − idle** as the headline; per-bucket split as % of
  active (idle as % of total); top-N self-time frames with `file:line` (idle
  excluded). `--json` for machine-readable output. The same speedscope JSON loads
  into speedscope.app for visual inspection.

### 4. `.claude/skills/startup-profile/SKILL.md` + references
- Procedure: confirm uv project → `uv run poe startup-profile` (capture) →
  `uv run poe startup-analyze --input <file>` (analyze; **no `--` separator** —
  poe forwards it literally) → interpret buckets.
- `references/speedscope-schema.md`: the sampled-speedscope shape + the
  bucket signature catalog the analyzer uses.

### Testing
`scripts/tests/test_startup_analyze.py` feeds synthetic speedscope JSON and
asserts self/inclusive time, bucket classification (incl. the idle and
subprocess-wait rules), active-time reporting, and JSON output.
`scripts/tests/test_startup_capture.py` covers the pure command-building helpers.
Capture-vs-vspeech integration was verified by hand end-to-end. (Same split as
`code-metrics`: the analyzer is the unit-tested surface.)

## As-built findings

- **py-spy vs uv trampoline (the big one).** py-spy first failed for *every*
  python with `Failed to find python version from target process` — including a
  trivial `time.sleep`. Root cause (not permissions/EDR, not a py-spy version):
  a uv `.venv\Scripts\python.exe` is a **trampoline** (270 KB, not a CPython
  copy) whose PE version resource isn't Python's, so py-spy can't identify it. It
  launches the real base interpreter (`sys._base_executable`) as a child, so
  **`--subprocesses`** lets py-spy follow into it. (`RUST_LOG=debug` + testing the
  base `C:\Program Files\Python311\python.exe` directly confirmed this.) This is
  why the `--pid` attach plan was dropped for spawn mode.
- **`--idle` is required.** Python releases the GIL during blocking I/O, so
  default on-CPU sampling misses DNS/socket/subprocess waits — the exact stalls
  this skill hunts. `--idle` captures them; the analyzer's `idle` bucket then
  removes the resulting event-loop / parked-thread noise (e.g. grpc pollers).
- **Classification rules came from real captures:** parked threads
  (`threading.py` run/lock) → `idle`, and subprocess waits (`subprocess.py`
  `communicate`/`_execute_child`) → `blocking-io`. A real run showed the actual
  startup blocker on this host is the `gcloud` credential subprocess (~1.9 s in
  `subprocess.communicate`), not a metadata HTTP probe.
- **poe arg passing:** `poe <task> -- <args>` forwards the `--` literally to the
  script; pass flags directly (`uv run poe startup-analyze --input <file>`).

## Out of scope (YAGNI)
- Continuous monitoring, regression detection, CI gating — this is on-demand
  investigation tooling.
- In-app profiling instrumentation — py-spy / app-unmodified is decided.
- Remediating the `translation` credential lookup itself — this skill only
  surfaces it; the fix is a follow-up decision.
