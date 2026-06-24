# Startup-profile skill — design

Date: 2026-06-25
Status: approved (design), pending implementation

## Problem

Launching the pipeline (`python -m vspeech --config …`) sometimes stalls for
several seconds before the first worker logs that it started. Investigation
(systematic debugging) traced this to **synchronous GCE metadata-server probes**
inside `google-auth`:

- `sender` called `get_id_token_credentials()` before logging "sender worker
  started"; with no GCP credentials configured it constructed
  `CeIdTokenCredentials(...)`, which synchronously hit
  `metadata.google.internal`. On a non-GCE host that blocks ~3–15 s
  (`getaddrinfo` failure × `retry_count=5` + backoff; measured 14.6 s) before
  raising `TransportError` (caught → `None`). Being synchronous in the coroutine,
  it stalled the whole event loop. **Already fixed** by gating on
  `config.use_ce_credentials` (separate change).
- `translation_worker_google` calls `get_credentials()` at worker startup
  (translation.py:76, before "translation worker [google] started"); with empty
  `[gcp]` creds it falls to `google_auth_default()`, which performs the same
  metadata probe. The real config (`config_transc_tts.toml`) has
  `translation.enable = true` with empty `[gcp.service_account_info]`, so this
  path is live.

Rather than hunt these one at a time, we want a **repeatable way to capture and
analyze where startup time goes** — especially time lost to blocking I/O
(DNS/network/SMB) versus imports versus model loading.

## Goal

An on-demand **project skill** that:
1. Launches `vspeech` under a sampling profiler with a minimal config and
   captures an analyzable flamegraph (speedscope JSON).
2. Analyzes that flamegraph to rank startup hotspots and classify them
   (blocking-I/O / import / compute), so the actionable startup-latency causes
   (e.g. the metadata probe, SMB log share) surface immediately.

Observation only — the skill never edits `vspeech` source (same philosophy as
`code-metrics`).

## Approach (decided)

- **Profiler: py-spy, app unmodified.** External sampling profiler run via
  `uvx py-spy`. Verified working on this Windows host (py-spy 0.4.2). py-spy
  surfaces blocking C calls (`getaddrinfo`, `connect`, `ssl`) as the Python
  frames that called them — exactly what localizes the stall — and emits
  speedscope JSON, which is both human-viewable (speedscope.app) and
  machine-analyzable. No app instrumentation, matching the "never edit source"
  rule. (Rejected: viztracer + an in-app exit hook — precise but touches app
  code; cProfile — weak for async + blocking I/O in flamegraph form.)
- **Layout: project skill (new convention).** The old
  `tools/<name>/` + `~/.claude/skills/` install + setup scripts pattern is
  retired. Executable code lives in the project's `scripts/`, wired as
  `[tool.poe.tasks]`; the skill (SKILL.md + references) lives in
  `.claude/skills/<name>/`, checked into the repo. Mirrors the current
  `code-metrics` skill (`scripts/metrics.py` + `.claude/skills/code-metrics/`).

## Components

### Layout
```
scripts/
  startup_capture.py              # the "task": py-spy run + child-tree kill
  startup_analyze.py              # speedscope analyzer (stdlib only, unit-tested)
  fixtures/minimal_startup.toml   # bundled minimal config
  tests/test_startup_analyze.py   # analyzer unit test (synthetic speedscope)
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
- `translation.enable = true` with empty `[gcp]` creds — reproduces the
  `get_credentials` → `google_auth_default()` metadata probe at startup
  (confirmed translation.py:76 runs it at worker start). `google.cloud.translate`
  is importable in the base venv (verified), so no extras needed.
- `tts/playback/recording/vc = false`, `telemetry.enable = false`,
  `log_file = ""` — removes audio/extras and the SMB log/telemetry shares.
- A throwaway `listen_port`.

`startup_capture.py --config <path>` profiles any config (e.g. the real one) too;
the bundled file is just the default.

### 2. `scripts/startup_capture.py` (the "task")
- Resolves the project venv python (`.venv/Scripts/python.exe` on win32,
  `.venv/bin/python` on posix) and config (default: the bundled fixture).
- Runs `uvx py-spy record -f speedscope -o <out> --subprocesses --
  <venv-python> -m vspeech --config <cfg>` so sampling starts at process
  creation (imports captured).
- Tees the child's stdout/stderr to `<out>.log` so the app's own
  "… worker started" lines and timestamps can be cross-referenced with frames.
- After `--duration` (default 30 s; ample with WHISPER disabled), guarantees the
  spawned **process tree** is terminated so no orphan gRPC server lingers.
  `capture.py` owns the lifecycle: it holds the py-spy child's PID and, because
  Windows reparents (does not auto-kill) grandchildren when py-spy exits, kills
  the whole tree explicitly — win32 `taskkill /PID <pyspy-pid> /T /F` issued
  while py-spy is still alive (or a Windows Job Object), posix process-group
  kill. As a fallback it locates the `vspeech` process by its marker (config
  path / `listen_port`) and kills it. stdlib only (no psutil dependency); the
  exact mechanism is settled in the implementation plan.
- Writes artifacts to `./.startup-profiles/` (gitignored). Prints the output
  paths and total wall time.

### 3. `scripts/startup_analyze.py` (analyzer, stdlib only)
- Parses speedscope JSON (`shared.frames`, `profiles[].samples` + `weights`),
  computes per-frame **self** and **inclusive** time.
- Classifies frames into three buckets by known signatures:
  - **blocking-I/O** — `getaddrinfo`, `socket`, `ssl`, `_metadata`,
    `google_auth_default`, UNC/SMB paths → the actionable startup-latency causes.
  - **import** — `<frozen importlib`, `torch`, `grpc`, `faster_whisper` →
    cold-cache / AV-scan cost.
  - **compute** — model loading, etc.
- Output: total startup wall-time, top-N self-time frames, per-bucket breakdown.
  `--json` for machine-readable output. SKILL.md notes the same JSON loads into
  speedscope.app for visual inspection.

### 4. `.claude/skills/startup-profile/SKILL.md` + references
- Procedure: confirm uv project → `uv run poe startup-profile` (capture) →
  `uv run poe startup-analyze` (analyze) → interpret buckets (blocking-I/O =
  fix targets like the metadata probe / SMB share; import = cold/AV; compute =
  model load).
- `references/speedscope-schema.md`: the speedscope JSON shape + the
  known-signature catalog the analyzer uses.

### Testing
`scripts/tests/test_startup_analyze.py` feeds a small synthetic speedscope JSON
and asserts ranking + bucket classification. (Capture is integration-shaped;
the analyzer is pure parsing and is the unit-tested surface — same split as
`code-metrics`.)

## Out of scope (YAGNI)
- Continuous monitoring, regression detection, CI gating — this is on-demand
  investigation tooling.
- In-app profiling instrumentation — py-spy / app-unmodified is decided.
- Fixing the `translation` metadata probe itself — this skill only surfaces it;
  remediation is a follow-up decision.
