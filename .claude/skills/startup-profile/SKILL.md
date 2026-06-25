---
name: startup-profile
description: Use when a uv-based Python app is slow to start — the first worker/log line takes several seconds, you want to see where launch time goes, or whether startup is blocked on imports, DNS/socket/TLS, an SMB share, or a credential subprocess (e.g. google-auth shelling out to gcloud). Captures a py-spy startup flamegraph and ranks the cost by bucket. Advisory: never edits source.
---

# Startup Profile

Capture a sampling profile of `python -m vspeech` startup and rank where the
time goes, so blocking stalls (DNS/socket/TLS, SMB shares, credential
subprocesses) stand out from import and model-load cost. Advisory only — this
never edits source.

The headline is **active time** (`total - idle`): event-loop and parked-thread
waiting is bucketed as `idle` and excluded so the slack at the tail of a fixed
sampling window doesn't dilute the real stalls.

## Procedure

1. Confirm the working directory is a uv project (`pyproject.toml` exists). If
   not, say so and stop.
2. **Capture** (the orchestrator lives at `scripts/startup_capture.py`, wired as
   `[tool.poe.tasks].startup-profile`):
   `uv run poe startup-profile`
   - Profiles the bundled minimal config (`scripts/fixtures/minimal_startup.toml`)
     by default. Profile a real config: `uv run poe startup-profile --config <path>`.
   - Tune the window: `... --duration 20` (default 30s). `--native` adds
     native frames; `--rate` changes sample Hz.
   - Writes `<stamp>.speedscope.json` + `<stamp>.log` to `./.startup-profiles/`
     (gitignored) and prints the analyze command.
3. **Analyze** (`scripts/startup_analyze.py`, wired as `startup-analyze`):
   `uv run poe startup-analyze --input .startup-profiles/<stamp>.speedscope.json`
   - `--json` for machine-readable output; `--top 30` for more rows.
4. **Interpret the buckets** (percentages are of *active* time):
   - **blocking-io** → the actionable stalls: DNS/socket/TLS, `\\host\share`
     (SMB/UNC) paths, or waiting on a credential subprocess. Remove these first.
   - **import** → module-load cost (cold disk cache / AV scanning / heavy deps
     like grpc, torch, faster_whisper).
   - **compute** → model loading.
   - **idle** → event-loop / parked-thread waiting; reported separately and
     excluded from the active headline.
5. Present the top frames with `file:line`, bucket, and self-time, plus the
   bucket split. This is input for the user's fix decision — do **not** edit code.

## Notes

- py-spy runs via `uvx` (no project dependency); the analyzer is pure stdlib.
- **Windows / uv gotcha:** a uv `.venv` ships a *trampoline* `python.exe`, not a
  CPython copy. py-spy can't identify it and fails with `Failed to find python
  version from target process`. The capture script passes `--subprocesses` so
  py-spy follows the trampoline into the real base interpreter it launches (and
  into credential helpers like `gcloud`). This is already baked in — don't drop
  it.
- `--idle` is on: Python releases the GIL during blocking I/O, so default on-CPU
  sampling would miss the very stalls this skill hunts. The analyzer's `idle`
  bucket then removes the resulting event-loop-wait noise.
- The minimal config disables transcription (WHISPER GPU load) and the
  telemetry/log network shares, but keeps `translation` enabled so the
  google-auth credential path runs at startup. Use `--config` for the real one.
- The speedscope JSON also opens at https://speedscope.app for a visual
  flamegraph. See `references/speedscope-schema.md` for the format and the
  bucket signature catalog.

## Rules

- Advisory only. Never gates a launch and never edits source or config — it
  reports where startup time goes.
- A missing `uvx`/py-spy is a SKIP with a message, not a failure.
- py-spy reads only the `sampled` speedscope format. pyinstrument emits
  `evented` speedscope, which this analyzer does not parse.
