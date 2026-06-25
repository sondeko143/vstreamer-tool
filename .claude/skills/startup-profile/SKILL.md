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

**Default to baseline.** For any "startup is slow / where does launch time go?"
request — or to profile a single real config — run `startup-profile` (baseline)
and stop there. It is fast, environment-independent, and measures the
import/infra floor every config pays. Only run `startup-sweep` when the user
**explicitly** asks to break the cost down per worker ("which worker is
heaviest?", "compare the workers", "profile each worker") — it costs ~7× the
window and is environment-specific. When unsure, do baseline and offer the sweep.

1. Confirm the working directory is a uv project (`pyproject.toml` exists). If
   not, say so and stop.
2. **Capture + analyze** (the orchestrator lives at `scripts/startup_capture.py`,
   wired as `[tool.poe.tasks].startup-profile`). Capture now **auto-analyzes
   inline** — one command both records and ranks the buckets:
   `uv run poe startup-profile`
   - **Default = baseline.** With no `--config` this profiles the bundled
     baseline (`scripts/fixtures/baseline_startup.toml`): every optional worker
     disabled, only the always-on gRPC `sender`/`receiver` run. That is the pure
     import/infra floor every configuration pays.
   - Profile a real config: `uv run poe startup-profile --config <path>`.
   - Tune the window: `... --duration 20` (default 30s). `--native` adds native
     frames; `--rate` changes sample Hz; `--top` sets frames listed.
   - Writes `<stamp>-<label>.speedscope.json` + `.log` to `./.startup-profiles/`
     (gitignored).
3. **Sweep** (opt-in — run *only* when asked to break startup cost down per
   worker; wired as `[tool.poe.tasks].startup-sweep`):
   `uv run poe startup-sweep`
   - Profiles the baseline **plus one single-worker config per optional worker**
     (`scripts/fixtures/sweep/`: recording, transcription, subtitle, translation,
     tts, playback), auto-analyzing each, then prints a **comparison table** of
     active time by config (sorted heaviest first). Subtract the baseline row to
     isolate each worker's own startup cost.
   - `vc` is **excluded**: its torch + CUDA model load dwarfs the window and
     needs real RVC model assets (`model_file`/`hubert_model_file`/
     `rmvpe_model_file`) to start at all. Measure it ad hoc with `--config`.
   - The window applies *per config*, so a sweep takes ~7 × `--duration`. Use a
     short window (e.g. `... --duration 12`) for a quick comparison.
   - Sweep fixtures are environment-specific by nature: recording/playback need
     audio devices, transcription(GCP)/translation need google credentials (and
     show the `gcloud` credential-subprocess stall), tts(VR2) needs VOICEROID2.
     A config whose assets are absent exits early — still a valid data point
     (its capture window is short and the analysis shows how far it got).
4. **Re-analyze** an existing profile if needed (`scripts/startup_analyze.py`,
   wired as `startup-analyze`):
   `uv run poe startup-analyze --input .startup-profiles/<stamp>-<label>.speedscope.json`
   - `--json` for machine-readable output; `--top 30` for more rows.
5. **Interpret the buckets** (percentages are of *active* time):
   - **blocking-io** → the actionable stalls: DNS/socket/TLS, `\\host\share`
     (SMB/UNC) paths, or waiting on a credential subprocess. Remove these first.
   - **import** → module-load cost (cold disk cache / AV scanning / heavy deps
     like grpc, torch, faster_whisper).
   - **compute** → model loading.
   - **idle** → event-loop / parked-thread waiting; reported separately and
     excluded from the active headline.
6. Present the top frames with `file:line`, bucket, and self-time, plus the
   bucket split (and the sweep comparison table, if run). This is input for the
   user's fix decision — do **not** edit code.

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
- The **baseline** config disables every optional worker and the telemetry/log
  network shares, so it isolates the import/infra floor with no devices,
  credentials, or models. The `startup-sweep` fixtures each turn on exactly one
  worker; the `translation` (and GCP `transcription`) fixtures are what exercise
  the google-auth `gcloud` credential subprocess. Use `--config` for a real one.
- The speedscope JSON also opens at https://speedscope.app for a visual
  flamegraph. See `references/speedscope-schema.md` for the format and the
  bucket signature catalog.

## Rules

- Advisory only. Never gates a launch and never edits source or config — it
  reports where startup time goes.
- A missing `uvx`/py-spy is a SKIP with a message, not a failure.
- py-spy reads only the `sampled` speedscope format. pyinstrument emits
  `evented` speedscope, which this analyzer does not parse.
