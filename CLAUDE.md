# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VStreamer Tool is a real-time speech pipeline for live streaming. It chains recording → transcription → translation → subtitles → text-to-speech → voice changing → playback, integrating external services (AmiVoice ACP, Google Cloud Speech/Translate, Whisper, VOICEROID2, VOICEVOX, RVC). The package name is `vspeech` (the project is `voicerecog` in `pyproject.toml`); a separate `gui/` package provides a Tkinter control panel.

## Commands

This project uses **uv** (migrated from Poetry). Python **3.12 only** (`>=3.12,<3.13`).

```sh
# Install. Base = transcription/subtitle only; add extras for other subsystems.
uv sync
uv sync --extra audio       # recording/playback (needs portaudio)
uv sync --extra whisper     # faster-whisper (needs CUDA 11.8/cu128)
uv sync --extra vroid2      # VOICEROID2 TTS
uv sync --extra voicevox    # VOICEVOX TTS
uv sync --extra rvc         # RVC voice changer (a lone --extra deselects the others; prefer 'uv sync --all-extras')
uv sync --extra gui         # ttkbootstrap GUI

# Run the pipeline (gRPC server + workers)
uv run python -m vspeech --config ./config.toml

# Run the GUI control panel
uv run python -m gui -c config.toml

# Tests
uv run pytest                              # all tests
uv run pytest tests/test_event_chains.py   # one file
uv run pytest tests/test_event_chains.py::test_worker_output_remotes  # one test

# Lint / format / type-check (ruff + ty are dev dependencies; both from Astral)
uv run ruff format .
uv run ruff check .
uv run ty check            # type checker (replaced pyright)

# Regenerate the Docker requirements file (Linux deploy image, voicevox extra)
make
```

`config.toml.example` documents every setting; `vspeech/config.py` is the source of truth for defaults and shapes. Config files (`config*.toml`, `config*.json`, `key.json`, `*.log`, `*.wav`) are gitignored.

[docs/follow-ups.md](docs/follow-ups.md) tracks review findings that were deliberately deferred out of a branch's scope — with the reason each was deferred. Check it before touching the code it names.

Every GPU-capable `onnxruntime` session (RVC decoder, HuBERT, RMVPE) is opened through the one `create_session` in [`vspeech/lib/onnx_session.py`](vspeech/lib/onnx_session.py). It honours the caller's `torch.device` — never key the execution provider off `torch.cuda.is_available()` alone, and never build a second copy of the factory: the provider guard then only ever gets fixed in one of them. Silero VAD (`vspeech/lib/vad.py`) is the deliberate exception — it pins `CPUExecutionProvider` and takes no device. `tests/test_onnx_session.py` enforces both, asserting `InferenceSession(...)` is constructed in exactly those two files.

A **gitleaks pre-commit gate** ([`.gitleaks.toml`](.gitleaks.toml), [`.pre-commit-config.yaml`](.pre-commit-config.yaml)) scans staged changes for secrets **and** environment-PII (private LAN IPs, `C:\Users\<name>` paths, AmiVoice `appkey`). It is local-only; activate with `uv tool install pre-commit && pre-commit install` plus a `gitleaks` binary on PATH. Use `<USER>`/`<NAS_HOST>` placeholders for machine-specific paths/hosts in committed docs. See [docs/secret-scanning.md](docs/secret-scanning.md).

### HuBERT assets (RVC only, offline)

The RVC content encoder runs as ONNX. Its assets are derived, gitignored, and built by two
one-shot offline steps whose dependencies live **only** in the poe task's `uv run --with`
overlay — never in `pyproject.toml` or `uv.lock`:

```sh
# hubert_base.pt -> hubert_contentvec/ (transformers asset). Needs fairseq, so it runs in a
# throwaway 3.11 environment. Keep ~/.config/vstreamer/hubert_base.pt: it is the input.
uv run poe convert-hubert --input ~/.config/vstreamer/hubert_base.pt \
    --output ./hubert_contentvec --golden ./hubert_golden

# hubert_contentvec/ -> hubert_fp32.onnx + hubert_fp16.onnx. Runs on the project env (cu128
# torch) because the fp16 graph is exported on CUDA. Self-verifies against the golden.
uv run poe export-hubert-onnx --asset ./hubert_contentvec --golden ./hubert_golden
```

`rvc.hubert_model_file` points at the **asset directory**, not a file. The runtime opens only
`hubert_*.onnx` + `mapping.json`; `vspeech/` never imports `fairseq` or `transformers`
(enforced by `tests/test_forbidden_imports.py`). Never run `uv sync --extra rvc` — it
uninstalls the other extras. Use `uv sync --all-extras`.

## Architecture

The system is an **event-driven graph of asyncio workers** connected by queues and a gRPC transport. There are no direct calls between workers — they are wired together only by `EventType` and routing chains.

### Process startup (`vspeech/main.py`)
`vspeech_coro` opens an `asyncio.TaskGroup` and conditionally spawns one task per enabled subsystem (gated by `config.<section>.enable`). Two infrastructure workers always run: `receiver` (gRPC server) and `sender` (gRPC client + local dispatcher). Workers are lazily imported so optional dependencies aren't required unless their feature is enabled. Shutdown propagates as a `WorkerShutdown` exception group caught with `except*`.

### The routing model (`vspeech/lib/command.py`, `vspeech/shared_context.py`)
This is the core abstraction and the part that requires reading several files to understand:

- External input arrives as a protobuf `Command` (from the `vstreamer-protos` dependency) containing **chains** — ordered lists of operations describing a pipeline (e.g. `transcription → translation → subtitle`).
- A `Command` is exploded into `WorkerInput`s, each with a `current_event` (the step to run now) and `following_events` (the remaining graph). `EventType` ↔ protobuf `Operation` conversion lives in `command.py` (`event_to_operation` / `operation_to_event`).
- A worker consumes a `WorkerInput`, does its job, and emits a `WorkerOutput` carrying `text` and/or `sound` plus the unprocessed `followings`. It puts this on `context.sender_queue`.
- The **sender** inspects each following step's `remote`: empty remote → dispatched locally via `process_command` (pushed onto the target worker's `in_queue`); non-empty remote → serialized back to a `Command` and sent over gRPC to another VStreamer instance. This is how pipelines span machines.
- `EventAddress.from_string` parses route strings of the form `<endpoint_uri>/<event_name>?<queries>` (e.g. `//host:8080/translation?t=en`). Query params have short aliases defined in `Params` (`t`=target_language, `s`=source_language, `p`=position, `i`=speaker_id, `v`=volume, `spd`=speed, `pit`=pitch). `EventType.from_string` also accepts shorthands (`sub`, `transc`, `transl`, `rec`, `play`, `fwd`).
- Pipelines are seeded two ways: the `recording` worker builds the initial `WorkerOutput` from `config.recording.routes_list` (`WorkerOutput.from_routes_list`), and inbound gRPC `Command`s arrive with their own chains.

### Control events vs. data events
Beyond data events (transcription/translation/subtitle/tts/vc/playback/recording), `process_command` handles control events synchronously: `pause`/`resume` toggle the global `context.running` `asyncio.Event` (a gate all workers respect), `reload` re-reads the config file and flags affected workers, `set_filters` swaps the text-replacement filters at runtime, `forward` re-emits text to its followings, and `ping` is a health check.

### Worker pattern
Every worker in `vspeech/worker/` follows the same shape:
- `async def <name>_worker(context, in_queue, out_queue)` — a loop that reads `in_queue`, processes, and puts results on `out_queue` (almost always `context.sender_queue`). It wraps `CancelledError` with `shutdown_worker(e)` so cancellation surfaces as `WorkerShutdown`.
- `def create_<name>_task(tg, context)` — registers the worker via `context.add_worker(event, configs_depends_on=[...])` and creates the task named after its `EventType`.

`configs_depends_on` lists the config sections a worker cares about; on a `reload` event, `process_command` compares old vs. new values and sets `need_reload` only for workers whose dependencies changed. Workers check `context.need_reload` mid-loop to rebuild clients/models (e.g. the transcription worker picks a different backend via `worker_type`). The `transcription` and `tts` workers dispatch to sub-implementations by `worker_type` (ACP/GCP/WHISPER, VR2/VOICEVOX) — add new backends there.

### `SharedContext`
The single shared object threaded through everything: holds the live `Config`, the `running` gate, the `sender_queue`, and the `workers` registry (`dict[str, WorkerMeta]`, keyed by event name, each owning its own `in_queue`). `put_queue(dest_event, request)` is how the local dispatcher routes work to a worker.

## Conventions & gotchas

- **Pydantic v2** (`pydantic>=2,<3`). Code uses v2 APIs: `model_config = ConfigDict(...)` / `SettingsConfigDict(...)`, `model_validate`, `model_dump`, `model_dump_json`, `model_validator(mode="after")`, `field_serializer`, `populate_by_name`, `from_attributes`, `AliasChoices`, `SecretStr`. `BaseSettings` is imported from **`pydantic-settings`**. SecretStr secrets that must serialize to plaintext JSON (for the GUI→main handoff) use `@field_serializer(..., when_used="json")` — note `json_encoders` does NOT affect `model_dump_json()` for SecretStr in v2. Do **not** reintroduce v1 APIs (`parse_obj`/`.dict()`/`.json()`/`root_validator`/`orm_mode`/`Field(env=)`/`json_encoders`).
- **Config loading** (`Config` in `config.py`): from a `--config` file (TOML, or JSON if the name ends in `.json`) or, with no file, from environment variables (prefix `vspeech_`, nested delimiter `__`). Secrets (`ami.appkey`, `gcp.service_account_info`) are `SecretStr`.
- **Imports are one-per-line** (`ruff` `force-single-line = true`) and auto-sorted on save. Type checking is **ty** (Astral, configured under `[tool.ty.environment]`, Python 3.12) — the project migrated off pyright. Custom type stubs live in `typings/`.
- **Platform constraints** (encoded in `pyproject.toml`): `torch`, `torchaudio`, `pyvcroid2` are **Windows-only** wheels pinned to specific release URLs in `[tool.uv.sources]`. `voicevox-core` is pinned per-platform there (a `marker`ed list: `win_amd64` for dev, `manylinux_2_34_x86_64` for the Docker image) — don't put a `sys_platform` marker back on the `voicevox` extra itself, or `uv export` silently drops it from `requirements-pod.txt`. `fairseq` and `transformers` are **not** runtime dependencies and are absent from both `pyproject.toml` and `uv.lock` — the HuBERT content encoder runs as ONNX, and the two offline conversion tools (`poe convert-hubert` / `poe export-hubert-onnx`) supply those libs from a `uv run --with` overlay (see *HuBERT assets*). Dev target is Windows; the Docker image targets Linux. The CPU `onnxruntime` is force-overridden out (`override-dependencies`) because it clobbers `onnxruntime-gpu`'s CUDA binary — both expose the same `onnxruntime` module. `voicevox-core` is pinned to **0.16.4** (`cp310-abi3`, pydantic-free). Its ONNX Runtime, OpenJTalk dictionary, and `.vvm` voice models are **not** bundled in the wheel — fetch them with the VOICEVOX downloader (`make voicevox-assets`) and point `voicevox.openjtalk_dir` / `model_dir` / `onnxruntime_path` at them. VOICEVOX uses its own `voicevox_onnxruntime` build, distinct from the `onnxruntime-gpu` used by whisper/rvc; set `onnxruntime_path` explicitly so the correct DLL is loaded.
- **Version-specific features in use**: `TaskGroup` and `except*` exception groups (3.11+); PEP 695 `type X = ...` aliases and `class C[T]` type parameters (3.12+, required by ruff's `UP040`/`UP046` since `requires-python` is 3.12). Don't lower the floor. Going **up** to 3.13 has **two** blockers, not one — see [docs/follow-ups.md](docs/follow-ups.md): (1) `audioop`, which PEP 594 removes; (2) the `numpy>=1.23,<2` cap — numpy 1.26.4 has no cp313 wheel and its `Requires-Python` is uncapped, so a resolver won't skip it, it'll fail building the sdist on 3.13. 3.13 buys **zero** runtime perf here (incremental GC was reverted before 3.13.0 final; free-threading is experimental until 3.14 and moot for this GPU/buffer-bound workload); if you ever move, target **3.14** (same enabling work, one more support year) — its one gap is `pyworld` (no cp314 wheel).
- `vstreamer-protos` (the gRPC/protobuf contract) is an external wheel dependency, not vendored here — changes to the wire format happen in that repo.
- Tests live in `tests/`, run under `pytest` with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed). The most load-bearing tests are `test_event_chains.py` (routing/`EventAddress`/`Command` conversion) — keep them green when touching `command.py` or `shared_context.py`.
