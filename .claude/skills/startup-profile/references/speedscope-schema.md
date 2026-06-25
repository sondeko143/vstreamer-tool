# speedscope format + bucket signature catalog

Reference for `scripts/startup_analyze.py`. The analyzer reads the **sampled**
speedscope profile that `py-spy record -f speedscope` emits.

## Sampled speedscope shape (what the analyzer parses)

```json
{
  "shared": { "frames": [ {"name": "...", "file": "...", "line": 12}, ... ] },
  "profiles": [
    {
      "type": "sampled",
      "unit": "seconds",
      "startValue": 0, "endValue": 23.6,
      "samples": [ [0, 1, 2], [0, 3], ... ],   // each is a stack of frame indices, leaf LAST
      "weights": [ 0.01, 0.01, ... ]           // time charged to each sample, in `unit`
    }
  ]
}
```

- `shared.frames` is global; every profile's sample indices point into it.
- One profile per thread (with `--subprocesses`, also per followed process).
  The analyzer **aggregates across all profiles** by frame index.
- **self** time = sum of weights where a frame is the leaf. **inclusive** = sum
  where it appears anywhere in the stack (counted once per sample).
- `total` = sum of all weights. `active` = `total - idle`.

**Not parsed:** the `evented` profile type (open/close events) that pyinstrument
emits. If `total` comes out `0.000`, the input is probably evented — re-capture
with py-spy.

## Bucket classification (leaf frame → bucket)

Precedence: **import → idle → blocking-io → compute → other**. (`<module>` and
importlib machinery win first so importing google.auth/torch reads as import
cost, not blocking/compute.)

| Bucket | Matches (name or file) |
|---|---|
| **import** | `<module>`, importlib machinery (`_find_and_load`, `exec_module`, `_call_with_frames_removed`, …), any `importlib` / `<frozen importlib...>` |
| **idle** | event loop: `selectors.py`, `windows_events.py`, `proactor_events.py`, `selector_events.py`, `GetQueuedCompletionStatus`, `epoll_wait`, `_poll`; parked threads: `threading.py` `run` / `_bootstrap_inner` / `_wait_for_tstate_lock` / `wait` / `acquire` |
| **blocking-io** | `getaddrinfo`, `create_connection`, `do_handshake`, `google_auth_default`, `get_service_account_info`, `_metadata`; `subprocess.py` `communicate` / `_execute_child` / `wait`; `\\host\share` UNC paths; `google/auth` files |
| **compute** | `torch`, `faster_whisper`, `ctranslate2`, `onnxruntime`, `numpy`, `fairseq`, `pyworld` |
| **other** | everything else |

These are heuristics tuned for vspeech startup. Update the token lists in
`scripts/startup_analyze.py` (with a failing test first) when a real profile
surfaces a frame in the wrong bucket — that is how the subprocess-wait and
parked-thread rules were added.

## Why `--idle` + an idle bucket

py-spy without `--idle` only samples threads holding the GIL, so it misses
blocking syscalls (the GIL is released during DNS/socket/subprocess waits) — the
exact stalls we want. `--idle` captures them but also samples the post-startup
event loop and parked background threads (e.g. grpc pollers). The `idle` bucket
absorbs that so the active-time headline stays meaningful regardless of how long
the sampling window runs past startup.
