import json

from scripts import startup_analyze as sa


def make_doc(frames, samples, weights, unit="seconds"):
    return {
        "shared": {"frames": frames},
        "profiles": [
            {
                "type": "sampled",
                "name": "thread 0",
                "unit": unit,
                "startValue": 0,
                "endValue": sum(weights),
                "samples": samples,
                "weights": weights,
            }
        ],
    }


FRAMES = [
    {"name": "main", "file": "vspeech/main.py", "line": 1},
    {"name": "get_credentials", "file": "vspeech/lib/gcp.py", "line": 40},
    {"name": "getaddrinfo", "file": "C:/Python311/Lib/socket.py", "line": 900},
    {"name": "_find_and_load", "file": "<frozen importlib._bootstrap>", "line": 1},
    {"name": "conv2d", "file": "C:/x/.venv/Lib/torch/nn/functional.py", "line": 5},
    {"name": "translation_worker", "file": "vspeech/worker/translation.py", "line": 79},
]
# leaf is the LAST frame index in each stack
SAMPLES = [[0, 1, 2], [0, 1, 2], [0, 3], [0, 4], [0, 5]]
WEIGHTS = [3.0, 2.0, 4.0, 1.0, 0.5]


def test_classify_frame_buckets():
    assert (
        sa.classify_frame("getaddrinfo", "C:/Python311/Lib/socket.py") == "blocking-io"
    )
    assert (
        sa.classify_frame("google_auth_default", "C:/x/google/auth/_default.py")
        == "blocking-io"
    )
    assert (
        sa.classify_frame("_find_and_load", "<frozen importlib._bootstrap>") == "import"
    )
    assert sa.classify_frame("<module>", "C:/x/torch/__init__.py") == "import"
    assert sa.classify_frame("conv2d", "C:/x/torch/nn/functional.py") == "compute"
    assert (
        sa.classify_frame("translation_worker", "vspeech/worker/translation.py")
        == "other"
    )


def test_classify_unc_path_is_blocking_io():
    assert (
        sa.classify_frame("write", "\\\\192.168.138.150\\d\\vs\\out.log")
        == "blocking-io"
    )


def test_classify_idle_event_loop_wait():
    assert sa.classify_frame("select", "C:/Python311/Lib/selectors.py") == "idle"
    assert (
        sa.classify_frame("_poll", "C:/Python311/Lib/asyncio/windows_events.py")
        == "idle"
    )
    assert (
        sa.classify_frame("GetQueuedCompletionStatus", "C:/x/_overlapped.pyd") == "idle"
    )
    assert sa.classify_frame("epoll_wait", "C:/x/selectors.py") == "idle"


def test_classify_parked_thread_is_idle():
    # A background thread sitting at its run/lock base is parked, not startup work.
    assert (
        sa.classify_frame("run", "C:/Program Files/Python311/Lib/threading.py")
        == "idle"
    )
    assert sa.classify_frame("_wait_for_tstate_lock", "C:/x/Lib/threading.py") == "idle"
    assert sa.classify_frame("wait", "C:/x/Lib/threading.py") == "idle"
    # but a same-named function elsewhere is not forced idle
    assert sa.classify_frame("run", "C:/x/vspeech/worker/x.py") == "other"


def test_classify_subprocess_wait_is_blocking_io():
    # Waiting on / launching an external process (e.g. gcloud credential lookup)
    # is an actionable startup stall, like network I/O.
    assert (
        sa.classify_frame("communicate", "C:/Program Files/Python311/Lib/subprocess.py")
        == "blocking-io"
    )
    assert (
        sa.classify_frame("_execute_child", "C:/x/Lib/subprocess.py") == "blocking-io"
    )


def test_render_summary_reports_active_excluding_idle():
    # A big chunk of event-loop idle (window slack after startup finished) must
    # not dilute the blocking-io headline, which is computed against active time.
    frames = FRAMES + [
        {"name": "select", "file": "C:/Python311/Lib/selectors.py", "line": 4}
    ]
    samples = SAMPLES + [[0, 6]]  # idx 6 = select (idle)
    weights = WEIGHTS + [20.0]
    stats, total, unit = sa.compute_frame_stats(make_doc(frames, samples, weights))
    assert total == 30.5
    out = sa.render_summary(stats, total, unit, top=10)
    assert "idle" in out
    # blocking-io self 5.0 over active (30.5 - 20.0 idle = 10.5) ~ 47.6%
    assert "47" in out or "48" in out


def test_compute_frame_stats_self_and_inclusive():
    stats, total, unit = sa.compute_frame_stats(make_doc(FRAMES, SAMPLES, WEIGHTS))
    assert unit == "seconds"
    assert total == 10.5
    by_name = {s.name: s for s in stats}

    # self time = weight where the frame is the leaf
    assert by_name["getaddrinfo"].self_weight == 5.0
    assert by_name["_find_and_load"].self_weight == 4.0
    assert by_name["conv2d"].self_weight == 1.0
    assert by_name["translation_worker"].self_weight == 0.5
    assert by_name["main"].self_weight == 0.0

    # inclusive time = weight where the frame appears anywhere in the stack
    assert by_name["main"].inclusive_weight == 10.5
    assert by_name["get_credentials"].inclusive_weight == 5.0
    assert by_name["getaddrinfo"].inclusive_weight == 5.0


def test_bucket_totals_sum_self_weight():
    stats, _total, _unit = sa.compute_frame_stats(make_doc(FRAMES, SAMPLES, WEIGHTS))
    totals = sa.bucket_totals(stats)
    assert totals["blocking-io"] == 5.0
    assert totals["import"] == 4.0
    assert totals["compute"] == 1.0
    assert totals["other"] == 0.5


def test_aggregates_across_multiple_profiles():
    doc = make_doc(FRAMES, SAMPLES, WEIGHTS)
    # second thread profile: another getaddrinfo leaf sample
    doc["profiles"].append(
        {
            "type": "sampled",
            "name": "thread 1",
            "unit": "seconds",
            "startValue": 0,
            "endValue": 1.0,
            "samples": [[0, 1, 2]],
            "weights": [1.0],
        }
    )
    stats, total, _unit = sa.compute_frame_stats(doc)
    by_name = {s.name: s for s in stats}
    assert total == 11.5
    assert by_name["getaddrinfo"].self_weight == 6.0


def test_rank_by_self_orders_descending():
    stats, _total, _unit = sa.compute_frame_stats(make_doc(FRAMES, SAMPLES, WEIGHTS))
    ranked = sa.rank_by_self(stats)
    assert [s.name for s in ranked[:2]] == ["getaddrinfo", "_find_and_load"]


def test_render_summary_surfaces_blocking_io():
    stats, total, unit = sa.compute_frame_stats(make_doc(FRAMES, SAMPLES, WEIGHTS))
    out = sa.render_summary(stats, total, unit, top=10)
    assert "blocking-io" in out
    assert "getaddrinfo" in out
    # blocking-io is the largest bucket (5.0 / 10.5 ~ 48%)
    assert "48" in out or "47" in out


def test_stats_to_json_valid_with_bucket():
    stats, total, unit = sa.compute_frame_stats(make_doc(FRAMES, SAMPLES, WEIGHTS))
    payload = json.loads(sa.stats_to_json(stats, total, unit))
    assert payload["total"] == 10.5
    assert payload["unit"] == "seconds"
    top = payload["frames"][0]
    assert top["name"] == "getaddrinfo"
    assert top["bucket"] == "blocking-io"
    assert "buckets" in payload


def test_main_reads_file_and_prints(tmp_path, capsys):
    p = tmp_path / "profile.speedscope.json"
    p.write_text(json.dumps(make_doc(FRAMES, SAMPLES, WEIGHTS)), encoding="utf-8")
    rc = sa.main(["--input", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "getaddrinfo" in out
    assert "blocking-io" in out
