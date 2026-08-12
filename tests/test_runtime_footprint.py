"""The outcome gate on the runtime's weight (ADR-0085, widened by ADR-0087).

This is the **only** protection this project has against a dependency getting heavy again.
There used to be a second one -- a list of package names that `vspeech/` was not allowed to
import and that the dependency table was not allowed to declare. ADR-0087 deleted it: the
goal was never to exclude particular packages but to catch unintended performance
regressions, and a deny-list of names has no principled membership rule, so it grows without
limit and its recorded justifications rot unnoticed (both demonstrated by measurement).
**Nothing here names a package in order to forbid it.** Names appear only as paths to
measure and as modules a path must reach.

Because it is the only gate, its coverage is its whole value. Measuring `vspeech.main`
alone -- which is all this file did before ADR-0087 -- leaves the hole that matters most:
every worker is imported lazily inside `vspeech_coro` behind `config.<section>.enable`, and
the transcription worker defers `faster_whisper` one step further into a function body. A
dependency that lands there makes every running pipeline heavier while the entry point
stays exactly as light as it was. So `MEASURED_PATHS` in
`scripts/runtime_startup_baseline.py` names the module chains a *running* worker reaches,
deferred imports included, and each path records what it must load for its coverage claim
to hold -- checked below, because a path that no longer loads the heavy thing it was added
for is worse than no path at all: it looks like coverage.

**Two of these paths were assertions of a different shape until ADR-0087.**
`stream_vc_consumer` (ADR-0055: a playback-only host must stay light) and `device_layer`
(ADR-0078: resolving a device must not drag an inference framework in) each used to assert
that a named framework was absent from `sys.modules`. The invariants are unchanged; what
changed is that they are now claims about weight, measured in the same shape as every other
path, so they also catch the *next* heavy thing rather than only the one that was named.

There are **two indicators per path, and neither is sufficient alone** (ADR-0085 measured
why): `pydantic_settings` costs +13.7 MB RSS / +176 modules on top of an already-loaded
pydantic, but only **+31 modules / ~1.5 MiB** unique to it on the real startup path
(716 -> 747 modules, N=10 with zero spread on either side; ADR-0066's "32 modules / about
1.6 MB" reproduced). A module check sees those 31; a resident-memory threshold loose enough
not to flap does not see 1.5 MiB. Each covers the other's blind spot.

Startup **time** is measured and printed by the recorder and is deliberately absent from
every verdict here (ADR-0085). The measurement runs behind this file reproduced why: the
same import, back to back on the same machine, took 1.01s and 3.69s.

**Out of scope, plainly**: anything that needs a GPU, model assets on disk or a config
file. A conversion session's real resident memory, CUDA context weight and warmed-model
cost are not measurable in the default suite; `uv run poe vc-footprint` measures a real
pipeline process on demand and stays the tool for that. What is measured here is import
weight, which is where a returning dependency shows up first.

Two entries in the recorded top-level lists look like noise and are not:
`81d243bd2c585b0f4821__mypyc` is charset_normalizer's mypyc-compiled runtime (the hex is a
build id) and `_cython_3_1_1` is a Cython version shim. Both change name when their package
is rebuilt, which is a legitimate baseline update like any other. Two others that *would*
have been noise are filtered out before they reach the baseline at all -- see
`PROVISIONING_ARTIFACTS`.

Re-recording is the fastest way to turn any snapshot gate green, so the record itself is
checked too: enough runs behind it, every measured path present, and a calibration showing
the resident-memory headroom still catches something. Neither check can tell a legitimate
re-record from one taken on a bloated tree -- reading the diff a re-record produces is what
does that.
"""

from functools import cache

import pytest

from scripts.runtime_startup_baseline import MEASURED_PATHS
from scripts.runtime_startup_baseline import MIN_RUNS_FOR_UPDATE
from scripts.runtime_startup_baseline import PATHS_BY_NAME
from scripts.runtime_startup_baseline import MeasuredPath
from scripts.runtime_startup_baseline import Measurement
from scripts.runtime_startup_baseline import _build_path_record
from scripts.runtime_startup_baseline import build_parser
from scripts.runtime_startup_baseline import load_baseline
from scripts.runtime_startup_baseline import measure_path
from scripts.runtime_startup_baseline import measure_startup

# A module that is emphatically not on any measured path and needs no extra installed --
# it stands in for "the heavy dependency nobody thought to name" in the proof below.
NEWCOMER = "sqlite3"

PATH_NAMES = [path.name for path in MEASURED_PATHS]

# One child process per path per session; every test below reads the same readings.
measured = pytest.mark.parametrize("path_name", PATH_NAMES)


@cache
def _baseline() -> dict:
    return load_baseline()


@cache
def _measure(path_name: str) -> Measurement:
    return measure_path(PATHS_BY_NAME[path_name])


def _record(path_name: str) -> dict:
    record = _baseline()["paths"].get(path_name)
    assert record is not None, (
        f"the gate measures a path named {path_name!r} that the recorded baseline does "
        f"not have. Adding a path means re-recording.\n{_regenerate_hint()}"
    )
    return record


def _regenerate_hint() -> str:
    return (
        "If the change is deliberate, review the diff and re-record the baseline:\n"
        f"    {_baseline()['regenerate_with']}"
    )


def _covers(path: MeasuredPath) -> str:
    return f"That path covers: {path.covers}"


def unlisted_top_level(measurement: Measurement, record: dict) -> list[str]:
    """Top-level modules a path loaded that its baseline record does not list.

    Kept separate from the assertion so the proof at the bottom of this file can drive it
    with a path that really does have a newcomer on it.
    """
    return sorted(measurement.top_level - frozenset(record["top_level_modules"]))


@measured
def test_path_pulls_in_no_top_level_module_outside_the_baseline(path_name: str):
    """The gate itself: nothing new arrived on this path, whatever its name."""
    path = PATHS_BY_NAME[path_name]
    added = unlisted_top_level(_measure(path_name), _record(path_name))
    assert not added, (
        f"{len(added)} top-level module(s) reached the {path_name} path that the baseline "
        f"does not list: {', '.join(added)}.\n"
        f"{_covers(path)}\n"
        "The runtime's weight is gated on what running it actually loads (ADR-0085/0087), "
        "not on a list of package names, so this fires for a dependency nobody thought to "
        "name.\n" + _regenerate_hint()
    )


@measured
def test_path_module_count_stays_within_budget(path_name: str):
    """The sub-tree half of the module indicator.

    The check above sees a package arriving; this one sees an already-present package
    growing a sub-tree, which is the shape ADR-0085 measured for `pydantic_settings`.
    """
    budget = _record(path_name)["module_count"]
    observed = len(_measure(path_name).modules)
    assert observed <= budget["budget"], (
        f"the {path_name} path now loads {observed} modules, over the budget of "
        f"{budget['budget']}.\n{_covers(PATHS_BY_NAME[path_name])}\n{budget['basis']}\n"
        + _regenerate_hint()
    )


@measured
def test_path_resident_memory_stays_within_budget(path_name: str):
    """The half a module count cannot see: native weight *on this path*.

    A native extension can be a handful of modules and hundreds of megabytes, which the
    module indicators would wave through almost unremarked. The recorded calibration
    measures how much of that the shared headroom actually stops; numpy, the lightest heavy
    native dependency installed here, is the case it is sized against.
    """
    budget = _record(path_name)["resident_memory_mib"]
    observed = _measure(path_name).working_set_mib
    assert observed <= budget["budget"], (
        f"the {path_name} path's working set is now {observed:.2f} MiB, over the budget of "
        f"{budget['budget']:.2f} MiB.\n{_covers(PATHS_BY_NAME[path_name])}\n"
        f"{budget['basis']}\n" + _regenerate_hint()
    )


@measured
def test_the_baseline_lists_nothing_the_path_no_longer_loads(path_name: str):
    """Baseline hygiene: a stale entry is a re-entry permit.

    Without this, a dependency removed today stays listed forever and could come back
    tomorrow with every gate green -- exactly the failure mode ADR-0085 found in the name
    list it replaces.
    """
    stale = sorted(
        frozenset(_record(path_name)["top_level_modules"])
        - _measure(path_name).top_level
    )
    assert not stale, (
        f"the baseline lists {len(stale)} top-level module(s) the {path_name} path no "
        f"longer loads: {', '.join(stale)}.\n"
        "Leaving them listed would let them return unnoticed.\n" + _regenerate_hint()
    )


@measured
def test_the_path_still_reaches_what_it_was_added_to_cover(path_name: str):
    """A path that stops loading its heavy dependency is not coverage, it is a decoration.

    Every entry in `MEASURED_PATHS` exists because something expensive is reachable along
    it. If an import chain moves -- a worker's deferred import is renamed, a back end is
    split out, a third-party package reorganises -- the path can go on passing all three
    indicators above while measuring a shell. Then a heavy dependency lands on the real
    chain and nothing sees it, which is strictly worse than never having claimed the
    coverage: the gate reads green and the hole is invisible.

    The names here are the modules a path **must** load. That is the opposite of a ban
    list, and it is the one place a name is load-bearing in this file.
    """
    path = PATHS_BY_NAME[path_name]
    loaded = _measure(path_name).top_level
    missing = sorted(name for name in path.reaches if name not in loaded)
    assert not missing, (
        f"the {path_name} path no longer loads {', '.join(missing)}, which it was added "
        f"to cover.\n{_covers(path)}\n"
        "Find where that import chain moved and point the path at it (imports/reaches in "
        "MEASURED_PATHS), then re-record. Do not drop the requirement: this path is what "
        "stands between the runtime and weight arriving there."
    )


def test_the_baseline_records_every_measured_path_and_nothing_else():
    """The record and the path set are one thing, and they drift apart silently.

    A path added to `MEASURED_PATHS` without a re-record has no budgets to check against;
    a path deleted from it but left in the record is a budget nobody measures. Neither
    shows up as a failure of any per-path test, because those iterate the code's list.
    """
    recorded = frozenset(_baseline()["paths"])
    measured_now = frozenset(PATH_NAMES)
    assert recorded == measured_now, (
        "the recorded baseline and the measured path set disagree.\n"
        f"measured but not recorded: {sorted(measured_now - recorded) or 'none'}\n"
        f"recorded but not measured: {sorted(recorded - measured_now) or 'none'}\n"
        + _regenerate_hint()
    )


def test_a_newcomer_on_a_measured_path_is_named():
    """The gate is proved to fire, on every run, not just the day it was written.

    Same child mechanism, same comparison, but with one more module on the path. A guard
    nobody has seen fail is not known to be a guard, and the one thing the deleted name
    list did well -- pointing straight at the culprit -- has to survive the move to an
    outcome measurement. One path suffices: every path is compared by the same code.
    """
    path = PATHS_BY_NAME["entry_point"]
    injected = measure_startup(path.imports + (NEWCOMER,))
    added = unlisted_top_level(injected, _record(path.name))
    assert NEWCOMER in added, (
        f"the gate did not name {NEWCOMER} after it was put on the {path.name} path; it "
        f"reported {added}.\n"
        f"If {NEWCOMER} has since become a legitimate part of that path it is in the "
        "baseline now and can no longer stand in for a newcomer -- pick another module "
        f"that is not, and change NEWCOMER."
    )
    assert len(injected.modules) > len(_measure(path.name).modules)


def test_the_recorded_baseline_rests_on_enough_runs():
    """Re-recording must not be a way to quietly narrow the budgets.

    A budget is anchored on the worst run seen, so fewer runs means a *tighter* budget, not
    merely a weaker one -- measured: an N=2 re-record of an unchanged runtime moved the
    entry point's resident-memory budget from 64.0 to 63.0 MiB, which is how a stable gate
    starts flapping. The CLI refuses `--update` below this floor; this pins the committed
    record so a hand-edit or a bypass is visible too.
    """
    for path_name, record in _baseline()["paths"].items():
        for indicator in ("module_count", "resident_memory_mib"):
            runs = record[indicator]["runs"]
            assert runs >= MIN_RUNS_FOR_UPDATE, (
                f"{path_name}'s {indicator} was recorded from {runs} run(s), below the "
                f"floor of {MIN_RUNS_FOR_UPDATE}. One run is not a measurement, and too "
                "few runs write a budget tighter than the code deserves.\n"
                + _regenerate_hint()
            )


def test_the_recorded_budget_is_calibrated_against_something_it_catches():
    """The tripwire on the resident-memory *headroom*.

    What this pins is `RSS_HEADROOM_MIB`, not any budget's level. `over_budget_mib` is
    `calibration - budget`, and both terms rise together with whatever the calibration path
    happened to cost on the day of the recording, so the difference reduces to the
    calibration module's marginal cost minus the headroom -- invariant to the base level.
    That is also why one calibration speaks for all eleven paths: every budget is the worst
    observed run plus the same headroom, so they all catch the same marginal cost.

    Widening a budget by re-recording on a bloated tree is therefore **not** what this
    catches: such a re-record writes a higher budget with the same `over_budget_mib` and
    reads identically here. What stops that is reviewing the diff the re-record produces,
    which is why every failure message above asks for it.

    What this does catch is a headroom grown until the budgets no longer stop the lightest
    heavy native dependency this project installs. At that point they guard nothing, and
    the record fails here rather than merely reading oddly.
    """
    calibration = _baseline()["budget_rule"]["calibration"]
    assert calibration["measured"], (
        f"the recorded baseline has no calibration: {calibration['module']} was not "
        "installed when it was taken, so nothing shows the resident-memory headroom still "
        "catches anything. Re-record with the extras synced "
        "(`uv sync --all-extras`).\n" + _regenerate_hint()
    )
    budget = _baseline()["paths"][calibration["path"]]["resident_memory_mib"]["budget"]
    assert calibration["over_budget_mib"] > 0, (
        f"the resident-memory headroom no longer catches {calibration['module']}: on the "
        f"{calibration['path']} path it measured "
        f"{calibration['working_set_mib']:.2f} MiB against a budget of {budget:.2f} MiB, "
        f"i.e. {calibration['over_budget_mib']:.2f} MiB over.\n"
        "A headroom that its own calibration walks under is guarding nothing. Tighten it "
        "(RSS_HEADROOM_MIB in scripts/runtime_startup_baseline.py) rather than accepting "
        "the record."
    )


def test_the_documented_regeneration_command_is_one_the_tool_accepts():
    """The instruction handed to a maintainer at the worst possible moment must work.

    It did not: `uv run poe runtime-baseline -- --update --runs 10` exits 2 with
    "unrecognized arguments" and writes nothing, because poe forwards the `--` separator to
    the task verbatim. Every failure message above quotes this string, so it is parsed here
    with the tool's own parser rather than trusted.
    """
    prefix = "uv run poe runtime-baseline "
    command = _baseline()["regenerate_with"]
    assert command.startswith(prefix), command
    tail = command[len(prefix) :].split()
    assert "--" not in tail, (
        f"{command!r} passes a `--` separator through poe to argparse, which rejects the "
        "whole tail (exit 2). Drop it."
    )
    args = build_parser().parse_args(tail)
    assert args.update, f"{command!r} would not re-record anything"
    assert args.runs >= MIN_RUNS_FOR_UPDATE, (
        f"{command!r} asks for {args.runs} run(s), which the tool refuses for --update"
    )


def test_a_measurement_that_disagrees_with_itself_is_not_recorded():
    """The recorder refuses to bake a flap into the baseline.

    The gate compares a *single* run against the recorded list, so a name seen in only some
    runs would be written down and then read as stale on the runs that lack it. Recording
    the intersection instead only moves the flap to the additions check. Any disagreement
    means the gate would flap either way, so the honest answer is to refuse and say which
    names moved.
    """
    path = MeasuredPath(
        name="synthetic", imports=("json",), covers="a fixture", reaches=("json",)
    )
    steady = Measurement(("vspeech", "json"), 50.0, 40.0, 1.0)
    wobbly = Measurement(("vspeech", "json", NEWCOMER), 50.0, 40.0, 1.0)
    with pytest.raises(SystemExit) as caught:
        _build_path_record(path, [steady, wobbly])
    assert NEWCOMER in str(caught.value)


def test_a_path_that_misses_its_own_coverage_claim_is_not_recorded():
    """The recorder half of the anti-decoration check.

    Without this, a path whose import chain has moved is re-recorded happily -- with a
    baseline taken from the shell it now measures -- and the per-path check above then
    passes against that record forever. Refusing at record time is what keeps the
    coverage claim from being repaired by re-recording it away.
    """
    path = MeasuredPath(
        name="synthetic",
        imports=("json",),
        covers="a fixture",
        reaches=("json", NEWCOMER),
    )
    steady = Measurement(("vspeech", "json"), 50.0, 40.0, 1.0)
    with pytest.raises(SystemExit) as caught:
        _build_path_record(path, [steady, steady])
    assert NEWCOMER in str(caught.value)
