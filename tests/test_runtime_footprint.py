"""The outcome gate on the runtime's weight (ADR-0085).

`test_forbidden_imports.py` protects that weight with a list of package names. That list
is a proxy for what is actually wanted -- a runtime that starts small and stays small --
and it has a hole a name list cannot close: a heavy dependency nobody thought to name
walks straight through it. This file closes that hole by measuring the outcome instead.

**"The runtime's startup path" here means importing `vspeech.main`** in a pristine child
process: everything `python -m vspeech` loads before it has read a config file. Every
worker past that point is imported lazily inside `vspeech_coro` behind
`config.<section>.enable` and needs a config file, model assets and often a GPU, which
ADR-0085 keeps out of the default suite. So this slice is the largest one measurable
without any of that, and it is the slice *every* pipeline pays whichever workers it
enables. The measurement itself lives in `scripts/runtime_startup_baseline.py`, which is
also the tool that re-records the baseline.

**This gate does not guard torch, and nothing here should be read as though it did.**
Nothing on this startup path imports `ctranslate2`: the transcription worker is imported
lazily, and it defers `faster_whisper` into a function body, so even
`import vspeech.worker.transcription` loads no ctranslate2 (both measured). ctranslate2 is
what picks up a merely *installed* torch, so `uv add torch` leaves every check in this
file green. **ADR-0084's dependency-table gate in `test_forbidden_imports.py` remains the
load-bearing guard for torch.** What this file adds is orthogonal: it catches weight that
lands on the unconditional startup path, which no name list covers. The same limit applies
to onnxruntime and to anything else reached only from a lazily-imported worker.

There are **two indicators, and neither is sufficient alone** (ADR-0085 measured why):
`pydantic_settings` costs +13.7 MB RSS / +176 modules on top of an already-loaded
pydantic (+18.75 MiB / +240 modules from a bare interpreter, re-measured at N=10 in
ADR-0086), but only about 32 modules and ~1.6 MB unique to it on the real startup path.
A module check sees those 32; a resident-memory threshold loose enough not to flap does
not see 1.6 MB. Each covers the other's blind spot.

Startup **time** is measured and printed by the script and is deliberately absent from
every verdict here. ADR-0085 rejected it on measured grounds, and the measurement runs
behind this file reproduced it: the same import, back to back on the same machine, took
1.01s and 3.69s.

Two entries in the recorded top-level list look like noise and are not:
`81d243bd2c585b0f4821__mypyc` is charset_normalizer's mypyc-compiled runtime (the hex is a
build id) and `_cython_3_1_1` is a Cython version shim. Both change name when their
package is rebuilt, which is a legitimate baseline update like any other. Two others that
*would* have been noise are filtered out before they reach the baseline at all -- see
`PROVISIONING_ARTIFACTS`.

Re-recording is the fastest way to turn any snapshot gate green, so the record itself is
checked too: enough runs behind it, and a calibration showing the resident-memory budget
still catches something.
"""

from functools import cache

import pytest

from scripts.runtime_startup_baseline import MIN_RUNS_FOR_UPDATE
from scripts.runtime_startup_baseline import Measurement
from scripts.runtime_startup_baseline import _build_baseline
from scripts.runtime_startup_baseline import build_parser
from scripts.runtime_startup_baseline import load_baseline
from scripts.runtime_startup_baseline import measure_startup

# A module that is emphatically not on the startup path and needs no extra installed --
# it stands in for "the heavy dependency nobody thought to ban" in the proof below.
NEWCOMER = "sqlite3"


@cache
def _baseline() -> dict:
    return load_baseline()


@cache
def _startup() -> Measurement:
    """One child process per test session; every test below reads the same reading."""
    return measure_startup()


def _regenerate_hint() -> str:
    return (
        "If the change is deliberate, review the diff and re-record the baseline:\n"
        f"    {_baseline()['regenerate_with']}"
    )


def unlisted_top_level(measurement: Measurement, baseline: dict) -> list[str]:
    """Top-level modules the startup path loaded that the baseline does not list.

    Kept separate from the assertion so the proof at the bottom of this file can drive it
    with a startup path that really does have a newcomer on it.
    """
    return sorted(measurement.top_level - frozenset(baseline["top_level_modules"]))


def test_startup_pulls_in_no_top_level_module_outside_the_baseline():
    """The gate itself: nothing new arrived, whatever its name."""
    added = unlisted_top_level(_startup(), _baseline())
    assert not added, (
        f"{len(added)} top-level module(s) reached the runtime's startup path that the "
        f"baseline does not list: {', '.join(added)}.\n"
        "The runtime's weight is gated on what starting it actually loads (ADR-0085), "
        "not on a list of package names, so this fires for a dependency nobody thought "
        "to ban.\n" + _regenerate_hint()
    )


def test_startup_module_count_stays_within_budget():
    """The sub-tree half of the module indicator.

    The check above sees a package arriving; this one sees an already-present package
    growing a sub-tree, which is the shape ADR-0085 measured for `pydantic_settings` on
    the real startup path.
    """
    budget = _baseline()["module_count"]
    observed = len(_startup().modules)
    assert observed <= budget["budget"], (
        f"the startup path now loads {observed} modules, over the budget of "
        f"{budget['budget']}.\n{budget['basis']}\n" + _regenerate_hint()
    )


def test_startup_resident_memory_stays_within_budget():
    """The half a module count cannot see: native weight *on this path*.

    A native extension can be a handful of modules and tens of megabytes, which the module
    indicators would wave through almost unremarked. The recorded calibration measures how
    much of that this budget actually stops; numpy, the lightest heavy native dependency
    installed here, is the case it is sized against.

    Read the scope narrowly. This says nothing about torch: torch arrives through
    ctranslate2, which nothing on this startup path imports, so it never reaches this
    measurement at all. ADR-0084's dependency-table gate is what stands between the runtime
    and torch's +476.7 MB (ADR-0080), not this assertion.
    """
    budget = _baseline()["resident_memory_mib"]
    observed = _startup().working_set_mib
    assert observed <= budget["budget"], (
        f"the startup path's working set is now {observed:.2f} MiB, over the budget of "
        f"{budget['budget']:.2f} MiB.\n{budget['basis']}\n" + _regenerate_hint()
    )


def test_the_baseline_lists_nothing_the_startup_no_longer_loads():
    """Baseline hygiene: a stale entry is a re-entry permit.

    Without this, a dependency removed today stays listed forever and could come back
    tomorrow with every gate green -- exactly the failure mode ADR-0085 found in the name
    list it replaces.
    """
    stale = sorted(frozenset(_baseline()["top_level_modules"]) - _startup().top_level)
    assert not stale, (
        f"the baseline lists {len(stale)} top-level module(s) the startup path no longer "
        f"loads: {', '.join(stale)}.\n"
        "Leaving them listed would let them return unnoticed.\n" + _regenerate_hint()
    )


def test_a_newcomer_on_the_startup_path_is_named():
    """The gate is proved to fire, on every run, not just the day it was written.

    Same child mechanism, same comparison, but with one more module on the startup path.
    A guard nobody has seen fail is not known to be a guard, and the one thing the name
    list did well -- pointing straight at the culprit -- has to survive the move to an
    outcome measurement.
    """
    injected = measure_startup(("vspeech.main", NEWCOMER))
    added = unlisted_top_level(injected, _baseline())
    assert NEWCOMER in added, (
        f"the gate did not name {NEWCOMER} after it was put on the startup path; it "
        f"reported {added}.\n"
        f"If {NEWCOMER} has since become a legitimate part of the startup path it is in "
        "the baseline now and can no longer stand in for a newcomer -- pick another "
        f"module that is not, and change NEWCOMER."
    )
    assert len(injected.modules) > len(_startup().modules)


def test_the_recorded_baseline_rests_on_enough_runs():
    """Re-recording must not be a way to quietly narrow the budgets.

    A budget is anchored on the worst run seen, so fewer runs means a *tighter* budget, not
    merely a weaker one -- measured: an N=2 re-record of an unchanged runtime moved the
    resident-memory budget from 64.0 to 63.0 MiB, which is how a stable gate starts
    flapping. The CLI refuses `--update` below this floor; this pins the committed record
    so a hand-edit or a bypass is visible too.
    """
    for indicator in ("module_count", "resident_memory_mib"):
        runs = _baseline()[indicator]["runs"]
        assert runs >= MIN_RUNS_FOR_UPDATE, (
            f"{indicator} was recorded from {runs} run(s), below the floor of "
            f"{MIN_RUNS_FOR_UPDATE}. One run is not a measurement, and too few runs write "
            f"a budget tighter than the code deserves.\n" + _regenerate_hint()
        )


def test_the_recorded_budget_is_calibrated_against_something_it_catches():
    """The tripwire on the resident-memory budget's usefulness.

    Nothing stops someone widening this budget by re-recording on a bloated tree, and the
    resulting record would read perfectly normally. The calibration is the check that it
    still stops something: `over_budget_mib` is by how much the budget catches the
    calibration module. If a re-record ever pushes that to zero or below, the budget has
    stopped guarding anything, and this fails instead of merely reading oddly.
    """
    calibration = _baseline()["resident_memory_mib"]["calibration"]
    assert calibration["measured"], (
        f"the recorded baseline has no calibration: {calibration['module']} was not "
        "installed when it was taken, so nothing shows the resident-memory budget still "
        "catches anything. Re-record with the extras synced "
        "(`uv sync --all-extras`).\n" + _regenerate_hint()
    )
    assert calibration["over_budget_mib"] > 0, (
        f"the resident-memory budget no longer catches {calibration['module']}: it "
        f"measured {calibration['working_set_mib']:.2f} MiB against a budget of "
        f"{_baseline()['resident_memory_mib']['budget']:.2f} MiB, i.e. "
        f"{calibration['over_budget_mib']:.2f} MiB over.\n"
        "A budget that its own calibration walks under is guarding nothing. Tighten it "
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
    steady = Measurement(("vspeech", "json"), 50.0, 40.0, 1.0)
    wobbly = Measurement(("vspeech", "json", "sqlite3"), 50.0, 40.0, 1.0)
    with pytest.raises(SystemExit) as caught:
        _build_baseline([steady, wobbly], None)
    assert "sqlite3" in str(caught.value)
