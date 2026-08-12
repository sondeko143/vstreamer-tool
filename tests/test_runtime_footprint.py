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
enables. A dependency that only a lazily-imported worker touches is out of reach by
construction -- a known limit, not an oversight. The measurement itself lives in
`scripts/runtime_startup_baseline.py`, which is also the tool that re-records the
baseline.

There are **two indicators, and neither is sufficient alone** (ADR-0085 measured why):
`pydantic_settings` costs +13.7 MB RSS / +176 modules imported in isolation, but only
about 32 modules and ~1.6 MB that are unique to it on the real startup path. A module
check sees those 32; a resident-memory threshold loose enough not to flap does not see
1.6 MB. Each covers the other's blind spot.

Startup **time** is measured and printed by the script and is deliberately absent from
every verdict here. ADR-0085 rejected it on measured grounds, and the measurement runs
behind this file reproduced it: the same import, back to back on the same machine, took
1.01s and 3.69s.

Two entries in the recorded top-level list look like noise and are not:
`81d243bd2c585b0f4821__mypyc` is charset_normalizer's mypyc-compiled runtime (the hex is a
build id) and `_cython_3_1_1` is a Cython version shim. Both change name when their
package is rebuilt, which is a legitimate baseline update like any other.
"""

from functools import cache

from scripts.runtime_startup_baseline import Measurement
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
    """The half a module count cannot see: native weight.

    A single native extension can be a handful of modules and hundreds of megabytes --
    torch was +476.7 MB (ADR-0080) -- so the module indicators alone would let it back in
    almost unremarked.
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
