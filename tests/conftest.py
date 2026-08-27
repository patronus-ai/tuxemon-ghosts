"""Make the vendored Tuxemon importable and its assets findable.

Two separate problems, both measured:
  * From the repo root, `import tuxemon` resolves to the clone directory as an
    empty namespace package -- it imports fine, then every submodule raises
    ModuleNotFoundError. Putting tuxemon/ on sys.path makes the real package
    (which has __init__.py) win, since a regular package beats a namespace
    portion regardless of path order.
  * Asset loading is relative to the working directory: from the repo root the
    game dies with "Metadata file missing: 'mods/tuxemon/mod.yaml'".
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Literal

import pytest

TUXEMON_DIR = Path(__file__).resolve().parent.parent / "tuxemon"

sys.path.insert(0, str(TUXEMON_DIR))
os.chdir(TUXEMON_DIR)


# --- Parked minor 7, whole-branch review: pin the ambient
# `patch_series_id` warning count -------------------------------------
#
# `tests/golden/walk_1234.tuxghost`'s header pins the `patch_series_id`
# that was live when it was recorded; `patches/` has grown since, so
# every `read()` of it warns (by design -- `tuxghost/trace.py` warns
# rather than refuses on this field). That is currently 7 ambient
# warnings in the fast tier (8 real `read(GOLDEN)` call sites total --
# 5 in `test_golden.py`, 3 in `test_execute.py`, ZERO in `test_trace.py`,
# correcting an earlier ledger entry that miscounted both the total and
# which files it came from -- 1 of the 5 in `test_golden.py` is behind
# the `slow` skip). Left alone, this is exactly the kind of "expected"
# noise a REAL regression -- a call site that stops warning because the
# golden trace quietly got re-recorded, or a new call site in one of
# these two files that reads it without being added here -- could hide
# inside without anyone noticing, which is a bad property for a
# determinism project's own test suite to have. Pinned per-nodeid, both
# directions: too few warnings from a known site, or ANY warning from an
# unpinned site IN THESE TWO FILES, both fail.
#
# Scoped to `test_golden.py`/`test_execute.py` only, deliberately:
# `test_trace.py` has its own tests that DELIBERATELY construct a
# mismatched `patch_series_id` and assert (via `recwarn`) that it warns
# -- e.g. `test_differing_patch_series_id_warns_rather_than_refuses` --
# that is the warning working as designed, already checked at its own
# call site, not ambient noise this tripwire owns. A first version of
# this check flagged those as "unexpected" too, which is wrong: it
# would have made this tripwire fail on ANY committed test of the
# warning path itself, in any file, forever.
_TRACKED_FILES = (
    "tests/test_golden.py::",
    "tests/test_execute.py::",
    # No pinned entries below for this file, deliberately: its
    # fixture (`tests/golden/claude_town_1234.tuxghost`) was recorded
    # against the CURRENT `patches/`, so it must warn ZERO times.
    # Tracking it means the day someone adds a seventh patch, the
    # live-capture fixture going stale FAILS here instead of quietly
    # joining the ambient noise.
    "tests/test_live_capture.py::",
    # No pinned entries below for this file either: it reads no committed
    # trace (both traces it compares are built live, in-process, from
    # `tests/fixtures/paper_town.save`), so it must warn ZERO times about
    # `patch_series_id`. Tracking it means a future seventh patch fails
    # loudly here rather than joining the ambient noise (task 1, S3 plan).
    "tests/test_convention_alignment.py::",
    # `walk_1234.tuxghost`'s `patch_series_id` differs from the current
    # build, so the two parametrized tests that read it via `lift` each
    # warn once (task 3, S3 plan); `claude_town_1234.tuxghost` is
    # recorded against the current `patches/` and does not warn, so it
    # gets no entry below -- the same asymmetry `test_golden.py` already
    # relies on.
    "tests/test_optimize_schedule.py::",
    # No pinned entries below for this file either: it reads
    # `claude_town_1234.tuxghost`, recorded against the current
    # `patches/`, so it must warn ZERO times. Tracking it means a future
    # seventh patch fails loudly here instead of the stale read joining
    # ambient noise -- the `test_live_capture.py` precedent (task 5, S3
    # plan).
    "tests/test_optimize_seal.py::",
    # No pinned entries below for this file either: it reads only
    # `claude_town_1234.tuxghost`, recorded against the current
    # `patches/`, so it must warn ZERO times. Tracking it means a future
    # seventh patch fails loudly here instead of the stale read joining
    # ambient noise -- the `test_optimize_seal.py` precedent (task 7, S3
    # plan).
    "tests/test_optimize_runner.py::",
    # No pinned entries below for this file either: it reads only
    # `claude_town_1234.tuxghost`, recorded against the current
    # `patches/`, so it must warn ZERO times. Tracking it means a future
    # seventh patch fails loudly here instead of the stale read joining
    # ambient noise -- the `test_optimize_runner.py` precedent (task 8, S3
    # plan).
    "tests/test_optimize_mutation.py::",
)
_EXPECTED_PATCH_SERIES_ID_WARNING_COUNTS: dict[str, int] = {
    "tests/test_golden.py::test_golden_trace_still_reaches_its_recorded_digest": 1,
    "tests/test_golden.py::test_golden_trace_verifies": 1,
    "tests/test_golden.py::test_golden_trace_reaches_a_nonempty_party": 1,
    "tests/test_golden.py::test_golden_trace_round_trips_through_record_execute_record": 1,
    "tests/test_golden.py::test_long_horizon_trace_is_stable": 1,
    "tests/test_execute.py::test_execute_boots_a_save_whose_current_map_omits_the_extension": 2,
    "tests/test_execute.py::test_verify_refuses_a_trace_whose_map_cannot_be_resolved": 1,
    "tests/test_optimize_schedule.py::test_lift_recovers_the_measured_shape[walk_1234-900-30-400-40]": 1,
    "tests/test_optimize_schedule.py::test_lower_lift_round_trips_to_the_same_schedule[walk_1234-900-30-400-40]": 1,
}

_patch_series_id_warning_counts: dict[str, int] = {}
_executed_nodeids: set[str] = set()
# All nodeids pytest actually COLLECTED this session, regardless of any
# `-k`/`-m` deselection applied afterward -- collection happens before
# that filtering, so an item deselected by `-k` still lands here, but an
# item that no longer EXISTS (deleted or renamed) never does. This is
# what lets the reachability check below (final residuals, item 3)
# distinguish "this pinned test wasn't selected this run" from "this
# pinned test is gone", the same distinction `tests/test_digest.py`'s
# `test_exemptions_are_all_reachable_in_the_digested_tree` makes for
# `EXEMPTIONS`.
_collected_nodeids: set[str] = set()


def pytest_itemcollected(item: pytest.Item) -> None:
    _collected_nodeids.add(item.nodeid)


def _file_of(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def pytest_warning_recorded(
    warning_message: warnings.WarningMessage,
    when: Literal["config", "collect", "runtest"],
    nodeid: str,
    location: tuple[str, int, str] | None,
) -> None:
    del when, location
    if "patch_series_id" in str(warning_message.message):
        _patch_series_id_warning_counts[nodeid] = (
            _patch_series_id_warning_counts.get(nodeid, 0) + 1
        )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":
        _executed_nodeids.add(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """A `pytest_sessionfinish` hook, deliberately NOT a fixture teardown:
    `_pytest.warnings.catch_warnings_for_item` wraps `pytest_runtest_
    protocol` (setup+call+teardown together) and only flushes captured
    warnings through `pytest_warning_recorded` in ITS OWN `finally`, which
    runs AFTER that whole wrapped protocol returns -- including after any
    session-scoped fixture's teardown that happens to finalize during the
    last item's teardown phase. A `_patch_series_id_warning_tripwire`
    fixture (an earlier version of this check) reliably saw 6 of 7
    warnings and silently missed the 7th for exactly this reason,
    confirmed with a debug print showing the hook fire AFTER the
    fixture's own teardown had already run and asserted. `pytest_session
    finish` fires once, strictly after every item's `pytest_runtest_
    protocol` -- and therefore every warning flush -- has completed."""
    mismatches = []
    seen = dict(_patch_series_id_warning_counts)
    for nodeid, expected in _EXPECTED_PATCH_SERIES_ID_WARNING_COUNTS.items():
        if nodeid not in _executed_nodeids:
            continue  # not selected this run -- nothing to check
        got = seen.pop(nodeid, 0)
        if got != expected:
            mismatches.append(
                f"{nodeid}: expected {expected} patch_series_id "
                f"warning(s), got {got}"
            )
    for nodeid, got in seen.items():
        if not nodeid.startswith(_TRACKED_FILES):
            continue  # e.g. test_trace.py's OWN deliberate warning tests
        mismatches.append(
            f"{nodeid}: {got} unexpected patch_series_id warning(s) -- "
            f"not in the pinned list; a new stale-fixture read appeared"
        )
    # Reachability gate (final residuals, item 3): unlike `EXEMPTIONS`
    # (guarded by `tests/test_digest.py`'s
    # `test_exemptions_are_all_reachable_in_the_digested_tree`), the
    # pinned table above had no check that each entry still names a real
    # test. Deleting or renaming a pinned test would silently leave a
    # dead entry that `nodeid not in _executed_nodeids` reads as "not
    # selected this run" forever -- nothing would ever flag it. A pinned
    # nodeid whose FILE was collected this session but which itself was
    # not is a dead entry, not a deselection; a pinned nodeid whose file
    # was never collected at all (e.g. a narrow `pytest tests/test_x.py`
    # invocation) is a genuinely out-of-scope run and must not be flagged.
    collected_files = {_file_of(nodeid) for nodeid in _collected_nodeids}
    for nodeid in _EXPECTED_PATCH_SERIES_ID_WARNING_COUNTS:
        if _file_of(nodeid) in collected_files and nodeid not in _collected_nodeids:
            mismatches.append(
                f"{nodeid}: pinned but was not collected at all this "
                "session -- the test was deleted or renamed; update or "
                "remove this pinned entry"
            )
    if mismatches:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        message = (
            "ambient patch_series_id warnings drifted from the pinned "
            "count (parked minor 7, whole-branch review -- this stops "
            "being safe to ignore the moment it changes):\n  "
            + "\n  ".join(mismatches)
        )
        if reporter is not None:
            reporter.write_line(message, red=True, bold=True)
        else:  # pragma: no cover -- always present under a normal pytest run
            print(message, file=sys.stderr)
        # Only overwrite a SUCCESSFUL incoming status (final residuals,
        # item 3): this tripwire must never launder a real interrupt or
        # pytest-internal error (a nonzero `exitstatus` already reported)
        # into "this tripwire's failure" by clobbering it with a flat 1.
        # The mismatch message above still prints either way, so a real
        # crash during a run that ALSO happens to have a warning
        # mismatch loses nothing -- only the exit code that a more
        # serious status already claimed is left alone.
        if not exitstatus:
            session.exitstatus = 1
