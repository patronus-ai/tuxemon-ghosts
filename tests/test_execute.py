"""Pins the executor's contract: `execute`/`verify` replay a trace through
the one stepping loop and compare against `header.final_digest` -- the
state the ORIGINAL recording reached, not a second execution -- and
`tuxghost.compare` can narrow a divergence down to a step and a field
rather than just a differing hash.

`verify` executes a trace EXACTLY ONCE. This is deliberately load-bearing
enough to get its own regression test
(`test_verify_executes_the_trace_exactly_once`): the brief's own sample
implementation type-checks and passes the happy-path test while still
comparing two independent executions against each other instead of
against the recorded `final_digest` -- indistinguishable from the correct
behaviour on a trace that replays cleanly, and exactly the bug this file's
process note ("write assertions that fail when the code is broken") warns
against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tuxghost.compare import describe_divergence, first_difference, first_divergent_step

GOLDEN = Path(__file__).parent / "golden" / "walk_1234.tuxghost"


def test_first_difference_names_the_field_path() -> None:
    a = {"npc_state": {"battles": [{"timestamp": 1.0}]}}
    b = {"npc_state": {"battles": [{"timestamp": 2.0}]}}
    assert first_difference(a, b) == "npc_state.battles[0].timestamp"


def test_first_difference_is_none_when_equal() -> None:
    assert first_difference({"a": [1, 2]}, {"a": [1, 2]}) is None


def test_first_difference_names_a_missing_key_not_just_len() -> None:
    """A key present on one side and absent on the other is not a list-
    length mismatch; it must name the key itself."""
    assert first_difference({"a": 1}, {"a": 1, "b": 2}) == "b"


def test_first_divergent_step_finds_the_earliest_checkpoint() -> None:
    a = [(100, "x"), (200, "y"), (300, "z")]
    b = [(100, "x"), (200, "DIFF"), (300, "z")]
    assert first_divergent_step(a, b) == 200
    assert first_divergent_step(a, a) is None


def test_first_divergent_step_reports_a_length_mismatch_not_none() -> None:
    """Fix round 1, finding 3: a naive `zip(a, b)` silently truncates to
    the shorter list, so two runs that plainly diverge by one having
    checkpoints the other never reached used to report `None` -- 'no
    divergence' -- instead of a real divergence at the point they stopped
    agreeing on length. `b` here shares every checkpoint `a` has and then
    keeps going; the extra checkpoint is the divergence, reported at its
    own step (400), not swallowed."""
    a = [(100, "x"), (200, "y")]
    b = [(100, "x"), (200, "y"), (300, "z"), (400, "w")]
    assert first_divergent_step(a, b) == 300
    assert first_divergent_step(b, a) == 300


def test_first_divergent_step_rejects_mismatched_cadences() -> None:
    """Same fix round: mismatched STEP NUMBERS at the same list position
    (as opposed to mismatched lengths) must raise rather than silently
    compare unrelated checkpoints -- was a bare `assert`, which `python
    -O` discards; now a real, caller-triggerable `ValueError`."""
    a = [(100, "x")]
    b = [(150, "x")]
    with pytest.raises(ValueError, match="cadences"):
        first_divergent_step(a, b)


def test_describe_divergence_names_step_field_and_both_values() -> None:
    """The target output shape from the task-12 brief: which step, which
    field, which two values -- not just 'the hash differs'."""
    a = {"npc_state": {"battles": [{"timestamp": 1787694198.869361}]}}
    b = {"npc_state": {"battles": [{"timestamp": 1787694387.998919}]}}
    report = describe_divergence(10000, a, b)
    assert report == (
        "diverged at step 10000\n"
        "  npc_state.battles[0].timestamp: 1787694198.869361 != 1787694387.998919"
    )


def test_value_at_rejects_a_malformed_path_segment() -> None:
    """Fix round 1, finding 4: `_value_at`'s path-segment parser used a
    bare `assert` for a segment `first_difference` itself would never
    produce but a caller resolving a hand-built path could still pass in
    -- now a real `ValueError`, checked directly rather than only via the
    happy path `describe_divergence` exercises."""
    from tuxghost.compare import _value_at

    with pytest.raises(ValueError, match="malformed path segment"):
        _value_at({"a": 1}, "a[not-a-number]")


def test_verify_returns_zero_for_a_self_consistent_trace() -> None:
    """The task-12 brief's own sample version of this test declares
    `step_count=120` but never actually calls `run_steps` before
    `finish()` -- `finish()` then digests the session at step 0, while
    `execute()` genuinely advances 120 steps from the same
    `initial_state`. Measured directly (`tuxghost.compare.first_difference`
    on `tuxghost.digest.state_of` before/after 120 idle steps): they
    disagree at `state_stack[len]` -- the world is not static even with an
    empty input schedule (a menu/dialogue state opens on its own). That
    made the brief's own sample `verify() == 0` assertion FAIL against
    correct code, for a reason that has nothing to do with `verify` itself
    -- exactly the "assertion that passes/fails for the wrong reason" this
    project's process note warns about, just inverted (a false failure
    instead of a false pass). Fixed by actually running the declared
    schedule before sealing the trace, matching
    `tests/test_record.py::test_recorder_output_is_stable_for_same_seed_
    and_clock_epoch`'s own pattern."""
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.execute import verify
    from tuxghost.loop import run_steps
    from tuxghost.record import Recorder

    seed_all(1234)
    pin_clock(1787694000)
    client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    )
    run_steps(client, 120)
    trace = recorder.finish(step_count=120)
    assert verify(trace) == 0


def test_verify_returns_one_when_the_recorded_final_digest_is_wrong() -> None:
    """A trace whose `header.final_digest` does not match what replay
    actually reaches must be reported as diverged (1), not refused (2) and
    not silently accepted (0). Tampering `final_digest` directly (rather
    than, say, corrupting `initial_state`) isolates exactly the comparison
    `verify` is responsible for -- `tuxghost.trace.read` never digest-
    checks `final_digest` the way it does `initial_state`, so a Trace
    object with a wrong one is a legitimate input to `verify`, not
    something an earlier layer would already have refused."""
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.execute import verify
    from tuxghost.record import Recorder

    seed_all(1234)
    pin_clock(1787694000)
    _client, session = build_client(seed=1234)
    trace = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    ).finish(step_count=10)

    tampered = trace.model_copy(
        update={
            "header": trace.header.model_copy(
                update={"final_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
            )
        }
    )
    assert verify(tampered) == 1


def test_verify_returns_two_when_the_trace_is_unrunnable_here() -> None:
    """`initial_state` with no `npc_state` at all validates fine as a
    `SaveData` (every field on it is optional/defaulted -- see
    `tuxemon/save_system/save_state.py`), but `boot_from_save` cannot
    restore a session from it: it asserts `npc_state is not None and
    npc_state.current_map is not None`. That is a real, reachable
    precondition failure distinct from anything `tuxghost.trace.read`
    already checks (which only ever sees `initial_state` as an opaque
    dict, not something it tries to restore a session from) -- `execute`
    must translate it into `Refused`, and `verify` into exit code 2, not
    let the `AssertionError` propagate raw."""
    from tuxghost.execute import execute, verify
    from tuxghost.trace import (
        FORMAT_VERSION,
        THIS_MOD_VERSION,
        THIS_PLATFORM,
        THIS_UPSTREAM_COMMIT,
        Provenance,
        Refused,
        Trace,
        TraceHeader,
        digest_of_initial_state,
        patch_series_id,
    )

    initial_state: dict[str, Any] = {}
    trace = Trace(
        format_version=FORMAT_VERSION,
        header=TraceHeader(
            upstream_commit=THIS_UPSTREAM_COMMIT,
            patch_series_id=patch_series_id(),
            platform=THIS_PLATFORM,
            mod_id="tuxemon",
            mod_version=THIS_MOD_VERSION,
            seed=1234,
            step_rate=60,
            clock_epoch=1787694000,
            initial_state_digest=digest_of_initial_state(initial_state),
            step_count=0,
            final_digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
        ),
        initial_state=initial_state,
        inputs=[],
        provenance=Provenance(recorder="offline-agent"),
    )

    with pytest.raises(Refused):
        execute(trace)
    assert verify(trace) == 2


def test_verify_executes_the_trace_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """`TraceHeader.final_digest` records the state the recorded run
    ACTUALLY reached. `verify` must compare a single execution against
    THAT recorded value -- not run the trace twice and compare the two
    runs against each other, which would only prove the engine is
    reproducible on this build and would pass even if the recorded trace
    itself no longer reaches `final_digest` at all (e.g. a stale trace
    replayed against a newer, behaviour-changing engine build). Pin the
    call count directly: a version of `verify` that executes twice must
    fail this even when it also happens to return the right exit code on
    the happy path above."""
    import tuxghost.execute as execute_module
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.loop import run_steps
    from tuxghost.record import Recorder

    seed_all(1234)
    pin_clock(1787694000)
    client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    )
    run_steps(client, 10)
    trace = recorder.finish(step_count=10)

    call_count = 0
    real_execute = execute_module.execute

    def counting_execute(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return real_execute(*args, **kwargs)

    monkeypatch.setattr(execute_module, "execute", counting_execute)

    assert execute_module.verify(trace) == 0
    assert call_count == 1, "verify() must call execute() exactly once"


def test_execute_populates_checkpoints_at_the_given_interval() -> None:
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.execute import execute
    from tuxghost.record import Recorder

    seed_all(1234)
    pin_clock(1787694000)
    _client, session = build_client(seed=1234)
    trace = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    ).finish(step_count=50)

    result = execute(trace, checkpoint=10)
    assert [step for step, _digest in result.checkpoints] == [10, 20, 30, 40]
    assert all(digest for _step, digest in result.checkpoints)


def test_execute_passes_clock_epoch_through_to_boot_from_save() -> None:
    """Session-time bookkeeping (`SessionSave.duration`/`total_playtime`/
    `start_time`) is invisible to `digest_of`/`state_of` (see
    `tuxghost.digest.state_of`), so `test_verify_returns_zero_for_a_self_
    consistent_trace` above cannot catch `execute` forgetting to pass
    `trace.header.clock_epoch` through to `boot_from_save` -- it would
    still pass. Pin the actual reproducibility guarantee directly, the
    same way `tests/test_boot.py
    ::test_boot_from_save_session_time_is_reproducible_with_clock_epoch`
    pins `boot_from_save` itself: dirty the shared session singleton's
    clock bookkeeping between recording and executing, then require
    `execute` to have restored it to the epoch-derived values regardless."""
    from datetime import UTC, datetime

    from tuxemon.core.clock import set_epoch
    from tuxemon.session import local_session

    from tuxghost.boot import build_client, snapshot_save
    from tuxghost.determinism import seed_all
    from tuxghost.execute import execute
    from tuxghost.record import Recorder

    epoch = 1787694000
    expected_start_time = (
        datetime.fromtimestamp(epoch, tz=UTC)
        .replace(tzinfo=None)
        .strftime("%Y-%m-%d %H:%M")
    )

    seed_all(4321)
    _client, session = build_client(seed=4321, clock_epoch=epoch)
    trace = Recorder(
        session, seed=4321, clock_epoch=epoch, recorder="offline-agent"
    ).finish(step_count=0)

    # Dirty the singleton's clock bookkeeping between recording and
    # executing -- see this test's own docstring and
    # tests/test_boot.py's matching test for why this step is required to
    # make the assertion below non-vacuous.
    set_epoch(None)
    snapshot_save(session)

    execute(trace)

    restored_state = json.loads(snapshot_save(local_session).model_dump_json())[
        "session_state"
    ]
    assert restored_state["duration"] == 0.0
    assert restored_state["total_playtime"] == 0.0
    assert restored_state["start_time"] == expected_start_time


def test_round_trip_record_write_read_execute_verify(tmp_path: Path) -> None:
    """The genuine end-to-end round trip the task's definition of done
    asks for: record a real trace, write it to disk, read it back (going
    through `tuxghost.trace.read`'s full refusal/warning matrix, not just
    constructing a `Trace` object directly the way the other tests in this
    file do), execute it, and confirm both `execute`'s own result and
    `verify` agree the replay reached what the recording reached."""
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.execute import execute, verify
    from tuxghost.loop import install_schedule, run_steps
    from tuxghost.record import Recorder
    from tuxghost.trace import read, write

    seed_all(1234)
    pin_clock(1787694000)
    client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    )
    install_schedule(client, {5: [(64, 1.0)], 9: [(64, 0.0)]})
    recorder.observe(5, 64, 1.0)
    recorder.observe(9, 64, 0.0)
    run_steps(client, 40)
    trace = recorder.finish(step_count=40)

    path = tmp_path / "roundtrip.tuxghost"
    write(trace, path)

    # This file's own build/record/write/read round trip is on the SAME
    # build that just ran it, so `read()` is expected not to warn --
    # already pinned directly by `tests/test_record.py
    # ::test_recorded_trace_round_trips_without_warning_or_refusal`. This
    # test's own assertions are the `execute`/`verify` results below.
    reread = read(path)

    result = execute(reread)
    assert result.final_digest == trace.header.final_digest
    assert verify(reread) == 0


def test_bisect_traces_locates_the_diverging_step_and_field() -> None:
    """Needs both halves working together: `first_divergent_step` narrows
    two executions down to a checkpoint, `first_difference` (via
    `describe_divergence`) names the field there. Two different seeds
    diverge almost immediately (different seeded ids/RNG draws from step
    0), which is enough to prove `bisect_traces` actually walks the
    checkpoint list and reports a real field -- not e.g. always reporting
    checkpoint 0 or a hardcoded path."""
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.execute import bisect_traces
    from tuxghost.record import Recorder

    def record(seed: int) -> Any:
        seed_all(seed)
        pin_clock(1787694000)
        _client, session = build_client(seed=seed, clock_epoch=1787694000)
        return Recorder(
            session, seed=seed, clock_epoch=1787694000, recorder="offline-agent"
        ).finish(step_count=40)

    trace_a = record(1234)
    trace_b = record(99)

    report = bisect_traces(trace_a, trace_b, checkpoint=10)
    assert report is not None
    assert report.startswith("diverged at step ")
    assert ": " in report.splitlines()[1]


def test_bisect_traces_finds_nothing_between_a_trace_and_itself() -> None:
    """Same-seed stability control for `bisect_traces`, per the project's
    process rule: a divergence-finding tool that always finds SOMETHING
    would be exactly as worthless as a digest that always discriminates.
    Executing the same trace against itself twice must find no divergent
    checkpoint."""
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.execute import bisect_traces
    from tuxghost.record import Recorder

    seed_all(1234)
    pin_clock(1787694000)
    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    trace = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    ).finish(step_count=40)

    assert bisect_traces(trace, trace, checkpoint=10) is None


def test_bisect_traces_rejects_a_non_positive_checkpoint_interval() -> None:
    """Fix round 1, finding 4: `bisect_traces` used to guard `checkpoint >
    0` with a bare `assert`, which `python -O` silently discards and which
    is not the right tool for a contract a caller can violate. Pin the
    real exception type directly rather than only the happy path."""
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.execute import bisect_traces
    from tuxghost.record import Recorder

    seed_all(1234)
    pin_clock(1787694000)
    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    trace = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    ).finish(step_count=10)

    with pytest.raises(ValueError, match="checkpoint"):
        bisect_traces(trace, trace, checkpoint=0)


def test_execute_boots_a_save_whose_current_map_omits_the_extension() -> None:
    """Pins the S1 defect: an extension-less `current_map` -- what every
    `spyder_*` map's own teleport script produces -- raised an uncaught
    OSError out of `fetch_asset`, so `verify()` raised instead of
    returning 2 and the CLI would have exited 1 ("diverged") for what is
    a refusal at worst and a bootable save at best."""
    from tuxghost.execute import execute
    from tuxghost.trace import read

    trace = read(GOLDEN)
    trace.header.step_count = 30
    with_ext = execute(trace).final_digest

    trace_no_ext = read(GOLDEN)
    trace_no_ext.header.step_count = 30
    trace_no_ext.initial_state["npc_state"]["current_map"] = "start_tuxemon"
    assert execute(trace_no_ext).final_digest == with_ext


def test_verify_refuses_a_trace_whose_map_cannot_be_resolved() -> None:
    """A genuinely missing map is exit 2 (refused), never an exception and
    never exit 1 (diverged) -- see tuxghost/cli.py's module docstring."""
    from tuxghost.execute import verify
    from tuxghost.trace import read

    trace = read(GOLDEN)
    trace.header.step_count = 30
    trace.initial_state["npc_state"]["current_map"] = "no_such_map_xyz"
    assert verify(trace) == 2
