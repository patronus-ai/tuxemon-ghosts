"""Pins `Recorder`'s contract: step-indexed inputs, a required
`final_digest` read from the live session, and a header that agrees with
`tuxghost.trace`'s own idea of "this build" closely enough that recording
and immediately reading back a trace never spuriously warns or refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tuxghost.record import Recorder
from tuxghost.trace import (
    THIS_MOD_VERSION,
    THIS_PLATFORM,
    THIS_UPSTREAM_COMMIT,
    Refused,
    Trace,
    patch_series_id,
    read,
    write,
)


def test_recorder_indexes_inputs_by_step_not_arrival_order() -> None:
    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    )
    recorder.observe(9, 64, 0.0)
    recorder.observe(5, 64, 1.0)
    trace = recorder.finish(step_count=12)

    assert trace.inputs == [(5, 64, 1.0), (9, 64, 0.0)]
    assert trace.header.step_rate == 60
    assert trace.header.clock_epoch == 1787694000
    assert trace.header.final_digest, "finish() must record the state reached"


def test_recorder_header_matches_this_builds_fingerprint() -> None:
    """The recorder must stamp the SAME environment fingerprint `read()`
    compares against -- not a hand-copied duplicate of it. If either drifts
    from `tuxghost.trace`'s `THIS_*` constants / `patch_series_id()`, a
    trace recorded and immediately read back on the same build would
    spuriously warn."""
    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    )
    trace = recorder.finish(step_count=0)

    assert trace.header.upstream_commit == THIS_UPSTREAM_COMMIT
    assert trace.header.mod_version == THIS_MOD_VERSION
    assert trace.header.platform == THIS_PLATFORM
    assert trace.header.patch_series_id == patch_series_id()


def test_recorded_trace_round_trips_without_warning_or_refusal(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """End-to-end: record a real session, write it, read it back on the
    same build. Nothing about "this build" changed between writing and
    reading, so none of the three warn rows (upstream_commit,
    patch_series_id, platform) may fire, and initial_state_digest -- a
    hard refusal -- must verify clean."""
    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    )
    recorder.observe(3, 64, 1.0)
    trace = recorder.finish(step_count=5)

    path = tmp_path / "recorded.tuxghost"
    write(trace, path)
    reread = read(path)

    assert reread.header == trace.header
    assert reread.inputs == trace.inputs
    assert not recwarn.list, [str(w.message) for w in recwarn.list]


def test_tampered_initial_state_refuses_on_read_even_with_allow_mismatch(
    tmp_path: Path,
) -> None:
    """A recorded trace whose `initial_state` is altered after the fact (a
    stray hand-edit, or corruption in transit) must be refused -- and must
    STAY refused under `allow_mismatch`, since that flag downgrades version
    quibbles, not the recorded state having changed underneath the trace."""
    import json

    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    )
    trace = recorder.finish(step_count=0)

    path = tmp_path / "recorded.tuxghost"
    write(trace, path)

    raw = json.loads(path.read_text())
    raw["initial_state"]["tile_pos"] = [999, 999]
    path.write_text(json.dumps(raw))

    with pytest.raises(Refused, match="initial_state_digest"):
        read(path)
    with pytest.raises(Refused, match="initial_state_digest"):
        read(path, allow_mismatch=True)


def test_recorder_output_is_stable_for_same_seed_and_clock_epoch() -> None:
    """A cross-run difference check alone cannot detect 'this value is
    effectively random' -- only a same-seed equality check on the same
    route can. Record twice from scratch with the same seed and clock
    epoch and require the sealed trace to match exactly (aside from
    `Provenance`, which carries no comparable content here).

    Relies on `build_client`'s own `clock_epoch` parameter to reset session
    time bookkeeping -- NOT on `Recorder` doing it, which would be a
    surprising side effect of construction (see the module docstring on
    `tuxghost.record`)."""
    from tuxghost.boot import build_client
    from tuxghost.loop import install_schedule, run_steps

    def record_once() -> Trace:
        _client, session = build_client(seed=1234, clock_epoch=1787694000)
        recorder = Recorder(
            session,
            seed=1234,
            clock_epoch=1787694000,
            recorder="offline-agent",
        )
        # `install_schedule` replaces upstream's real event polling, which
        # otherwise reads actual OS/pygame events -- a nondeterminism leak
        # this test must not (re)introduce. See the module docstring on
        # `tuxghost.loop`.
        install_schedule(_client, {5: [(64, 1.0)]})
        recorder.observe(5, 64, 1.0)
        run_steps(_client, 8)
        return recorder.finish(step_count=8)

    first = record_once()
    second = record_once()

    assert first.header == second.header
    assert first.initial_state == second.initial_state
    assert first.inputs == second.inputs


# --- Non-button platform events (the crash that destroyed a real session) --
#
# `tuxemon.platform.const.events` shares `PlayerInput.button` with real
# buttons: QUIT (20000), UNICODE (20001), BACKSPACE (20002). A UNICODE
# event carries the typed CHARACTER as its value. Before `observe`
# filtered them, a human who pressed SPACE during `tuxghost play` put
# `(step, 20001, ' ')` into the trace and `finish()` died in pydantic --
# at WRITE time, on quit, after the session had already been played.


def test_unicode_event_does_not_destroy_the_trace_at_write_time() -> None:
    """The bug, verbatim: `float_parsing` on `input_value=' '`.

    Pinned against `observe`'s filter. Removing it restores
    `ValidationError: ... inputs.N.2 Input should be a valid number,
    unable to parse string as a number [input_value=' ', input_type=str]`
    raised from `finish()` -- i.e. the whole recorded session lost.
    """
    from tuxemon.platform.const import buttons, events

    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="human"
    )
    recorder.observe(0, buttons.DOWN, 1.0)
    recorder.observe(1, events.UNICODE, " ")  # the space bar
    recorder.observe(2, events.BACKSPACE, 0.0)
    recorder.observe(3, buttons.DOWN, 0.0)

    trace = recorder.finish(step_count=4)

    assert [tuple(i) for i in trace.inputs] == [
        (0, buttons.DOWN, 1.0),
        (3, buttons.DOWN, 0.0),
    ]


def test_dropping_a_non_button_event_taints_the_trace() -> None:
    """Dropping keeps the session writable; the taint keeps it honest.

    A run whose text entry was discarded will not replay faithfully, and
    nobody downstream should have to guess that. Pinned against the
    `_taints.append`: without it the trace comes back clean and silently
    claims to be a faithful record.
    """
    from tuxemon.platform.const import events

    from tuxghost.boot import build_client
    from tuxghost.record import DROPPED_NON_BUTTON

    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="human"
    )
    recorder.observe(0, events.UNICODE, "a")
    recorder.observe(1, events.UNICODE, "b")

    trace = recorder.finish(step_count=2)
    # Recorded ONCE, not once per dropped event.
    assert trace.provenance.taints == [DROPPED_NON_BUTTON]


def test_a_clean_recording_is_not_tainted() -> None:
    """The control for the test above: the taint must mean something.

    An assertion that a taint APPEARS proves nothing on its own if the
    recorder taints unconditionally -- this project has shipped exactly
    that class of vacuous test before.
    """
    from tuxemon.platform.const import buttons

    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="human"
    )
    recorder.observe(0, buttons.DOWN, 1.0)

    assert recorder.finish(step_count=1).provenance.taints == []


def test_a_real_button_with_a_non_numeric_value_is_also_refused() -> None:
    """The value guard is separate from the button guard on purpose: a
    real button arriving with a non-numeric value is a different defect
    and must not reach `Trace` either. `bool` counts as non-numeric --
    it is an `int` subclass, and `float(True)` would silently record 1.0.
    """
    from tuxemon.platform.const import buttons

    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="human"
    )
    recorder.observe(0, buttons.DOWN, "1.0")
    recorder.observe(1, buttons.DOWN, True)

    assert list(recorder.finish(step_count=2).inputs) == []
