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
