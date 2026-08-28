"""tests/test_ghost_pump.py"""

from pathlib import Path
from typing import Any

import pytest

from tuxghost.ghost.pump import steps_owed
from tuxghost.loop import FIXED_DT


def test_a_short_frame_owes_no_steps() -> None:
    steps, rest = steps_owed(0.0, FIXED_DT / 4, cap=5)
    assert steps == 0
    assert 0.0 < rest < FIXED_DT


def test_a_long_frame_owes_several_steps() -> None:
    steps, rest = steps_owed(0.0, FIXED_DT * 3.5, cap=5)
    assert steps == 3
    assert rest == pytest.approx(FIXED_DT * 0.5)


def test_the_cap_drops_TIME_not_steps() -> None:
    """Past the cap we discard wall-clock time rather than skipping step
    indices. This is the property that makes frame drops harmless: the
    game feels slow, but the recording's step indices stay exact and the
    ghost -- indexed by step, never by time -- cannot desync.

    A version that skipped steps instead would silently corrupt every
    trace recorded on a loaded machine.
    """
    steps, rest = steps_owed(0.0, FIXED_DT * 100, cap=5)
    assert steps == 5
    assert rest == 0.0, "leftover time must be discarded, not banked"


EPOCH = 1787659200


def _boot() -> tuple[Any, Any]:
    import json
    from pathlib import Path

    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save

    save = SaveData.model_validate(
        json.loads(
            (Path(__file__).parent / "fixtures" / "paper_town.save").read_text()
        )
    )
    return boot_from_save(save, seed=1234, clock_epoch=EPOCH)


def test_a_recorded_session_replays_faithfully(tmp_path: Path) -> None:
    """The whole point of the module split: this exercises the recording
    path with NO window, so it runs in the gate.

    Synthetic events stand in for a keyboard. What is being tested is not
    pygame but the wiring: that an event delivered at step N is recorded
    at step N, so replay delivers it at step N.
    """
    from tuxemon.platform.const import buttons

    from tuxghost.execute import verify
    from tuxghost.ghost.pump import install_recording_events
    from tuxghost.loop import PRESSED, RELEASED, run_steps
    from tuxghost.record import Recorder
    from tuxghost.trace import write

    client, session = _boot()

    rec = Recorder(session, seed=1234, clock_epoch=EPOCH, recorder="human")
    state = {"step": 0}
    scripted = {
        8: [(buttons.DOWN, PRESSED)],
        24: [(buttons.DOWN, RELEASED)],
    }
    install_recording_events(
        client, rec, lambda: state["step"], source=scripted
    )

    def hook(i: int) -> None:
        state["step"] = i

    run_steps(client, 64, hook=hook)
    trace = rec.finish(step_count=64)

    out = tmp_path / "played.tuxghost"
    write(trace, out)
    assert verify(trace) == 0
    assert (8, buttons.DOWN, PRESSED) in trace.inputs


def test_no_timestamp_reaches_the_recorded_trace(tmp_path: Path) -> None:
    """Asserted on the written FILE, not on an in-memory object: the
    format is what must stay clean.

    `PlayerInput`'s upstream default for `timestamp` is `time.time()`.
    Live play is the only path in this project where a real wall-clock
    value is sitting in the object being observed, so this is the only
    place it can leak in.
    """
    import json

    from tuxemon.platform.const import buttons

    from tuxghost.ghost.pump import install_recording_events
    from tuxghost.loop import PRESSED, RELEASED, run_steps
    from tuxghost.record import Recorder
    from tuxghost.trace import write

    client, session = _boot()

    rec = Recorder(session, seed=1234, clock_epoch=EPOCH, recorder="human")
    state = {"step": 0}
    scripted = {
        8: [(buttons.DOWN, PRESSED)],
        24: [(buttons.DOWN, RELEASED)],
    }
    install_recording_events(
        client, rec, lambda: state["step"], source=scripted
    )
    run_steps(client, 64, hook=lambda i: state.__setitem__("step", i))

    out = tmp_path / "played.tuxghost"
    write(rec.finish(step_count=64), out)

    written = json.loads(out.read_text())
    assert written["inputs"], "a vacuous pass over zero inputs"
    assert "timestamp" not in json.dumps(written["inputs"])
    for entry in written["inputs"]:
        assert len(entry) == 3, entry
