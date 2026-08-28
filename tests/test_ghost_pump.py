"""tests/test_ghost_pump.py"""

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
