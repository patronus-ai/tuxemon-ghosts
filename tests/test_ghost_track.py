"""tests/test_ghost_track.py"""

from pathlib import Path

from tuxghost.ghost.track import build_track
from tuxghost.trace import read

PARENT = Path(__file__).parent / "golden" / "scripted_town_1234.tuxghost"


def test_the_track_has_one_frame_per_step_plus_the_final_state() -> None:
    """`run_steps` calls `hook(i)` BEFORE `client.update(FIXED_DT)`, so a
    track built from the hook alone observes the state ENTERING each step
    and never sees the state after the final update -- it is one sample
    short.

    Asserted ARITHMETICALLY, never via a digest. This project has been
    bitten by this exact class twice: `lift` produced a 432-step script
    for a 442-step trace, and `seal`'s identity control dropped the
    parent's last 10 steps. In BOTH cases the digest matched anyway,
    because the dropped steps changed nothing already settled.
    """
    trace = read(PARENT)
    track = build_track(trace)
    assert len(track.frames) == trace.header.step_count + 1


def test_the_cheap_hook_agrees_with_state_of() -> None:
    """`build_track` reads three attributes directly instead of calling
    `state_of`. If the fast path ever disagrees with the canonical one,
    the ghost is drawing a lie -- so they are compared at every sampled
    step, not spot-checked at the end."""
    from tuxghost.execute import execute

    trace = read(PARENT)
    track = build_track(trace)
    result = execute(trace, checkpoint=16, capture_states=True)

    assert result.checkpoint_states, "fixture must produce checkpoints"
    for step, state in result.checkpoint_states.items():
        frame = track.at(step)
        assert frame is not None, step
        assert frame.map_name == state["map"], step
        assert list(frame.tile) == state["tile_pos"], step
        assert frame.facing == state["facing"], step


def test_build_track_reuses_a_given_context_instead_of_reinitialising() -> None:
    """The windowed-render defect S4's manual acceptance found.

    Given a context, `build_track` must NOT reach
    `tuxghost.boot.headless_context()`, because that calls
    `pg.display.set_mode()` a second time and a second `set_mode`
    invalidates every surface already converted against the first
    display. Windowed, that turns the map into blank rectangles with the
    player and every NPC missing.

    This asserts the CALL, not the pixels, deliberately: the suite runs
    entirely headless, where there is no first display to invalidate and
    both paths render identically. A pixel assertion here would pass
    whether or not the bug were present -- exactly the vacuous-test trap
    this project keeps falling into. The pixel evidence is in the commit
    message and STATUS.org, measured windowed.
    """
    from tuxghost import boot

    calls: list[int] = []
    real = boot.headless_context

    def counting() -> object:
        calls.append(1)
        return real()

    trace = read(PARENT)
    boot.headless_context = counting
    try:
        # Without a context: the headless path is taken, as it must be
        # for every non-windowed caller in this project.
        build_track(trace)
        assert len(calls) == 1, calls

        # With one: reused, never re-initialised.
        ctx = real()
        build_track(trace, context=ctx)
        assert len(calls) == 1, calls
    finally:
        boot.headless_context = real
