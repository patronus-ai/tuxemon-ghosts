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
