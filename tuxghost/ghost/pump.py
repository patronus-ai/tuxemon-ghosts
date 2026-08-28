"""The real-time half of S4: a fixed-timestep accumulator, and a wrapper
that records live input into a normal trace.

`tuxghost.loop.run_steps` takes exactly N steps with no wall clock
involved at all -- exactly right for offline replay/record, but it says
nothing about how many steps a REAL frame, arriving after some measured
`elapsed` wall-clock time, owes the simulation. `steps_owed` is that
accumulator maths.

`install_recording_events` is the write side of live play: it intercepts
`client.input_manager.process_events` (the same interception point
`tuxghost.loop.install_schedule` uses for playback) and feeds every
input it sees to a `tuxghost.record.Recorder`, indexed by the exact step
it arrived on -- never by `PlayerInput.timestamp`, which must never enter
the trace format (see `tuxghost.trace`'s module docstring).
"""

from __future__ import annotations

from typing import Any


def steps_owed(accumulator: float, elapsed: float, cap: int) -> tuple[int, float]:
    """How many fixed steps a frame owes, and the time left over.

    Past `cap`, leftover time is DISCARDED rather than banked. Banking it
    is the spiral of death: a slow frame owes more steps, which take
    longer, which owes more still. Discarding makes the game run slow
    under load -- but every step index stays exact, so the recorded trace
    and the ghost, both indexed by step rather than by time, stay
    correct. Frame drops degrade smoothness, never correctness.
    """
    from tuxghost.loop import FIXED_DT

    total = accumulator + elapsed
    steps = int(total // FIXED_DT)
    if steps > cap:
        return cap, 0.0
    return steps, total - steps * FIXED_DT


def install_recording_events(
    client: Any,
    recorder: Any,
    step_source: Any,
    source: dict[int, list[tuple[int, float]]] | None = None,
) -> None:
    """Record every input the session delivers, at the step it arrives.

    Replaces `client.input_manager.process_events`, the same interception
    `tuxghost.loop.install_schedule` uses -- established practice here,
    not a new mechanism.

    `source` exists ONLY so this can be tested without a keyboard: given
    one, events are synthesised from it; given none, real events are read
    from the wrapped `process_events`. The recording path is identical
    either way, which is the point -- a test that exercised a different
    recording path would prove nothing about live play.

    `PlayerInput.timestamp` is never read or recorded. `CLAUDE.md`: the
    upstream `time.time()` default deliberately never enters the format,
    and live play is the one path where a real timestamp is sitting there
    to be captured by accident.

    Fix round 1 (task 7 review): the synthetic-`source` branch below
    must construct its `PlayerInput` the same way `tuxghost.loop
    .install_schedule` does -- `timestamp=0.0` explicit (the third
    positional argument is `hold_time`, not `timestamp`; leaving
    `timestamp` unset lets the upstream `time.time()` default through,
    inert today only because nothing downstream of this wholesale
    replacement happens to read it) and `.triggered = bool(value)` set
    explicitly (read by `tuxemon/states/input.py:286`). Matching
    `install_schedule`'s construction keeps the two interception points
    this codebase uses in agreement with each other.
    """
    from tuxemon.platform.events import PlayerInput

    original = client.input_manager.process_events

    def process_events() -> Any:
        step = int(step_source())
        if source is not None:
            for button, value in source.get(step, []):
                recorder.observe(step, button, value)
                event = PlayerInput(
                    button, value, 1 if value else 0, timestamp=0.0
                )
                event.triggered = bool(value)
                yield event
            return
        for event in original():
            recorder.observe(step, event.button, event.value)
            yield event

    client.input_manager.process_events = process_events
