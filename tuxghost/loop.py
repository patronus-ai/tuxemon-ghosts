"""The one stepping loop. A second one would fail nothing, so there is not one.

Upstream's `HeadlessClient.main()` derives its accumulator from
`time.time()` and calls `time.sleep(0.001)` -- it cannot produce
reproducible execution. This loop takes exactly N steps: no wall clock, no
sleep, no draw pass. Recording and replay both drive the game through this
one function.

Upstream also already has an input record/playback system
(`InputRecorder`, wired into `BaseClient` via the `input_record` /
`input_playback` / `input_save` / `input_load` event actions). We
deliberately do not reuse its playback path: in playback mode,
`InputManager.process_events()` yields exactly one recorded event per
frame, regardless of when that event was originally recorded. That
silently rescales every trace -- a 3-second walk and a 3-minute walk would
replay identically. `install_schedule` replaces `process_events` outright
so an event scheduled for step 5 arrives at step 5, not "whenever the
frame counter happens to get there."
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

STEP_RATE = 60
FIXED_DT = 1.0 / STEP_RATE

# step index -> list of (button, value) to deliver at that exact step.
InputSchedule = dict[int, list[tuple[int, float]]]


def install_schedule(client: Any, schedule: InputSchedule) -> None:
    """Deliver scheduled inputs on exact step indices.

    Replaces `client.input_manager.process_events` outright. Upstream's
    playback path yields one recorded event per frame regardless of its
    recorded time, which silently rescales every trace; we must not
    inherit that behaviour.
    """
    from tuxemon.platform.events import PlayerInput

    state = {"step": 0}

    def process_events() -> Iterator[Any]:
        for button, value in schedule.get(state["step"], []):
            event = PlayerInput(button, value, 1 if value else 0)
            event.triggered = bool(value)
            yield event
        state["step"] += 1

    client.input_manager.process_events = process_events


def run_steps(
    client: Any, n: int, hook: Callable[[int], None] | None = None
) -> None:
    """Advance exactly `n` logical steps. No wall clock, no sleep, no draw."""
    for i in range(n):
        if hook is not None:
            hook(i)
        client.update(FIXED_DT)
