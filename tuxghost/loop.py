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

PRESSED = 1.0
RELEASED = 0.0


def add_edge(
    schedule: InputSchedule, step: int, button: int, value: float
) -> None:
    """Insert one `(button, value)` edge at `step`, appending rather than
    assigning, and keeping the list at that key sorted by `(button,
    value)`.

    `schedule[step] = [...]` (this function's predecessor) SILENTLY
    OVERWRITES rather than accumulates. `validate_actions` permits
    `settle == 0`, and an action with `settle=0` schedules its release at
    exactly the step the *next* action's press lands on (`release = step
    + action.hold` equals the following iteration's starting `step`).
    Task 5 review round 1 found this reachable from an entirely ordinary
    policy: the trace (built by `Recorder.observe`, which only ever
    appends) kept both edges, but the live schedule kept only the last
    write -- so `result.schedule != _schedule_of(result.trace)`, and
    worse, REPLAY would then deliver a button release the RECORDING never
    actually delivered to the engine.

    Sorted by `(button, value)`, on every insert, rather than left in
    whatever order calls happened to arrive. The canonical order a
    replay sees comes from `Recorder.finish` (`tuxghost/record.py`),
    which writes `inputs=sorted(self._inputs)` -- sorted by the full
    `(step, button, value)` tuple -- at RECORD time. `_schedule_of`
    (`tuxghost.execute`) itself does no sorting at all: it just appends
    each `trace.inputs` entry, in the order it finds them, into
    `schedule.setdefault(step, []).append(...)`. It only reconstructs
    ascending `(button, value)` order per step because `Recorder` already
    wrote the file that way. This is a WRITER-SIDE CONVENTION, not a
    format-enforced guarantee: `tuxghost/trace.py`'s `read()` neither
    sorts nor validates input ordering, so a hand-edited or third-party
    trace with unsorted same-step inputs would replay in file order,
    whatever that happens to be (task 5 review round 2 -- pre-existing
    format surface, not something this function introduces or closes).
    What matters here is only that THIS module's own writer
    (`Recorder`, fed by this very function) and THIS module's own live
    delivery agree with each other, and sorting on insert is how that
    agreement is kept.

    Matching that order is not just for the equality check:
    `install_schedule`'s `process_events` yields a step's edges in LIST
    ORDER, and delivers them to the live engine as it goes. An
    insertion-order list could still compare `==` to `_schedule_of`'s
    output as a Python list (if it happened to already be in the same
    order) while, on a different policy, delivering a press/release pair
    to the ENGINE in the opposite order replay would -- a divergence no
    equality check on the schedule alone would catch, only matching,
    canonical ordering does. Safe to re-sort on every insert (not just
    once, lazily) because a step's list is only ever written before
    `run_steps` reaches that step, never mutated after the engine has
    already consumed it. See
    `tests/test_agent_runner.py::test_a_zero_settle_action_schedules_edges_in_canonical_order`,
    which pins this specifically: reversing the collision's button order
    relative to the append-only regression test above is what makes a
    missing `edges.sort()` actually fail (task 5 review round 2 --
    `buttons.DOWN < buttons.RIGHT` made the append-only test's insertion
    order already ascending, so deleting the sort left it green).

    Lives here, next to `InputSchedule`, rather than in either consumer:
    `tuxghost.agent.runner` (S2) and `tuxghost.optimize.schedule` (S3)
    both write schedules, and putting the writer in either one would make
    the other depend on it for a single function. `tuxghost.loop` defines
    the type, owns `install_schedule`, and is already imported by both.
    """
    edges = schedule.setdefault(step, [])
    edges.append((button, value))
    edges.sort()


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
            # `timestamp` defaults to `time.time()` upstream (a falsy
            # check would have been overridden by 0.0; the constructor
            # uses `is not None`, so 0.0 sticks). Inputs are ground truth
            # and step-indexed -- no wall clock may leak into them, or a
            # trace stops being a pure function of (schedule, step).
            event = PlayerInput(button, value, 1 if value else 0, timestamp=0.0)
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
