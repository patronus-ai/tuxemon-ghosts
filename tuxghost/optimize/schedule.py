"""A trace's inputs as an editable action list, and back again.

`lift` and `lower` are inverses for every trace `lift` accepts, pinned
both ways in `tests/test_optimize_schedule.py`. `lower` writes through
`tuxghost.loop.add_edge` -- the project's one canonical schedule writer
-- rather than assembling a dict itself, because same-step edge order is
a writer-side convention (`tuxghost/trace.py`'s `read()` neither sorts
nor validates it) and a second writer would be a second convention.

Refusals are `ValueError`, matching the `ValueError`/`TypeError` taxonomy
`validate_actions` and `actions_from_json` already use, so the CLI's
boundary can stay as narrow as `_agent`'s.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

from tuxghost.agent.types import Action
from tuxghost.loop import PRESSED, RELEASED, InputSchedule, add_edge


@dataclass(frozen=True)
class ActionScript:
    """A trace's inputs as actions, plus the wait before the first press.

    INVARIANT: `lead_in + sum(a.hold + a.settle for a in actions) ==
    step_count` of the trace this came from. Both ends are load-bearing
    and NEITHER is present in `Trace.inputs`: `lead_in` is the gap before
    the first press (measured 30 steps on `walk_1234`) and the last
    action's `settle` is the gap after the last release (measured 400 on
    `walk_1234`, 44% of that trace). A version of `lift` that ignored
    them produced a 432-step script for a 442-step trace whose replay
    still matched the parent digest -- see this module's test docstring.
    """

    lead_in: int
    actions: tuple[Action, ...]

    def cost(self) -> int:
        """Total steps this script occupies -- what `seal` runs and what
        the resulting trace's `step_count` must equal."""
        return self.lead_in + sum(a.hold + a.settle for a in self.actions)


def lift(
    inputs: Sequence[tuple[int, int, float]], step_count: int
) -> ActionScript:
    """Edges plus `step_count` -> actions. `step_count` is NOT optional.

    Refuses rather than reinterprets: a button left held, a release with
    no press, a double press, two buttons held at once, a value that is
    neither edge, or a `step_count` that ends before the last release.
    """
    held: dict[int, int] = {}
    pairs: list[tuple[int, int, int]] = []
    for step, button, value in sorted(inputs):
        if value == PRESSED:
            if button in held:
                raise ValueError(
                    f"step {step}: button {button} pressed while already "
                    f"held since step {held[button]}; this trace is not "
                    "expressible as a sequential action list"
                )
            held[button] = step
        elif value == RELEASED:
            press = held.pop(button, None)
            if press is None:
                raise ValueError(
                    f"step {step}: button {button} released without a "
                    "press; refusing rather than inventing one"
                )
            pairs.append((press, button, step))
        else:
            raise ValueError(
                f"step {step}: input value {value!r} is neither "
                f"{PRESSED} (press) nor {RELEASED} (release)"
            )
    if held:
        raise ValueError(
            f"buttons still held when the trace ends: {sorted(held)}; an "
            "action list has no way to express a press that never releases"
        )

    pairs.sort()
    for (p1, b1, r1), (p2, b2, _r2) in pairwise(pairs):
        if p2 < r1:
            raise ValueError(
                f"overlapping holds: button {b1} is held [{p1}, {r1}] "
                f"while button {b2} presses at {p2}; an action list is "
                "sequential and cannot express two buttons held at once"
            )

    if not pairs:
        return ActionScript(lead_in=step_count, actions=())

    last_release = pairs[-1][2]
    if step_count < last_release:
        raise ValueError(
            f"step_count {step_count} ends before the last release at "
            f"step {last_release}; the trace is internally inconsistent"
        )

    lead_in = pairs[0][0]
    actions: list[Action] = []
    prev_release = lead_in
    for press, button, release in pairs:
        if actions:
            actions[-1] = replace(actions[-1], settle=press - prev_release)
        actions.append(Action(button=button, hold=release - press, settle=0))
        prev_release = release
    actions[-1] = replace(actions[-1], settle=step_count - prev_release)

    script = ActionScript(lead_in=lead_in, actions=tuple(actions))
    if script.cost() != step_count:
        raise AssertionError(  # pragma: no cover -- arithmetic invariant
            f"lift produced a {script.cost()}-step script for a "
            f"{step_count}-step trace; the invariant is broken"
        )
    return script


def lower(script: ActionScript) -> InputSchedule:
    """Actions -> a canonical schedule, written through `add_edge`.

    Step indices are computed here, never carried by an edit, which is
    what makes an insert unable to corrupt every action after it.
    """
    schedule: InputSchedule = {}
    step = script.lead_in
    for action in script.actions:
        add_edge(schedule, step, action.button, PRESSED)
        add_edge(schedule, step + action.hold, action.button, RELEASED)
        step += action.hold + action.settle
    return schedule
