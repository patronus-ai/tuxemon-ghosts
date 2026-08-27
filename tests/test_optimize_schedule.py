"""Task 3: the edit unit, and the two quantities a trace's edge list does
not contain.

`lift` REQUIRES `step_count`. A lead-in before the first press and a tail
after the last release leave no edge behind, and a version of `lift`
written without `step_count` produced a 432-step script for the 442-step
`claude_town_1234` trace -- whose replay then matched the parent's digest
anyway, because the ten dropped steps changed nothing already settled.
That is this project's attractor lesson (docs/STATUS.org, "Settled end
states cannot detect a perturbation in this game") applied to `lift`: the
digest cannot be trusted to catch a dropped tail, so the invariant is
asserted arithmetically instead.

Both committed traces are tested, never just one: `claude_town_1234` has
`lead_in == 0`, so a `lift` that drops lead-ins entirely passes against it
and fails only against `walk_1234`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tuxemon.platform.const import buttons

from tuxghost.agent.types import Action
from tuxghost.execute import _schedule_of
from tuxghost.loop import PRESSED, RELEASED
from tuxghost.optimize.schedule import ActionScript, lift, lower
from tuxghost.trace import read

GOLDEN = Path(__file__).parent / "golden"

#: (name, step_count, lead_in, tail, action count) -- measured, not guessed.
COMMITTED = [
    ("claude_town_1234", 442, 0, 10, 18),
    ("walk_1234", 900, 30, 400, 40),
]


@pytest.mark.parametrize("name,step_count,lead_in,tail,n", COMMITTED)
def test_lift_recovers_the_measured_shape(
    name: str, step_count: int, lead_in: int, tail: int, n: int
) -> None:
    trace = read(GOLDEN / f"{name}.tuxghost")
    assert trace.header.step_count == step_count
    script = lift(trace.inputs, trace.header.step_count)
    assert script.lead_in == lead_in
    assert len(script.actions) == n
    assert script.actions[-1].settle == tail
    assert script.cost() == step_count


@pytest.mark.parametrize("name,step_count,lead_in,tail,n", COMMITTED)
def test_lower_lift_round_trips_to_the_same_schedule(
    name: str, step_count: int, lead_in: int, tail: int, n: int
) -> None:
    del step_count, lead_in, tail, n
    trace = read(GOLDEN / f"{name}.tuxghost")
    script = lift(trace.inputs, trace.header.step_count)
    assert lower(script) == _schedule_of(trace)


def test_an_input_free_trace_lifts_to_a_bare_wait() -> None:
    script = lift([], 300)
    assert script == ActionScript(lead_in=300, actions=())
    assert script.cost() == 300
    assert lower(script) == {}


def test_lower_places_the_first_press_at_lead_in() -> None:
    script = ActionScript(lead_in=30, actions=(Action(buttons.A, 10, 5),))
    assert lower(script) == {30: [(buttons.A, PRESSED)], 40: [(buttons.A, RELEASED)]}


def test_lower_keeps_a_zero_settle_collision_intact() -> None:
    """A `settle=0` action releases on exactly the step the next action
    presses. Both edges must survive -- the defect `add_edge` exists to
    prevent, re-checked from this side of the boundary."""
    script = ActionScript(
        lead_in=0,
        actions=(Action(buttons.DOWN, 4, 0), Action(buttons.RIGHT, 4, 0)),
    )
    assert lower(script)[4] == [
        (buttons.DOWN, RELEASED),
        (buttons.RIGHT, PRESSED),
    ]


def test_lift_refuses_a_button_left_held() -> None:
    with pytest.raises(ValueError, match="still held"):
        lift([(0, buttons.A, PRESSED)], 100)


def test_lift_refuses_a_release_without_a_press() -> None:
    with pytest.raises(ValueError, match="released without"):
        lift([(0, buttons.A, RELEASED)], 100)


def test_lift_refuses_a_double_press() -> None:
    with pytest.raises(ValueError, match="already held"):
        lift([(0, buttons.A, PRESSED), (2, buttons.A, PRESSED)], 100)


def test_lift_refuses_overlapping_holds() -> None:
    """Two DIFFERENT buttons held at once cannot be expressed as a
    sequential action list. Refused loudly rather than reinterpreted --
    neither committed trace has one, but a human-recorded trace could."""
    inputs = [
        (0, buttons.DOWN, PRESSED),
        (4, buttons.A, PRESSED),
        (8, buttons.DOWN, RELEASED),
        (12, buttons.A, RELEASED),
    ]
    with pytest.raises(ValueError, match="overlap"):
        lift(inputs, 100)


def test_lift_refuses_a_value_that_is_neither_press_nor_release() -> None:
    with pytest.raises(ValueError, match="neither"):
        lift([(0, buttons.A, 0.5)], 100)


def test_lift_refuses_a_step_count_shorter_than_the_last_release() -> None:
    with pytest.raises(ValueError, match="step_count"):
        lift([(0, buttons.A, PRESSED), (50, buttons.A, RELEASED)], 20)
