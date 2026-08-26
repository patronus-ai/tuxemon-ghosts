"""A policy is untrusted input -- a model can and will emit nonsense. Every
rejection here is a boundary check, per this project's standing preference
for validation over convenience."""

from __future__ import annotations

import pytest
from tuxemon.platform.const import buttons

from tuxghost.agent.types import (
    HOLD_CAP,
    SETTLE_CAP,
    Action,
    validate_actions,
)


def test_valid_actions_pass_through_unchanged() -> None:
    actions = (Action(buttons.DOWN, hold=12, settle=6),)
    assert validate_actions(actions) == actions


def test_empty_means_stop_and_is_not_an_error() -> None:
    assert validate_actions(()) == ()


@pytest.mark.parametrize(
    "action, message",
    [
        (Action(buttons.A, hold=0, settle=1), "hold"),
        (Action(buttons.A, hold=-1, settle=1), "hold"),
        (Action(buttons.A, hold=HOLD_CAP + 1, settle=1), "hold"),
        (Action(buttons.A, hold=1, settle=-1), "settle"),
        (Action(buttons.A, hold=1, settle=SETTLE_CAP + 1), "settle"),
        (Action(999_999, hold=1, settle=1), "button"),
    ],
)
def test_invalid_actions_are_rejected_at_the_boundary(
    action: Action, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_actions((action,))


def test_a_non_action_in_the_sequence_is_rejected() -> None:
    """A model's JSON decodes to dicts, so `validate_actions` must reject a
    dict that merely LOOKS like an Action rather than duck-typing it."""
    with pytest.raises(TypeError, match="Action"):
        # The following line deliberately passes the wrong type, which is the
        # test itself. Passing a dict instead of an Action is correct usage here.
        # Mypy is correct to object at the call site, so we suppress it.
        validate_actions(({"button": 64, "hold": 1, "settle": 1},))  # type: ignore[arg-type]
