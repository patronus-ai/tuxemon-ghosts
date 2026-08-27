"""Task 4: the edit vocabulary.

Edits are applied SEQUENTIALLY -- each index sees the list as the previous
edit left it. Pinned here because the alternative (all indices against the
original list) is equally defensible and silently different, and because
an editor author reading only `apply_edits`'s signature could reasonably
assume either.
"""

from __future__ import annotations

import pytest
from tuxemon.platform.const import buttons

from tuxghost.agent.types import Action
from tuxghost.optimize.edits import Delete, Insert, Replace, apply_edits
from tuxghost.optimize.schedule import ActionScript

A = Action(buttons.A, 10, 5)
DOWN = Action(buttons.DOWN, 16, 8)
UP = Action(buttons.UP, 16, 8)


def _script(*actions: Action) -> ActionScript:
    return ActionScript(lead_in=0, actions=tuple(actions))


def test_replace_swaps_one_action() -> None:
    out = apply_edits(_script(A, DOWN), [Replace(1, UP)])
    assert out.actions == (A, UP)


def test_insert_puts_the_action_before_the_index() -> None:
    out = apply_edits(_script(A, DOWN), [Insert(1, UP)])
    assert out.actions == (A, UP, DOWN)


def test_insert_at_the_end_appends() -> None:
    out = apply_edits(_script(A), [Insert(1, UP)])
    assert out.actions == (A, UP)


def test_delete_removes_one_action() -> None:
    out = apply_edits(_script(A, DOWN, UP), [Delete(1)])
    assert out.actions == (A, UP)


def test_edits_apply_sequentially_not_against_the_original() -> None:
    """Two `Delete(0)`s remove the FIRST TWO actions, because the second
    one sees the list the first one left. The documented semantics."""
    out = apply_edits(_script(A, DOWN, UP), [Delete(0), Delete(0)])
    assert out.actions == (UP,)


def test_lead_in_is_preserved_and_never_edited() -> None:
    out = apply_edits(ActionScript(lead_in=30, actions=(A,)), [Replace(0, UP)])
    assert out.lead_in == 30


def test_an_empty_edit_list_returns_an_equal_script() -> None:
    script = ActionScript(lead_in=7, actions=(A, DOWN))
    assert apply_edits(script, []) == script


def test_out_of_range_index_is_refused() -> None:
    with pytest.raises(ValueError, match="index 5"):
        apply_edits(_script(A), [Replace(5, UP)])


def test_insert_beyond_the_end_is_refused() -> None:
    """`Insert(len)` appends; `Insert(len + 1)` is a hole, not an append."""
    with pytest.raises(ValueError, match="index 3"):
        apply_edits(_script(A), [Insert(3, UP)])


def test_negative_index_is_refused_rather_than_wrapping() -> None:
    """Python would happily index from the end. An editor that emitted -1
    meaning "the last one" and got it is an editor whose next -1 means
    something else."""
    with pytest.raises(ValueError, match="index -1"):
        apply_edits(_script(A), [Delete(-1)])


def test_an_out_of_range_action_is_refused_by_validate_actions() -> None:
    with pytest.raises(ValueError, match="hold"):
        apply_edits(_script(A), [Replace(0, Action(buttons.A, 10_000, 0))])


def test_an_invalid_button_is_refused() -> None:
    with pytest.raises(ValueError, match="button"):
        apply_edits(_script(A), [Insert(0, Action(999_999, 10, 0))])


def test_deleting_every_action_is_allowed() -> None:
    """A script with no actions is legal -- it is a bare wait -- so an
    editor is allowed to reach it. `seal` still runs `lead_in` steps."""
    out = apply_edits(ActionScript(lead_in=5, actions=(A,)), [Delete(0)])
    assert out == ActionScript(lead_in=5, actions=())
