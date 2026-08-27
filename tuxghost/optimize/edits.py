"""What an editor is allowed to say.

Three operations over an `ActionScript`'s action list. `lead_in` is NOT
editable in v1 (see the spec's Open questions): it is preserved exactly,
so a run whose only problem is starting too early cannot be fixed here.

Indices are interpreted SEQUENTIALLY -- each edit sees the list as the
previous edit in the same batch left it. `[Delete(0), Delete(0)]`
therefore removes the first two actions. The alternative (every index
against the original list, applied atomically) is equally defensible;
this one composes trivially and an editor that wants the other semantics
can propose one edit per round.

Every refusal is a `ValueError`, matching `validate_actions`, so
`tuxghost.optimize.runner` can catch exactly `(ValueError, TypeError)` --
the same narrow boundary `tuxghost.cli._agent` uses -- and treat a bad
proposal as a rejected round rather than a crash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from tuxghost.agent.types import Action, validate_actions
from tuxghost.optimize.schedule import ActionScript


@dataclass(frozen=True)
class Insert:
    index: int
    action: Action


@dataclass(frozen=True)
class Delete:
    index: int


@dataclass(frozen=True)
class Replace:
    index: int
    action: Action


Edit = Insert | Delete | Replace


def _check_index(index: int, length: int, *, insert: bool) -> None:
    limit = length if insert else length - 1
    if index < 0 or index > limit:
        raise ValueError(
            f"index {index} is out of range for {length} action(s) "
            f"(valid: 0..{max(limit, 0)}); negative indices are refused "
            "rather than wrapping to the end"
        )


def apply_edits(
    script: ActionScript, edits: Sequence[Edit]
) -> ActionScript:
    """Apply `edits` in order and revalidate the result.

    Raises `ValueError` for an out-of-range index or an action
    `validate_actions` rejects. Returns a new `ActionScript`; the input is
    frozen and untouched.
    """
    actions = list(script.actions)
    for edit in edits:
        match edit:
            case Insert(index, action):
                _check_index(index, len(actions), insert=True)
                actions.insert(index, action)
            case Delete(index):
                _check_index(index, len(actions), insert=False)
                del actions[index]
            case Replace(index, action):
                _check_index(index, len(actions), insert=False)
                actions[index] = action
    return replace(script, actions=validate_actions(actions))
