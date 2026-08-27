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

Each variant carries an `op` DISCRIMINATOR and serializes through
`to_json`, never through bare `dataclasses.asdict` (whole-branch review,
Important 2). `asdict` recurses by field and the field sets of `Insert`
and `Replace` are identical, so an `asdict`-logged round rendered
`Insert(0, Action(2, 8, 4))` and `Replace(0, Action(2, 8, 4))` as the
same object and fed back through
`tuxghost.optimize.editors.replay.edits_from_json` raised
`ValueError: edit 0: unknown op None` (measured). The spec calls
`ReplayEditor` "what makes an LLM-driven optimization auditable after the
fact", and that claim is only true if what `optimize.jsonl` WRITES is
what `--edits FILE` can READ -- so `to_json`'s output shape is exactly
`edits_from_json`'s accepted shape, pinned by
`tests/test_optimize_edits.py` and, end to end, by
`tests/test_optimize_cli.py`'s round-trip assertion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from tuxghost.agent.types import Action, validate_actions
from tuxghost.optimize.schedule import ActionScript


def _action_json(action: Action) -> dict[str, int]:
    return {
        "button": action.button,
        "hold": action.hold,
        "settle": action.settle,
    }


@dataclass(frozen=True)
class Insert:
    #: A `ClassVar`, so it is NOT a dataclass field: `__match_args__`
    #: stays `("index", "action")` and the positional `case Insert(index,
    #: action)` patterns in `apply_edits` below are unaffected.
    op: ClassVar[str] = "insert"
    index: int
    action: Action

    def to_json(self) -> dict[str, Any]:
        return {"op": self.op, "index": self.index,
                "action": _action_json(self.action)}


@dataclass(frozen=True)
class Delete:
    op: ClassVar[str] = "delete"
    index: int

    def to_json(self) -> dict[str, Any]:
        return {"op": self.op, "index": self.index}


@dataclass(frozen=True)
class Replace:
    op: ClassVar[str] = "replace"
    index: int
    action: Action

    def to_json(self) -> dict[str, Any]:
        return {"op": self.op, "index": self.index,
                "action": _action_json(self.action)}


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
