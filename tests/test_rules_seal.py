"""tests/test_rules_seal.py -- seal's per-step rules observer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tuxghost.optimize.schedule import lift
from tuxghost.optimize.seal import seal
from tuxghost.rules import Observation, TuxemonFirstBattleRules
from tuxghost.trace import read

PARENT = Path(__file__).parent / "golden" / "scripted_town_1234.tuxghost"


class _AlwaysAtGoal:
    """Test-local `GameRules` stub: `at_goal` is True from step 0 onward.

    Exists to pin `goal_step`'s FIRST-hit latching against a specific
    real regression: overwriting `goal_step = i` on every `at_goal` step
    (dropping the `goal_step is None` guard) reports the run's LAST step
    instead of its first, which is exactly backwards for a field whose
    whole purpose is "when did the goal first become true". See the
    task-3 report for the mutate/observe/restore demonstration.
    """

    def observe(self, session: Any, step: int) -> Observation:
        del session
        return {"progress": step, "at_goal": True, "dead": False}


def test_seal_records_max_progress_when_given_rules() -> None:
    parent = read(PARENT)
    script = lift(parent.inputs, parent.header.step_count)
    result = seal(script, parent, rules=TuxemonFirstBattleRules())
    assert result.max_progress > 0
    assert result.goal_step is None      # this parent wins no battle
    assert result.died is False


def test_seal_without_rules_is_unchanged() -> None:
    """The observer is OPTIONAL. Every existing caller passes no rules and
    must keep working, with the new fields at their defaults."""
    parent = read(PARENT)
    script = lift(parent.inputs, parent.header.step_count)
    result = seal(script, parent)
    assert result.max_progress == 0
    assert result.goal_step is None
    assert result.died is False
    assert result.trace.header.final_digest == parent.header.final_digest


def test_the_observer_never_serializes_the_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE COST CONSTRAINT, pinned structurally rather than by timing.

    `observe` runs every step. If it called `state_of` it would serialize
    the entire save (party, items, monsters) per step. The S4 plan made
    exactly this mistake. A timing assertion would be flaky; making
    `state_of` raise is not.

    `seal` legitimately calls `state_of` ONCE at the end for
    `final_state`, so the patch is installed only for the stepping
    window -- see the module under test for where that boundary is.

    NOTE: the monkeypatch below is INERT as a check on `rules.py` --
    `rules.py` never imports `state_of` by name, so patching this
    attribute on `tuxghost.rules` pins nothing there (it guards only a
    hypothetical future `from tuxghost.digest import state_of` in that
    module). The AST scan below is the assertion that actually
    discriminates -- it checks for real `state_of` REFERENCES (a `Name`
    or `Attribute` node), not a plain substring scan: `rules.py`'s own
    module docstring mentions `state_of` in prose (explaining exactly
    this constraint), so a naive `"state_of" not in inspect.getsource(...)`
    check -- verified against the real file -- false-fails on correct
    code. Parsing the AST and checking for name/attribute references
    instead ignores string literals (docstrings, comments) and only
    catches an actual `state_of(...)` call or `import ... state_of`.
    """
    import ast

    import tuxghost.rules as rules_mod

    def boom(*a: Any, **k: Any) -> Any:  # pragma: no cover - must never be reached
        raise AssertionError("observe() called state_of(); it must not")

    monkeypatch.setattr(rules_mod, "state_of", boom, raising=False)

    parent = read(PARENT)
    script = lift(parent.inputs, parent.header.step_count)
    result = seal(script, parent, rules=TuxemonFirstBattleRules())
    assert result.max_progress > 0

    import inspect

    tree = ast.parse(inspect.getsource(rules_mod))
    referenced = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "state_of" not in referenced, (
        "rules.py must not reference state_of in actual code"
    )


def test_goal_step_latches_the_first_at_goal_step_not_the_last() -> None:
    """`goal_step` must latch the FIRST step `at_goal` was true.

    Regression for the exact bug this task's brief calls out: dropping
    the `goal_step is None` guard and overwriting on every `at_goal`
    step reports the run's LAST step instead of its first. See the
    task-3 report for the real mutate/fail/restore/pass demonstration.
    """
    parent = read(PARENT)
    script = lift(parent.inputs, parent.header.step_count)
    result = seal(script, parent, rules=_AlwaysAtGoal())
    assert result.goal_step == 0
