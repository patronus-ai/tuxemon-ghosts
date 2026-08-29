"""tests/test_rules_seal.py -- seal's per-step rules observer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tuxghost.boot import boot_from_save
from tuxghost.optimize.editors.scripted import ScriptedEditor
from tuxghost.optimize.objective import RulesObjective
from tuxghost.optimize.runner import optimize
from tuxghost.optimize.schedule import lift
from tuxghost.optimize.seal import CandidateResult, seal
from tuxghost.rules import CRITICAL_PATH, Observation, TuxemonFirstBattleRules
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


# --- Task 4: RulesObjective, and threading `rules` through `optimize` ---


def _candidate(
    steps: int,
    max_progress: int,
    goal_step: int | None = None,
    died: bool = False,
) -> CandidateResult:
    trace: Any = None  # never read by an objective
    return CandidateResult(
        trace=trace, steps=steps, final_state={},
        max_progress=max_progress, goal_step=goal_step, died=died,
    )


def test_reaching_the_goal_beats_any_amount_of_progress() -> None:
    obj = RulesObjective()
    reached = obj.score(_candidate(999, 1, goal_step=10))
    unreached = obj.score(_candidate(10, 10**12))
    assert reached > unreached


def test_among_goal_reaching_candidates_fewer_steps_wins() -> None:
    obj = RulesObjective()
    assert obj.score(_candidate(100, 5, goal_step=1)) > obj.score(
        _candidate(200, 5, goal_step=1))


def test_among_non_goal_candidates_higher_progress_wins() -> None:
    """Ours must give a gradient BEFORE the goal is reachable. Their loop
    rejects every non-goal candidate outright, which it can afford
    because it seeds from a gold that already clears the level; our
    parent does not, so progress is the only signal the search has."""
    obj = RulesObjective()
    assert obj.score(_candidate(100, 50)) > obj.score(_candidate(100, 10))


def test_a_dead_candidate_scores_below_everything() -> None:
    obj = RulesObjective()
    dead = obj.score(_candidate(1, 10**12, goal_step=1, died=True))
    assert dead < obj.score(_candidate(10**6, 0))


def test_the_score_length_is_constant() -> None:
    """`runner` refuses a varying tuple length -- a real check at
    runner.py:286."""
    obj = RulesObjective()
    a = obj.score(_candidate(1, 1))
    b = obj.score(_candidate(2, 2, goal_step=1, died=True))
    assert len(a) == len(b) == 3


def test_round_zero_carries_rules_max_progress_through_optimize() -> None:
    """`optimize`'s round 0 seals the PARENT baseline, at runner.py:216 --
    a call site distinct from every candidate's seal at runner.py:158.
    If `rules` reaches only the second, the parent scores
    `max_progress=0` against candidates with real progress and round 0's
    score is not comparable to anything the search produces. This
    exercises the real path end to end (not a stubbed `seal`), so it
    would also fail if `rules` were dropped in `CandidateResult`
    plumbing rather than just in the call site.
    """
    parent = read(PARENT)
    result = optimize(
        parent,
        ScriptedEditor([]),  # STOP at round 1: only round 0 boots.
        RulesObjective(),
        rounds=1,
        patience=1,
        max_rejections=1,
        max_cost=10_000,
        rules=TuxemonFirstBattleRules(),
    )
    assert result.rounds[0].index == 0
    assert result.rounds[0].accepted is True
    assert result.best_round == 0
    assert result.best.max_progress > 0


# --- Whole-branch review fix wave: I1 (per-run vs per-instance leak) ---


def test_a_shared_rules_instance_does_not_leak_furthest_map_across_seals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I1, THE REAL BUG. `tuxghost.optimize.runner.optimize` threads ONE
    shared `rules` object into every `seal()` call (round 0 and every
    candidate via `_prepare`). `TuxemonRules._map_index` tracks its high
    water mark on `self._furthest_map` -- INSTANCE state -- so without a
    per-run reset, candidate B (fresh, only ever at `CRITICAL_PATH[0]`)
    inherits candidate A's `CRITICAL_PATH[2]` floor when the same `rules`
    object seals both.

    `session.client.get_map_name` is monkeypatched via a wrapped
    `boot_from_save` (rather than a fake session/client) so this exercises
    the REAL `seal()` call path end to end -- the exact site the bug
    lives at -- not just `TuxemonRules` in isolation.
    """
    import tuxghost.optimize.seal as seal_mod

    parent = read(PARENT)
    script = lift(parent.inputs, parent.header.step_count)
    rules = TuxemonFirstBattleRules()

    current_map = [CRITICAL_PATH[0]]
    real_boot = boot_from_save

    def wrapped_boot(*args: Any, **kwargs: Any) -> Any:
        client, session = real_boot(*args, **kwargs)
        client.get_map_name = lambda: current_map[0]
        return client, session

    monkeypatch.setattr(seal_mod, "boot_from_save", wrapped_boot)

    # Candidate A genuinely reaches CRITICAL_PATH[2].
    current_map[0] = CRITICAL_PATH[2]
    result_a = seal(script, parent, rules=rules)
    assert result_a.max_progress >= 2_000, result_a.max_progress

    # Candidate B is fresh, at CRITICAL_PATH[0] for its entire run, and
    # uses the SAME `rules` instance as candidate A.
    current_map[0] = CRITICAL_PATH[0]
    result_b = seal(script, parent, rules=rules)

    assert result_b.max_progress < result_a.max_progress, (
        "candidate B's own max_progress leaked candidate A's furthest "
        f"map from the shared rules instance: A={result_a.max_progress} "
        f"B={result_b.max_progress}"
    )
    # CRITICAL_PATH[0]'s own index contributes 0 to the map term, so B's
    # progress is party_count * 10 only -- the map term must be exactly
    # absent, not merely lower than A's.
    party_term = result_b.max_progress
    assert party_term % 1_000 == party_term, (
        "candidate B's progress still carries a nonzero map term"
    )


# --- Whole-branch review fix wave: I2 (the vacuity trap) ---


def test_optimize_refuses_rules_objective_without_rules() -> None:
    """I2. `RulesObjective` reads `CandidateResult.max_progress`/
    `.goal_step`/`.died`, which are populated ONLY when `seal` is given
    `rules=...` -- both call sites in `runner.optimize` (round 0 and every
    candidate). Combining `RulesObjective` with no `rules` silently scores
    every candidate `(0.0, 0.0, -steps)`, degenerating the search into
    "shortest wins" with no error. `optimize` must refuse this combination
    outright, before booting anything.
    """
    parent = read(PARENT)
    with pytest.raises(ValueError, match="RulesObjective"):
        optimize(
            parent,
            ScriptedEditor([]),
            RulesObjective(),
            rounds=1,
            patience=1,
            max_rejections=1,
            max_cost=10_000,
        )
