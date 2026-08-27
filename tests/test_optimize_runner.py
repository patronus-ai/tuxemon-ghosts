"""Task 7: the edit -> seal -> score loop.

Round 0 is the PARENT, sealed and scored, so "did anything improve" is
answered against a measured baseline rather than against the parent's own
recorded header.

Improvement is asserted on the round-by-round score SEQUENCE, never on
the final best alone: a settled tile is an attractor in this game, so an
endpoint comparison cannot distinguish a run that improved from one that
wandered and came back.

WHAT "IMPROVING" MEANS AGAINST THIS PARENT (measured 2026-08-27, and the
reason the plan's own edits are inverted here). The live capture ends at
tile (11, 14) on `spyder_paper_town.tmx` after 442 steps with
`state_stack == ['DialogState', 'WorldState']`, and that trailing dialog
swallows every input: one appended `UP`/`DOWN`/`LEFT`/`RIGHT`, and `A`
then `DOWN`, and `B` then `DOWN`, ALL leave the tile at (11, 14). So no
append-only edit to this parent can move the player, and every accepted
round in this module is accepted on `ReachTile`'s `steps` term.
`ReachTile`'s `distance` term is NOT frozen -- across the nine-run tuning
sweep it varied from -2 (the parent's own) down to -9, from mid-script
edits that change the route before the dialogue opens -- but no candidate
ever scored BETTER than the parent on it, so it has never been the term
that carried an acceptance. See docs/STATUS.org, "The consequence, and it
is a limit not a feature", which measured the earlier "frozen" reading
false.

That makes the improving edit a DELETE (442 -> 416 -> 390 -> 364 steps,
same tile) and the worsening edit an APPEND (442 -> 642 -> 842). Fewer
steps to the same place is a real improvement on a real term of the
objective; the target is NOT moved to (11, 14) to make the distance term
trivially satisfied, which would pass while proving nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from tuxemon.platform.const import buttons

from tuxghost.agent.types import Action
from tuxghost.optimize.editors.scripted import ScriptedEditor
from tuxghost.optimize.edits import Delete, Edit, Insert, Replace
from tuxghost.optimize.objective import ReachTile
from tuxghost.optimize.runner import optimize
from tuxghost.optimize.schedule import ActionScript, lift
from tuxghost.optimize.seal import CandidateResult, OverBudget
from tuxghost.trace import Trace, read

LIVE = Path(__file__).parent / "golden" / "claude_town_1234.tuxghost"
#: The live capture ends at tile (11, 14) on this map, immovably (see the
#: module docstring). The target is left two tiles away so the distance
#: term is a real, unsatisfied term rather than one relocated onto the
#: parent's own end position.
TARGET = ReachTile("spyder_paper_town.tmx", (11, 16))
#: A worsening append: +200 steps (hold 100 + settle 100), same tile.
BIG = Action(buttons.DOWN, 100, 100)
DOWN = Action(buttons.DOWN, 16, 8)


def _parent() -> Trace:
    return read(LIVE)


def _script_of(parent: Trace) -> ActionScript:
    return lift(parent.inputs, parent.header.step_count)


def _n_actions(parent: Trace) -> int:
    return len(_script_of(parent).actions)


def _delete_last(n: int, times: int) -> list[list[Delete]]:
    """`times` rounds of "delete the last action", against a script that
    shrinks by one each accepted round."""
    return [[Delete(n - 1 - i)] for i in range(times)]


def test_round_zero_is_the_parent_scored() -> None:
    result = optimize(
        _parent(),
        ScriptedEditor([]),
        TARGET,
        rounds=1,
        patience=1,
        max_rejections=1,
        max_cost=10_000,
    )
    assert result.rounds[0].index == 0
    assert result.rounds[0].edits == ()
    assert result.rounds[0].accepted is True
    assert result.best.trace.header.final_digest == read(LIVE).header.final_digest
    assert result.best_round == 0


def test_an_improving_edit_is_accepted_and_becomes_the_new_best() -> None:
    """Dropping trailing actions reaches the same tile in fewer steps."""
    parent = _parent()
    editor = ScriptedEditor(_delete_last(_n_actions(parent), 2))
    result = optimize(
        parent,
        editor,
        TARGET,
        rounds=3,
        patience=3,
        max_rejections=1,
        max_cost=10_000,
    )
    scores = [r.score for r in result.rounds if r.score is not None]
    assert scores == sorted(scores), f"score sequence went backwards: {scores}"
    assert scores[-1] > scores[0]
    assert result.best_round > 0
    # The improvement is on the `steps` term specifically -- the tile is
    # immovable against this parent, so a distance-term improvement here
    # would mean the measurement above is wrong, not that the loop works.
    assert [r.steps for r in result.rounds if r.steps is not None] == [442, 416, 390]


def test_a_worsening_edit_is_kept_out_of_best_but_recorded() -> None:
    parent = _parent()
    n = _n_actions(parent)
    # `best_script` never advances (both candidates lose), so both rounds
    # append at the same index.
    editor = ScriptedEditor([[Insert(n, BIG)], [Insert(n, BIG)]])
    result = optimize(
        parent,
        editor,
        TARGET,
        # `rounds=2`, not the plan's 3: with 3 the editor runs out and a
        # third STOP round (score None) is appended, which the plan's own
        # `all(r.score is not None ...)` assertion would then fail on.
        rounds=2,
        patience=3,
        max_rejections=1,
        max_cost=10_000,
    )
    assert result.best_round == 0
    assert [r.accepted for r in result.rounds] == [True, False, False]
    assert all(r.score is not None for r in result.rounds)
    assert [r.steps for r in result.rounds] == [442, 642, 642]


def test_an_empty_proposal_stops_the_run() -> None:
    parent = _parent()
    editor = ScriptedEditor([[Delete(_n_actions(parent) - 1)], []])
    result = optimize(
        parent,
        editor,
        TARGET,
        rounds=10,
        patience=10,
        max_rejections=1,
        max_cost=10_000,
    )
    assert result.stop_reason == "editor returned STOP"
    # round 0 (parent) + round 1 (the accepted edit) + round 2 (the STOP,
    # which IS appended before the break so the log records why it ended)
    assert len(result.rounds) == 3
    assert result.rounds[-1].index == 2
    assert result.rounds[-1].rejected_reason == "STOP"


def test_patience_stops_a_run_that_stops_improving() -> None:
    parent = _parent()
    editor = ScriptedEditor([[Insert(_n_actions(parent), BIG)]] * 10)
    result = optimize(
        parent,
        editor,
        TARGET,
        rounds=10,
        patience=2,
        max_rejections=10,
        max_cost=10_000,
    )
    assert result.stop_reason == "patience exhausted"
    assert len(result.rounds) == 3  # round 0 + two non-improving


def test_the_rounds_cap_stops_a_run_that_keeps_improving() -> None:
    parent = _parent()
    editor = ScriptedEditor(_delete_last(_n_actions(parent), 10))
    result = optimize(
        parent,
        editor,
        TARGET,
        rounds=2,
        patience=10,
        max_rejections=1,
        max_cost=10_000,
    )
    assert result.stop_reason == "rounds exhausted"
    assert len(result.rounds) == 3  # round 0 + 2
    assert [r.accepted for r in result.rounds] == [True, True, True]


def test_an_invalid_proposal_is_a_rejected_round_not_a_crash() -> None:
    parent = _parent()
    editor = ScriptedEditor(
        [[Replace(999, DOWN)], [Delete(_n_actions(parent) - 1)]]
    )
    result = optimize(
        parent,
        editor,
        TARGET,
        rounds=3,
        patience=3,
        max_rejections=2,
        max_cost=10_000,
    )
    assert result.rounds[1].accepted is False
    assert result.rounds[1].score is None
    assert "index 999" in (result.rounds[1].rejected_reason or "")
    assert result.rounds[2].accepted is True  # the run continued


def test_consecutive_rejections_stop_the_run() -> None:
    result = optimize(
        _parent(),
        ScriptedEditor([[Replace(999, DOWN)]] * 5),
        TARGET,
        rounds=5,
        patience=5,
        max_rejections=2,
        max_cost=10_000,
    )
    assert result.stop_reason == "editor rejected 2 rounds consecutively"


def test_an_over_budget_proposal_is_rejected_rather_than_run() -> None:
    """`max_cost` is the guard against an editor describing a
    12,000,000-step candidate: HOLD_CAP and SETTLE_CAP are 600 each, so a
    few thousand inserted actions is hours of engine time in one call."""
    editor = ScriptedEditor([[Insert(18, Action(buttons.DOWN, 600, 600))]])
    result = optimize(
        _parent(),
        editor,
        TARGET,
        rounds=1,
        patience=1,
        max_rejections=1,
        max_cost=500,
    )
    assert result.rounds[1].accepted is False
    assert "max_cost" in (result.rounds[1].rejected_reason or "")


def test_an_unrunnable_parent_propagates_not_a_rejected_round() -> None:
    """Ruling F11. A malformed `parent.initial_state` makes `seal` raise a
    `ValueError` (`pydantic.ValidationError` IS one) on EVERY round. A
    single broad `except ValueError` around `seal` would launder that into
    a rejected round and report the run as "the editor kept proposing
    rubbish" -- so it must propagate instead. Round 0 seals the parent
    before the loop starts, which is where this surfaces.
    """
    parent = _parent()
    broken = parent.model_copy(
        update={"initial_state": {"npc_state": "nonsense"}}
    )
    editor = ScriptedEditor([[Delete(_n_actions(parent) - 1)]])

    with pytest.raises(ValueError) as excinfo:
        optimize(
            broken,
            editor,
            TARGET,
            rounds=2,
            patience=2,
            max_rejections=1,
            max_cost=10_000,
        )
    assert not isinstance(excinfo.value, OverBudget)
    assert "max_cost" not in str(excinfo.value)


def test_a_seal_failure_that_is_not_over_budget_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruling F11, the in-loop half. `seal` also raises a plain
    `ValueError` from `boot_from_save` when `current_map` cannot be
    resolved. Inside the loop that must propagate, not become a rejected
    round -- only `OverBudget` is a statement about the proposal.

    `seal` is substituted rather than a broken parent constructed,
    because a broken parent fails at round 0 (above) and so never reaches
    the loop's `except` at all.
    """
    parent = _parent()
    stub = CandidateResult(
        trace=parent,
        steps=parent.header.step_count,
        final_state={"map": "spyder_paper_town.tmx", "tile_pos": [11, 14]},
    )
    calls = {"n": 0}

    def fake_seal(
        candidate_script: ActionScript, from_trace: Trace, **kwargs: object
    ) -> CandidateResult:
        del candidate_script, from_trace, kwargs
        calls["n"] += 1
        if calls["n"] == 1:
            return stub
        raise ValueError("could not resolve current_map 'nowhere.tmx'")

    monkeypatch.setattr("tuxghost.optimize.runner.seal", fake_seal)

    with pytest.raises(ValueError, match="current_map"):
        optimize(
            parent,
            ScriptedEditor([[Delete(_n_actions(parent) - 1)]]),
            TARGET,
            rounds=2,
            patience=2,
            max_rejections=1,
            max_cost=10_000,
        )
    assert calls["n"] == 2


def test_an_objective_with_an_unstable_score_length_is_refused() -> None:
    class Wobbly:
        def __init__(self) -> None:
            self.n = 0

        def score(self, candidate: object) -> tuple[float, ...]:
            del candidate
            self.n += 1
            return tuple(0.0 for _ in range(self.n))

    parent = _parent()
    with pytest.raises(ValueError, match="score length"):
        optimize(
            parent,
            ScriptedEditor([[Delete(_n_actions(parent) - 1)]]),
            Wobbly(),
            rounds=2,
            patience=2,
            max_rejections=1,
            max_cost=10_000,
        )


def test_round_zero_is_sealed_without_max_cost() -> None:
    """`max_cost` bounds what an EDITOR may describe; the parent is the
    baseline, not a proposal.

    Review round 1, Important 1: every other `max_cost` in this file
    (10_000, 500, 0) sits either above the parent's 442-step cost or
    below the `>= 1` bound check, so passing `max_cost` through at round
    0 would have left the whole suite green and a reasoned decision
    deletable by a "consistency" cleanup. `max_cost=100` is below 442, so
    it separates the two: round 0 must still be sealed and SCORED, while
    round 1's 416-step candidate is refused.
    """
    parent = _parent()
    result = optimize(
        parent,
        ScriptedEditor(_delete_last(_n_actions(parent), 2)),
        TARGET,
        rounds=2,
        patience=2,
        max_rejections=1,
        max_cost=100,
    )
    assert result.rounds[0].score is not None
    assert result.rounds[0].steps == 442  # the parent ran in full
    assert result.rounds[0].accepted is True
    assert result.rounds[1].accepted is False
    assert result.rounds[1].score is None
    assert "max_cost" in (result.rounds[1].rejected_reason or "")
    assert result.best_round == 0


class _BrokenThenGoodEditor:
    """Raises on its first proposal, then proposes a real improvement.

    Stands in for Task 10's `ClaudeEditor`, whose `parse_response` raises
    `ValueError` when a model reply carries no JSON fence -- and
    `json.JSONDecodeError` is itself a `ValueError` subclass.
    """

    def __init__(self, edit: Edit) -> None:
        self._edit = edit
        self.calls = 0

    def propose(
        self,
        script: ActionScript,
        candidate: CandidateResult,
        score: tuple[float, ...],
    ) -> Sequence[Edit]:
        del script, candidate, score
        self.calls += 1
        if self.calls == 1:
            raise ValueError("no JSON fence in the model's reply")
        return [self._edit]


def test_a_propose_failure_is_a_rejected_round_not_the_end_of_the_run() -> None:
    """Review round 1, Important 2. The spec makes containment a
    requirement: "An editor is untrusted input. A proposal that fails
    validation is a rejected round ... one malformed answer must not kill
    a 20-round run." `propose` was outside every boundary, so it did.

    A raised proposal must NOT be routed through the STOP path either: it
    counts toward `max_rejections`/`patience` and the loop continues.
    """
    parent = _parent()
    editor = _BrokenThenGoodEditor(Delete(_n_actions(parent) - 1))
    result = optimize(
        parent,
        editor,
        TARGET,
        rounds=2,
        patience=2,
        max_rejections=2,
        max_cost=10_000,
    )
    assert editor.calls == 2
    assert result.rounds[1].accepted is False
    assert result.rounds[1].score is None
    assert result.rounds[1].edits == ()
    reason = result.rounds[1].rejected_reason or ""
    assert "no JSON fence" in reason
    assert "propose" in reason
    # NOT the STOP path: the run survived and the next round was scored.
    assert reason != "STOP"
    assert result.stop_reason == "rounds exhausted"
    assert len(result.rounds) == 3
    assert result.rounds[2].accepted is True
    assert result.best_round == 2


def test_an_editor_raising_a_programming_error_still_crashes() -> None:
    """The boundary is `(ValueError, TypeError)`, not `Exception`. An
    `AttributeError` from a buggy editor is a programming error; a tidy
    rejected round would hide it for the whole run."""

    class Buggy:
        def propose(
            self,
            script: ActionScript,
            candidate: CandidateResult,
            score: tuple[float, ...],
        ) -> Sequence[Edit]:
            del script, candidate, score
            raise AttributeError(
                "'NoneType' object has no attribute 'actions'"
            )

    with pytest.raises(AttributeError):
        optimize(
            _parent(),
            Buggy(),
            TARGET,
            rounds=2,
            patience=2,
            max_rejections=2,
            max_cost=10_000,
        )


def test_rounds_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="rounds"):
        optimize(
            _parent(),
            ScriptedEditor([]),
            TARGET,
            rounds=0,
            patience=1,
            max_rejections=1,
            max_cost=10_000,
        )


@pytest.mark.parametrize(
    ("name", "patience", "max_rejections", "max_cost"),
    [
        ("patience", 0, 1, 10_000),
        ("max_rejections", 1, 0, 10_000),
        ("max_cost", 1, 1, 0),
    ],
)
def test_the_other_bounds_below_one_are_refused(
    name: str, patience: int, max_rejections: int, max_cost: int
) -> None:
    """The three bounds `optimize` checks alongside `rounds`. Refused
    before anything boots, so a caller that passes `patience=0` gets a
    message rather than a run that stops after round 1 for no stated
    reason."""
    with pytest.raises(ValueError, match=name):
        optimize(
            _parent(),
            ScriptedEditor([]),
            TARGET,
            rounds=1,
            patience=patience,
            max_rejections=max_rejections,
            max_cost=max_cost,
        )


def test_a_negative_checkpoint_is_refused() -> None:
    """Whole-branch review, Also-fix 1. `optimize()` validated its other
    four bounds and not this one, and the failure is silent rather than
    loud: `seal`'s hook tests `i % checkpoint == 0`, and `i % -1` is 0
    for EVERY step, so `checkpoint=-1` took a `digest_of` AND a
    `state_of` snapshot on all 442 steps of every round without saying
    so. The CLI guarded it; a library caller had nothing. Refused before
    the engine boots, so this test costs no engine time.

    `>= 0`, not `>= 1`: 0 is the legal "do not sample" value, pinned by
    `test_checkpoint_zero_samples_nothing`-style coverage in
    `tests/test_optimize_seal.py`.
    """
    with pytest.raises(ValueError, match="checkpoint"):
        optimize(
            _parent(),
            ScriptedEditor([]),
            TARGET,
            rounds=1,
            patience=1,
            max_rejections=1,
            max_cost=10_000,
            checkpoint=-1,
        )


def test_checkpoint_and_model_reach_seal() -> None:
    """Whole-branch review, Also-fix 3: both were plumbed through
    `optimize()` to `seal` and neither was asserted anywhere, so a
    version that dropped either kept the whole suite green.

    Asserted behaviourally on round 0's own candidate rather than by
    intercepting `seal`: `checkpoint` shows up as sampled checkpoints
    that a `checkpoint=0` run would not have, and `model` shows up in the
    sealed trace's provenance. A stubbed `seal` would only prove the
    keyword was forwarded, not that it did anything.
    """
    result = optimize(
        _parent(),
        ScriptedEditor([]),  # STOP at round 1: only round 0 boots.
        TARGET,
        rounds=1,
        patience=1,
        max_rejections=1,
        max_cost=10_000,
        checkpoint=64,
        model="a-model-1",
    )
    assert [s for s, _ in result.best.checkpoints] == [
        64, 128, 192, 256, 320, 384,
    ]
    assert result.best.trace.provenance.model == "a-model-1"
