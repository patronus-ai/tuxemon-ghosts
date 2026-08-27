"""Task 7: the edit -> seal -> score loop.

Round 0 is the PARENT, sealed and scored, so "did anything improve" is
answered against a measured baseline rather than against the parent's own
recorded header.

Improvement is asserted on the round-by-round score SEQUENCE, never on
the final best alone: a settled tile is an attractor in this game, so an
endpoint comparison cannot distinguish a run that improved from one that
wandered and came back.

WHAT "IMPROVING" MEANS DEPENDS ON THE PARENT, and this module runs both.
Neither pair of edits is assumed; both were measured, and the second
INVERTS the first.

`scripted_town_1234` (the default parent, 176 steps, ends tile (16, 14)
on a bare `WorldState`). Appended input reaches the player here, so with
the target three tiles east at (19, 14) the improving edit is an APPEND
toward it and the worsening edit an APPEND away from it:

    parent            (0.0, -3.0, -176.0)
    +RIGHT            (0.0, -2.0, -198.0)   accepted, on DISTANCE
    +RIGHT +RIGHT     (0.0, -1.0, -220.0)   accepted, on DISTANCE
    +LEFT             (0.0, -4.0, -198.0)   worse on both terms
    delete last       (0.0, -4.0, -154.0)   FEWER steps, but further

Note the last line: against this parent a DELETE is a WORSENING edit. It
buys steps at the cost of distance, and distance outranks steps.

`claude_town_1234` (the retired parent, kept as a parametrized case, 442
steps). It ends with `state_stack == ['DialogState', 'WorldState']` and
that trailing dialog swallows every APPENDED input: `UP`/`DOWN`/`LEFT`/
`RIGHT`, `A` then `DOWN`, and `B` then `DOWN` all leave the tile at
(11, 14). So no append-only edit can move the player and every
acceptance rides on `steps`, which makes the improving edit a DELETE
(442 -> 416 -> 390, same tile) and the worsening edit an APPEND
(442 -> 642). Exactly the reverse of the parent above.

It is kept rather than deleted because an input-swallowing parent is a
real adversarial case: a loop that quietly depended on edits always
moving the player would pass against the new parent alone. (Its
`distance` term is not frozen -- across the nine-run tuning sweep it
varied from -2 down to -9, from MID-script edits that change the route
before the dialogue opens -- but no candidate ever scored better than
the parent on it. See docs/STATUS.org.)

In neither case is the target moved onto the parent's own end tile,
which would satisfy the distance term trivially and prove nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

_GOLDEN = Path(__file__).parent / "golden"
#: The default parent: 176 steps, ends tile (16, 14) on a bare
#: `WorldState`, so appended input reaches the player.
LIVE = _GOLDEN / "scripted_town_1234.tuxghost"
#: The retired parent, kept as a parametrized case (see module docstring).
RETIRED = _GOLDEN / "claude_town_1234.tuxghost"

#: Three tiles east of where `LIVE` ends, so the distance term is a real
#: unsatisfied term rather than one relocated onto the parent's own end
#: position.
TARGET = ReachTile("spyder_paper_town.tmx", (19, 14))
#: Two tiles from where `RETIRED` ends, for the same reason.
RETIRED_TARGET = ReachTile("spyder_paper_town.tmx", (11, 16))

#: Improving against `LIVE`: one tile toward the target, +22 steps.
RIGHT_STEP = Action(buttons.RIGHT, 16, 6)
#: Worsening against `LIVE`: one tile away from it, also +22 steps -- so
#: the two differ in DIRECTION alone and a loop that accepted on step
#: count could not tell them apart.
LEFT_STEP = Action(buttons.LEFT, 16, 6)
#: A worsening append against `RETIRED`: +200 steps (hold 100 + settle
#: 100), same tile.
BIG = Action(buttons.DOWN, 100, 100)
DOWN = Action(buttons.DOWN, 16, 8)


def _parent() -> Trace:
    return read(LIVE)


def _script_of(parent: Trace) -> ActionScript:
    return lift(parent.inputs, parent.header.step_count)


def _n_actions(parent: Trace) -> int:
    return len(_script_of(parent).actions)


def _delete_last(n: int, times: int) -> list[list[Edit]]:
    """`times` rounds of "delete the last action", against a script that
    SHRINKS by one each accepted round. Improving against `RETIRED`,
    worsening against `LIVE` -- see the module docstring."""
    return [[Delete(n - 1 - i)] for i in range(times)]


def _append_right(n: int, times: int) -> list[list[Edit]]:
    """`times` rounds of "append one step toward the target", against a
    script that GROWS by one each accepted round -- hence `n + i`, where
    `_delete_last` uses `n - 1 - i`. Improving against `LIVE`."""
    return [[Insert(n + i, RIGHT_STEP)] for i in range(times)]


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


#: (label, trace, target, improving-edit factory, worsening action,
#: improving step sequence, worsening step sequence) -- every number
#: MEASURED against that parent, never carried over from the other one.
#: The two rows invert each other: what improves against one worsens
#: against the other. See the module docstring.
_EDIT_CASES = [
    pytest.param(
        LIVE, TARGET, _append_right, LEFT_STEP,
        [176, 198, 220], [176, 198, 198],
        id="scripted_town-append",
    ),
    pytest.param(
        RETIRED, RETIRED_TARGET, _delete_last, BIG,
        [442, 416, 390], [442, 642, 642],
        id="claude_town-delete",
    ),
]


@pytest.mark.parametrize(
    "trace, target, improving, worsening, improving_steps, worsening_steps",
    _EDIT_CASES,
)
def test_an_improving_edit_is_accepted_and_becomes_the_new_best(
    trace: Path,
    target: ReachTile,
    improving: Callable[[int, int], list[list[Edit]]],
    worsening: Action,
    improving_steps: list[int],
    worsening_steps: list[int],
) -> None:
    """Two rounds of that parent's own improving edit are both accepted
    and the score sequence rises monotonically.

    `worsening`/`worsening_steps` are unused here; the parametrization is
    shared with the test below so the two halves of a parent's measured
    edit pair cannot drift apart in separate tables.
    """
    del worsening, worsening_steps
    parent = read(trace)
    editor = ScriptedEditor(improving(_n_actions(parent), 2))
    result = optimize(
        parent,
        editor,
        target,
        rounds=3,
        patience=3,
        max_rejections=1,
        max_cost=10_000,
    )
    scores = [r.score for r in result.rounds if r.score is not None]
    assert scores == sorted(scores), f"score sequence went backwards: {scores}"
    assert scores[-1] > scores[0]
    assert result.best_round > 0
    assert [
        r.steps for r in result.rounds if r.steps is not None
    ] == improving_steps


@pytest.mark.parametrize(
    "trace, target, improving, worsening, improving_steps, worsening_steps",
    _EDIT_CASES,
)
def test_a_worsening_edit_is_kept_out_of_best_but_recorded(
    trace: Path,
    target: ReachTile,
    improving: Callable[[int, int], list[list[Edit]]],
    worsening: Action,
    improving_steps: list[int],
    worsening_steps: list[int],
) -> None:
    del improving, improving_steps
    parent = read(trace)
    n = _n_actions(parent)
    # `best_script` never advances (both candidates lose), so both rounds
    # append at the same index.
    editor = ScriptedEditor([[Insert(n, worsening)], [Insert(n, worsening)]])
    result = optimize(
        parent,
        editor,
        target,
        # `rounds=2`, not 3: with 3 the editor runs out and a third STOP
        # round (score None) is appended, which the `all(r.score is not
        # None ...)` assertion below would then fail on.
        rounds=2,
        patience=3,
        max_rejections=1,
        max_cost=10_000,
    )
    assert result.best_round == 0
    assert [r.accepted for r in result.rounds] == [True, False, False]
    assert all(r.score is not None for r in result.rounds)
    assert [r.steps for r in result.rounds] == worsening_steps


def test_an_empty_proposal_stops_the_run() -> None:
    parent = _parent()
    editor = ScriptedEditor([[Insert(_n_actions(parent), RIGHT_STEP)], []])
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
    editor = ScriptedEditor(_append_right(_n_actions(parent), 10))
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
        [[Replace(999, DOWN)], [Insert(_n_actions(parent), RIGHT_STEP)]]
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
    parent = _parent()
    editor = ScriptedEditor(
        [[Insert(_n_actions(parent), Action(buttons.DOWN, 600, 600))]]
    )
    result = optimize(
        parent,
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
        final_state={"map": "spyder_paper_town.tmx", "tile_pos": [16, 14]},
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
    (10_000, 500, 0) sits either above the parent's cost or below the
    `>= 1` bound check, so passing `max_cost` through at round 0 would
    have left the whole suite green and a reasoned decision deletable by
    a "consistency" cleanup. `max_cost=100` is below the parent's 176, so
    it separates the two: round 0 must still be sealed and SCORED, while
    round 1's 198-step candidate is refused.
    """
    parent = _parent()
    result = optimize(
        parent,
        ScriptedEditor(_append_right(_n_actions(parent), 2)),
        TARGET,
        rounds=2,
        patience=2,
        max_rejections=1,
        max_cost=100,
    )
    assert result.rounds[0].score is not None
    assert result.rounds[0].steps == 176  # the parent ran in full
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
    # An APPEND toward the target, not a delete: against this parent a
    # delete worsens the score, so the second round would not improve
    # either and the run would end "patience exhausted" -- a green-
    # looking pass that stopped testing what this test is named for.
    editor = _BrokenThenGoodEditor(Insert(_n_actions(parent), RIGHT_STEP))
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
    `state_of` snapshot on all 176 steps of every round without saying
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
        checkpoint=16,
        model="a-model-1",
    )
    # `checkpoint=16` against a 176-step parent: every multiple below
    # `step_count`, and NOT `step_count` itself even though 176 is an
    # exact multiple of 16. Same convention `tests/test_optimize_seal.py`
    # pins directly.
    assert [s for s, _ in result.best.checkpoints] == [
        16, 32, 48, 64, 80, 96, 112, 128, 144, 160,
    ]
    assert result.best.trace.provenance.model == "a-model-1"
