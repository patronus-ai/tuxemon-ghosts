"""Task 8: the seeded mutation baseline.

Reproducibility is asserted on the round-by-round score SEQUENCE and the
winning trace's digest, not on the final score alone -- two runs that
wandered differently and happened to settle on the same tile would look
identical to an endpoint check.

The convergence test is demonstrated against a NO-OP editor: if it passes
with an editor that proposes nothing, it was measuring the harness.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TypedDict, cast

from tuxghost.optimize.editors.mutation import MutationEditor
from tuxghost.optimize.objective import ReachTile
from tuxghost.optimize.runner import optimize
from tuxghost.optimize.seal import CandidateResult
from tuxghost.trace import read

LIVE = Path(__file__).parent / "golden" / "scripted_town_1234.tuxghost"
#: Three tiles east of where the parent ends ([16, 14]), so the distance
#: term is a real unmet term the mutator can actually climb. Against the
#: RETIRED parent no target was climbable at all -- its trailing dialog
#: swallowed every input -- so every acceptance there rode on `steps`
#: alone (handoff item A1).
TARGET = ReachTile("spyder_paper_town.tmx", (19, 14))


#: `mypy --strict` refuses to `**`-splat a bare `dict[str, int]` into
#: `optimize`'s keyword-only parameters (it cannot see that every key
#: matches a parameter name), so this is a `TypedDict` rather than a plain
#: dict literal -- same four bounds the brief specifies, typed so the
#: splat below type-checks.
class _Bounds(TypedDict):
    rounds: int
    patience: int
    max_rejections: int
    max_cost: int


BOUNDS: _Bounds = {
    "rounds": 12,
    "patience": 12,
    "max_rejections": 6,
    "max_cost": 4_000,
}

#: `propose`'s `candidate` parameter is typed `CandidateResult`, but these
#: pure/no-engine tests never read it (the mutator ignores `candidate` and
#: `score` by design -- see the module docstring). `cast` rather than
#: `object`, so `mypy --strict` sees a value of the declared type instead
#: of rejecting the call outright.
_FAKE_CANDIDATE = cast(CandidateResult, None)


def test_the_same_seed_produces_the_same_proposals() -> None:
    """Pure, no engine: the editor is a function of its seed and inputs."""
    from tuxghost.optimize.schedule import lift

    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    a = MutationEditor(seed=7)
    b = MutationEditor(seed=7)
    for _ in range(20):
        assert a.propose(script, _FAKE_CANDIDATE, (0.0,)) == b.propose(
            script, _FAKE_CANDIDATE, (0.0,)
        )


def test_different_seeds_diverge() -> None:
    from tuxghost.optimize.schedule import lift

    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    a = [
        MutationEditor(seed=7).propose(script, _FAKE_CANDIDATE, (0.0,))
        for _ in range(20)
    ]
    b = [
        MutationEditor(seed=8).propose(script, _FAKE_CANDIDATE, (0.0,))
        for _ in range(20)
    ]
    assert a != b


def test_the_editor_never_touches_the_global_rng() -> None:
    """The engine's determinism depends on the global `random` stream that
    `seed_all` pins. An editor drawing from it would perturb the game."""
    random.seed(99)
    before = random.random()
    random.seed(99)
    from tuxghost.optimize.schedule import lift

    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    editor = MutationEditor(seed=7)
    for _ in range(50):
        editor.propose(script, _FAKE_CANDIDATE, (0.0,))
    assert random.random() == before, (
        "MutationEditor consumed from the global RNG stream"
    )


def test_a_seeded_run_is_reproducible_end_to_end() -> None:
    a = optimize(read(LIVE), MutationEditor(seed=7), TARGET, **BOUNDS)
    b = optimize(read(LIVE), MutationEditor(seed=7), TARGET, **BOUNDS)
    assert [r.score for r in a.rounds] == [r.score for r in b.rounds]
    assert a.best.trace.header.final_digest == b.best.trace.header.final_digest
    assert a.best_round == b.best_round and a.stop_reason == b.stop_reason


def test_the_mutation_editor_improves_on_the_parent() -> None:
    """Seed 7 is not a lucky pick: seeds 7, 8 and 9 were all RE-MEASURED
    against this parent/target/BOUNDS combination when the fixture was
    swapped (handoff item A1), and every one still beats the parent
    within 12 rounds -- best_round 9, 6 and 12 respectively, from
    parent `(0.0, -3.0, -176.0)` to `(0.0, -3.0, -124.0)`,
    `(0.0, -1.0, -175.0)` and `(0.0, -3.0, -142.0)`. So the mutator
    genuinely searches rather than winning by chance on one seed. 7 is
    pinned only because a test needs one fixed seed to be reproducible.

    (Against the RETIRED parent the same three seeds measured best_round
    10, 8 and 10. The numbers moved; the claim did not.)
    """
    result = optimize(read(LIVE), MutationEditor(seed=7), TARGET, **BOUNDS)
    scores = [r.score for r in result.rounds if r.score is not None]
    assert result.best_round > 0, (
        f"no round beat the parent; score sequence was {scores}"
    )
    assert scores[0] < max(scores)


def test_an_acceptance_can_be_carried_by_the_distance_term() -> None:
    """The search finds a route improvement, not just a shorter route.

    Seed 8 is pinned because it is the measured seed whose winner
    improves DISTANCE, `-3.0` to `-1.0`: the mutator re-routed the player
    two tiles nearer the target rather than merely arriving sooner.
    Asserted on the distance term alone, because a whole-tuple assertion
    passes on a pure `steps` win and would say nothing about routing.

    WHAT THIS DOES *NOT* SHOW, recorded because the first draft of this
    test claimed it and was wrong. It does not discriminate this parent
    from the retired `claude_town_1234` -- run against that fixture with
    this target, it still passes. Only APPENDED input is swallowed by
    that parent's trailing dialog; `MutationEditor` edits MID-script, and
    `docs/STATUS.org` records that mid-script edits change the route
    before the dialogue opens, so distance moves there too. The claim
    that appended input cannot move the retired parent is pinned where it
    is actually demonstrable, in
    `tests/test_scripted_parent.py::test_the_distance_term_is_live_
    against_this_parent`.

    What this DOES pin is the objective and the mutator staying wired
    together: a `ReachTile.score` whose distance term went constant, or a
    mutator that stopped producing route-changing edits, fails here.
    """
    result = optimize(read(LIVE), MutationEditor(seed=8), TARGET, **BOUNDS)
    scores = [r.score for r in result.rounds if r.score is not None]
    parent_distance = scores[0][1]
    best_distance = max(scores)[1]
    assert best_distance > parent_distance, (
        "no accepted round improved the DISTANCE term; the search is "
        f"winning on steps alone. Sequence was {scores}"
    )


def test_every_proposal_is_structurally_valid() -> None:
    """A mutator that proposed out-of-range actions would burn the whole
    budget on rejected rounds and still look like it 'ran'."""
    result = optimize(read(LIVE), MutationEditor(seed=7), TARGET, **BOUNDS)
    reasons = [r.rejected_reason for r in result.rounds if r.rejected_reason]
    assert not [r for r in reasons if r != "STOP"], reasons
