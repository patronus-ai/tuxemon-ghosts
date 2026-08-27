"""Task 6: scoring.

Scores are TUPLES compared lexicographically, higher is better, because
any objective mixing "did it get there" with "how fast" into a single
float needs an invented weighting constant.

`ReachTile`'s middle term (distance) is not decoration. A binary
reached/didn't objective gives `MutationEditor` nothing to climb: every
failing candidate scores identically, the search degenerates to a random
walk, and Task 8's convergence test would be measuring luck.
"""

from __future__ import annotations

from typing import Any

import pytest

from tuxghost.optimize.objective import ReachTile
from tuxghost.optimize.seal import CandidateResult


def _candidate(map_name: str, tile: tuple[int, int], steps: int) -> CandidateResult:
    """A CandidateResult with only the fields `ReachTile` reads. The
    `trace` field is never touched by an objective, so `None` typed as
    `Any` keeps the fixture honest about that rather than fabricating a
    Trace this test does not use."""
    trace: Any = None
    return CandidateResult(
        trace=trace,
        steps=steps,
        final_state={"map": map_name, "tile_pos": [tile[0], tile[1]]},
    )


TARGET = ReachTile("spyder_paper_town.tmx", (12, 20))


def test_reaching_the_target_beats_being_near_it() -> None:
    on = _candidate("spyder_paper_town.tmx", (12, 20), 400)
    near = _candidate("spyder_paper_town.tmx", (12, 19), 100)
    assert TARGET.score(on) > TARGET.score(near)


def test_closer_beats_farther_on_the_same_map() -> None:
    near = _candidate("spyder_paper_town.tmx", (12, 18), 400)
    far = _candidate("spyder_paper_town.tmx", (12, 12), 400)
    assert TARGET.score(near) > TARGET.score(far)


def test_any_same_map_candidate_beats_a_wrong_map_one() -> None:
    """Even an absurdly distant same-map candidate. `manhattan` across
    maps is meaningless, so the map term leads."""
    same = _candidate("spyder_paper_town.tmx", (900, 900), 400)
    other = _candidate("spyder_bedroom.tmx", (12, 20), 400)
    assert TARGET.score(same) > TARGET.score(other)


def test_fewer_steps_breaks_a_tie() -> None:
    quick = _candidate("spyder_paper_town.tmx", (12, 20), 100)
    slow = _candidate("spyder_paper_town.tmx", (12, 20), 400)
    assert TARGET.score(quick) > TARGET.score(slow)


def test_steps_never_outweigh_distance() -> None:
    """The gradient must dominate the tie-break, or the optimizer learns
    to do nothing quickly rather than something slowly."""
    closer_slower = _candidate("spyder_paper_town.tmx", (12, 19), 10_000)
    farther_faster = _candidate("spyder_paper_town.tmx", (12, 12), 1)
    assert TARGET.score(closer_slower) > TARGET.score(farther_faster)


def test_the_score_is_a_stable_length_tuple() -> None:
    a = TARGET.score(_candidate("spyder_paper_town.tmx", (12, 20), 1))
    b = TARGET.score(_candidate("spyder_bedroom.tmx", (0, 0), 999))
    assert len(a) == len(b) == 3


def test_distance_is_symmetric_in_both_axes() -> None:
    left = _candidate("spyder_paper_town.tmx", (10, 20), 400)
    below = _candidate("spyder_paper_town.tmx", (12, 22), 400)
    assert TARGET.score(left) == TARGET.score(below)


def _bare_candidate(final_state: dict[str, Any], steps: int = 400) -> CandidateResult:
    """Like `_candidate`, but lets a test hand `final_state` a malformed
    `tile_pos` directly rather than always producing a valid two-element
    one."""
    trace: Any = None
    return CandidateResult(trace=trace, steps=steps, final_state=final_state)


def test_missing_tile_pos_is_a_refusal_not_a_default() -> None:
    candidate = _bare_candidate({"map": "spyder_paper_town.tmx"})
    with pytest.raises(ValueError, match="tile_pos"):
        TARGET.score(candidate)


def test_empty_tile_pos_is_a_refusal_not_a_default() -> None:
    candidate = _bare_candidate({"map": "spyder_paper_town.tmx", "tile_pos": []})
    with pytest.raises(ValueError, match="tile_pos"):
        TARGET.score(candidate)


def test_short_tile_pos_is_a_refusal_not_a_default() -> None:
    candidate = _bare_candidate({"map": "spyder_paper_town.tmx", "tile_pos": [5]})
    with pytest.raises(ValueError, match="tile_pos"):
        TARGET.score(candidate)


def test_terms_names_every_score_term_in_order() -> None:
    """`ReachTile.TERMS` is sent to `ClaudeEditor` as the legend for the
    score tuple (whole-branch review, Important 1), so a term added to
    `score` without a name here would mislabel every number after it.
    Pinned on LENGTH; the SIGN convention each name states is pinned by
    `test_every_term_that_can_go_negative_is_named_as_a_cost` below."""
    score = TARGET.score(_candidate("spyder_paper_town.tmx", (12, 18), 400))
    assert len(ReachTile.TERMS) == len(score)


def test_every_term_that_can_go_negative_is_named_as_a_cost() -> None:
    """`TERMS[0]` read `on_target_map` while `score`'s term 0 is `0.0`
    when the candidate IS on the target map and `-1.0` when it is not.
    A model shown `[0.0, -2.0, -442.0]` and told term 0 is
    `on_target_map` reads `0.0` as FALSE -- the exact inversion of the
    truth (handoff item A2).

    Asserted as an INVARIANT over every term rather than as a string
    comparison against the fixed word: a term whose value goes negative
    for the worse of two candidates is a cost, and its name must say so
    with a leading `-`. A fourth term wired up backwards fails here too,
    which a `TERMS[0] == "-off_target_map"` assertion would not catch.
    """
    best = _candidate("spyder_paper_town.tmx", (12, 18), 400)
    #: One candidate per term, each strictly worse than `best` in that
    #: one term and identical in the others.
    worse_by_term = (
        _candidate("cotton_town.tmx", (12, 18), 400),       # 0: off map
        _candidate("spyder_paper_town.tmx", (12, 30), 400),  # 1: further
        _candidate("spyder_paper_town.tmx", (12, 18), 900),  # 2: slower
    )
    assert len(worse_by_term) == len(ReachTile.TERMS)

    baseline = TARGET.score(best)
    for index, candidate in enumerate(worse_by_term):
        name = ReachTile.TERMS[index]
        degraded = TARGET.score(candidate)[index]
        assert degraded < baseline[index], (
            f"term {index} ({name}) was supposed to get WORSE for this "
            "candidate but did not; the fixture no longer isolates it"
        )
        assert degraded < 0.0, (
            f"term {index} ({name}) is worse at {degraded}, which is not "
            "negative, so it is not a negated cost"
        )
        assert name.startswith("-"), (
            f"term {index} goes negative ({degraded}) for a worse "
            f"candidate, so it is a COST, but its name {name!r} does not "
            "say so -- a model reading the tuple will invert its meaning"
        )
