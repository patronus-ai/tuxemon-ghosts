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
    Pinned on LENGTH and on the sign convention each name states, not on
    the exact strings, which are prose for a prompt."""
    score = TARGET.score(_candidate("spyder_paper_town.tmx", (12, 18), 400))
    assert len(ReachTile.TERMS) == len(score)
    # The trailing two terms are negated in `score`; their names say so,
    # because a model told "higher is better" and shown `-442.0` has to
    # know that number is a cost.
    assert ReachTile.TERMS[1].startswith("-")
    assert ReachTile.TERMS[2].startswith("-")
    assert not ReachTile.TERMS[0].startswith("-")
