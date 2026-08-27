"""What "better" means, and the one built-in that says so.

An `Objective` returns a TUPLE, compared lexicographically, higher is
better. A single float would need an invented constant to weigh "arrived"
against "arrived quickly"; tuple comparison gives that priority for free,
and a single-criterion objective returns a 1-tuple.

`ReachTile` is openly a PROXY. It measures position, not progress: it
says nothing about whether the game advanced, which is exactly where S5
takes over. S5 supplies its own `Objective` and nothing else changes.
"""

from __future__ import annotations

from typing import Protocol

from tuxghost.optimize.seal import CandidateResult


class Objective(Protocol):
    def score(self, candidate: CandidateResult) -> tuple[float, ...]:
        """Higher is better; compared lexicographically against other
        scores from the SAME objective. The tuple length must not vary
        between candidates -- `tuxghost.optimize.runner` checks it."""
        ...


class ReachTile:
    """Get to `tile` on `map_name`, in as few steps as possible.

    Score: `(on_the_right_map, -distance, -steps)`.

    The distance term is what makes this searchable. Without it every
    candidate that has not arrived scores identically and a mutation
    search has no gradient to climb.

    `manhattan` between two different maps is meaningless, which is why
    the map term leads: a candidate that wanders onto a neighbouring map
    scores below EVERY same-map candidate however physically close it
    ends. Deliberate -- a trace that left the map has broken the thing
    being optimized -- but it means `ReachTile` cannot reward a route
    that legitimately passes THROUGH another map. Such a target needs a
    state predicate, not a tile.

    The two terms handle missing data asymmetrically, on purpose. A
    missing/mismatched `"map"` fails SAFE -- it reads as off-target, the
    worst score, which is the right default for a term whose whole job
    is a boolean comparison. A missing, empty, or short `tile_pos` gets
    no such default: there is no safe numeric stand-in for "we don't
    know where the candidate ended up", because inventing one (e.g. the
    origin) can silently outscore a genuine candidate that is merely far
    from the target. So a malformed `tile_pos` is a REFUSAL --
    `ValueError`, in the style of `tuxghost.optimize.schedule.lift` --
    not a default. Do not "helpfully" restore a fallback here.
    """

    #: This objective's score terms, in the order `score` returns them.
    #: It exists so a prompt can LABEL the tuple: `ClaudeEditor` was
    #: sent a bare `[0.0, -2.0, -442.0]` and had to guess the order and
    #: the signs (whole-branch review, Important 1). Kept here, beside
    #: `score`, rather than in the CLI, so the names cannot drift out of
    #: step with the tuple they describe -- pinned by
    #: `tests/test_optimize_objective.py`. NOT part of the `Objective`
    #: protocol: an objective that wants no legend simply omits it, and
    #: `ClaudeEditor.score_legend` defaults to none.
    TERMS = ("on_target_map", "-distance_to_target", "-steps")

    def __init__(self, map_name: str, tile: tuple[int, int]) -> None:
        self.map_name = map_name
        self.tile = tile

    def score(self, candidate: CandidateResult) -> tuple[float, ...]:
        state = candidate.final_state
        on_map = state.get("map") == self.map_name
        pos = state.get("tile_pos")
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            raise ValueError(
                f"final_state['tile_pos'] is {pos!r}; expected a "
                "two-element (x, y) sequence, refusing rather than "
                "inventing a position"
            )
        distance = abs(pos[0] - self.tile[0]) + abs(pos[1] - self.tile[1])
        return (
            0.0 if on_map else -1.0,
            -float(distance),
            -float(candidate.steps),
        )
