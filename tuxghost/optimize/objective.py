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
    """

    def __init__(self, map_name: str, tile: tuple[int, int]) -> None:
        self.map_name = map_name
        self.tile = tile

    def score(self, candidate: CandidateResult) -> tuple[float, ...]:
        state = candidate.final_state
        on_map = state.get("map") == self.map_name
        pos = state.get("tile_pos") or [0, 0]
        distance = abs(pos[0] - self.tile[0]) + abs(pos[1] - self.tile[1])
        return (
            0.0 if on_map else -1.0,
            -float(distance),
            -float(candidate.steps),
        )
