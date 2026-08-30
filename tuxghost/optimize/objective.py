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
    #:
    #: Term 0 is `-off_target_map`, NOT `on_target_map`, and the
    #: difference is not cosmetic: `score` returns `0.0` when the
    #: candidate IS on the target map and `-1.0` when it is not, so a
    #: model told the term is `on_target_map` reads the good value
    #: (`0.0`) as false -- exactly backwards. Naming it as a cost makes
    #: all three terms read uniformly as "negated cost, 0 is best", so
    #: `[0.0, -2.0, -442.0]` is unambiguous: on the map, 2 away, 442
    #: steps. `test_every_term_that_can_go_negative_is_named_as_a_cost`
    #: pins the convention against the values themselves rather than
    #: against these strings.
    TERMS = ("-off_target_map", "-distance_to_target", "-steps")

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


class RulesObjective:
    """Adapts a `GameRules` run to this project's lexicographic scoring.

    (goal_state, max_progress, -steps), higher better:
      * a goal-reaching candidate beats any non-goal candidate;
      * among non-goal candidates, more progress wins -- this is what
        gives the search a gradient before the goal is reachable at all;
      * among goal-reaching candidates, fewer steps wins -- and ONLY
        among those. Before the goal is reached the step term is a flat
        0.0.

    That last clause is the fix for a measured defect, not a nicety.
    `-steps` used to apply unconditionally, and the docstring above has
    always claimed otherwise. The consequence: every tile inside
    `classic_gym_granite` is one rank of the critical path, so walking
    its thirteen-tile corridor toward the leader left `max_progress`
    unchanged at 1020 and `-steps` strictly worse. The objective
    therefore preferred the candidate that stepped through the door and
    STOPPED, and `run13` shows it plainly: rounds 6 through 19, fourteen
    consecutive candidates all scoring 1020, all rejected for length,
    while the model walked deeper each time. The only route to the goal
    was a region of pure cost, and the score forbade crossing it.

    Shortening pressure belongs on a run that has already WON, which is
    what the docstring said all along.

    A dead candidate scores (-1.0, ...), below every live one.

    TERMS names the tuple for `ClaudeEditor`'s prompt, the same way
    `ReachTile.TERMS` does. Unlike `ReachTile.TERMS`, term 0 here is NOT
    a negated cost -- `goal_state` is -1/0/1 (dead/not yet/reached), not
    a cost that is 0 when best, so naming it with a leading `-` would
    misdescribe it. `max_progress` and `-steps` follow the same
    negated-cost convention `ReachTile` uses.
    """

    TERMS = ("goal_state", "max_progress", "-steps")

    def score(self, candidate: CandidateResult) -> tuple[float, ...]:
        if candidate.died:
            first = -1.0
        elif candidate.goal_step is None:
            first = 0.0
        else:
            first = 1.0
        # Flat 0.0 until the goal is reached: see the class docstring.
        # Written against `goal_step` rather than `first` so a DEAD
        # candidate is also spared the penalty -- it already loses on
        # term 0 against everything alive, and letting length break ties
        # among corpses would rank one failure above another for a
        # reason nothing here cares about.
        steps = (
            0.0
            if candidate.goal_step is None
            else -float(candidate.steps)
        )
        return (first, float(candidate.max_progress), steps)
