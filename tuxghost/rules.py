"""Per-game progress rules, mirroring videogamebench's game_rules.py.

Each game supplies a `GameRules` turning live session state into the
three signals a progress-based optimizer needs each step:

    observe(session, step) -> {"progress": int, "at_goal": bool, "dead": bool}

    progress : monotone scalar, higher = further. The caller tracks its max.
    at_goal  : True once the goal is achieved (latches goal_step).
    dead     : True to end the run as a failure.

Deliberately one module with several rules classes, matching their
`scripts/game_rules.py` rather than inventing a package layout --
matching their shape is the point of S5.

COST CONSTRAINT, load-bearing: `observe` runs EVERY STEP, so it may read
only cheap fields. It must never call `tuxghost.digest.state_of`, which
serializes the entire save (party, items, monsters). The S4 plan made
exactly this mistake -- `execute(checkpoint=1, capture_states=True)`
would have run a full digest plus save-serialization per step to recover
three fields. Pinned by a test, not by this comment.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class Observation(TypedDict):
    progress: int
    at_goal: bool
    dead: bool


class GameRules(Protocol):
    def observe(self, session: Any, step: int) -> Observation: ...


#: Map ordinals used to SHAPE progress. The NAMES are VERIFIED --
#: `spyder_route1.tmx` cross-references all three of `spyder_paper_town
#: .tmx`, `spyder_cotton_town.tmx` and `spyder_brideswood.tmx` by direct
#: grep of the shipped `.tmx` map data (whole-branch review, M1), unlike
#: an earlier version of this list (`route1.tmx`, `taba_town.tmx`) which
#: named a different, unreachable map family -- those two maps exist in
#: `mods/tuxemon/maps/` but nothing connects them to `spyder_paper_town
#: .tmx`, so the map term was permanently 0 and `progress` reduced to
#: `party_count * 10` with no gradient at all.
#:
#: THE ORDER IS NOW MEASURED, not assumed. Every `transition_teleport
#: player,<map>.tmx` in all 263 shipped `.tmx` files was parsed into a
#: directed graph (472 edges over 199 maps with outgoing links) and
#: breadth-first searched from the starting save's own map. Hop counts:
#: `spyder_paper_town` 0, `spyder_route1` 1, `spyder_cotton_town` 2 --
#: strictly increasing, exactly the order below. This replaced the
#: earlier "ORDER still VERIFY" marker, which was correct to raise and
#: turned out to be correct as written.
#: `tests/test_rules_mapgraph.py` re-derives this from the map data on
#: every run, so a vendor bump that reorders the route fails rather than
#: silently reshaping the gradient.
CRITICAL_PATH: tuple[str, ...] = (
    "spyder_paper_town.tmx",     # VERIFIED name, ORDER VERIFIED (0 hops)
    "spyder_route1.tmx",         # VERIFIED name, ORDER VERIFIED (1 hop)
    "spyder_cotton_town.tmx",    # VERIFIED name, ORDER VERIFIED (2 hops)
)

#: All 13 leaders are named `classic_gym_leader_<name>`; nothing in the
#: shipped data orders them, which is why the goal is "any gym won".
GYM_LEADER_PREFIX = "classic_gym_leader_"


class TuxemonRules:
    """Shared progress; subclasses supply the goal.

    progress = n_gyms_won * 1e9 + furthest_critical_path_map * 1e3
             + party_count * 10

    Same three-term shape as their `PokemonRedBrockRules`
    (n_badges * 1e9 + furthest_critical_path_map * 1e3 + party_count * 10)
    so the numbers read the same way.
    """

    def __init__(self) -> None:
        self._furthest_map = 0

    def reset(self) -> None:
        """Clear per-run state. `seal` calls this (when present -- see
        below) before stepping a candidate.

        `_furthest_map` must persist WITHIN one run: `seal`'s hook calls
        `observe` once per step, and progress must not decrease as a
        candidate walks forward and back across maps it has already
        visited (see `test_map_index_does_not_decrease_when_a_run_
        backtracks`). But it must NOT persist ACROSS runs --
        `tuxghost.optimize.runner.optimize` threads exactly ONE `rules`
        instance into round 0 (runner.py:220-222) and into every
        candidate's `seal` call via `_prepare` (runner.py:158-167), so
        without this reset candidate N inherits candidate N-1's
        furthest-map floor: a fresh candidate that never left the start
        map would still report the previous candidate's high water mark
        (whole-branch review, I1 -- demonstrated: candidate B, freshly
        at `CRITICAL_PATH[0]`, reported `furthest == 2` because candidate
        A had genuinely reached `CRITICAL_PATH[2]` on the SAME shared
        instance). Once `CRITICAL_PATH` is calibrated this makes
        accept/reject depend on candidate ORDER rather than candidate
        quality -- a silent correctness bug, not a cosmetic one.

        Deliberately NOT part of the `GameRules` Protocol: a minimal
        test double with no persistent state (e.g. this module's own
        `_AlwaysAtGoal`-style stand-ins used in tests) has nothing to
        reset and should not be forced to grow a no-op method just to
        satisfy an interface. `seal` calls this only when it exists
        (`hasattr`), which keeps `rules.py` honest about which
        implementers actually carry per-run state -- rather than forcing
        every future `GameRules` implementation, stateful or not, to
        answer a question that does not apply to it.
        """
        self._furthest_map = 0

    def _n_gyms_won(self, session: Any) -> int:
        battles = session.player.battle_handler.get_battles()
        won = set()
        for battle in battles:
            state = battle.get_state()
            if (state.get("outcome") == "won"
                    and str(state.get("opponent", "")).startswith(
                        GYM_LEADER_PREFIX)):
                won.add(state["opponent"])
        return len(won)

    def _map_index(self, session: Any) -> int:
        name = session.client.get_map_name()
        if name in CRITICAL_PATH:
            self._furthest_map = max(
                self._furthest_map, CRITICAL_PATH.index(name)
            )
        return self._furthest_map

    def progress(self, session: Any) -> int:
        return (self._n_gyms_won(session) * 1_000_000_000
                + self._map_index(session) * 1_000
                + len(session.player.monsters) * 10)

    def at_goal(self, session: Any) -> bool:
        raise NotImplementedError

    def dead(self, session: Any) -> bool:
        """A wipe: party non-empty and every monster at 0 HP.

        `party` is checked for non-emptiness first because `all([])` is
        True -- an unassembled party is not a wipe.
        """
        party = session.player.monsters
        return bool(party) and all(m.current_hp == 0 for m in party)

    def observe(self, session: Any, step: int) -> Observation:
        del step
        return {
            "progress": self.progress(session),
            "at_goal": self.at_goal(session),
            "dead": self.dead(session),
        }


class TuxemonGymRules(TuxemonRules):
    """at_goal = ANY gym leader defeated -- the Boulder Badge analogue.

    "Any", not "the first": all 13 leaders are named
    `classic_gym_leader_<name>` and NOTHING in the shipped data orders
    them, so "the first gym" is not implementable. "Your first badge"
    only requires that one has been won.

    NOT REACHABLE by any trace this project can currently record -- and
    the reason is STRONGER than the spec's "horizon problem" framing,
    which implies a long-but-possible search. It is not long. It is
    impossible from the shipped save.

    Measured over the map graph (see `tests/test_rules_mapgraph.py`):
    all 13 `classic_gym_*.tmx` maps sit in a 22-map connected component
    that contains NO `spyder_*` map at all, and the starting save's map
    `spyder_paper_town.tmx` reaches 88 maps directed / 94 undirected,
    zero of them a gym. The `classic_*` campaign and the `spyder_*`
    campaign are disconnected. No amount of search time crosses that.

    So this class is aspirational until a `classic_*` starting save
    exists. `TuxemonFirstBattleRules` is the goal anything actually runs
    against today.
    """

    def at_goal(self, session: Any) -> bool:
        return self._n_gyms_won(session) >= 1


class TuxemonFirstBattleRules(TuxemonRules):
    """at_goal = any battle won. The SHIPPED, TESTED goal.

    Reachable: `tests/test_combat_determinism.py` drives a battle to a
    clean CombatState exit at step 4945.
    """

    def at_goal(self, session: Any) -> bool:
        return any(
            b.get_state().get("outcome") == "won"
            for b in session.player.battle_handler.get_battles()
        )
