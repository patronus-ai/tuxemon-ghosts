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

#: The SAME shaping for the `classic_*` campaign, where the gym goal is
#: actually reachable (`tests/fixtures/hearthrock_city.save`).
#:
#: THIS LIST WAS WRONG TWICE, and both errors were found by running the
#: optimizer rather than by reading it. Recorded because each failure
#: mode is easy to reintroduce:
#:
#:   1. Absent entirely. The map term was permanently 0 from that save
#:      and `progress` collapsed to `party_count * 10` -- the identical
#:      dead-gradient defect M1 was, relocated. Every round of the first
#:      real run scored `[0.0, 20.0, -steps]`; that 20.0 is 2 monsters
#:      x 10, with nothing from the map.
#:   2. Cities only. Re-run: every round STILL scored 20.0. The nearest
#:      credited city is two hops out, so the search faced a two-map
#:      plateau with no signal anywhere on it. A gradient that pays only
#:      on arrival at a city is not climbable by an editor proposing
#:      button sequences. (The `spyder_*` path above had this right by
#:      accident -- it lists `spyder_route1`.)
#:
#: GYM MAPS ARE INCLUDED, and excluding them was the second mistake's
#: twin. They were left out as "the goal, not the path", which sounds
#: principled and is wrong here: measured from the committed save's own
#: spawn tile (23,14), holding UP walks the player straight into
#: `classic_gym_granite` -- the gym door is directly above. Entering a
#: gym is the single most reachable act of progress available, and
#: omitting it denied credit for the only move the search can easily
#: find. Gyms sit at their real hop distance like everything else.
#:
#: Ordered by measured hop distance from `classic_hearthrock_city.tmx`,
#: every hop 0..8 populated. Equal-hop entries are real BRANCHES and are
#: listed contiguously; `tests/test_rules_mapgraph.py` requires exactly
#: that and re-derives every number here from the map data.
CLASSIC_CRITICAL_PATH: tuple[str, ...] = (
    "classic_hearthrock_city.tmx",       # 0
    "classic_gym_granite.tmx",           # 1
    "classic_gym_mila.tmx",              # 1
    "classic_route_1.tmx",               # 1
    "classic_route_8.tmx",               # 1
    "classic_steamshore_city.tmx",       # 2
    "classic_valorhold_city.tmx",        # 2
    "classic_gym_bravion.tmx",           # 3
    "classic_gym_marin.tmx",             # 3
    "classic_gym_pyra.tmx",              # 3
    "classic_route_2.tmx",               # 3
    "classic_route_7.tmx",               # 3
    "classic_thornwood_city.tmx",        # 4
    "classic_route_3.tmx",               # 5
    "classic_route_5.tmx",               # 5
    "classic_aerolume_city.tmx",         # 6
    "classic_route_4.tmx",               # 6
    "classic_route_6.tmx",               # 7
    "classic_stormpeak_city.tmx",        # 7
    "classic_gym_astra.tmx",             # 8
    "classic_gym_voltessa.tmx",          # 8
    "classic_umbrastar_city.tmx",        # 8
)

#: Every campaign's path. The two campaigns are disjoint map components
#: (measured), so a map name belongs to at most one of these and the
#: lookup below cannot be ambiguous.
CRITICAL_PATHS: tuple[tuple[str, ...], ...] = (
    CRITICAL_PATH,
    CLASSIC_CRITICAL_PATH,
)

#: All 13 leaders are named `classic_gym_leader_<name>`; nothing in the
#: shipped data orders them, which is why the goal is "any gym won".
GYM_LEADER_PREFIX = "classic_gym_leader_"


#: The three terms of `TuxemonRules.progress`, as named constants rather
#: than magic numbers inline in the sum. They are constants because TWO
#: things must agree about them: `progress()`, which computes the score,
#: and `describe_progress()`, which tells the editor what the score
#: rewards. A prose description typed by hand beside a numeric literal is
#: exactly the pair that drifts apart silently -- this file has already
#: shipped three comments that went stale that way. Deriving both from
#: one constant removes the failure mode instead of documenting it.
#:
#: The magnitudes are a strict lexicographic ordering, not a weighting to
#: be tuned: one gym outranks every map, and one map outranks any party.
GYM_POINTS = 1_000_000_000
MAP_POINTS = 1_000
PARTY_POINTS = 10


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
        """Furthest rank reached along whichever campaign's path this run
        is on.

        Searches every path in `CRITICAL_PATHS`, not just the `spyder_*`
        one. The campaigns are disjoint map components, so at most one
        path can contain `name` and there is nothing to disambiguate.
        """
        name = session.client.get_map_name()
        for path in CRITICAL_PATHS:
            if name in path:
                self._furthest_map = max(
                    self._furthest_map, path.index(name)
                )
                break
        return self._furthest_map

    def progress(self, session: Any) -> int:
        return (self._n_gyms_won(session) * GYM_POINTS
                + self._map_index(session) * MAP_POINTS
                + len(session.player.monsters) * PARTY_POINTS)

    @classmethod
    def describe_progress(cls) -> str:
        """What `progress` rewards, in words, for the editor's prompt.

        GENERATED from the same constants `progress` sums, never typed
        beside them. Before this existed the editor was handed the score
        term NAMES only -- `"goal_state, max_progress, -steps"` -- and
        asked to maximise a number nothing ever explained. It knew where
        it was standing (the prompt has carried map, tile and a
        checkpoint trail all along) but not that walking onto a new map
        was worth anything at all. Across four runs and 37 rounds from
        `hearthrock_city.save`, `max_progress` never once moved off 20.

        Deliberately says nothing about WHICH maps or where they are:
        naming the route would hand over the answer and a successful run
        would then demonstrate execution rather than search.
        """
        return (
            "max_progress is a sum, and higher is better:\n"
            f"  {GYM_POINTS:>13,}  per gym leader defeated\n"
            f"  {MAP_POINTS:>13,}  per new map reached along the "
            "campaign route\n"
            f"  {PARTY_POINTS:>13,}  per party member\n"
            "Reaching a map you have not been to before is the cheapest "
            "way to raise it."
        )

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
