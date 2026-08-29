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


#: Map ordinals used to SHAPE progress. Marked VERIFY exactly as their
#: Pokemon CRITICAL_PATH is: the shipped order below is inferred from
#: where the (dead) `badge` gate is checked -- `route1.tmx` and
#: `taba_town.tmx` -- and has NOT been confirmed against a real route.
#: Dump the map sequence over a reference run and fix this list before
#: trusting the shaping. The GOAL does not depend on it; only the shape.
CRITICAL_PATH: tuple[str, ...] = (
    "spyder_paper_town.tmx",   # VERIFY
    "route1.tmx",              # VERIFY
    "taba_town.tmx",           # VERIFY
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
        return False

    def observe(self, session: Any, step: int) -> Observation:
        del step
        return {
            "progress": self.progress(session),
            "at_goal": self.at_goal(session),
            "dead": self.dead(session),
        }
