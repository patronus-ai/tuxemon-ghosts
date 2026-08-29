"""tests/test_rules.py -- S5 progress rules."""

import json
from pathlib import Path
from typing import Any

from tuxghost.rules import (
    CRITICAL_PATH,
    TuxemonFirstBattleRules,
    TuxemonGymRules,
    TuxemonRules,
)

SAVE = Path(__file__).parent / "fixtures" / "paper_town.save"
#: Read BEFORE bootstrapping: `_bootstrap_vendored_tuxemon` chdirs into
#: the vendored tuxemon/ directory, so a relative path resolves wrong
#: afterwards. This is the same hazard `_PATH_ARGS` exists for in the CLI.
_RAW = json.loads(SAVE.read_text())


def booted() -> Any:
    """A fresh session on the committed fixture. Returns the session.

    `boot_from_save` returns a (client, session) TUPLE and takes a
    validated `SaveData`, not a raw dict -- both verified against the
    tree. It also RESETS the module-level `local_session` singleton, so
    anything captured from a previous session goes stale: build what you
    need from THIS return value, never from an earlier one. That aliasing
    is a documented cause of vacuous tests in this project.
    """
    from tuxghost.execute import _bootstrap_vendored_tuxemon

    _bootstrap_vendored_tuxemon()
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save

    _client, session = boot_from_save(
        SaveData.model_validate(_RAW), seed=1234, clock_epoch=1787659200
    )
    return session


def test_progress_never_decreases_across_a_run() -> None:
    """`progress` is the search's gradient, so a decrease would tell the
    optimizer a candidate got WORSE when it merely wandered. Their
    docstring calls the equivalent 'monotone-as-possible'; we assert it
    outright because every term we use is monotone by construction
    (battles only append, the map index is max-tracked, party count only
    grows in normal play)."""
    from tuxghost.loop import run_steps

    session = booted()
    rules = TuxemonRules()
    seen: list[int] = []
    run_steps(
        session.client, 120,
        hook=lambda i: seen.append(rules.progress(session)),
    )
    assert seen, "no observations recorded"
    assert seen == sorted(seen), f"progress went backwards: {seen}"


def test_progress_discriminates_a_further_run() -> None:
    """THE ANTI-VACUITY TEST. This project's original state digest was
    'deterministic' across nine runs for the worst possible reason: it
    captured nothing that changed. A progress function that returns a
    constant would pass the monotonicity test above and be useless.

    A party gaining a monster must raise progress, because `party_count`
    is a real term.
    """
    session = booted()
    rules = TuxemonRules()
    before = rules.progress(session)

    party = session.player.monsters
    party.append(party[0])                     # crude but real: party_count += 1
    after = rules.progress(session)

    assert after > before, (before, after)


def test_map_index_does_not_decrease_when_a_run_backtracks() -> None:
    """Task 1's monotonicity test above only walks forward across maps
    for 120 steps and never revisits an earlier one, so it would pass
    even against a `_map_index` that just returned the CURRENT map's
    ordinal instead of the max-tracked one. Force an actual backtrack
    directly against `_map_index`, which only ever calls
    `session.client.get_map_name()` -- a minimal duck-typed stub
    exercises the real max-tracking logic without inventing a fixture.
    """

    class _FakeClient:
        def __init__(self, name: str) -> None:
            self.map_name = name

        def get_map_name(self) -> str:
            return self.map_name

    class _FakeSession:
        def __init__(self, name: str) -> None:
            self.client = _FakeClient(name)

    rules = TuxemonRules()
    session = _FakeSession(CRITICAL_PATH[2])
    forward = rules._map_index(session)
    assert forward == 2

    session.client.map_name = CRITICAL_PATH[0]  # backtrack to an earlier map
    after_backtrack = rules._map_index(session)
    assert after_backtrack == forward, "map index decreased on backtrack"


def test_the_gym_goal_is_exercised_even_though_no_trace_reaches_one() -> None:
    """The TRUE Boulder Badge analogue, and unreachable today: a gym needs
    overworld navigation plus an interior plus the fight, and our longest
    recorded run is a wild encounter at step 4945.

    So it is exercised the way `tests/test_digest.py` exercises
    `persistent_npc_state` -- by hand-constructing the state rather than
    pretending a trace produces it. Without this the goal would be the
    FIFTH piece of correct, permanently inert plumbing in this project.
    """
    from tuxemon.db import OutputBattle  # members UPPERCASE, values lowercase

    session = booted()
    rules = TuxemonGymRules()
    assert rules.observe(session, 0)["at_goal"] is False

    session.player.battle_handler.record_battle(
        "classic_gym_leader_granite", OutputBattle.WON
    )
    obs = rules.observe(session, 1)
    assert obs["at_goal"] is True
    assert obs["progress"] >= 1_000_000_000, "a gym win must dominate progress"


def test_a_non_gym_win_is_not_the_gym_goal() -> None:
    """Discrimination: beating anything else must NOT latch the gym goal,
    or `at_goal` would be 'won any battle' wearing a gym's name."""
    from tuxemon.db import OutputBattle

    session = booted()
    rules = TuxemonGymRules()
    session.player.battle_handler.record_battle("random_npc", OutputBattle.WON)
    assert rules.observe(session, 1)["at_goal"] is False


def test_the_first_battle_goal_latches_on_any_win() -> None:
    from tuxemon.db import OutputBattle

    session = booted()
    rules = TuxemonFirstBattleRules()
    assert rules.observe(session, 0)["at_goal"] is False
    session.player.battle_handler.record_battle("random_npc", OutputBattle.WON)
    assert rules.observe(session, 1)["at_goal"] is True


def test_a_loss_is_not_a_goal() -> None:
    from tuxemon.db import OutputBattle

    session = booted()
    rules = TuxemonFirstBattleRules()
    session.player.battle_handler.record_battle("random_npc", OutputBattle.LOST)
    assert rules.observe(session, 1)["at_goal"] is False


def test_a_wiped_party_is_dead() -> None:
    """`dead` ships INERT -- nothing we can currently record wipes a
    party. Exercised by hand-zeroing HP, the same technique
    test_digest.py uses for persistence. Their Pokemon rules leave the
    equivalent as `dead = False  # TODO: detect blackout`; ours is real
    code, just not yet reachable in production."""
    session = booted()
    rules = TuxemonFirstBattleRules()
    assert rules.observe(session, 0)["dead"] is False

    for monster in session.player.monsters:
        monster.current_hp = 0
    assert rules.observe(session, 1)["dead"] is True


def test_an_empty_party_is_not_dead() -> None:
    """Guard against `all([]) is True`: a party that has not been
    assembled yet is not a wipe."""
    session = booted()
    rules = TuxemonFirstBattleRules()
    session.player.monsters.clear()
    assert rules.observe(session, 1)["dead"] is False
