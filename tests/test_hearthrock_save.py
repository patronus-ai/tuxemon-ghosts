"""Pins `tests/fixtures/hearthrock_city.save`, the `classic_*` campaign
starting state.

It exists because `paper_town.save` sits in the `spyder_*` campaign,
which is topologically disconnected from the one holding all 13 gyms --
so `TuxemonGymRules` could never fire from it. The reachability half of
that claim is asserted in `tests/test_rules_mapgraph.py`, against the map
graph; this file asserts what the SAVE itself boots into.

Regenerate with `tests/fixtures/make_hearthrock_save.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"
SAVE = FIXTURES / "hearthrock_city.save"
MAPS = Path(__file__).resolve().parent.parent / "tuxemon/mods/tuxemon/maps"

EPOCH = 1787659200
SEED = 1234


def _boot() -> Any:
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save
    from tuxghost.determinism import pin_clock, seed_all

    seed_all(SEED)
    pin_clock(EPOCH)
    save_data = SaveData.model_validate(json.loads(SAVE.read_text()))
    _client, session = boot_from_save(
        save_data, seed=SEED, clock_epoch=EPOCH
    )
    return session


def test_the_save_boots_to_a_bare_world_state_in_hearthrock() -> None:
    """A bare `WorldState` is the whole contract of a start fixture.

    The generator runs through a cold boot, so `ChoiceState` and
    `ImageState` are on the stack while it works -- measured. `SaveData`
    carries no state stack, so they must not survive into the artifact.
    This asserts they did not, rather than trusting the docstring that
    says they cannot.
    """
    session = _boot()
    client = session.client
    assert client.get_map_name() == "classic_hearthrock_city.tmx"
    assert tuple(int(v) for v in session.player.tile_pos) == (23, 14)
    assert [
        type(s).__name__ for s in client.state_manager.active_states
    ] == ["WorldState"]


def test_the_party_can_actually_meet_the_gym_leaders() -> None:
    """Reachable is not the same as winnable.

    A level-5 party like `paper_town.save`'s would make the gym goal
    reachable in topology and unwinnable in practice, which moves the
    problem rather than solving it. The required level is READ FROM THE
    MAP DATA -- both Hearthrock leaders field `hydrone,20` and
    `nudikill,20` -- never hardcoded here, so a vendor bump that raises
    the leaders' levels fails this instead of silently making the
    fixture too weak.
    """
    levels: set[int] = set()
    for gym in ("classic_gym_granite", "classic_gym_mila"):
        text = (MAPS / f"{gym}.tmx").read_text(errors="replace")
        levels.update(
            int(m) for m in re.findall(r"add_monster \w+,(\d+)", text)
        )
    assert levels, "no leader monsters found -- has the map data moved?"
    leader_max = max(levels)

    session = _boot()
    party = [(m.slug, m.level) for m in session.player.monsters]
    assert len(party) >= 2, party
    assert min(lvl for _, lvl in party) >= leader_max, (party, leader_max)


def test_booting_the_save_twice_reaches_the_same_digest() -> None:
    """The fixture is only useful if runs from it are comparable."""
    from tuxghost.digest import digest_of
    from tuxghost.loop import run_steps

    digests = []
    for _ in range(2):
        session = _boot()
        run_steps(session.client, 120)
        digests.append(digest_of(session))
    assert digests[0] == digests[1], digests
