"""The start fixture must boot to a bare, PLAYABLE overworld.

"Playable" is the load-bearing word: a save that restores to a WorldState
the player cannot move in would pass every "does it boot" assertion while
making the agent harness useless, and the walk assertion below is the only
thing that would notice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tuxemon.platform.const import buttons
from tuxemon.save_system.save_state import SaveData
from tuxemon.states.world_state import WorldState

from tuxghost.boot import boot_from_save
from tuxghost.determinism import pin_clock
from tuxghost.loop import InputSchedule, install_schedule, run_steps

FIXTURE = Path(__file__).parent / "fixtures" / "paper_town.save"
EPOCH = 1787659200


def _boot() -> tuple[Any, Any]:
    pin_clock(EPOCH)
    save = SaveData.model_validate(json.loads(FIXTURE.read_text()))
    return boot_from_save(save, seed=1234, clock_epoch=EPOCH)


def test_fixture_boots_to_a_bare_world_state_in_the_town() -> None:
    client, _session = _boot()
    assert list(client.active_state_names) == ["WorldState"]
    assert client.get_map_name() == "spyder_paper_town.tmx"


def test_fixture_has_a_party() -> None:
    client, _session = _boot()
    world = client.get_state_by_name(WorldState)
    assert [m.slug for m in world.player.monsters] == ["rockitten", "budaye"]


def test_player_can_walk_from_the_fixture_spawn() -> None:
    client, _session = _boot()
    world = client.get_state_by_name(WorldState)
    schedule: InputSchedule = {}
    install_schedule(client, schedule)
    start = world.player.tile_pos

    moved = None
    for button in (buttons.DOWN, buttons.UP, buttons.LEFT, buttons.RIGHT):
        schedule.clear()
        base = 5
        schedule[base] = [(button, 1.0)]
        schedule[base + 60] = [(button, 0.0)]
        run_steps(client, 120)
        if world.player.tile_pos != start:
            moved = button
            break
    assert moved is not None, (
        f"player could not leave tile {start} in any direction; the "
        "fixture spawn is boxed in -- regenerate it at a different tile"
    )
