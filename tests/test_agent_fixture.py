"""The start fixture must boot to a bare, PLAYABLE overworld.

"Playable" is the load-bearing word: a save that restores to a WorldState
the player cannot move in would pass every "does it boot" assertion while
making the agent harness useless, and the walk assertion below is the only
thing that would notice.
"""

from __future__ import annotations

import json
from pathlib import Path

# `Any` for the untyped upstream client/session pair -- the established
# convention in this repo (see e.g. tuxghost/boot.py's own return types)
# for objects `tuxemon` itself leaves unannotated.
from typing import Any

from tuxemon.platform.const import buttons
from tuxemon.save_system.save_state import SaveData
from tuxemon.states.world_state import WorldState

from tuxghost.boot import boot_from_save
from tuxghost.determinism import pin_clock
from tuxghost.loop import InputSchedule, install_schedule, run_steps

FIXTURE = Path(__file__).parent / "fixtures" / "paper_town.save"
EPOCH = 1787659200
DIRECTIONS = (buttons.DOWN, buttons.UP, buttons.LEFT, buttons.RIGHT)


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


def _moves_the_player(button: int, start: tuple[int, int]) -> bool:
    """Boots the fixture fresh and holds `button` for one held/release
    cycle, returning whether the player left `start`.

    A FRESH boot per direction, deliberately, rather than one client
    reused across all four directions in sequence (the fix-round-1
    review's Important #1 finding, reproduced and fixed here):
    `install_schedule`'s delivery counter (`tuxghost/loop.py`) is
    monotonic across `run_steps` calls, so reusing one client with a
    fixed step base across iterations delivers scheduled events to only
    the FIRST direction tried -- every later direction silently receives
    no input at all, and the original `break`-on-first-success form never
    surfaced this because it never reached iteration 2. Advancing the
    base per iteration (or reinstalling the schedule) would fix delivery,
    but a single reused client would still let an earlier successful
    direction move the player away from `start`, contaminating every
    later direction's comparison against the ORIGINAL spawn tile with
    wherever the player happens to be standing by then. A fresh boot per
    direction keeps every direction's result independently anchored to
    the fixture's actual spawn tile.
    """
    client, _session = _boot()
    world = client.get_state_by_name(WorldState)
    assert world.player.tile_pos == start
    schedule: InputSchedule = {5: [(button, 1.0)], 65: [(button, 0.0)]}
    install_schedule(client, schedule)
    run_steps(client, 120)
    return bool(world.player.tile_pos != start)


def test_player_can_walk_from_the_fixture_spawn() -> None:
    client, _session = _boot()
    world = client.get_state_by_name(WorldState)
    start = world.player.tile_pos

    moved_via = {b for b in DIRECTIONS if _moves_the_player(b, start)}

    # Pin the actual measured set, not merely "at least one direction
    # works": a fixture that regressed to being walkable in only ONE
    # direction (e.g. a future map edit narrowing the open corridor)
    # would still pass an `assert moved_via` check but silently make the
    # harness far more fragile than this fixture was built to be.
    assert moved_via == {buttons.DOWN, buttons.UP, buttons.RIGHT}, (
        f"player could not leave tile {start} in the expected set of "
        f"directions (measured DOWN/UP/RIGHT open, LEFT blocked by "
        f"collision geometry); got {moved_via!r} instead -- the fixture "
        "spawn's surroundings changed, or is boxed in differently; "
        "regenerate at a different tile if this is now empty"
    )
