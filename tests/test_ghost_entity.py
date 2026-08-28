"""tests/test_ghost_entity.py"""

import json
from pathlib import Path
from typing import Any

import pytest

from tuxghost.boot import boot_from_save
from tuxghost.determinism import pin_clock, seed_all
from tuxghost.ghost.entity import GHOST_SLUG, install_ghost
from tuxghost.ghost.track import build_track
from tuxghost.trace import read

SAVE = Path(__file__).parent / "fixtures" / "paper_town.save"
PARENT = Path(__file__).parent / "golden" / "scripted_town_1234.tuxghost"

SEED = 1234
CLOCK_EPOCH = 1787659200


def _session() -> Any:
    """Boot a fresh session from the committed save fixture.

    `boot_from_save` returns `(client, session)` and takes a validated
    `SaveData`, not a raw dict -- mirrors the preamble in
    `tuxghost/ghost/track.py::build_track`: seed, pin the clock, THEN
    validate and boot.
    """
    from tuxemon.save_system.save_state import SaveData

    seed_all(SEED)
    pin_clock(CLOCK_EPOCH)

    raw = json.loads(SAVE.read_text())
    save_data = SaveData.model_validate(raw)

    _client, session = boot_from_save(save_data, seed=SEED, clock_epoch=CLOCK_EPOCH)
    return session


def test_install_refuses_when_the_reserved_slug_is_taken() -> None:
    """A slug collision would mean `install_ghost` silently DISPLACING a
    real map NPC -- `NPCRepository` is keyed by slug, so adding under a
    taken slug overwrites. Refusing is the same choice `optimize` makes
    for an unused `--seed`: never silently do something other than what
    the caller asked.

    `track` is built BEFORE `_session()`, not after: `build_track`
    itself calls `boot_from_save` (see its own preamble), which resets
    the `local_session` module-level singleton and swaps in a fresh
    `client` as a side effect. Building the track first, then booting
    the session under test, guarantees the LAST boot in this test is
    `_session()`'s own -- so `session.client` stays the same object for
    the rest of the test instead of being silently replaced partway
    through by an unrelated `build_track` call.
    """
    track = build_track(read(PARENT))
    session = _session()

    from tuxemon.entity.npc import NPC

    squatter = NPC.create(session, "allie")
    squatter.slug = GHOST_SLUG
    session.client.npc_manager.add_npc(squatter)

    with pytest.raises(ValueError, match=GHOST_SLUG):
        install_ghost(session, track)


def test_hiding_removes_the_ghost_from_the_drawn_set() -> None:
    """Asserted on `npc_manager.npcs` -- the exact collection the
    renderer iterates -- and not on anything about the camera or what
    happens to be on screen.

    `track` is built BEFORE `_session()`, not after -- see the matching
    comment on `test_install_refuses_when_the_reserved_slug_is_taken`.
    Building it inline as `install_ghost(session, build_track(...))`,
    as the brief originally sketched, evaluates `build_track` (which
    reboots the `local_session` singleton) AFTER `session`/`client`
    were already captured, so the `client` local goes stale and every
    assertion below silently checks the wrong, abandoned client.
    """
    from tuxghost.ghost.entity import hide_ghost, show_ghost

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)
    assert GHOST_SLUG in client.npc_manager.npcs

    hide_ghost(client, npc)
    assert GHOST_SLUG not in client.npc_manager.npcs

    show_ghost(client, npc)
    assert GHOST_SLUG in client.npc_manager.npcs
