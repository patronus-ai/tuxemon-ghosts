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


def test_the_ghost_is_hidden_past_the_end_of_its_track() -> None:
    """Hidden, not frozen and not teleported: a ghost standing motionless
    at its final tile forever reads as a bug, and one left at a stale
    position is worse -- it is drawing a position the trace never had.

    `track` is built BEFORE `_session()` -- see the matching comment on
    `test_install_refuses_when_the_reserved_slug_is_taken`.
    """
    from tuxghost.ghost.entity import advance_ghost

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)

    advance_ghost(client, npc, track, len(track.frames), "spyder_paper_town.tmx")
    assert GHOST_SLUG not in client.npc_manager.npcs


def test_the_ghost_is_hidden_on_a_different_map() -> None:
    """See the matching comment on
    `test_install_refuses_when_the_reserved_slug_is_taken` for why
    `track` is built before `_session()`."""
    from tuxghost.ghost.entity import advance_ghost

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)

    advance_ghost(client, npc, track, 10, "somewhere_else.tmx")
    assert GHOST_SLUG not in client.npc_manager.npcs


def test_the_ghost_follows_its_track_on_the_matching_map() -> None:
    """See the matching comment on
    `test_install_refuses_when_the_reserved_slug_is_taken` for why
    `track` is built before `_session()`."""
    from tuxghost.ghost.entity import advance_ghost

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)

    frame = track.at(100)
    assert frame is not None
    advance_ghost(client, npc, track, 100, frame.map_name)
    assert GHOST_SLUG in client.npc_manager.npcs
    assert (int(npc.tile_pos[0]), int(npc.tile_pos[1])) == frame.tile


def test_the_ghost_animates_a_walk_cycle_when_it_moves() -> None:
    """R5 (controller ruling): position-only movement is a SLIDING
    STATUE, measured in `docs/2026-08-27-ghost-probes.org` probe 2 --
    `MapRenderer._get_sprites` (`tuxemon/map/view.py`) hands back the
    *same* Surface object every step unless `npc.mover.state` enters
    `WALKING` and the matching walk `SurfaceAnimation` is playing.

    Asserted on Surface OBJECT IDENTITY (`id()`), not pixel content, per
    probe 2's own method: a full-frame or per-sprite pixel diff would
    differ from step to step purely because the ghost's on-screen
    position moved, regardless of whether the pose is animating -- only
    identity distinguishes "the exact same static pose, redrawn
    elsewhere" from "actually cycling frames".

    Frames 16 -> 17 of this golden track are a real, measured moved
    transition on the same map: `(12, 12)` -> `(12, 13)`. A real
    `MapRenderer` is installed via `tuxghost.observe.FrameRenderer`
    (the same, already-existing seam Task 1's probe used) because the
    headless boot's default `NullRenderer` has no `_get_sprites` at
    all. 90 real client ticks (`tuxghost.loop.run_steps`, `FIXED_DT`
    each) is comfortably more than one full walk cycle at this sprite's
    measured `frame_duration` (~0.178s, i.e. ~10-11 ticks per frame).
    """
    from tuxghost.ghost.entity import advance_ghost
    from tuxghost.loop import run_steps
    from tuxghost.observe import FrameRenderer

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)
    FrameRenderer(client)

    frame16 = track.at(16)
    frame17 = track.at(17)
    assert frame16 is not None and frame17 is not None
    assert frame16.map_name == frame17.map_name
    assert frame16.tile != frame17.tile

    advance_ghost(client, npc, track, 16, frame16.map_name)
    advance_ghost(client, npc, track, 17, frame17.map_name)

    surface_ids = set()
    for _ in range(90):
        run_steps(client, 1)
        surfaces = client.map_renderer._get_sprites(npc, 0)
        surface_ids.add(id(surfaces[-1].surface))

    assert len(surface_ids) > 1, (
        "the ghost's character-sprite Surface identity never changed "
        "across 90 ticks after a moved step -- a sliding statue, not a "
        "walk cycle"
    )


def test_the_ghost_stands_still_when_it_does_not_move() -> None:
    """The mirror image of the walk-cycle test above: when the tile does
    NOT change between calls, the ghost must return to (or stay at) a
    single static pose -- the same Surface object identity every tick,
    not a walk cycle left running in place.
    """
    from tuxghost.ghost.entity import advance_ghost
    from tuxghost.loop import run_steps
    from tuxghost.observe import FrameRenderer

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)
    FrameRenderer(client)

    frame0 = track.at(0)
    assert frame0 is not None

    advance_ghost(client, npc, track, 0, frame0.map_name)

    surface_ids = set()
    for _ in range(90):
        run_steps(client, 1)
        surfaces = client.map_renderer._get_sprites(npc, 0)
        surface_ids.add(id(surfaces[-1].surface))

    assert len(surface_ids) == 1, (
        "the ghost's character-sprite Surface identity changed while "
        "standing still -- a walk animation left running in place"
    )
