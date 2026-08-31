"""tests/test_ghost_entity.py"""

import json
from pathlib import Path
from typing import Any

import pytest

from tuxghost.boot import boot_from_save
from tuxghost.determinism import pin_clock, seed_all
from tuxghost.ghost.entity import GHOST_SLUG, install_ghost
from tuxghost.ghost.track import GhostFrame, GhostTrack, build_track
from tuxghost.trace import read

SAVE = Path(__file__).parent / "fixtures" / "paper_town.save"
PARENT = Path(__file__).parent / "golden" / "scripted_town_1234.tuxghost"

SEED = 1234
CLOCK_EPOCH = 1787659200


def _adversarial_tracks(
    session: Any, client: Any, step_count: int, metadata: GhostTrack
) -> tuple[GhostTrack, GhostTrack]:
    """Two tracks for the two safety tests below (I4, whole-branch
    review): `decoy` is what `install_ghost` places the ghost at, and
    `adversarial` is what drives `advance_ghost` every tick thereafter,
    parked on the tile directly below the player's spawn for the WHOLE
    run -- the parent trace's first four actions are DOWN, so this is
    squarely in its path, for as long as the run lasts, not just its
    first step.

    Two tracks, not one, because of a MEASURED timing fact: instrumenting
    `Pathfinder.is_tile_traversable` shows the collision check for the
    player's very first move (spawn -> one tile down) fires during
    `client.update` for step 0 -- i.e. the ghost must already occupy that
    tile BEFORE `run_steps`' hook(0) call returns. `install_ghost` itself
    places the ghost via plain `set_position` (a different line, outside
    `advance_ghost`, unaffected by the mutation this fixture exists to
    catch), so parking the ghost on the adversarial tile from the start
    via `install_ghost` alone would never reach `advance_ghost`'s own
    per-step line at all. `decoy` (a single frame, far from the action)
    is what `install_ghost` uses instead, so the FIRST real
    `advance_ghost` call -- at hook(0), before step 0's `client.update` --
    is a genuine tile change (decoy tile -> adversarial tile), reaching
    the mutated line before the critical check rather than a same-tile
    no-op `_stand()` that never touches it.
    """
    adversarial_tile = (
        int(session.player.tile_pos[0]),
        int(session.player.tile_pos[1]) + 1,
    )
    map_name = client.get_map_name()
    decoy_track = GhostTrack(
        frames=(GhostFrame(map_name=map_name, tile=(0, 0), facing="Direction.DOWN"),),
        source_digest=metadata.source_digest,
        recorder=metadata.recorder,
        model=metadata.model,
    )
    adversarial_frame = GhostFrame(
        map_name=map_name, tile=adversarial_tile, facing="Direction.DOWN"
    )
    adversarial_track = GhostTrack(
        frames=(adversarial_frame,) * (step_count + 1),
        source_digest=metadata.source_digest,
        recorder=metadata.recorder,
        model=metadata.model,
    )
    return decoy_track, adversarial_track


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


def test_the_ghost_stands_at_its_final_tile_past_the_end_of_its_track() -> None:
    """REVERSES S4's original decision, and the reversal is the point.

    S4 hid the ghost past the end of its track, reasoning that "a ghost
    standing motionless at its final tile forever reads as a bug". S4's
    manual acceptance had never been run when that was written. When a
    human finally ran it, PARENT's 176 steps -- 2.9 seconds at 60fps --
    meant the ghost finished its route and vanished while the window was
    still opening, and the report was "I don't see the ghosts". Every
    automated test passed throughout, this one included: it asserted the
    disappearance was correct.

    The final tile is a REAL recorded position, not the stale mid-track
    one the original reasoning worried about. A ghost waiting at its
    finish line is what a ghost race looks like.

    `track` is built BEFORE `_session()` -- see the matching comment on
    `test_install_refuses_when_the_reserved_slug_is_taken`.
    """
    from tuxghost.ghost.entity import advance_ghost

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)
    final = track.frames[-1]

    for step in (len(track.frames), len(track.frames) + 500):
        advance_ghost(client, npc, track, step, "spyder_paper_town.tmx")
        assert GHOST_SLUG in client.npc_manager.npcs, step
        assert (int(npc.tile_pos[0]), int(npc.tile_pos[1])) == final.tile, step


def test_an_empty_track_still_hides_the_ghost() -> None:
    """The linger above reads `track.frames[-1]`, so an EMPTY track has
    no final tile to stand on. It must fall through to hiding rather
    than raise `IndexError` or invent a position.
    """
    from tuxghost.ghost.entity import advance_ghost
    from tuxghost.ghost.track import GhostTrack

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)

    empty = GhostTrack(
        frames=(),
        source_digest=track.source_digest,
        recorder=track.recorder,
        model=track.model,
    )
    advance_ghost(client, npc, empty, 0, "spyder_paper_town.tmx")
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

    Whole-branch review, I1: an earlier version of this test called
    `advance_ghost` exactly once, at step 0, where the ghost has never
    walked -- `npc.mover.state` was already IDLE and no animation was
    running, so the standing assertion held trivially, without
    exercising `_stand` at all. PROVEN vacuous: no-op'ing `_stand`
    (`tuxghost/ghost/entity.py`) left that version passing.

    Fixed by first driving the ghost into a WALKING state (frames 16 ->
    17 of this golden track, the same real moved transition
    `test_the_ghost_animates_a_walk_cycle_when_it_moves` uses -- `(12,
    12) -> (12, 13)` on the same map), asserting it actually got there
    (`npc.mover.is_moving_state`, the sanity check that makes the next
    assertion meaningful rather than accidental), and only THEN advancing
    to frame 18, where the tile does not move again (`17 -> 18` is
    `(12, 13) -> (12, 13)`), and asserting the WALKING state was actually
    undone. A no-op `_stand` leaves `mover.state` at WALKING here, so this
    assertion is the one that would have failed."""
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
    frame18 = track.at(18)
    assert frame16 is not None and frame17 is not None and frame18 is not None
    assert frame16.map_name == frame17.map_name == frame18.map_name
    assert frame16.tile != frame17.tile, "16 -> 17 must be a real move"
    assert frame17.tile == frame18.tile, "17 -> 18 must NOT move"

    advance_ghost(client, npc, track, 16, frame16.map_name)
    advance_ghost(client, npc, track, 17, frame17.map_name)
    assert npc.mover.is_moving_state, (
        "sanity check: the ghost must actually enter a WALKING state "
        "after a moved step, or the no-move assertion below would hold "
        "vacuously regardless of whether _stand does anything"
    )

    advance_ghost(client, npc, track, 18, frame18.map_name)
    assert not npc.mover.is_moving_state, (
        "the ghost was still in a WALKING state after a step where its "
        "tile did not change -- _stand never ran, or ran and had no "
        "effect"
    )

    surface_ids = set()
    for _ in range(90):
        run_steps(client, 1)
        surfaces = client.map_renderer._get_sprites(npc, 0)
        surface_ids.add(id(surfaces[-1].surface))

    assert len(surface_ids) == 1, (
        "the ghost's character-sprite Surface identity changed while "
        "standing still -- a walk animation left running in place"
    )


def test_a_ghost_does_not_change_the_per_step_digest_sequence() -> None:
    """The claim that makes a real engine entity acceptable.

    Asserted on the PER-STEP digest sequence, never on the final digest
    alone. A final-digest comparison would be vacuous here: this project
    measured candidates 146 steps apart reaching the identical digest,
    because a settled end state cannot detect a perturbation. See
    docs/STATUS.org, "Settled end states cannot detect a perturbation".

    `_session()` is called AFTER `build_track`, inside `digests()`, each
    time it runs -- `build_track` itself boots (and so resets)
    `local_session`, and `digests()` is called twice (once per branch),
    so each branch must get its OWN fresh boot rather than share one
    stale `client`/`session` pair with the other branch.

    Whole-branch review, I4: an earlier version of this test only ever
    called `install_ghost` -- a static, one-time placement -- and never
    `advance_ghost`, the ONLY per-step ghost API `tuxghost.play` actually
    uses. Proven hole: mutating `advance_ghost` (`tuxghost/ghost/
    entity.py:162`) from `set_position` to `complete_tile_entry`, which
    registers the ghost in `CollisionManager._entity_map`, left this test
    (and the collision test below) passing regardless -- because neither
    one ever called `advance_ghost` at all, that line was simply never
    reached.

    `_adversarial_tracks` (see its docstring) places the ghost on the
    tile directly below spawn -- squarely in the parent trace's path --
    for the WHOLE run, and `advance_ghost` is driven from this test's own
    `hook`, every step, the same per-step call `tuxghost.play` makes.
    Co-locating the ghost with the player's own CURRENT tile (tried
    first, mirroring the player's trajectory) turned out NOT to
    discriminate the mutation below at all, for any lead time tried (1
    through 10 steps): collision is decided once, at the very first
    `client.update`, well before a moving ghost following the player's
    OWN path could ever get there first. A fixed adversarial tile, held
    for the whole run, is what actually catches it.
    """
    from tuxghost.digest import digest_of
    from tuxghost.execute import _schedule_of
    from tuxghost.ghost.entity import advance_ghost
    from tuxghost.loop import install_schedule, run_steps

    trace = read(PARENT)
    metadata = build_track(trace)

    def digests(with_ghost: bool) -> list[str]:
        session = _session()
        client = session.client
        install_schedule(client, _schedule_of(trace))
        npc = None
        adversarial_track = None
        if with_ghost:
            decoy_track, adversarial_track = _adversarial_tracks(
                session, client, trace.header.step_count, metadata
            )
            npc = install_ghost(session, decoy_track)
        seen: list[str] = []

        def hook(i: int) -> None:
            if npc is not None:
                assert adversarial_track is not None
                advance_ghost(client, npc, adversarial_track, i, client.get_map_name())
            seen.append(digest_of(session))

        run_steps(client, trace.header.step_count, hook=hook)
        seen.append(digest_of(session))
        return seen

    assert digests(with_ghost=True) == digests(with_ghost=False)


def test_the_player_walks_through_the_ghost() -> None:
    """The claim that would silently corrupt every recorded trace. A
    solid ghost blocks the player, changing where they end up -- so the
    ghost is placed DIRECTLY in the player's path and the run is compared
    against the same run with no ghost at all.

    `_session()` is called AFTER `build_track`, inside `end_tile()`, each
    time it runs -- see the matching comment on
    `test_a_ghost_does_not_change_the_per_step_digest_sequence` for why:
    two boots are needed here (with-ghost, without-ghost), and each must
    read its own session, not a stale reference left over from the other
    or from `build_track`'s internal boot.

    Whole-branch review, I4: an earlier version of this test placed the
    ghost adversarially exactly ONCE, via a static `set_position` call
    right after `install_ghost`, and never called `advance_ghost` again
    for the rest of the run -- so only the very first step was ever
    actually adversarial, and the ONLY per-step ghost API `tuxghost.play`
    uses (`advance_ghost`) went completely unexercised. Proven hole: see
    `test_a_ghost_does_not_change_the_per_step_digest_sequence`'s
    docstring -- the same `complete_tile_entry` mutation left this test
    passing too.

    `_adversarial_tracks` (see its docstring on the digest test above)
    places the ghost one tile below spawn -- squarely in the parent
    trace's path, since its first four actions are DOWN -- for the WHOLE
    run, and `advance_ghost` is driven from this test's own `hook`,
    every step, matching the ONLY per-step call `tuxghost.play` makes.
    The golden track (a DIFFERENT recording) never crosses this player's
    path, and even the player's OWN trajectory, mirrored with a lead of
    1 through 10 steps, failed to discriminate the mutation this test
    exists to catch (collision is decided once, at the very first
    `client.update`) -- a fixed adversarial tile held for the whole run
    is what actually works.
    """
    from tuxghost.execute import _schedule_of
    from tuxghost.ghost.entity import advance_ghost
    from tuxghost.loop import install_schedule, run_steps

    trace = read(PARENT)
    metadata = build_track(trace)

    def end_tile(with_ghost: bool) -> tuple[int, int]:
        session = _session()
        client = session.client
        install_schedule(client, _schedule_of(trace))
        npc = None
        adversarial_track = None
        if with_ghost:
            decoy_track, adversarial_track = _adversarial_tracks(
                session, client, trace.header.step_count, metadata
            )
            npc = install_ghost(session, decoy_track)

        def hook(i: int) -> None:
            if npc is not None:
                assert adversarial_track is not None
                advance_ghost(client, npc, adversarial_track, i, client.get_map_name())

        run_steps(client, trace.header.step_count, hook=hook)
        return (int(session.player.tile_pos[0]), int(session.player.tile_pos[1]))

    assert end_tile(with_ghost=True) == end_tile(with_ghost=False) == (16, 14)


def test_loop_replays_the_track_instead_of_lingering() -> None:
    """`loop=True` is what the browser build uses, and it exists because
    lingering was measured to fail with a real human.

    The browser boots for over a minute while the ghost's whole walk
    lasts seconds, so a viewer reliably arrives after the motion and
    finds a stationary figure. The first person to see it asked "why is
    there a random ghost there?" -- see docs/STATUS.org, "The two human
    checks".

    Pinned against the `loop` branch: with it removed the ghost stays on
    its final tile forever and the assertion below fails.

    `track` is built BEFORE `_session()` -- see the matching comment on
    `test_install_refuses_when_the_reserved_slug_is_taken`.
    """
    from tuxghost.ghost.entity import advance_ghost

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)
    n = len(track.frames)

    # One full track-length past the end lands back on frame 0.
    advance_ghost(client, npc, track, n, "spyder_paper_town.tmx", loop=True)
    assert GHOST_SLUG in client.npc_manager.npcs
    assert (int(npc.tile_pos[0]), int(npc.tile_pos[1])) == track.frames[0].tile

    # ...and it keeps moving rather than parking: a step partway into the
    # second lap sits where that step sits on the first.
    mid = n + 60
    advance_ghost(client, npc, track, mid, "spyder_paper_town.tmx", loop=True)
    expected = track.frames[mid % n].tile
    assert (int(npc.tile_pos[0]), int(npc.tile_pos[1])) == expected
    assert expected != track.frames[-1].tile, (
        "this fixture's frame 60 coincides with its last frame, so the "
        "assertion above cannot tell looping from lingering -- pick "
        "another offset"
    )


def test_lingering_is_still_the_default() -> None:
    """The control. `tuxghost.play` relies on the default, and changing a
    documented behaviour by side effect is how this function's earlier
    reversal got expensive.
    """
    from tuxghost.ghost.entity import advance_ghost

    track = build_track(read(PARENT))
    session = _session()
    client = session.client
    npc = install_ghost(session, track)

    advance_ghost(
        client, npc, track, len(track.frames) + 60, "spyder_paper_town.tmx"
    )
    assert (int(npc.tile_pos[0]), int(npc.tile_pos[1])) == track.frames[-1].tile
