"""The ghost NPC's construction, translucency, and show/hide lifecycle.

Three properties make a ghost safe to add to a live session, none of
which needs an engine patch:

* DRAWN, with correct occlusion, because `MapRenderer._get_npc_surfaces`
  (`tuxemon/map/view.py`) iterates `npc_manager.npcs` and feeds the
  layered draw -- `NPCManager.add_npc` is enough to make that happen.
* NOT SOLID, because `CollisionManager` keeps a separate `_entity_map`
  populated only by explicit `add_collision` calls, which this module
  never makes. A solid ghost would BLOCK the player, moving them,
  changing `tile_pos`, and corrupting every recorded trace it replays
  against.
* INVISIBLE TO THE DIGEST, because `tuxghost.digest`'s
  `get_persistent_npc_states` filters on `npc.persistence`, and
  `install_ghost` overrides it to `False` explicitly below (see
  `install_ghost`'s docstring for why this is not left to the sprite's
  own database default).
"""

from __future__ import annotations

from typing import Any

from tuxghost.ghost.track import GhostTrack

#: Reserved, and asserted free before use. Chosen so it cannot collide
#: with the player slug (`npc_red` in the committed save) or any map NPC.
GHOST_SLUG = "ghost"


def _apply_translucency(npc: Any, alpha: int = 128) -> None:
    """Make `npc`'s own sprite surfaces render semi-transparent.

    Measured (not guessed) in `docs/2026-08-27-ghost-probes.org`, probe
    1: `ghost.sprite_controller.get_sprite_renderer()` hands back the
    SAME `Surface` object on every render call for both `.standing`
    (the static per-facing frames) and, per `SurfaceAnimation`
    reachable from `.sprite`, its `._frame_manager.images` list (the
    walk-cycle frames) -- so calling `set_alpha()` on each once, here,
    at install time, is sufficient; there is no per-frame
    re-application needed, and no engine patch either. There is no
    public accessor to enumerate a `SurfaceAnimation`'s frames, so this
    reaches `._frame_manager.images`, a private attribute, as probe 1's
    "consequence for design" section calls out explicitly as an accepted
    trade-off rather than an oversight.

    Deliberately does not raise if a collection is missing or empty:
    a ghost that fails to go translucent should still be a working
    ghost, drawn opaque rather than not drawn (or not installed) at
    all -- translucency is cosmetic, occlusion/non-solidity/digest
    invisibility are not.
    """
    renderer = getattr(npc, "sprite_controller", None)
    if renderer is None:
        return
    sprite_renderer = renderer.get_sprite_renderer()

    standing = getattr(sprite_renderer, "standing", None) or {}
    for surface in standing.values():
        surface.set_alpha(alpha)

    sprites = getattr(sprite_renderer, "sprite", None) or {}
    for animation in sprites.values():
        frame_manager = getattr(animation, "_frame_manager", None)
        if frame_manager is None:
            continue
        for surface in getattr(frame_manager, "images", None) or []:
            surface.set_alpha(alpha)


def install_ghost(session: Any, track: GhostTrack, sprite_slug: str = "allie") -> Any:
    """Add the ghost NPC to the session and return it.

    Refuses (a `ValueError` naming `GHOST_SLUG`) if the slug is already
    taken on the map: `NPCRepository` is keyed by slug, so adding under
    a taken slug would silently DISPLACE whatever real NPC is already
    there instead of raising. Checked via `npc_exists`, which reads the
    on-map repository the same way `NPCManager.add_npc` itself resolves
    slugs.

    `persistence` is assigned in `NPC.__init__` from the npc DATABASE
    entry for `sprite_slug` (via `NpcModel.lookup`), not from an
    argument passed here, so it is overridden explicitly after
    construction rather than trusted to whatever that sprite's shipped
    data happens to default to.
    """
    from tuxemon.entity.npc import NPC

    manager = session.client.npc_manager
    if manager.npc_exists(GHOST_SLUG):
        raise ValueError(
            f"an NPC with the reserved slug {GHOST_SLUG!r} is already on "
            "this map; refusing to displace it"
        )

    npc = NPC.create(session, sprite_slug)
    npc.slug = GHOST_SLUG
    npc.persistence = False

    first = track.at(0)
    if first is not None:
        npc.set_position([float(first.tile[0]), float(first.tile[1])])

    _apply_translucency(npc)

    manager.add_npc(npc)
    return npc


def hide_ghost(client: Any, npc: Any) -> None:
    """Move the ghost off-map, which REMOVES it from the drawn set.

    `NPCManager` keeps two repositories, and the `npcs` property -- the
    one `MapRenderer._get_npc_surfaces` iterates -- returns `_on_map`
    only. So this is a real hide, not a park behind the camera: the NPC
    object and its state survive, but nothing draws it.
    """
    client.npc_manager.add_npc_off_map(npc)


def show_ghost(client: Any, npc: Any) -> None:
    """Move the ghost back on-map, restoring it to the drawn set."""
    client.npc_manager.add_npc(npc)


def advance_ghost(
    client: Any, npc: Any, track: GhostTrack, step: int, player_map: str
) -> None:
    """Place the ghost for `step`, or hide it when it cannot be placed
    honestly -- past the end of its track, or on a different map.

    Hidden, never frozen at a stale position and never teleported: a
    ghost standing motionless on the wrong map reads as a bug, and one
    left at a stale tile is drawing a position the trace never had.

    Also drives ANIMATION STATE, not just position -- this is the part
    the brief's original sketch got wrong, per Task 1's probe 2
    (`docs/2026-08-27-ghost-probes.org`). Measured there: assigning
    `set_position` alone, every step, leaves `npc.mover.is_moving_state`
    False forever, so `MapRenderer._get_sprites`
    (`tuxemon/map/view.py`) always takes its static `get_facing_frame`
    branch and hands back the *identical* Surface object every step --
    a sliding statue, not a walk cycle. The probe's measured fix, with
    no engine patch, is forcing `npc.mover.state` into `WALKING` and
    playing the matching walk `SurfaceAnimation`
    (`SpriteController.play_animation`, existing public API) whenever
    the ghost's tile actually changed since the call before this one,
    and returning it to `IDLE`/stopped otherwise. "Since the call
    before this one" is read off `npc.tile_pos` itself -- whatever this
    function last set it to -- rather than tracked separately, since
    the NPC object already carries exactly that state between calls.
    """
    frame = track.at(step)
    if frame is None or frame.map_name != player_map:
        hide_ghost(client, npc)
        return

    show_ghost(client, npc)

    previous_tile = (int(npc.tile_pos[0]), int(npc.tile_pos[1]))
    npc.set_position([float(frame.tile[0]), float(frame.tile[1])])

    direction = _direction_from_facing(frame.facing)
    if direction is not None:
        npc.set_facing(direction)

    if previous_tile != frame.tile:
        _walk(npc, direction)
    else:
        _stand(npc)


def _walk(npc: Any, direction: Any) -> None:
    """Enter the walking state and play the matching walk animation.

    `direction` may be `None` (an unrecognised facing, see
    `_direction_from_facing`) -- state still moves to WALKING so
    `MapRenderer._get_sprites` takes the animated branch, but no
    specific animation is selected in that case, matching this
    project's general choice (see `_direction_from_facing`) that a
    cosmetic gap should never escalate into a crash.
    """
    from tuxemon.entity.entity import EntityState

    npc.mover.set_state(EntityState.WALKING)
    if direction is not None:
        npc.sprite_controller.play_animation(direction)


def _stand(npc: Any) -> None:
    """Return the ghost to its idle standing pose.

    Both calls matter: `set_state(IDLE)` is what flips
    `npc.mover.is_moving_state` back to False, which is what
    `MapRenderer._get_sprites` branches on; `stop_animation()` halts
    the walk `SurfaceAnimation` so it does not keep silently advancing
    (unobserved, since the static branch is what actually gets drawn)
    while the ghost stands still.
    """
    from tuxemon.entity.entity import EntityState

    npc.mover.set_state(EntityState.IDLE)
    npc.sprite_controller.stop_animation()


def _direction_from_facing(facing: str) -> Any:
    """`facing` is stored as `str(Direction.X)` -- confirmed, not
    guessed: `Direction` (`tuxemon.db`) is a `str, Enum` mixin, and on
    this project's Python version `str(Direction.UP) ==
    "Direction.UP"`, not the bare value `"up"` -- matching
    `tuxghost.digest`'s own `str(player.facing)`. Converts back by
    name, over `tuxemon.db.Direction` (the brief's guessed
    `tuxemon.map` does not export it).

    Returns `None` for an unrecognised value rather than raising: a
    facing this build does not know is a cosmetic problem, and a ghost
    that crashes the player's session over a sprite direction is a
    worse one.
    """
    from tuxemon.db import Direction

    for direction in Direction:
        if str(direction) == facing:
            return direction
    return None
