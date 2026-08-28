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
