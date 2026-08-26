"""Canonical game-state digest.

The exemption register below is a list of defects not yet fixed, not a list
of things that are allowed to vary. Its target size is zero.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Keys are dotted PATHS into the tree `state_of()` builds, not bare field
# names -- a bare-name register would strip every field with that name at
# every nesting level, and upstream reuses common names (e.g.
# `status.lifecycle.Lifecycle.duration`, a real per-status turn counter)
# for meanings that must NOT be exempted just because a wall-clock
# `duration` elsewhere shares the name. List elements do not get an index
# segment in their path, since exempting "the `instance_id` of any
# monster" is the intent, not "the `instance_id` of monster 0".
#
# `SessionSave` fields (`uuid`, `start_time`, `duration`, `total_playtime`)
# are deliberately NOT listed here even though patch 0004 will touch
# `start_time`/`duration`/`total_playtime` upstream: `state_of()` never
# digests `session.session_state`, only `npc_state` and `world_state`, so
# none of those paths are reachable in the tree below and an entry for
# them would be dead -- untestable and a trap for
# `test_exemptions_are_all_reachable_in_the_digested_tree`. If a later task
# starts digesting `SessionSave`, add the exemption then, at the real path.
# (`SessionSave.uuid` itself needs no entry at all, seeded or not: patch
# 0003's `AbstractSession.reset()` already draws it from the seeded id
# factory -- see `tuxemon/session.py` -- so it would be reproducible even
# if `session_state` became reachable.)
#
# The four entries patch 0003 closed (`npc_state.instance_id`,
# `npc_state.monsters.instance_id`, `npc_state.monsters.moves.instance_id`,
# `npc_state.game_variables.add_monster`) are gone: NPC (`entity.py`),
# Monster (`monster.py`), Technique -- a monster's moves -- (`technique.py`),
# Status (`status.py`), Battle (`battle.py`), and Item (`item.py`)
# `instance_id`s, plus the `chosen_tech` game variable that stores a
# Technique's `instance_id.hex`, all now come from
# `tuxemon.core.ids.new_id()`, seeded by `seed_all`/`build_client`/
# `boot_from_save`. Patch 0003 also fixed two load-bearing call sites the
# brief specifying this patch didn't list: `tuxemon/monster/monster.py`'s
# own `uuid4()` (Monster does not subclass Entity, so `entity.py`'s fix
# alone did not cover it -- without this one, `npc_state.monsters
# .instance_id` and `npc_state.game_variables.add_monster` would still have
# diverged) and `tuxemon/item/item.py`'s own `uuid4()`, found in fix-round-1
# review (`npc_state.items[].instance_id` diverges on the same
# `add_item`-exercising route as `add_monster`'s own stability test above).
#
# This list is a statement about what `EXEMPTIONS == {}` actually covers,
# not a claim that every `uuid4()` in the vendored tree is gone: two sites
# still call it raw, deliberately left unpatched because no route any test
# in this suite drives ever reaches them (an exemption for either would be
# a dead entry, caught by `test_exemptions_are_all_reachable_in_the_digested
# _tree`, and a fix would be an unpinned hunk -- see fix-round-1 finding 1's
# corrective, which is exactly why `item.py` was NOT left in this same
# category): `tuxemon/mission/mission.py:47`
# (`npc_state.missions[].instance_id`, never populated -- no test route
# ever completes a mission) and `tuxemon/event/eventparser.py:52` (an
# internal event-bus id, not a persisted/digested state field at all). If
# a future task starts exercising either, route it through `new_id()` and
# update this note.
EXEMPTIONS: dict[str, str] = {}


def _canonical(obj: Any, path: str = "") -> Any:
    if isinstance(obj, dict):
        result = {}
        for k, v in sorted(obj.items()):
            child_path = f"{path}.{k}" if path else k
            if child_path in EXEMPTIONS:
                continue
            result[k] = _canonical(v, child_path)
        return result
    if isinstance(obj, (list, tuple)):
        # List elements share their parent's path -- see the note above
        # EXEMPTIONS on why index segments are deliberately omitted.
        return [_canonical(v, path) for v in obj]
    if isinstance(obj, float):
        return repr(obj)  # exact bits; no formatting loss
    return obj


def state_of(session: Any) -> dict[str, Any]:
    """Snapshot the comparable game state via the game's own serialisation.

    Covers `npc_state` (the player) and `world_state`, plus
    `persistent_npc_state` -- every OTHER npc the save system considers
    persistent (`NPCManager.get_persistent_npc_states`, the same call
    `save_system.save.get_save_data` makes to populate
    `SaveData.persistent_state`). Task 12 found this tree entirely absent
    from an earlier version of this function: NPC position/AI divergence
    among non-player characters was invisible to `verify`, discovered by
    moving a persistent NPC via `entity.Entity.set_position` and observing
    the pre-widening digest not move (`tests/test_digest.py
    ::test_widened_digest_discriminates_persistent_npc_state`, which pins
    the old, blind behaviour as a regression test against reintroducing
    it). Not left as a documented limit: the fix is a direct parallel of
    the existing `npc_state`/`world_state` pattern (call the game's own
    `get_state`, canonicalize, hash) rather than new introspection, and
    the fix DOES discriminate -- see that test.

    `state_stack` remains a KNOWN, DOCUMENTED gap, deliberately not
    widened here: it records only state NAMES
    (`session.client.state_manager.active_states`), so a `CombatState`
    with a different turn count or a different opponent's HP is invisible
    to `verify` except indirectly, through the player's own party's HP
    (which IS covered, via `npc_state.monsters`). `CombatState`
    (`tuxemon/states/combat_state.py`, ~1300 lines) holds its live turn/
    battle bookkeeping on plain Python attributes alongside sprites,
    animations and other non-JSON-serializable objects with no existing
    `get_state()`-style serializer of its own (unlike NPC/World, which
    upstream already round-trips through `SaveData`) -- there is no
    established, low-risk seam to hook the way `persistent_npc_state`
    above hooks one. Widening it would mean hand-picking and serializing
    specific `CombatState` internals, which is real engine-shaped work
    outside a pure-Python task, and an unvetted subset chosen under time
    pressure risks exactly the "worthless probe" failure mode this project
    has already hit once (`tests/test_digest.py`'s
    `test_digest_discriminates_between_seeds` docstring): a widened digest
    that looks more thorough but has not been proven to discriminate is
    worse than a narrow one that has. See `tuxghost.execute`'s module
    docstring for what a `verify() == 0` result does and does not certify
    as a consequence.
    """
    player = session.player
    persistent_npcs = session.client.npc_manager.get_persistent_npc_states(session)
    state: dict[str, Any] = {
        "map": session.client.get_map_name(),
        "tile_pos": list(player.tile_pos),
        "facing": str(player.facing),
        "state_stack": [s.name for s in session.client.state_manager.active_states],
        "npc_state": _canonical(
            json.loads(player.get_state(session).model_dump_json()),
            path="npc_state",
        ),
        "world_state": _canonical(
            json.loads(session.world.get_state(session).model_dump_json()),
            path="world_state",
        ),
        "persistent_npc_state": _canonical(
            [json.loads(npc.model_dump_json()) for npc in persistent_npcs],
            path="persistent_npc_state",
        ),
    }
    return state


def digest_of(session: Any) -> str:
    blob = json.dumps(state_of(session), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
