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
# `npc_state.game_variables.add_monster`) are gone: every `instance_id` in
# the digested tree -- NPC, Monster, Technique (a monster's moves), Status,
# Battle -- and the `chosen_tech` game variable that stores a Technique's
# `instance_id.hex` now come from `tuxemon.core.ids.new_id()`, seeded by
# `seed_all`/`build_client`. Patch 0003 also fixed a load-bearing call site
# the brief specifying this patch didn't list:
# `tuxemon/monster/monster.py`'s own `uuid4()` (Monster does not subclass
# Entity, so entity.py's fix alone did not cover it) -- without it,
# `npc_state.monsters.instance_id` and `npc_state.game_variables.add_monster`
# would still have diverged.
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
    """Snapshot the comparable game state via the game's own serialisation."""
    player = session.player
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
    }
    return state


def digest_of(session: Any) -> str:
    blob = json.dumps(state_of(session), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
