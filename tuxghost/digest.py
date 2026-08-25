"""Canonical game-state digest.

The exemption register below is a list of defects not yet fixed, not a list
of things that are allowed to vary. Its target size is zero.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

EXEMPTIONS: dict[str, str] = {
    "uuid": "session identity is uuid4; closed by patch 0003",
    "instance_id": "entity ids are uuid4; closed by patch 0003",
    "start_time": "wall clock in SessionSave; closed by patch 0004",
    "duration": "wall clock in SessionSave; closed by patch 0004",
    "total_playtime": "wall clock in SessionSave; closed by patch 0004",
}


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _canonical(v) for k, v in sorted(obj.items()) if k not in EXEMPTIONS
        }
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
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
            json.loads(player.get_state(session).model_dump_json())
        ),
        "world_state": _canonical(
            json.loads(session.world.get_state(session).model_dump_json())
        ),
    }
    return state


def digest_of(session: Any) -> str:
    blob = json.dumps(state_of(session), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
