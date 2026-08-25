#!/usr/bin/env python3
"""THROWAWAY determinism probe for Tuxemon. Not production code.

Question: can two headless runs, from the same seed and the same
step-indexed input schedule, reach byte-identical game state?

Run:  probe_determinism.py --seed 1234 --steps 600 --out digest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tuxemon"))

# volatile-by-construction fields: wall clock and random identity
VOLATILE = {
    "uuid", "instance_id", "start_time", "duration", "total_playtime",
    "steps", "time", "game_time",
}


def strip_volatile(obj):
    """Recursively drop fields that cannot be deterministic by construction."""
    if isinstance(obj, dict):
        return {
            k: strip_volatile(v) for k, v in sorted(obj.items())
            if k not in VOLATILE
        }
    if isinstance(obj, (list, tuple)):
        return [strip_volatile(v) for v in obj]
    if isinstance(obj, float):
        return repr(obj)  # exact float bits, no formatting loss
    return obj


def build_client(seed: int | None):
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    from tuxemon.platform import platform
    platform.init()

    from tuxemon.prepare import headless_init
    context = headless_init()

    # Upstream headless_init() never calls set_mode(), so convert_alpha()
    # raises on the first sprite load. Dummy driver -> no window.
    import pygame as pg
    pg.init()
    from tuxemon.user_config import CONFIG as _C
    pg.display.set_mode(_C.resolution)

    from tuxemon.user_config import CONFIG
    config = CONFIG.copy()

    from tuxemon.headless_client import HeadlessClient
    from tuxemon.session import local_session

    client = HeadlessClient(config, context)
    local_session.set_client(client)

    # Seed AFTER construction, immediately before any gameplay randomness.
    if seed is not None:
        random.seed(seed)

    from tuxemon.database.runtime import db
    from tuxemon.launcher import GameLauncher

    meta = db.mod_metadata.get_mod_metadata("tuxemon")
    GameLauncher(client).launch(local_session, meta)
    return client, local_session


def digest_state(session) -> dict:
    """Canonical state snapshot via the game's own serialization."""
    out = {}
    player = session.player
    out["tile_pos"] = list(player.tile_pos)
    out["facing"] = str(player.facing)
    out["map"] = session.client.get_map_name()
    out["state_stack"] = [
        st.name for st in session.client.state_manager.active_states
    ]
    try:
        out["npc_state"] = strip_volatile(
            json.loads(player.get_state(session).model_dump_json())
        )
    except Exception as e:
        out["npc_state_error"] = f"{type(e).__name__}: {e}"
    try:
        out["world_state"] = strip_volatile(
            json.loads(session.world.get_state(session).model_dump_json())
        )
    except Exception as e:
        out["world_state_error"] = f"{type(e).__name__}: {e}"
    try:
        out["variables"] = strip_volatile(
            dict(session.client.variables.items())
        )
    except Exception as e:
        out["variables_error"] = f"{type(e).__name__}: {e}"
    return out


def install_input_schedule(client, schedule):
    """Yield scheduled PlayerInputs on exact step indices.

    Replaces InputManager.process_events wholesale: upstream's playback
    path yields one event per FRAME regardless of when it was recorded,
    which is the timing bug this probe must not inherit.
    """
    from tuxemon.platform.events import PlayerInput

    state = {"step": 0}

    def process_events():
        for button, value in schedule.get(state["step"], []):
            ev = PlayerInput(button, value, 1 if value else 0)
            ev.triggered = bool(value)
            yield ev
        state["step"] += 1

    client.input_manager.process_events = process_events
    return state


def mash(button, start, count, period=10, hold=2):
    """Schedule `count` presses of `button` every `period` steps."""
    sched = {}
    for i in range(count):
        t = start + i * period
        sched.setdefault(t, []).append((button, 1))
        sched.setdefault(t + hold, []).append((button, 0))
    return sched


def merge(*scheds):
    out = {}
    for sc in scheds:
        for k, v in sc.items():
            out.setdefault(k, []).extend(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="omit to run UNSEEDED (discrimination control)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    client, session = build_client(args.seed)

    from tuxemon.platform.const import buttons

    # Mash A to clear intro dialogs, then walk a fixed route.
    schedule = merge(
        mash(buttons.A, start=30, count=40, period=12),
        mash(buttons.DOWN, start=560, count=30, period=8, hold=6),
        mash(buttons.RIGHT, start=820, count=30, period=8, hold=6),
        mash(buttons.A, start=1100, count=60, period=10),
    )
    install_input_schedule(client, schedule)

    FIXED_DT = 1.0 / 60.0
    for _ in range(args.steps):
        client.update(FIXED_DT)

    state = digest_state(session)
    blob = json.dumps(state, sort_keys=True, separators=(",", ":"))
    args.out.write_text(
        json.dumps(
            {"sha256": hashlib.sha256(blob.encode()).hexdigest(),
             "state": state},
            indent=2, sort_keys=True,
        )
    )
    print(hashlib.sha256(blob.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
