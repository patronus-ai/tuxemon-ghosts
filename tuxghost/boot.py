"""Headless bootstrap for Tuxemon."""

from __future__ import annotations

import os
from typing import Any


def headless_context() -> Any:
    """Initialise pygame headlessly and return upstream's DisplayContext.

    Sets both SDL drivers to dummy before importing pygame so no window or
    audio device is ever opened.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

    from tuxemon.platform import platform

    platform.init()

    from tuxemon.prepare import headless_init

    return headless_init()


def build_client(seed: int) -> tuple[Any, Any]:
    """Build a headless client on a fresh game at the given seed."""
    import random

    context = headless_context()

    from tuxemon.core.ids import seed_ids
    from tuxemon.database.runtime import db
    from tuxemon.launcher import GameLauncher
    from tuxemon.main import headless_world
    from tuxemon.session import local_session
    from tuxemon.user_config import CONFIG

    # Seed the id factory before anything below can draw from it --
    # `local_session.reset()` immediately below is one such draw (see its
    # docstring). Matches the `random.seed(seed)` call further down and
    # `config.deterministic_seed = seed` next: this build's own `seed`
    # argument must be authoritative for every entropy source it touches,
    # not left to whatever `tuxghost.determinism.seed_all` last left on
    # process-wide state (or never set at all, if a caller never called
    # it -- e.g. every test in `tests/test_digest.py`, which builds
    # straight from `build_client` and would otherwise draw ids from raw
    # OS entropy).
    seed_ids(seed)

    # `local_session` is a module-level singleton shared across every
    # `build_client` call in this process. Without resetting it first, a
    # second build in the same process starts from whatever the previous
    # build's session already accumulated (its player, its leftover state
    # stack, ...) instead of a genuinely fresh session -- silent
    # contamination, not an error. `boot_from_save` already resets for the
    # same reason; do it here too instead of leaving every caller (tests,
    # the executor, later tasks) to rediscover the hazard for themselves.
    local_session.reset()

    # `config.deterministic_seed` is what patch 0002 threads into
    # `WorldWeatherManager`, the only instance-RNG in the codebase (see
    # `tuxghost/determinism.py`). It is set here, directly on this build's
    # own config copy, rather than left to whatever `tuxghost.determinism
    # .seed_all` last wrote onto the process-wide `CONFIG` singleton:
    # `CONFIG` persists across every `build_client` call in a process, so a
    # test (or caller) that never calls `seed_all` would otherwise silently
    # inherit whichever seed a *previous, unrelated* call happened to leave
    # behind -- a disagreement between this build's explicit `seed`
    # argument and its weather RNG. Setting it here makes `build_client`'s
    # `seed` parameter authoritative for everything it constructs,
    # matching the `random.seed(seed)` call below.
    config = CONFIG.copy()
    config.deterministic_seed = seed
    client = headless_world(config, context)

    random.seed(seed)

    meta = db.mod_metadata.get_mod_metadata("tuxemon")
    GameLauncher(client).launch(local_session, meta)
    return client, local_session


def snapshot_save(session: Any) -> Any:
    """Serialise the live session into upstream's SaveData model."""
    from tuxemon.save_system import save

    return save.get_save_data(session)


def boot_from_save(save_data: Any, seed: int) -> tuple[Any, Any]:
    """Boot a headless client and restore `save_data` into it."""
    import random

    context = headless_context()

    from tuxemon.constants.asset_loader import fetch_asset
    from tuxemon.core.ids import seed_ids
    from tuxemon.entity.npc import NPC
    from tuxemon.main import headless_world
    from tuxemon.platform.const.sizes import PLAYER_NPC
    from tuxemon.session import local_session
    from tuxemon.user_config import CONFIG

    # See the matching comment in `build_client`: seed the id factory
    # before `local_session.reset()` (the next line) can draw from it.
    seed_ids(seed)

    # `local_session` is a module-level singleton shared with any prior
    # session in this process (e.g. the one that produced `save_data`).
    # Reset it so `create_player` below actually creates a fresh NPC
    # instead of reusing whatever player/world/client is already
    # attached -- otherwise restoration would trivially "work" by
    # aliasing the old objects rather than by `load_state` doing
    # anything.
    local_session.reset()

    # See the matching comment in `build_client`: set the seed directly on
    # this build's own config copy so it is authoritative for the weather
    # RNG, independent of whatever `tuxghost.determinism.seed_all` last
    # left on the process-wide `CONFIG` singleton.
    config = CONFIG.copy()
    config.deterministic_seed = seed
    client = headless_world(config, context)
    random.seed(seed)

    npc_state = save_data.npc_state
    assert npc_state is not None and npc_state.current_map is not None

    NPC.create_player(local_session, slug=npc_state.player_slug or PLAYER_NPC)
    client.push_state(
        "WorldState",
        session=local_session,
        map_name=fetch_asset("maps", npc_state.current_map),
    )
    local_session.load_state(save_data)
    return client, local_session
