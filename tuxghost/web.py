"""The browser entry point: a windowed session with a ghost, advanced one
frame at a time so the browser stays responsive.

`step_once` is synchronous and runs natively under `make check`; only
`main`'s `await` is browser-only. That split is the whole reason this
module exists rather than the loop living in JavaScript.

THE `ssl` STUB. `tuxemon.base_client` imports `NetworkManager` at module
level, which imports `ssl`, which browsers do not have. Multiplayer over
raw sockets cannot work in a browser anyway. Measured (spike stage D):
with this stub and an OTHERWISE UNPATCHED engine, every tuxemon import
succeeds. Deliberately NOT a seventh patch -- `patch_series_id` digests
`patches/*.patch`, so a new patch would make three golden fixtures stale
at once, including the very ghost this module plays, and
`tests/conftest.py` turns that into a failure on purpose.

BOOT ORDER is `tuxghost.play`'s, and each step of it was paid for:
  1. `pygame_init()` FIRST, then import `tuxemon.graphics` and
     `tuxemon.map.view`, so they bind the windowed scale (play.py's
     "Ordering note 2").
  2. `build_track(..., context=ctx)` -- the context argument is REQUIRED.
     Without it the ghost's own boot calls `pg.display.set_mode()` a
     second time and invalidates every already-converted surface. That
     cost a whole session; see commit 8840756.
  3. Seed, pin the clock, THEN boot the live session -- `boot_from_save`
     resets the `local_session` singleton, so the track must already
     exist (play.py's "Ordering note 1").

The browser is a WINDOWED consumer, like `tuxghost play`, never a
headless one: SDL's `dummy` driver does not exist under Emscripten
(measured: `pygame.error: dummy not available`). This module must never
call `tuxghost.boot.headless_context`, and a test enforces it.
"""

from __future__ import annotations

import json
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tuxghost.ghost.entity import advance_ghost, install_ghost
from tuxghost.ghost.pump import steps_owed
from tuxghost.ghost.track import GhostTrack, build_track
from tuxghost.loop import run_steps
from tuxghost.trace import read

#: Same bound `tuxghost.play` uses. Past this, elapsed time is DISCARDED
#: rather than banked -- see `steps_owed`.
CATCH_UP_CAP = 5


@dataclass
class WebSession:
    """Everything one frame needs. Mutable: `step` and `accumulator`
    carry across frames, which is why this is a dataclass and not a
    tuple."""

    client: Any
    session: Any
    npc: Any
    track: GhostTrack
    frames: Any
    display: Any
    step: int = 0
    accumulator: float = 0.0


def boot(
    save: Path, ghost: Path, *, seed: int, clock_epoch: int
) -> WebSession:
    """Open a window, build the ghost's track, boot the save, install the
    ghost. See the module docstring for why this order is not negotiable.
    """
    sys.modules.setdefault("ssl", types.ModuleType("ssl"))

    from tuxemon.prepare import pygame_init

    context = pygame_init()

    import tuxemon.graphics
    import tuxemon.map.view  # noqa: F401
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.observe import FrameRenderer

    track = build_track(read(ghost), context=context)

    seed_all(seed)
    pin_clock(clock_epoch)
    save_data = SaveData.model_validate(json.loads(save.read_text()))
    _client, session = boot_from_save(
        save_data, seed=seed, clock_epoch=clock_epoch, context=context
    )
    npc = install_ghost(session, track)
    return WebSession(
        client=session.client,
        session=session,
        npc=npc,
        track=track,
        frames=FrameRenderer(session.client, upscale=1),
        display=context.screen,
    )


def step_once(state: WebSession, elapsed: float) -> int:
    """Advance by however many fixed steps `elapsed` owes, then draw.

    Returns the number of steps taken, which is 0 on a frame that arrived
    faster than one fixed step -- normal, not an error.
    """
    import pygame as pg

    steps, state.accumulator = steps_owed(
        state.accumulator, elapsed, CATCH_UP_CAP
    )
    for _ in range(steps):
        run_steps(state.client, 1)
        state.step += 1
        advance_ghost(
            state.client,
            state.npc,
            state.track,
            state.step,
            state.client.get_map_name(),
        )
    state.display.blit(state.frames.surface(), (0, 0))
    pg.display.flip()
    return steps


async def main(
    save: Path, ghost: Path, *, seed: int, clock_epoch: int
) -> None:
    """The browser's entry point.

    `await asyncio.sleep(0)` is the ONLY browser-specific line in this
    module. Without it the loop never returns control and the tab
    freezes; with it, Pyodide gets to run the event loop and deliver the
    keyboard events the game reads through `client.update`.

    Pacing is `steps_owed` against `time.monotonic()`, exactly as
    `tuxghost.play` does, rather than trusting the browser's frame
    timing. It already handles catch-up and already carries
    `CATCH_UP_CAP`.
    """
    import asyncio

    state = boot(save, ghost, seed=seed, clock_epoch=clock_epoch)
    last = time.monotonic()
    while state.client.is_running:
        now = time.monotonic()
        step_once(state, now - last)
        last = now
        await asyncio.sleep(0)
