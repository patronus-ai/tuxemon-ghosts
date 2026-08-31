"""The browser entry point: a windowed session with a ghost, advanced one
frame at a time so the browser stays responsive.

`step_once` is synchronous and runs natively under `make check`; only
`main`'s `await` is browser-only. That split is the whole reason this
module exists rather than the loop living in JavaScript.

THE `ssl` STUB. `tuxemon.base_client` imports `NetworkManager` at module
level, which imports `ssl`, which Pyodide's default build does not have.
Multiplayer over raw sockets cannot work in a browser anyway. The stub
only fires via `except ImportError` -- it must never override a real
`ssl` module, because this project's pinned `websockets==17.0.1` breaks
against a hollow one: `import ssl` (real) is fine, but a bare
`sys.modules.setdefault("ssl", types.ModuleType("ssl"))` unconditionally
shadowing an already-real `ssl` makes
`websockets.asyncio.client` -- exactly what this engine reaches via
`client.py -> base_client.py -> network/manager.py ->
network/client.py -> network/websocket_client.py` -- fail with
`AttributeError: module 'ssl' has no attribute 'SSLWantReadError'`, measured
directly against this repo's pin. Measured (spike stages E and F): with
this shim and an OTHERWISE UNPATCHED engine, `build_client` boots and
the ghost walks its route to completion. Deliberately NOT a seventh
patch -- `patch_series_id` digests `patches/*.patch`, so a new patch
would make three golden fixtures stale at once, including the very
ghost this module plays, and `tests/conftest.py` turns that into a
failure on purpose.

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

#: How often `main` refreshes `measured_rate`, in seconds.
RATE_WINDOW = 1.0

#: The game loop's OWN throughput, in fixed steps/sec, refreshed by
#: `main` roughly every `RATE_WINDOW` seconds. This is what
#: `web/index.html`'s FPS readout displays -- round 1 of this review
#: shipped a JS-side `requestAnimationFrame` counter instead, which
#: ticks at the browser's paint rate regardless of whether THIS loop is
#: making any progress at all, so it could read a healthy 60 on a
#: stalled build. This number cannot: it IS the loop's progress, and it
#: drops to 0 the moment the loop stops taking steps, whatever the
#: browser's own paint rate happens to be doing. See
#: `test_measured_rate_reflects_the_loops_own_throughput_and_drops_on_stall`
#: in `tests/test_web.py`, which pins exactly that.
measured_rate: float = 0.0


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
    try:
        import ssl  # noqa: F401
    except ImportError:
        sys.modules["ssl"] = types.ModuleType("ssl")

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
            # The browser boots for over a minute while the ghost's walk
            # lasts seconds, so a viewer reliably arrives after it has
            # finished. Lingering was measured to read as "a random
            # ghost" to the first human who saw it; looping keeps the
            # ghost legible as something that MOVES.
            loop=True,
        )
    state.display.blit(state.frames.surface(), (0, 0))
    pg.display.flip()
    return steps


async def main(
    save: Path, ghost: Path, *, seed: int, clock_epoch: int
) -> None:
    """The browser's entry point.

    `await asyncio.sleep(0)` is the ONLY browser-specific line in this
    module. Without it the loop never returns control at all and the
    tab freezes -- Pyodide never gets a turn to run its event loop.

    What was actually measured, and NOT together: the design spec's
    measurement table records DOM KEYDOWN/KEYUP reaching `pygame.event`
    in isolation, and spike stage F records this ghost loop walking to
    completion in isolation. Neither run exercised both at once, so
    "this `await` is what lets Pyodide deliver keyboard events the game
    reads through `client.update`" is a plausible mechanism, not a
    measured one -- see the spec's "What the gate CANNOT cover".

    Pacing is `steps_owed` against `time.monotonic()`, exactly as
    `tuxghost.play` does, rather than trusting the browser's frame
    timing. It already handles catch-up and already carries
    `CATCH_UP_CAP`.

    Also refreshes the module-level `measured_rate` roughly every
    `RATE_WINDOW` seconds -- real fixed steps taken divided by real
    elapsed wall-clock time. Kept thin deliberately: three extra local
    variables and one `if`, no new control flow, no change to
    `step_once`'s own synchronous, natively-testable signature.
    """
    import asyncio

    global measured_rate

    state = boot(save, ghost, seed=seed, clock_epoch=clock_epoch)
    last = time.monotonic()
    window_start = last
    window_steps = 0
    while state.client.is_running:
        now = time.monotonic()
        window_steps += step_once(state, now - last)
        last = now
        if now - window_start >= RATE_WINDOW:
            measured_rate = window_steps / (now - window_start)
            window_start = now
            window_steps = 0
        await asyncio.sleep(0)
