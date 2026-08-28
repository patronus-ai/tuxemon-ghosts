"""The one windowed entry point in this project, and its only exception
to the dummy-SDL rule.

`CLAUDE.md` requires `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy` on
every invocation touching the client, because a headless run must not
block on window creation or vary machine to machine. This module is the
single deliberate exception: a human at a real window is the one consumer
that needs one.

It does NOT threaten determinism. `tuxghost.boot.build_client`'s own
docstring records the measurement that settles it (`boot_from_save`
merely cross-references the same result) -- record and replay may use
different display contexts, and a 300-step per-step digest sequence
(`tuxghost.digest.digest_of`, not just the final digest) was identical
at every index between the two. A window changes what is DRAWN, not
what HAPPENS.

Kept deliberately thin: `make check` runs headless, so nothing here is
gated. Every decision lives in `tuxghost.ghost`, which is fully tested.

Two ordering constraints, in tension, both required:

1. The ghost track is built BEFORE the real session boots. `boot_from_save`
   restores state onto `tuxemon.session.local_session`, a process-wide
   singleton, resetting it on every call -- built the other way around,
   the ghost's own throwaway boot would silently reset the live player's
   session out from under them.
2. `tuxemon.map.view`/`tuxemon.graphics` are imported right after
   `pygame_init()`, before that same ghost-track build, to force their
   one-time `DISPLAY_CONTEXT` binding (a plain module global; a later
   reassignment cannot reach a name another module already bound at its
   own first import -- same guarantee `tuxghost.observe.scaled_context`
   relies on) to lock in while the real, windowed context is current.
   Without this, the ghost's build -- always headless, scale 1 -- would
   be the first thing in the process to import these two modules, and
   the real window would silently render at the ghost's scale instead
   of its own. Measured, not guessed: see this task's fix-round-1 report
   for a before/after probe.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tuxghost.determinism import pin_clock, seed_all
from tuxghost.ghost.entity import advance_ghost, install_ghost
from tuxghost.ghost.pump import install_recording_events, steps_owed
from tuxghost.ghost.track import build_track
from tuxghost.loop import run_steps
from tuxghost.record import Recorder
from tuxghost.trace import read, write

#: Catch-up bound. Past this, wall-clock time is dropped rather than
#: banked -- see `steps_owed`.
CATCH_UP_CAP = 5


def play(
    ghost_trace: Path | None,
    save: Path,
    seed: int,
    clock_epoch: int,
    out: Path,
    run_dir: Path,
) -> int:
    import pygame as pg
    from tuxemon.prepare import pygame_init
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save
    from tuxghost.observe import FrameRenderer

    context = pygame_init()

    # Lock in this real, windowed scale on these two modules' own bound
    # names BEFORE anything headless (the ghost track build, next) gets a
    # chance to import them first at scale 1. See "Ordering note 2" above.
    import tuxemon.graphics
    import tuxemon.map.view  # noqa: F401

    track = build_track(read(ghost_trace)) if ghost_trace is not None else None

    seed_all(seed)
    pin_clock(clock_epoch)
    save_data = SaveData.model_validate(json.loads(save.read_text()))
    _client, session = boot_from_save(
        save_data, seed=seed, clock_epoch=clock_epoch, context=context
    )
    client = session.client

    npc = (
        # No explicit `sprite_slug`: defaults to the session's own
        # player's sprite (M1, whole-branch review -- see
        # `install_ghost`'s docstring). The spec requires the ghost read
        # as "another you", not a hardcoded stranger.
        install_ghost(session, track)
        if track is not None
        else None
    )

    recorder = Recorder(session, seed=seed, clock_epoch=clock_epoch, recorder="human")
    state: dict[str, int] = {"step": 0}
    install_recording_events(client, recorder, lambda: state["step"])

    frames = FrameRenderer(client, upscale=1)
    display = context.screen

    accumulator = 0.0
    last = time.monotonic()
    while client.is_running:
        now = time.monotonic()
        steps, accumulator = steps_owed(accumulator, now - last, CATCH_UP_CAP)
        last = now

        for _ in range(steps):
            run_steps(client, 1)
            state["step"] += 1
            if npc is not None and track is not None:
                advance_ghost(
                    client, npc, track, state["step"], client.get_map_name()
                )

        display.blit(frames.surface(), (0, 0))
        pg.display.flip()

    run_dir.mkdir(parents=True, exist_ok=True)
    write(recorder.finish(step_count=state["step"]), out)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "clock_epoch": clock_epoch,
                "steps": state["step"],
                "ghost": str(ghost_trace) if ghost_trace is not None else None,
                "ghost_digest": track.source_digest if track is not None else None,
            },
            indent=2,
        )
    )
    return 0
