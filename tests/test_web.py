"""Native tests for `tuxghost.web`, the browser entry point.

Everything here runs under the dummy SDL drivers like the rest of the
suite. `test_main_yields_between_frames` does exercise `main`'s
`await asyncio.sleep(0)` natively. What it cannot exercise is the
Pyodide boot and genuine browser scheduling around that `await` -- see
the spec's "What the gate CANNOT cover".
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"
SAVE = FIXTURES / "paper_town.save"
GHOST = GOLDEN / "scripted_town_1234.tuxghost"
SEED = 1234
CLOCK_EPOCH = 1787659200


def _booted() -> Any:
    from tuxghost.web import boot

    return boot(SAVE, GHOST, seed=SEED, clock_epoch=CLOCK_EPOCH)


def test_boot_installs_the_ghost_on_the_players_map() -> None:
    """The whole point of the page: a ghost beside you when it loads."""
    from tuxghost.ghost.entity import GHOST_SLUG

    state = _booted()
    assert GHOST_SLUG in state.client.npc_manager.npcs
    assert state.client.get_map_name() == "spyder_paper_town.tmx"
    assert len(state.track.frames) == 177


def test_step_once_walks_the_ghost_along_its_track() -> None:
    """`elapsed` drives how many fixed steps a frame owes, so a frame
    worth 60 steps must move the ghost 60 frames along its track."""
    state = _booted()
    start = tuple(int(v) for v in state.npc.tile_pos)
    taken = step_once_many(state, 60)
    assert taken == 60, taken
    assert state.step == 60
    moved = tuple(int(v) for v in state.npc.tile_pos)
    assert moved == state.track.at(60).tile, (moved, state.track.at(60))
    assert moved != start, "the ghost never moved"


def step_once_many(state: Any, n: int) -> int:
    """Drive exactly `n` fixed steps, one frame's worth at a time.

    Deliberately a loop of `n` single-frame `step_once` calls rather than
    one call worth `FIXED_DT * n` elapsed seconds: `step_once`'s own
    `CATCH_UP_CAP` (5) would silently truncate a single big call to at
    most 5 steps, discarding the rest per `steps_owed`'s catch-up-bound
    contract -- exactly the behaviour a REAL browser frame wants (a slow
    frame should not spiral), but not what a test driving the ghost a
    fixed `n` steps for an assertion wants. One fixed step per call never
    approaches the cap.
    """
    from tuxghost.loop import FIXED_DT
    from tuxghost.web import step_once

    taken = 0
    for _ in range(n):
        taken += step_once(state, FIXED_DT)
    return taken


def test_the_ghost_does_not_perturb_the_digest() -> None:
    """S4's invariant, re-run in this new caller. The ghost is drawn but
    must stay invisible to `state_of`.

    The control is a session that NEVER had a ghost -- not one where the
    ghost is removed afterwards. Removing it would leave whatever the
    install itself did behind, so the comparison would be against the
    wrong thing and would pass even if `install_ghost` perturbed state.
    """
    import json

    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.digest import digest_of
    from tuxghost.loop import run_steps

    with_ghost = _booted()
    step_once_many(with_ghost, 60)
    a = digest_of(with_ghost.session)

    seed_all(SEED)
    pin_clock(CLOCK_EPOCH)
    save_data = SaveData.model_validate(json.loads(SAVE.read_text()))
    _c, plain = boot_from_save(
        save_data, seed=SEED, clock_epoch=CLOCK_EPOCH
    )
    run_steps(plain.client, 60)
    b = digest_of(plain)
    assert a == b, (a, b)


def test_web_never_uses_the_headless_context() -> None:
    """SDL's `dummy` driver does not exist under Emscripten, so
    `headless_context()` would break the browser build while every native
    test still passed -- the worst possible failure shape.

    An AST scan, not a substring search: the module DOCSTRING names
    `headless_context` in prose, so `"headless_context" in source` would
    false-fail on correct code. S5 shipped exactly that bug once.

    The source path is resolved from `__file__`, not a bare relative
    `Path("tuxghost/web.py")`: `tests/conftest.py` `chdir`s the process
    into the vendored `tuxemon/` clone before any test runs (to make its
    assets findable), so a relative path anchored on the repo root would
    silently fail to resolve under that cwd -- the exact "test only ever
    used a path that happened to still work" hazard CLAUDE.md documents
    for `tuxghost.cli`'s own relative-path bug.
    """
    source = (
        Path(__file__).resolve().parent.parent / "tuxghost" / "web.py"
    ).read_text()
    tree = ast.parse(source)
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "headless_context" not in names, sorted(names)


def test_main_yields_between_frames() -> None:
    """The `await` is the entire reason this module is not JavaScript: a
    loop that never yields freezes the browser tab. Pinned by counting
    yields, since the browser cannot be here to notice.

    Termination is driven by `counting_step`, not `counting_sleep`: if it
    depended on the patched `asyncio.sleep` raising, deleting the
    `await` under test would delete the only exit path along with it,
    and this test would hang instead of failing. `step_once` runs every
    iteration regardless of the `await`, so it is the one place a stop
    condition is guaranteed to fire either way.
    """
    import asyncio

    from tuxghost import web

    calls = {"steps": 0, "yields": 0}
    real_sleep = asyncio.sleep

    async def counting_sleep(delay: float) -> None:
        calls["yields"] += 1
        await real_sleep(delay)

    def counting_step(state: Any, elapsed: float) -> int:
        # The stop condition lives HERE, not in `counting_sleep`:
        # `main`'s loop calls `step_once` every iteration whether or not
        # the `await` under test is present, so this still terminates --
        # and the yield-count assertion below still fails with a real
        # message -- even when the defect this test pins (a missing
        # `await`) removes every call to `asyncio.sleep` from the loop.
        if calls["steps"] >= 3:
            raise KeyboardInterrupt  # stop the loop deterministically
        calls["steps"] += 1
        return 1

    original_step, original_sleep = web.step_once, asyncio.sleep
    web.step_once, asyncio.sleep = counting_step, counting_sleep  # type: ignore[assignment]
    try:
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(
                web.main(SAVE, GHOST, seed=SEED, clock_epoch=CLOCK_EPOCH)
            )
    finally:
        web.step_once, asyncio.sleep = original_step, original_sleep

    assert calls["yields"] == 3, calls
    assert calls["steps"] == 3, calls
