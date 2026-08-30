"""Native tests for `tuxghost.web`, the browser entry point.

Everything here runs under the dummy SDL drivers like the rest of the
suite. `test_main_yields_between_frames` does exercise `main`'s
`await asyncio.sleep(0)` natively. What it cannot exercise is the
Pyodide boot and genuine browser scheduling around that `await` -- see
the spec's "What the gate CANNOT cover".

THE DUMMY SDL DRIVERS ARE SET HERE, DELIBERATELY, BEFORE ANY OTHER
IMPORT. `tuxghost/web.py`'s `boot()` calls upstream's `pygame_init()` --
a WINDOWED init (`tuxemon/tuxemon/prepare.py` does `pg.init()` then
`pg.display.set_mode(...)` with whatever `SDL_VIDEODRIVER` the
environment happens to hold). Nothing in `tests/conftest.py` or the
Makefile's `test:` target sets these vars; this suite stayed green only
because ~13 alphabetically-earlier test files import `tuxghost.boot`,
whose `headless_context()` sets both vars as a side effect, before
pytest ever collects this file. Run `pytest tests/test_web.py` alone and,
without the two lines below, it opens a real window and a real audio
device -- exactly the collection-order-dependent failure shape this
project catalogues (see CLAUDE.md's "never run the game without dummy
SDL drivers"). `tuxghost/observe.py` already proves `set_mode` works
fine under the dummy driver, so there is no reason `boot()` needs a real
window natively, only in the browser.
"""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

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


def _top_level_import_modules(tree: ast.Module) -> set[str]:
    """Every module named by a top-level `import x` / `from x import y`
    in `tree`'s body -- deliberately NOT `ast.walk`, which would also
    catch imports nested inside a function or class. Only TOP-LEVEL
    imports run at module-import time, before `tuxghost.web.boot`'s
    try/except ssl shim has had a chance to run; a deferred import
    inside a function (every tuxemon import `web.py` itself makes) runs
    only when that function is later called, by which point the shim
    has already run."""
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
        ):
            modules.add(node.module)
    return modules


def _module_path(root: Path, dotted: str) -> Path | None:
    base = root / Path(*dotted.split("."))
    leaf = base.with_suffix(".py")
    if leaf.is_file():
        return leaf
    init = base / "__init__.py"
    if init.is_file():
        return init
    return None


def test_ssl_shim_runs_before_any_tuxemon_import_reaches_the_page() -> None:
    """`tuxghost.web.boot`'s ssl shim (see the module docstring) only
    protects imports that happen AFTER it runs. Everything `tuxghost.web`
    imports at ITS OWN top level runs at MODULE IMPORT TIME -- before
    `boot()` is ever called -- so if any module reachable from those
    top-level imports itself top-level-imports anything from `tuxemon`,
    that import runs with no shim in place at all. Under Pyodide, where
    `ssl` genuinely does not always exist, that breaks only the browser
    build, silently, while every native test (which has a real `ssl`)
    stays green -- exactly the failure shape I2 found for the shim
    itself, one hop further out.

    AST-based, in the style of `test_web_never_uses_the_headless_context`:
    this must not actually trigger any of the imports it inspects --
    importing IS the thing under test, and doing so here would prove
    nothing about import ORDER.
    """
    root = Path(__file__).resolve().parent.parent
    web_path = root / "tuxghost" / "web.py"
    web_tree = ast.parse(web_path.read_text())

    offenders: list[str] = []
    seen: set[str] = {"tuxghost.web"}
    queue = list(_top_level_import_modules(web_tree))
    for m in list(queue):
        if m == "tuxemon" or m.startswith("tuxemon."):
            offenders.append(f"tuxghost.web top-level-imports {m}")

    while queue:
        dotted = queue.pop()
        if dotted in seen or not (
            dotted == "tuxghost" or dotted.startswith("tuxghost.")
        ):
            continue
        seen.add(dotted)
        path = _module_path(root, dotted)
        if path is None:
            continue
        tree = ast.parse(path.read_text())
        for m in _top_level_import_modules(tree):
            if m == "tuxemon" or m.startswith("tuxemon."):
                offenders.append(f"{dotted} top-level-imports {m}")
            else:
                queue.append(m)

    assert not offenders, offenders


def test_web_references_neither_trace_write_nor_recorder() -> None:
    """S6 is PLAY ONLY: the browser records no trace (see the design
    spec's scope decision 1 and "Invariants at close-out"). `write`
    (`tuxghost.trace`) and `Recorder` (`tuxghost.record`) are the only
    two names that could put a trace on disk, so their total absence
    from `tuxghost/web.py` is the whole invariant, checkable directly.

    AST-based like the two tests above: a name/attribute/import-alias
    scan, not a substring search on the source text, so a future
    docstring that merely MENTIONS "write" or "Recorder" in prose (as
    this very test's own docstring does) cannot false-fail the check.
    Import aliases are collected separately from `ast.Name`/
    `ast.Attribute`: an unused `from tuxghost.trace import write` binds
    no `Name` node anywhere in the tree (an import statement's `alias`
    is not a `Name`), so a scan of only Name/Attribute nodes would miss
    the import itself and catch only a later CALL to the imported name
    -- exactly the way this defect would actually land, one commit
    before anyone wires up the call.
    """
    root = Path(__file__).resolve().parent.parent
    source = (root / "tuxghost" / "web.py").read_text()
    tree = ast.parse(source)
    names = (
        {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        | {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
    )
    assert "write" not in names, sorted(names)
    assert "Recorder" not in names, sorted(names)


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
    # Monkeypatching test doubles over `web.step_once` (module function)
    # and `asyncio.sleep` (stdlib coroutine function) in one statement --
    # mypy sees each target's real, narrower signature and flags the
    # swap as `assignment`; both are restored in `finally` below, so the
    # narrowing is deliberately only ever violated for this test's
    # duration.
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


def test_measured_rate_reflects_the_loops_own_throughput_and_drops_on_stall() -> (
    None
):
    """Round 2 of this review: `web/index.html`'s FPS readout must show
    the GAME LOOP's own throughput, not a browser paint-rate proxy that
    would keep reading a healthy number on a stalled build. `measured_rate`
    is that number; this pins both halves the coordinator asked for --
    a healthy loop reads ~60 steps/sec, and a stalled one visibly drops.

    Fakes the clock and `step_once` to drive `main` through two whole
    `RATE_WINDOW`s without any real sleep: 60 iterations advancing 1/60s
    each with `step_once` reporting 1 step taken (a healthy second),
    then 2 iterations advancing 0.5s each reporting 0 steps taken (a
    stalled second). `asyncio.sleep` is replaced with a recorder, not a
    real sleep, so this test costs milliseconds -- it records
    `measured_rate` once per completed iteration, giving a full trace of
    the value across both windows rather than only its value at the end.

    The clock fake replaces `web.time` OUTRIGHT with a
    `types.SimpleNamespace` exposing only `monotonic`, rather than
    mutating the real `time` module's `monotonic` attribute in place.
    `web.py` resolves `time.monotonic()` by looking up the module-global
    name `time` in ITS OWN namespace, so rebinding `web.time` redirects
    only what `web.py` sees. Mutating the real, process-wide
    `time.monotonic` was tried first and broke: `asyncio`'s own event
    loop calls `time.monotonic()` internally for its own scheduling, so
    a shared fake exhausted the iterator on calls this test never issued
    (measured: consumption reached call 65, then kept climbing through
    67 during the interpreter's own cleanup, against 64 calls this
    test's own arithmetic accounts for) -- a real, reproducible failure
    mode of monkeypatching a stdlib module in place that this docstring
    records so nobody re-tries it.

    The stop condition lives in the fake `step_once`, exactly as
    `test_main_yields_between_frames` above established: the loop must
    terminate deterministically regardless of what happens to `sleep`.
    """
    import asyncio
    import types

    from tuxghost import web

    # 1 initial `time.monotonic()` call for `last`, then one call per
    # loop iteration: 60 healthy iterations (i/60 for i in 1..60,
    # landing exactly on window_start + RATE_WINDOW at i=60), 2 stalled
    # iterations (1.5, 2.0 -- landing exactly on the next window's
    # close), then one more call for the 63rd iteration, which is the
    # one where `step_once` raises and stops the loop before any of its
    # own timing math runs.
    times = iter(
        [0.0] + [i / 60 for i in range(1, 61)] + [1.5, 2.0] + [2.5]
    )
    # 60 steps taken (healthy), then 2 iterations taking 0 steps
    # (stalled) -- exhausted after 62 entries, at which point the 63rd
    # `step_once` call raises to stop the loop.
    steps = iter([1] * 60 + [0, 0])
    rate_history: list[float] = []

    def fake_monotonic() -> float:
        return next(times)

    def fake_step_once(state: Any, elapsed: float) -> int:
        try:
            return next(steps)
        except StopIteration:
            raise KeyboardInterrupt from None

    async def recording_sleep(delay: float) -> None:
        rate_history.append(web.measured_rate)

    # `time` is a plain stdlib import inside `tuxghost.web`, not
    # something that module explicitly re-exports, so mypy --strict
    # treats `web.time` as an attribute access it cannot vouch for
    # (`attr-defined`) rather than the module-global rebind it actually
    # is; the swap itself is also flagged (`assignment`) since
    # `types.SimpleNamespace` isn't statically the real `time` module.
    # Both are the deliberate point of this monkeypatch, restored in
    # `finally` below.
    original_time = web.time  # type: ignore[attr-defined]
    original_step_once = web.step_once
    original_sleep = asyncio.sleep
    web.time = types.SimpleNamespace(monotonic=fake_monotonic)  # type: ignore[attr-defined, assignment]
    web.step_once = fake_step_once
    asyncio.sleep = recording_sleep  # type: ignore[assignment]
    try:
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(
                web.main(SAVE, GHOST, seed=SEED, clock_epoch=CLOCK_EPOCH)
            )
    finally:
        web.time = original_time  # type: ignore[attr-defined]
        web.step_once = original_step_once
        asyncio.sleep = original_sleep

    assert len(rate_history) == 62, rate_history
    # First window closes at iteration 60 (wall-clock reaches exactly
    # 1.0s, 60 steps taken): a healthy ~60 steps/sec.
    assert rate_history[59] == 60.0, rate_history
    # Second window closes at iteration 62 (wall-clock reaches exactly
    # 2.0s from a 1.0s start, 0 steps taken across it): the number MUST
    # visibly drop, all the way to 0 -- this is the exact failure mode a
    # `requestAnimationFrame`-based readout (round 1) could never show.
    assert rate_history[-1] == 0.0, rate_history
