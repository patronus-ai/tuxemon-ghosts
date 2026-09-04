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
from types import SimpleNamespace
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"
SAVE = FIXTURES / "paper_town.save"
# No longer shipped -- the page passes no ghost (see `tuxghost/web.py`).
# Kept, and every ghost test below kept with it, because `boot`/`main`
# still ACCEPT one: this is the trace that comes back if anybody turns
# the ghost on again, so the path stays exercised rather than rotting.
GHOST = GOLDEN / "claude_town_1234.tuxghost"
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
    assert len(state.track.frames) == 443


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


def _booted_without_a_ghost() -> Any:
    """`boot` with no trace at all -- what `web/index.html` now does."""
    from tuxghost.web import boot

    return boot(SAVE, seed=SEED, clock_epoch=CLOCK_EPOCH)


def test_cold_boot_starts_at_the_campaign_entry() -> None:
    from tuxghost.web import boot_cold

    state = boot_cold(seed=SEED, clock_epoch=CLOCK_EPOCH)
    assert state.client.get_map_name() == "start_tuxemon.tmx"
    assert state.session.player.slug == "npc_red"
    assert state.npc is None
    assert state.track is None


def _intro_state(
    *,
    states: tuple[str, ...],
    map_name: str = "spyder_bedroom.tmx",
    intro: str | None = None,
    can_move: bool = False,
) -> Any:
    removed: list[str] = []
    player = SimpleNamespace(game_variables={"intro_scoop": intro})
    state = SimpleNamespace(
        client=SimpleNamespace(
            active_state_names=states,
            get_map_name=lambda: map_name,
            remove_state_by_name=removed.append,
            movement_manager=SimpleNamespace(
                is_movement_allowed=lambda char: can_move,
            ),
        ),
        session=SimpleNamespace(player=player),
    )
    return state, removed


def test_finished_intro_background_is_removed_before_free_play() -> None:
    """The setup background must not cover the world after starter choice."""
    from tuxghost.web import clear_finished_intro_background

    state, removed = _intro_state(
        states=("ImageState", "WorldState"), intro="done"
    )

    assert clear_finished_intro_background(state)
    assert removed == ["ImageState"]


@pytest.mark.parametrize(
    ("map_name", "intro", "can_move", "states"),
    [
        # Handed control back on a map the intro walked on to.
        ("spyder_paper_scoop.tmx", None, True, ("ImageState", "WorldState")),
        # Intro flagged done, background still up, even off the bedroom.
        ("spyder_paper_town.tmx", "done", False, ("ImageState", "WorldState")),
        # More than one background stacked over the world.
        (
            "spyder_bedroom.tmx",
            "done",
            False,
            ("ImageState", "ImageState", "WorldState"),
        ),
    ],
)
def test_stuck_intro_background_is_cleared_once_control_returns(
    map_name: str,
    intro: str | None,
    can_move: bool,
    states: tuple[str, ...],
) -> None:
    from tuxghost.web import clear_finished_intro_background

    state, removed = _intro_state(
        states=states, map_name=map_name, intro=intro, can_move=can_move
    )

    assert clear_finished_intro_background(state)
    assert removed == ["ImageState"] * (len(states) - 1)


@pytest.mark.parametrize(
    ("map_name", "intro", "can_move", "states"),
    [
        # A scene is still playing: something interactive sits on top.
        (
            "spyder_bedroom.tmx",
            "done",
            True,
            ("DialogState", "ImageState", "WorldState"),
        ),
        # An intentional image-only cutscene beat: intro unfinished and
        # controls still locked -- must be left on screen.
        ("spyder_bedroom.tmx", None, False, ("ImageState", "WorldState")),
        # No background over the world at all.
        ("spyder_bedroom.tmx", "done", True, ("WorldState",)),
    ],
)
def test_intro_background_cleanup_leaves_active_scenes_alone(
    map_name: str,
    intro: str | None,
    can_move: bool,
    states: tuple[str, ...],
) -> None:
    from tuxghost.web import clear_finished_intro_background

    state, removed = _intro_state(
        states=states, map_name=map_name, intro=intro, can_move=can_move
    )

    assert not clear_finished_intro_background(state)
    assert removed == []


def test_boot_without_a_ghost_installs_no_ghost() -> None:
    """The shipped page passes no trace, so nothing may be installed.

    All three assertions, not just the first: `npc`/`track` coming back
    `None` is what `step_once` branches on, while `GHOST_SLUG` being
    absent from `npc_manager` is what a human actually sees. A defect
    that installed the ghost but forgot to record it on the session
    would satisfy either check alone.
    """
    from tuxghost.ghost.entity import GHOST_SLUG

    state = _booted_without_a_ghost()
    assert state.npc is None, state.npc
    assert state.track is None, state.track
    assert GHOST_SLUG not in state.client.npc_manager.npcs, sorted(
        state.client.npc_manager.npcs
    )


def test_step_once_without_a_ghost_matches_a_plain_session() -> None:
    """A ghostless boot must be a REAL session, not merely a
    non-crashing one.

    Compared against the same hand-built control
    `test_the_ghost_does_not_perturb_the_digest` uses -- a session that
    never went near `tuxghost.web` -- so this pins the whole ghostless
    path (skipped `build_track`, skipped `install_ghost`, skipped
    `advance_ghost`) as equivalent to plain play, rather than just
    asserting that `step_once` returned a number. Sixty steps, so the
    comparison is against a session that has actually run.
    """
    import json

    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.digest import digest_of
    from tuxghost.loop import run_steps

    ghostless = _booted_without_a_ghost()
    taken = step_once_many(ghostless, 60)
    assert taken == 60, taken
    assert ghostless.step == 60, ghostless.step
    a = digest_of(ghostless.session)

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


def test_run_stops_when_the_first_gym_is_entered() -> None:
    """The embed owns boot separately from play, then ends this benchmark
    as soon as the first gym map loads. This pins that split without a
    browser: the map probe becomes true after two frames.
    """
    import asyncio

    from tuxghost import web

    state: Any = SimpleNamespace(client=SimpleNamespace(is_running=True))
    calls = {"steps": 0, "yields": 0}
    times = iter([0.0, 0.1, 0.2])

    def fake_step_once(_state: Any, _elapsed: float) -> int:
        calls["steps"] += 1
        return 1

    def fake_goal(_state: Any) -> bool:
        return calls["steps"] >= 2

    async def fake_sleep(_delay: float) -> None:
        calls["yields"] += 1

    original_step = web.step_once
    original_goal = web.first_gym_entered
    original_time = web.time  # type: ignore[attr-defined]
    original_sleep = asyncio.sleep
    web.step_once = fake_step_once  # type: ignore[assignment]
    web.first_gym_entered = fake_goal  # type: ignore[assignment]
    web.time = SimpleNamespace(monotonic=lambda: next(times))  # type: ignore[attr-defined, assignment]
    asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        elapsed = asyncio.run(web.run(state, stop_on_gym_entry=True))
    finally:
        web.step_once = original_step
        web.first_gym_entered = original_goal
        web.time = original_time  # type: ignore[attr-defined]
        asyncio.sleep = original_sleep

    assert elapsed == pytest.approx(0.2)
    assert calls == {"steps": 2, "yields": 1}


def test_first_gym_probe_is_exact() -> None:
    from tuxghost.web import first_gym_entered

    gym = SimpleNamespace(
        client=SimpleNamespace(
            get_map_name=lambda: "spyder_leather_gym.tmx"
        )
    )
    town = SimpleNamespace(
        client=SimpleNamespace(
            get_map_name=lambda: "spyder_leather_town.tmx"
        )
    )
    assert first_gym_entered(gym)  # type: ignore[arg-type]
    assert not first_gym_entered(town)  # type: ignore[arg-type]


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
