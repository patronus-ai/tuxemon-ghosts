# Tuxemon Deterministic Trace Core (S1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable, step-indexed Tuxemon trace format and an offline headless executor that replays a trace and verifies it reached the same state.

**Architecture:** Upstream Tuxemon stays a vendored checkout at `59a34164f442ddecbaee4c436e3f1a5ba9474e29`, modified only by a numbered patch series in `patches/`. All new code lives in a separate `tuxghost/` package that imports Tuxemon, owns the single stepping loop, and provides the trace format, recorder, executor and CLI.

**Tech Stack:** Python 3.12, pygame-ce 2.5.7 (dummy SDL drivers), pydantic v2, pytest, mypy (strict), ruff. Dependency management via `uv`.

**Spec:** `docs/superpowers/specs/2026-08-25-tuxemon-trace-core-design.org`

**Evidence:** `docs/2026-08-25-determinism-spike.org` — every determinism claim in this plan was measured there.

## Global Constraints

- Upstream commit is `59a34164f442ddecbaee4c436e3f1a5ba9474e29`. Never edit `tuxemon/` directly; every engine change is a file in `patches/` applied by `make patch`.
- Docs are org-mode (`.org`). Plans and code comments are the exception; this plan is `.md` per the skill default.
- Trace format is **version 1**. Any other value is rejected outright, never coerced.
- Traces are **step-indexed, never wall-clock-indexed**. `step_rate` is 60 (upstream's existing `FIXED_DT = 1/60`).
- Never run the game without `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`, or it opens a window.
- All game runs must set `PYTHONHASHSEED=0` in tests for reproducible reporting, even though the spike showed hash seed does not affect state.
- **A regression test must be demonstrated to fail against the bug it pins.** Mutate, watch it fail, restore, watch it pass, report real values. An honest gap beats hollow coverage.
- `mypy --strict`, `ruff check` at zero warnings, and `pytest` all block. No `# type: ignore` or `# noqa` without a written reason.
- Python is invoked as `./.venv/bin/python`; the venv already exists at repo root.

---

### Task 1: Repo scaffold, CI gates, and the test harness

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `tuxghost/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_scaffold.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tuxghost` importable package; `make check` runs mypy + ruff + pytest as one blocking gate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
def test_package_imports() -> None:
    import tuxghost

    assert tuxghost.__version__ == "0.1.0"


def test_tuxemon_submodules_are_importable() -> None:
    """From the repo root `import tuxemon` yields an EMPTY namespace package
    (__file__ is None) and every submodule import fails. conftest.py fixes
    both the path and the working directory; this pins that it stays fixed."""
    import tuxemon
    from tuxemon.session import local_session

    assert tuxemon.__file__ is not None
    assert local_session is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuxghost'`

- [ ] **Step 3: Create the package and config**

```python
# tuxghost/__init__.py
"""Deterministic trace core for Tuxemon."""

__version__ = "0.1.0"
```

```python
# tests/conftest.py
"""Make the vendored Tuxemon importable and its assets findable.

Two separate problems, both measured:
  * From the repo root, `import tuxemon` resolves to the clone directory as an
    empty namespace package -- it imports fine, then every submodule raises
    ModuleNotFoundError. Putting tuxemon/ on sys.path makes the real package
    (which has __init__.py) win, since a regular package beats a namespace
    portion regardless of path order.
  * Asset loading is relative to the working directory: from the repo root the
    game dies with "Metadata file missing: 'mods/tuxemon/mod.yaml'".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

TUXEMON_DIR = Path(__file__).resolve().parent.parent / "tuxemon"

sys.path.insert(0, str(TUXEMON_DIR))
os.chdir(TUXEMON_DIR)
```

```toml
# pyproject.toml
[project]
name = "tuxghost"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.10.6"]

[project.optional-dependencies]
dev = ["pytest", "mypy", "ruff"]

[tool.setuptools.packages.find]
include = ["tuxghost*"]

[tool.mypy]
strict = true
warn_unreachable = true
# tuxemon upstream is unannotated and not ours to fix
[[tool.mypy.overrides]]
module = "tuxemon.*"
ignore_missing_imports = true
follow_imports = "skip"

[tool.ruff]
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: drives a real game loop for minutes; opt-in"]
```

```makefile
# Makefile
PY := ./.venv/bin/python

.PHONY: check check-fast lint types test slow
check: lint types test slow
check-fast: lint types test

lint:
	$(PY) -m ruff check tuxghost tests

types:
	$(PY) -m mypy tuxghost tests

test:
	PYTHONHASHSEED=0 $(PY) -m pytest -q

slow:
	# exit 5 means "no tests collected", which is correct until Task 6 adds
	# the first slow test. Any other non-zero status is a real failure.
	PYTHONHASHSEED=0 TUXGHOST_RUN_SLOW=1 $(PY) -m pytest -q -m slow || [ $$? -eq 5 ]
```

- [ ] **Step 4: Install dev deps and run the gate**

Run: `VIRTUAL_ENV=.venv uv pip install -e ".[dev]" && make check-fast`
Expected: ruff clean, mypy clean, 1 test PASSES.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml Makefile tuxghost tests
git commit -m "feat: scaffold the tuxghost package with blocking CI gates"
```

---

### Task 2: Patch 0001 — make headless able to load graphics

Stock `headless_init()` calls neither `pg.init()` nor `pg.display.set_mode()`, so the first sprite load raises `pygame.error: No convert format has been set` and `pygame_menu` asserts `pygame is not initialized`. Creating the player NPC loads a sprite sheet, so **stock headless cannot create a player at all**.

**Files:**
- Create: `patches/0001-headless-init-graphics.patch`
- Create: `tuxghost/boot.py`
- Create: `tests/test_boot.py`
- Modify (via patch): `tuxemon/tuxemon/prepare.py` in `headless_init()`

**Interfaces:**
- Consumes: nothing.
- Produces: `tuxghost.boot.headless_context() -> DisplayContext` — initialises pygame headlessly and returns upstream's `DisplayContext`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_boot.py
def test_headless_context_can_convert_surfaces() -> None:
    """A converted surface is what sprite loading needs; without set_mode
    pygame raises 'No convert format has been set'."""
    import pygame as pg

    from tuxghost.boot import headless_context

    ctx = headless_context()
    assert ctx.resolution[0] > 0
    surface = pg.Surface((8, 8), pg.SRCALPHA)
    converted = surface.convert_alpha()
    assert converted.get_size() == (8, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_boot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuxghost.boot'`

- [ ] **Step 3: Write the patch**

```diff
--- a/tuxemon/prepare.py
+++ b/tuxemon/prepare.py
@@ def headless_init() -> DisplayContext:
     os.environ["SDL_VIDEODRIVER"] = "dummy"
 
     core_init()
 
-    pg.display.init()
+    pg.init()
+    pg.display.init()
     pg.font.init()
 
+    # A display mode must exist before any Surface.convert_alpha(), which
+    # every sprite load performs. The dummy driver opens no window.
+    pg.display.set_mode(CONFIG.resolution)
+
     screen = pg.Surface(CONFIG.resolution)
```

- [ ] **Step 4: Write the boot module**

```python
# tuxghost/boot.py
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
```

- [ ] **Step 5: Apply the patch and run the test**

Run: `cd tuxemon && git apply ../patches/0001-headless-init-graphics.patch && cd .. && PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_boot.py -v`
Expected: PASS

- [ ] **Step 6: Prove the patch is load-bearing (mutation gate)**

Run: `cd tuxemon && git stash && cd .. && PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_boot.py -v; cd tuxemon && git stash pop`
Expected: FAIL without the patch. **Record the real error text in the commit message.** If it passes without the patch, the test is not pinning anything — stop and fix the test.

- [ ] **Step 7: Commit**

```bash
git add patches/0001-headless-init-graphics.patch tuxghost/boot.py tests/test_boot.py
git commit -m "feat: patch 0001 so headless can load graphics"
```

---

### Task 3: Boot from a save (patch 0006) — a defined starting world

The spike started the game via `GameLauncher`, which lands on the intro map and leaves a blocking `InputMenu`. Popping states to get past it broke twice with `ValueError: Missing state SinkState`, because the intro script later runs `unlock_controls`, which looks that state up by name. The executor must start from a **saved world** instead.

**Files:**
- Create: `patches/0006-headless-entry-from-save.patch`
- Modify: `tuxghost/boot.py`
- Modify: `tests/test_boot.py`

**Interfaces:**
- Consumes: `tuxghost.boot.headless_context()`.
- Produces:
  - `tuxghost.boot.build_client(seed: int) -> tuple[Any, Any]` returning `(client, session)`.
  - `tuxghost.boot.boot_from_save(save: SaveData, seed: int) -> tuple[Any, Any]` — loads a save into a headless client and pushes `WorldState`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_boot.py
def test_boot_from_save_restores_position_and_party() -> None:
    from tuxghost.boot import boot_from_save, build_client, snapshot_save

    client, session = build_client(seed=1234)
    session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    saved = snapshot_save(session)

    client2, session2 = boot_from_save(saved, seed=1234)
    assert [m.slug for m in session2.player.monsters] == ["rockitten"]
    assert session2.player.tile_pos == session.player.tile_pos
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_boot.py::test_boot_from_save_restores_position_and_party -v`
Expected: FAIL with `ImportError: cannot import name 'boot_from_save'`

- [ ] **Step 3: Add the upstream entry point as patch 0006**

```diff
--- a/tuxemon/main.py
+++ b/tuxemon/main.py
@@
+def headless_world(
+    config: TuxemonConfig, context: DisplayContext
+) -> HeadlessClient:
+    """Build a headless client able to run the overworld.
+
+    Unlike headless(), this does not push HeadlessServerState -- that state
+    loads no world and returns None from process_event, so it cannot play.
+    """
+    client = HeadlessClient(config, context)
+    local_session.set_client(client)
+    return client
```

- [ ] **Step 4: Implement the boot helpers**

```python
# append to tuxghost/boot.py
def build_client(seed: int) -> tuple[Any, Any]:
    """Build a headless client on a fresh game at the given seed."""
    import random

    context = headless_context()

    from tuxemon.main import headless_world
    from tuxemon.session import local_session
    from tuxemon.user_config import CONFIG

    client = headless_world(CONFIG.copy(), context)

    random.seed(seed)

    from tuxemon.database.runtime import db
    from tuxemon.launcher import GameLauncher

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
    from tuxemon.entity.npc import NPC
    from tuxemon.main import headless_world
    from tuxemon.session import local_session
    from tuxemon.user_config import CONFIG

    client = headless_world(CONFIG.copy(), context)
    random.seed(seed)

    NPC.create_player(local_session, slug="npc_red")
    npc_state = save_data.npc_state
    assert npc_state is not None and npc_state.current_map is not None
    client.push_state(
        "WorldState",
        session=local_session,
        map_name=fetch_asset("maps", npc_state.current_map),
    )
    local_session.load_state(save_data)
    return client, local_session
```

- [ ] **Step 5: Run the test**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_boot.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 6: Commit**

```bash
git add patches/0006-headless-entry-from-save.patch tuxghost/boot.py tests/test_boot.py
git commit -m "feat: boot a headless world from a SaveData, not a state unwind"
```

---

### Task 4: The single stepping loop and step-indexed input

Upstream's `HeadlessClient.main()` derives its accumulator from `time.time()` and calls `time.sleep(0.001)`, so it cannot be used for reproducible execution. `InputManager.process_events()` in playback mode yields **one event per frame** regardless of when it was recorded, destroying timing. This task replaces both with one loop that both recording and execution use.

**Files:**
- Create: `tuxghost/loop.py`
- Create: `tests/test_loop.py`

**Interfaces:**
- Consumes: `build_client` from Task 3.
- Produces:
  - `tuxghost.loop.InputSchedule` — `dict[int, list[tuple[int, float]]]`, mapping step -> list of `(button, value)`.
  - `tuxghost.loop.install_schedule(client, schedule: InputSchedule) -> None`
  - `tuxghost.loop.run_steps(client, n: int, hook: Callable[[int], None] | None = None) -> None`
  - `tuxghost.loop.STEP_RATE = 60` and `FIXED_DT = 1.0 / 60.0`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loop.py
from tuxghost.loop import FIXED_DT, STEP_RATE, install_schedule, run_steps


def test_step_rate_matches_upstream_fixed_dt() -> None:
    assert STEP_RATE == 60
    assert FIXED_DT == 1.0 / 60.0


def test_schedule_delivers_events_on_exact_steps() -> None:
    """Timing is preserved: an event scheduled for step 5 arrives at step 5,
    not on the next frame the way upstream's playback does."""
    from tuxghost.boot import build_client

    client, _session = build_client(seed=1234)
    seen: list[tuple[int, int]] = []
    install_schedule(client, {5: [(64, 1.0)], 9: [(64, 0.0)]})

    original = client.input_manager.process_events

    def spy() -> object:
        for event in original():
            seen.append((step_counter[0], event.button))
            yield event

    step_counter = [0]
    client.input_manager.process_events = spy
    run_steps(client, 12, hook=lambda i: step_counter.__setitem__(0, i))

    assert [s for s, _b in seen] == [5, 9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuxghost.loop'`

- [ ] **Step 3: Implement the loop**

```python
# tuxghost/loop.py
"""The one stepping loop. A second one would fail nothing, so there is not one."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

STEP_RATE = 60
FIXED_DT = 1.0 / STEP_RATE

InputSchedule = dict[int, list[tuple[int, float]]]


def install_schedule(client: Any, schedule: InputSchedule) -> None:
    """Deliver scheduled inputs on exact step indices.

    Replaces InputManager.process_events entirely. Upstream's playback path
    yields one recorded event per frame regardless of its recorded time,
    which silently rescales every trace; we must not inherit that.
    """
    from tuxemon.platform.events import PlayerInput

    state = {"step": 0}

    def process_events() -> Iterator[Any]:
        for button, value in schedule.get(state["step"], []):
            event = PlayerInput(button, value, 1 if value else 0)
            event.triggered = bool(value)
            yield event
        state["step"] += 1

    client.input_manager.process_events = process_events


def run_steps(
    client: Any, n: int, hook: Callable[[int], None] | None = None
) -> None:
    """Advance exactly n logical steps. No wall clock, no sleep, no draw."""
    for i in range(n):
        if hook is not None:
            hook(i)
        client.update(FIXED_DT)
```

- [ ] **Step 4: Run the test**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_loop.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add tuxghost/loop.py tests/test_loop.py
git commit -m "feat: one step-indexed stepping loop for record and execute"
```

---

### Task 5: The state digest and its discrimination test

**This is the task that keeps the whole project honest.** During the spike the first digest reported "deterministic" across nine runs — including unseeded runs and three different seeds — because it captured nothing that changed. A digest that cannot discriminate makes every downstream test vacuous.

**Files:**
- Create: `tuxghost/digest.py`
- Create: `tests/test_digest.py`

**Interfaces:**
- Consumes: `build_client`, `run_steps`, `install_schedule`.
- Produces:
  - `tuxghost.digest.EXEMPTIONS: dict[str, str]` — field name -> the defect it waits on. Target size zero.
  - `tuxghost.digest.state_of(session) -> dict[str, Any]`
  - `tuxghost.digest.digest_of(session) -> str` (sha256 hex)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_digest.py
import pytest

from tuxghost.boot import build_client
from tuxghost.digest import EXEMPTIONS, digest_of, state_of
from tuxghost.loop import install_schedule, run_steps
from tuxemon.platform.const import buttons


def _run(seed: int, steps: int = 900) -> str:
    client, session = build_client(seed=seed)
    schedule = {}
    for i in range(40):
        schedule[30 + i * 12] = [(buttons.A, 1.0)]
        schedule[32 + i * 12] = [(buttons.A, 0.0)]
    install_schedule(client, schedule)
    run_steps(client, steps)
    return digest_of(session)


def test_digest_is_stable_for_one_seed() -> None:
    assert _run(1234) == _run(1234)


def test_digest_discriminates_between_seeds() -> None:
    """A digest that cannot tell seeds apart makes every determinism test
    vacuous. This is the control that caught a worthless probe in the spike."""
    client_a, session_a = build_client(seed=1234)
    session_a.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    client_b, session_b = build_client(seed=99)
    session_b.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    assert digest_of(session_a) != digest_of(session_b)


def test_digest_sees_gameplay_state() -> None:
    client, session = build_client(seed=1234)
    before = digest_of(session)
    session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    assert digest_of(session) != before


def test_exemptions_each_name_a_defect() -> None:
    for field, reason in EXEMPTIONS.items():
        assert reason.strip(), f"exemption {field!r} has no stated reason"
    assert "timestamp" not in EXEMPTIONS, (
        "Battle.timestamp becomes deterministic under patch 0004; exempting "
        "it would re-hide the bug the spike spent a bisect finding"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_digest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuxghost.digest'`

- [ ] **Step 3: Implement the digest**

```python
# tuxghost/digest.py
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
            k: _canonical(v)
            for k, v in sorted(obj.items())
            if k not in EXEMPTIONS
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
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_digest.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add tuxghost/digest.py tests/test_digest.py
git commit -m "feat: canonical state digest with a discrimination control"
```

---

### Task 6: Close the spec's open question — battle to world transition

The spec parks one measurement that could still change the format: whether combat stays deterministic through a **completed battle, exited back to the world**. Every spike run ended with `CombatState` still on the stack, so capture, evolution and the battle exit are unmeasured. If this diverges, `verify` needs a divergence-tolerant mode and the format changes; if it holds, the format is safe.

**Files:**
- Create: `tests/test_combat_determinism.py`
- Create: `docs/2026-08-25-battle-exit-measurement.org`

**Interfaces:**
- Consumes: `build_client`, `install_schedule`, `run_steps`, `digest_of`.
- Produces: a recorded measurement; no new API.

- [ ] **Step 1: Write the test that drives a battle to its exit**

```python
# tests/test_combat_determinism.py
import os

import pytest

from tuxghost.boot import build_client
from tuxghost.digest import digest_of
from tuxghost.loop import install_schedule, run_steps
from tuxemon.platform.const import buttons

pytestmark = pytest.mark.slow


def _mash(button: int, count: int, period: int) -> dict[int, list[tuple[int, float]]]:
    schedule: dict[int, list[tuple[int, float]]] = {}
    for i in range(count):
        schedule.setdefault(i * period, []).append((button, 1.0))
        schedule.setdefault(i * period + 2, []).append((button, 0.0))
    return schedule


def _battle_run(seed: int, steps: int) -> tuple[str, bool]:
    client, session = build_client(seed=seed)
    install_schedule(client, _mash(buttons.A, 60, 10))
    run_steps(client, 700)

    execute = client.event_engine.execute_action
    execute("add_monster", ("rockitten", 12))
    execute("add_monster", ("budaye", 10))
    # _start_battle returns early, and silently, when no environment is active
    execute("set_environment", ("grass",))
    # skip=True runs start() without spinning: EventAction.run() never advances
    # the client, so a normal call deadlocks headless forever
    execute("random_battle", (2, 8, 14), True)

    install_schedule(client, _mash(buttons.A, steps // 5 + 2, 5))
    exited = False
    was_in = False

    def watch(i: int) -> None:
        nonlocal exited, was_in
        names = [s.name for s in client.state_manager.active_states]
        if "CombatState" in names:
            was_in = True
        elif was_in:
            exited = True

    run_steps(client, steps, hook=watch)
    return digest_of(session), exited


@pytest.mark.skipif(
    not os.environ.get("TUXGHOST_RUN_SLOW"),
    reason="drives a full battle; minutes per run",
)
def test_battle_is_deterministic_through_its_exit() -> None:
    a, exited_a = _battle_run(1234, 200_000)
    b, exited_b = _battle_run(1234, 200_000)
    assert exited_a == exited_b
    assert a == b, "combat diverged; verify needs a divergence-tolerant mode"
```

- [ ] **Step 2: Run it and record what actually happens**

Run: `PYTHONHASHSEED=0 TUXGHOST_RUN_SLOW=1 ./.venv/bin/python -m pytest tests/test_combat_determinism.py -v -s`
Expected: PASS. **If `exited_a` is False, the battle still did not end** — raise the step budget and rerun before believing the result. Report the real values.

- [ ] **Step 3: Write up the measurement**

Create `docs/2026-08-25-battle-exit-measurement.org` recording: the step budget needed, whether the battle actually exited, both digests, and — if it diverged — the field-level diff. State plainly whether the format is safe.

- [ ] **Step 4: Commit**

```bash
git add tests/test_combat_determinism.py docs/2026-08-25-battle-exit-measurement.org
git commit -m "test: measure determinism through a completed battle exit"
```

---

### Task 7: Patch 0002 — seed the weather RNG

`WorldWeatherManager` is constructed seedless at `base_client.py:189`, and `world/weather.py:133` then builds `random.Random()` from OS entropy. It is the **only** instance-RNG in the codebase, and it accounted for all 152 unseeded draws the spike measured on a walking route. It already accepts a `seed`; it is simply never given one.

**Files:**
- Create: `patches/0002-seed-weather-rng.patch`
- Create: `tuxghost/determinism.py`
- Create: `tests/test_determinism_sources.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tuxghost.determinism.seed_all(seed: int) -> None` — seeds every entropy source from one master seed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_determinism_sources.py
import random

from tuxghost.determinism import seed_all


def test_weather_rng_is_seeded_from_the_master_seed() -> None:
    from tuxghost.boot import build_client

    seed_all(1234)
    client_a, _ = build_client(seed=1234)
    draws_a = [client_a.weather_manager._rng.random() for _ in range(5)]

    seed_all(1234)
    client_b, _ = build_client(seed=1234)
    draws_b = [client_b.weather_manager._rng.random() for _ in range(5)]

    assert draws_a == draws_b


def test_master_seed_controls_the_global_generator() -> None:
    seed_all(7)
    first = [random.random() for _ in range(3)]
    seed_all(7)
    assert [random.random() for _ in range(3)] == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_determinism_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuxghost.determinism'`

- [ ] **Step 3: Write the patch**

```diff
--- a/tuxemon/base_client.py
+++ b/tuxemon/base_client.py
@@
-        self.weather_manager = WorldWeatherManager()
+        self.weather_manager = WorldWeatherManager(
+            seed=config.deterministic_seed
+        )
```

```diff
--- a/tuxemon/config.py
+++ b/tuxemon/config.py
@@
+        # None keeps upstream behaviour: weather seeds from OS entropy.
+        self.deterministic_seed: int | None = None
```

- [ ] **Step 4: Implement seed_all**

```python
# tuxghost/determinism.py
"""Seed every entropy source from one master seed.

The spike found exactly three: the global `random` generator,
WorldWeatherManager's instance RNG, and uuid4. One master seed derives all
three; separate seeds would let a trace be 'seeded' while one source stayed
live, which is the state in which a probe reported nine identical hashes and
meant nothing.
"""

from __future__ import annotations

import random


def seed_all(seed: int) -> None:
    random.seed(seed)

    from tuxemon.user_config import CONFIG

    CONFIG.deterministic_seed = seed
```

- [ ] **Step 5: Apply the patch and run the tests**

Run: `cd tuxemon && git apply ../patches/0002-seed-weather-rng.patch && cd .. && PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_determinism_sources.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 6: Mutation gate — prove the patch is load-bearing**

Run: `cd tuxemon && git stash && cd .. && PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_determinism_sources.py::test_weather_rng_is_seeded_from_the_master_seed -v; cd tuxemon && git stash pop`
Expected: FAIL. Record the real draws in the commit message.

- [ ] **Step 7: Commit**

```bash
git add patches/0002-seed-weather-rng.patch tuxghost/determinism.py tests/test_determinism_sources.py
git commit -m "feat: patch 0002 seeds the only instance RNG in the codebase"
```

---

### Task 8: Patch 0003 — seeded uuid factory

With weather seeded, three same-seed spike runs still diverged by exactly one field: a monster `instance_id`. `uuid4()` draws from OS entropy and ignores `random.seed()` entirely.

**Files:**
- Create: `patches/0003-seeded-uuid-factory.patch`
- Modify: `tuxghost/determinism.py`
- Modify: `tests/test_determinism_sources.py`

**Interfaces:**
- Consumes: `seed_all`.
- Produces: `tuxemon.core.ids.new_id() -> UUID` (added by the patch); `seed_all` also seeds it.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_determinism_sources.py
def test_entity_ids_are_reproducible() -> None:
    from tuxghost.boot import build_client

    def ids(seed: int) -> list[str]:
        seed_all(seed)
        _client, session = build_client(seed=seed)
        session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
        return [str(m.instance_id) for m in session.player.monsters]

    assert ids(1234) == ids(1234)
    assert ids(1234) != ids(99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_determinism_sources.py::test_entity_ids_are_reproducible -v`
Expected: FAIL — the two `ids(1234)` lists differ.

- [ ] **Step 3: Write the patch**

Create `tuxemon/core/ids.py` and route the five load-bearing call sites through it (`entity/entity.py:218`, `battle.py:29`, `technique/technique.py:56`, `status/status.py:46`, `session.py:42`):

```diff
--- /dev/null
+++ b/tuxemon/core/ids.py
+"""Central identifier factory, so ids can be made reproducible."""
+from __future__ import annotations
+
+import random
+import uuid
+
+_rng: random.Random | None = None
+
+
+def seed_ids(seed: int | None) -> None:
+    """Seed the id factory. None restores OS entropy."""
+    global _rng
+    _rng = None if seed is None else random.Random(seed)
+
+
+def new_id() -> uuid.UUID:
+    if _rng is None:
+        return uuid.uuid4()
+    return uuid.UUID(int=_rng.getrandbits(128), version=4)
```

```diff
--- a/tuxemon/entity/entity.py
+++ b/tuxemon/entity/entity.py
@@
-        self.instance_id = instance_id or uuid4()
+        self.instance_id = instance_id or new_id()
```

- [ ] **Step 4: Extend seed_all**

```python
# in tuxghost/determinism.py, inside seed_all(), after CONFIG assignment
    from tuxemon.core.ids import seed_ids

    seed_ids(seed)
```

- [ ] **Step 5: Apply and run**

Run: `cd tuxemon && git apply ../patches/0003-seeded-uuid-factory.patch && cd .. && PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_determinism_sources.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Mutation gate**

Revert patch 0003 only, rerun `test_entity_ids_are_reproducible`. Expected: FAIL. Restore.

- [ ] **Step 7: Drop the now-closed exemptions**

Remove `uuid` and `instance_id` from `EXEMPTIONS` in `tuxghost/digest.py`. Run `make check-fast`; the digest tests must still pass.

- [ ] **Step 8: Commit**

```bash
git add patches/0003-seeded-uuid-factory.patch tuxghost/determinism.py tuxghost/digest.py tests/test_determinism_sources.py
git commit -m "feat: patch 0003 makes entity ids reproducible; two exemptions closed"
```

---

### Task 9: Patch 0004 — the injectable clock

The highest-value patch. `TimeHandler.get_current_time()` is the single `datetime.now()` choke point — both `get_ordinal()` and `get_time_variables()` route through it — and `TimeHandler` is constructed in exactly one place, `session.py:43`. Separately, `battle.py:33` stamps `time.time()` into every persisted battle record. That field was the **entire** divergence between two 120,000-step runs.

**Files:**
- Create: `patches/0004-injectable-clock.patch`
- Modify: `tuxghost/determinism.py`
- Create: `tests/test_clock.py`

**Interfaces:**
- Consumes: `seed_all`.
- Produces: `tuxemon.core.clock.set_epoch(epoch: float | None) -> None` and `now() -> float`; `tuxghost.determinism.pin_clock(epoch: int) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clock.py
from tuxghost.boot import build_client
from tuxghost.determinism import pin_clock, seed_all


def test_time_variables_come_from_the_pinned_epoch() -> None:
    seed_all(1234)
    pin_clock(1787694000)  # 2026-08-25T12:20:00Z
    _client, session = build_client(seed=1234)
    variables = dict(session.player.game_variables.items())
    first = (variables.get("day_of_year"), variables.get("hour"))

    seed_all(1234)
    pin_clock(1787694000)
    _client2, session2 = build_client(seed=1234)
    variables2 = dict(session2.player.game_variables.items())
    assert (variables2.get("day_of_year"), variables2.get("hour")) == first


def test_a_different_epoch_changes_the_day() -> None:
    seed_all(1234)
    pin_clock(1787694000)
    _c, session = build_client(seed=1234)
    day_a = dict(session.player.game_variables.items()).get("day_of_year")

    seed_all(1234)
    pin_clock(1787694000 + 86400 * 30)
    _c2, session2 = build_client(seed=1234)
    day_b = dict(session2.player.game_variables.items()).get("day_of_year")
    assert day_a != day_b


def test_battle_timestamps_are_pinned() -> None:
    from tuxemon.battle import Battle

    pin_clock(1787694000)
    assert Battle().timestamp == Battle().timestamp == 1787694000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_clock.py -v`
Expected: FAIL with `ImportError: cannot import name 'pin_clock'`

- [ ] **Step 3: Write the patch**

```diff
--- /dev/null
+++ b/tuxemon/core/clock.py
+"""Injectable wall clock, so recorded state can be reproduced.
+
+Unpinned, wall clock leaks into persisted game state (day_of_year, hour,
+Battle.timestamp) and a trace silently stops reproducing the next day.
+"""
+from __future__ import annotations
+
+import time as _time
+from datetime import datetime
+
+_epoch: float | None = None
+
+
+def set_epoch(epoch: float | None) -> None:
+    global _epoch
+    _epoch = epoch
+
+
+def now() -> float:
+    return _time.time() if _epoch is None else _epoch
+
+
+def now_datetime() -> datetime:
+    return datetime.fromtimestamp(now())
```

```diff
--- a/tuxemon/time_handler.py
+++ b/tuxemon/time_handler.py
@@
     def get_current_time(self) -> datetime:
-        """Returns the real current datetime."""
-        return datetime.now()
+        """Returns the current datetime, honouring any pinned epoch."""
+        return now_datetime()
```

```diff
--- a/tuxemon/battle.py
+++ b/tuxemon/battle.py
@@
-        self.timestamp: float = time.time()
+        self.timestamp: float = now()
```

Apply the same substitution to the three `time.time()` reads in `tuxemon/session.py` (`_start_timestamp`, and both in `get_state`).

- [ ] **Step 4: Implement pin_clock**

```python
# append to tuxghost/determinism.py
def pin_clock(epoch: int) -> None:
    """Pin all wall-clock reads to a fixed epoch (seconds since 1970)."""
    from tuxemon.core.clock import set_epoch

    set_epoch(float(epoch))
```

- [ ] **Step 5: Apply and run**

Run: `cd tuxemon && git apply ../patches/0004-injectable-clock.patch && cd .. && PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_clock.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Mutation gate**

Revert patch 0004 only, rerun `test_battle_timestamps_are_pinned`. Expected: FAIL, the two timestamps differ. Restore.

- [ ] **Step 7: Empty the exemption register**

Remove `start_time`, `duration` and `total_playtime` from `EXEMPTIONS`, leaving `EXEMPTIONS = {}`. Run `make check-fast`. All digest tests must still pass — if any now fails, a real defect has surfaced; fix it rather than re-adding the exemption.

- [ ] **Step 8: Commit**

```bash
git add patches/0004-injectable-clock.patch tuxghost/determinism.py tuxghost/digest.py tests/test_clock.py
git commit -m "feat: patch 0004 pins the clock; exemption register now empty"
```

---

### Task 10: The trace format and its preconditions

**Files:**
- Create: `tuxghost/trace.py`
- Create: `tests/test_trace.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `tuxghost.trace.Trace` (pydantic model) with `.header`, `.initial_state`, `.inputs`, `.provenance`
  - `tuxghost.trace.TraceHeader`, `tuxghost.trace.Provenance`
  - `tuxghost.trace.read(path: Path, allow_mismatch: bool = False) -> Trace`
  - `tuxghost.trace.write(trace: Trace, path: Path) -> None`
  - `tuxghost.trace.Refused` (exception), `tuxghost.trace.FORMAT_VERSION = 1`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trace.py
import json

import pytest

from tuxghost.trace import FORMAT_VERSION, Refused, read, write


def _minimal(tmp_path, **overrides):
    data = {
        "format_version": FORMAT_VERSION,
        "header": {
            "upstream_commit": "59a34164f442ddecbaee4c436e3f1a5ba9474e29",
            "patch_series_id": "sha256:abc",
            "mod_id": "tuxemon",
            "mod_version": "0.4.35",
            "seed": 1234,
            "step_rate": 60,
            "clock_epoch": 1787694000,
            "initial_state_digest": "sha256:def",
            "step_count": 10,
            "final_digest": "sha256:000",
        },
        "initial_state": {},
        "inputs": [[5, 64, 1.0]],
        "provenance": {"recorder": "offline-agent"},
    }
    data.update(overrides)
    path = tmp_path / "t.tuxghost"
    path.write_text(json.dumps(data))
    return path


def test_reads_a_valid_trace(tmp_path) -> None:
    trace = read(_minimal(tmp_path))
    assert trace.header.seed == 1234
    assert trace.inputs == [(5, 64, 1.0)]


def test_rejects_an_unknown_format_version(tmp_path) -> None:
    with pytest.raises(Refused, match="format_version"):
        read(_minimal(tmp_path, format_version=2))


def test_refuses_a_trace_with_no_clock_epoch(tmp_path) -> None:
    """An unpinned clock silently stops reproducing at midnight. It corrupted
    a comparison twice during the spike while every outcome looked correct,
    so this is a refusal, not a warning."""
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    del data["header"]["clock_epoch"]
    path.write_text(json.dumps(data))
    with pytest.raises(Refused, match="clock_epoch"):
        read(path)


def test_allow_mismatch_does_not_downgrade_clock_epoch(tmp_path) -> None:
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    del data["header"]["clock_epoch"]
    path.write_text(json.dumps(data))
    with pytest.raises(Refused, match="clock_epoch"):
        read(path, allow_mismatch=True)


def test_mod_version_mismatch_refuses_but_allow_mismatch_taints(tmp_path) -> None:
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    data["header"]["mod_version"] = "0.4.99"
    path.write_text(json.dumps(data))

    with pytest.raises(Refused, match="mod_version"):
        read(path)

    trace = read(path, allow_mismatch=True)
    assert "trace_mismatch" in trace.provenance.taints


def test_differing_upstream_commit_warns_rather_than_refuses(tmp_path, recwarn) -> None:
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    data["header"]["upstream_commit"] = "0" * 40
    path.write_text(json.dumps(data))

    trace = read(path)
    assert trace.header.upstream_commit == "0" * 40
    assert any("upstream_commit" in str(w.message) for w in recwarn)


def test_round_trips(tmp_path) -> None:
    trace = read(_minimal(tmp_path))
    out = tmp_path / "out.tuxghost"
    write(trace, out)
    assert read(out).header == trace.header
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_trace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuxghost.trace'`

- [ ] **Step 3: Implement the format**

```python
# tuxghost/trace.py
"""Trace format version 1.

Inputs are ground truth and are step-indexed. PlayerInput.timestamp, which
upstream defaults to time.time(), deliberately does not enter this format.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

FORMAT_VERSION = 1

#: What this build is. A trace recorded elsewhere is still runnable, but the
#: reader says so rather than pretending the environments match.
THIS_UPSTREAM_COMMIT = "59a34164f442ddecbaee4c436e3f1a5ba9474e29"
THIS_MOD_VERSION = "0.4.35"
THIS_STEP_RATE = 60

#: Refused outright: reading on would produce a silently wrong comparison.
_REFUSE_IF_MISSING = ("seed", "clock_epoch")


class Refused(Exception):
    """A precondition failed; the trace will not be executed here."""


class TraceHeader(BaseModel):
    upstream_commit: str
    patch_series_id: str
    mod_id: str
    mod_version: str
    seed: int
    step_rate: int
    clock_epoch: int
    initial_state_digest: str
    step_count: int
    final_digest: str


class Provenance(BaseModel):
    recorder: Literal["cu-agent", "offline-agent", "human"]
    model: str | None = None
    claimed_outcome: str | None = None
    taints: list[str] = Field(default_factory=list)
    recorded_at: str | None = None


class Trace(BaseModel):
    format_version: int
    header: TraceHeader
    initial_state: dict[str, Any]
    inputs: list[tuple[int, int, float]]
    provenance: Provenance


def read(path: Path, allow_mismatch: bool = False) -> Trace:
    raw = json.loads(Path(path).read_text())

    version = raw.get("format_version")
    if version != FORMAT_VERSION:
        raise Refused(
            f"format_version {version!r} is not {FORMAT_VERSION}; refusing "
            "rather than guessing at an unknown layout"
        )

    header = raw.get("header", {})
    for field in _REFUSE_IF_MISSING:
        if header.get(field) is None:
            raise Refused(
                f"header.{field} is required. allow_mismatch does not "
                f"downgrade this: silencing a version quibble must not also "
                f"silence the world changing underneath the trace."
            )

    # Refusals that allow_mismatch may downgrade -- a version quibble, not a
    # changed world.
    downgradable = {
        "mod_version": (header.get("mod_version"), THIS_MOD_VERSION),
        "step_rate": (header.get("step_rate"), THIS_STEP_RATE),
    }
    tainted = False
    for field, (got, want) in downgradable.items():
        if got != want:
            if not allow_mismatch:
                raise Refused(
                    f"header.{field} is {got!r}, this build is {want!r}. "
                    f"Pass allow_mismatch to run anyway; the trace will be "
                    f"tainted."
                )
            tainted = True

    # Environment differences that are reported, never fatal.
    if header.get("upstream_commit") != THIS_UPSTREAM_COMMIT:
        warnings.warn(
            f"upstream_commit {header.get('upstream_commit')!r} differs from "
            f"this build {THIS_UPSTREAM_COMMIT!r}",
            stacklevel=2,
        )

    trace = Trace.model_validate(raw)
    if tainted:
        trace.provenance.taints.append("trace_mismatch")
    return trace


def write(trace: Trace, path: Path) -> None:
    Path(path).write_text(trace.model_dump_json(indent=2))
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_trace.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add tuxghost/trace.py tests/test_trace.py
git commit -m "feat: trace format v1 with refuse-on-missing-clock preconditions"
```

---

### Task 11: The recorder

**Files:**
- Create: `tuxghost/record.py`
- Create: `tests/test_record.py`

**Interfaces:**
- Consumes: `Trace`, `TraceHeader`, `Provenance`, `snapshot_save`, `digest_of`, `run_steps`.
- Produces: `tuxghost.record.Recorder` with `.observe(step, button, value)` and `.finish(step_count) -> Trace`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_record.py
from tuxghost.record import Recorder


def test_recorder_indexes_inputs_by_step_not_arrival_order() -> None:
    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234)
    recorder = Recorder(session, seed=1234, clock_epoch=1787694000,
                        recorder="offline-agent")
    recorder.observe(9, 64, 0.0)
    recorder.observe(5, 64, 1.0)
    trace = recorder.finish(step_count=12)

    assert trace.inputs == [(5, 64, 1.0), (9, 64, 0.0)]
    assert trace.header.step_rate == 60
    assert trace.header.clock_epoch == 1787694000
    assert trace.header.final_digest, "finish() must record the state reached"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_record.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuxghost.record'`

- [ ] **Step 3: Implement the recorder**

```python
# tuxghost/record.py
"""Records a step-indexed trace from a live session."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from tuxghost.boot import snapshot_save
from tuxghost.trace import FORMAT_VERSION, Provenance, Trace, TraceHeader

UPSTREAM_COMMIT = "59a34164f442ddecbaee4c436e3f1a5ba9474e29"
PATCHES_DIR = Path(__file__).resolve().parent.parent / "patches"


def patch_series_id() -> str:
    """Digest of the applied patch series, so a trace records which engine
    it was produced against."""
    import hashlib

    digest = hashlib.sha256()
    for patch in sorted(PATCHES_DIR.glob("*.patch")):
        digest.update(patch.name.encode())
        digest.update(patch.read_bytes())
    return "sha256:" + digest.hexdigest()


class Recorder:
    def __init__(
        self,
        session: Any,
        seed: int,
        clock_epoch: int,
        recorder: Literal["cu-agent", "offline-agent", "human"],
        model: str | None = None,
    ) -> None:
        self._session = session
        self._seed = seed
        self._clock_epoch = clock_epoch
        self._recorder = recorder
        self._model = model
        self._inputs: list[tuple[int, int, float]] = []
        self._initial = json.loads(snapshot_save(session).model_dump_json())

    def observe(self, step: int, button: int, value: float) -> None:
        self._inputs.append((step, button, value))

    def finish(self, step_count: int) -> Trace:
        """Seal the trace, recording the state the run actually reached.

        Without a recorded outcome, `verify` could only prove a trace is
        reproducible -- not that it still reaches what it originally reached,
        which is what verification means.
        """
        import hashlib

        from tuxghost.digest import digest_of
        from tuxghost.loop import STEP_RATE

        blob = json.dumps(self._initial, sort_keys=True, separators=(",", ":"))
        return Trace(
            format_version=FORMAT_VERSION,
            header=TraceHeader(
                upstream_commit=UPSTREAM_COMMIT,
                patch_series_id=patch_series_id(),
                mod_id="tuxemon",
                mod_version="0.4.35",
                seed=self._seed,
                step_rate=STEP_RATE,
                clock_epoch=self._clock_epoch,
                initial_state_digest="sha256:"
                + hashlib.sha256(blob.encode()).hexdigest(),
                step_count=step_count,
                final_digest=digest_of(self._session),
            ),
            initial_state=self._initial,
            inputs=sorted(self._inputs),
            provenance=Provenance(
                recorder=self._recorder, model=self._model
            ),
        )
```

- [ ] **Step 4: Run the test**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_record.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tuxghost/record.py tests/test_record.py
git commit -m "feat: step-indexed trace recorder"
```

---

### Task 12: The executor, verify, and divergence reporting

**Files:**
- Create: `tuxghost/execute.py`
- Create: `tuxghost/compare.py`
- Create: `tests/test_execute.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `tuxghost.execute.execute(trace: Trace, checkpoint: int = 0) -> ExecutionResult` with `.final_digest`, `.checkpoints: list[tuple[int, str]]`
  - `tuxghost.execute.verify(trace: Trace) -> int` returning 0 / 1 / 2
  - `tuxghost.compare.first_difference(a: dict, b: dict) -> str | None` — dotted field path
  - `tuxghost.compare.first_divergent_step(a: list[tuple[int, str]], b: list[tuple[int, str]]) -> int | None` — earliest checkpoint whose digests differ

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_execute.py
from tuxghost.compare import first_difference


def test_first_difference_names_the_field_path() -> None:
    a = {"npc_state": {"battles": [{"timestamp": 1.0}]}}
    b = {"npc_state": {"battles": [{"timestamp": 2.0}]}}
    assert first_difference(a, b) == "npc_state.battles[0].timestamp"


def test_first_difference_is_none_when_equal() -> None:
    assert first_difference({"a": [1, 2]}, {"a": [1, 2]}) is None


def test_first_divergent_step_finds_the_earliest_checkpoint() -> None:
    from tuxghost.compare import first_divergent_step

    a = [(100, "x"), (200, "y"), (300, "z")]
    b = [(100, "x"), (200, "DIFF"), (300, "z")]
    assert first_divergent_step(a, b) == 200
    assert first_divergent_step(a, a) is None


def test_verify_returns_zero_for_a_self_consistent_trace(tmp_path) -> None:
    from tuxghost.execute import verify
    from tuxghost.record import Recorder
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all

    seed_all(1234)
    pin_clock(1787694000)
    _client, session = build_client(seed=1234)
    trace = Recorder(session, seed=1234, clock_epoch=1787694000,
                     recorder="offline-agent").finish(step_count=120)
    assert verify(trace) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_execute.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuxghost.compare'`

- [ ] **Step 3: Implement comparison**

```python
# tuxghost/compare.py
"""Field-level divergence reporting."""

from __future__ import annotations

from typing import Any


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    """Return the dotted path of the first differing field, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else key
            if key not in a or key not in b:
                return sub
            found = first_difference(a[key], b[key], sub)
            if found is not None:
                return found
        return None

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}[len]"
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found is not None:
                return found
        return None

    return None if a == b else path


def first_divergent_step(
    a: list[tuple[int, str]], b: list[tuple[int, str]]
) -> int | None:
    """Earliest checkpoint step at which two runs' digests differ.

    Narrowing to a step before naming a field is what turned 'the hash
    differs' into 'npc_state.battles[0].timestamp' during the spike.
    """
    for (step_a, digest_a), (step_b, digest_b) in zip(a, b):
        assert step_a == step_b, "checkpoint cadences must match"
        if digest_a != digest_b:
            return step_a
    return None
```

- [ ] **Step 4: Implement the executor**

```python
# tuxghost/execute.py
"""Offline executor. Exit codes: 0 matched, 1 diverged, 2 refused."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tuxghost.boot import boot_from_save
from tuxghost.determinism import pin_clock, seed_all
from tuxghost.digest import digest_of, state_of
from tuxghost.loop import InputSchedule, install_schedule, run_steps
from tuxghost.trace import Refused, Trace


@dataclass
class ExecutionResult:
    final_digest: str
    final_state: dict[str, Any]
    checkpoints: list[tuple[int, str]] = field(default_factory=list)


def _schedule_of(trace: Trace) -> InputSchedule:
    schedule: InputSchedule = {}
    for step, button, value in trace.inputs:
        schedule.setdefault(step, []).append((button, value))
    return schedule


def execute(trace: Trace, checkpoint: int = 0) -> ExecutionResult:
    """Replay a trace. Takes no map, seed or save: all of it is the header."""
    from tuxemon.save_system.save_state import SaveData

    seed_all(trace.header.seed)
    pin_clock(trace.header.clock_epoch)

    _client, session = boot_from_save(
        SaveData.model_validate(trace.initial_state), seed=trace.header.seed
    )
    client = session.client
    install_schedule(client, _schedule_of(trace))

    result = ExecutionResult(final_digest="", final_state={})

    def hook(i: int) -> None:
        if checkpoint and i and i % checkpoint == 0:
            result.checkpoints.append((i, digest_of(session)))

    run_steps(client, trace.header.step_count, hook=hook)
    result.final_state = state_of(session)
    result.final_digest = digest_of(session)
    return result


def verify(trace: Trace) -> int:
    """Execute `trace` and compare against itself, in process.

    0 every compared field matched, 1 at least one differed, 2 refused.
    """
    try:
        result = execute(trace)
    except Refused:
        return 2
    return 0 if result.final_digest == trace.header.final_digest else 1
```

- [ ] **Step 5: Run the tests**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_execute.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add tuxghost/execute.py tuxghost/compare.py tests/test_execute.py
git commit -m "feat: offline executor with verify and field-level divergence"
```

---

### Task 13: The CLI

**Files:**
- Create: `tuxghost/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `read`, `write`, `execute`, `verify`, `first_difference`.
- Produces: `tuxghost.cli.main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
import pytest

from tuxghost.cli import main


def test_execute_and_verify_together_are_refused() -> None:
    assert main(["execute", "x.tuxghost", "--verify"]) == 2


def test_unknown_command_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_info_refuses_a_missing_clock_epoch(tmp_path) -> None:
    import json

    path = tmp_path / "bad.tuxghost"
    path.write_text(json.dumps({"format_version": 1, "header": {"seed": 1}}))
    assert main(["info", str(path)]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuxghost.cli'`

- [ ] **Step 3: Implement the CLI**

```python
# tuxghost/cli.py
"""tuxghost command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tuxghost")
    sub = parser.add_subparsers(dest="command", required=True)

    p_exec = sub.add_parser("execute")
    p_exec.add_argument("trace", type=Path)
    p_exec.add_argument("--record", type=Path, default=None)
    p_exec.add_argument("--verify", action="store_true")

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("trace", type=Path)

    p_info = sub.add_parser("info")
    p_info.add_argument("trace", type=Path)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("a", type=Path)
    p_cmp.add_argument("b", type=Path)

    p_rec = sub.add_parser("record")
    p_rec.add_argument("out", type=Path)
    p_rec.add_argument("--from-save", type=Path, required=True)
    p_rec.add_argument("--seed", type=int, required=True)
    p_rec.add_argument("--clock-epoch", type=int, required=True)
    p_rec.add_argument("--steps", type=int, required=True)

    args = parser.parse_args(argv)

    if args.command == "execute" and args.verify:
        print(
            "refused: --verify and execute do different things; run "
            "`tuxghost verify` instead",
            file=sys.stderr,
        )
        return 2

    from tuxghost.trace import Refused, read

    if args.command == "compare":
        return _compare(args.a, args.b)

    if args.command == "record":
        return _record(args)

    try:
        trace = read(args.trace)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    if args.command == "info":
        print(trace.header.model_dump_json(indent=2))
        print(trace.provenance.model_dump_json(indent=2))
        return 0

    from tuxghost.execute import execute, verify

    if args.command == "verify":
        return verify(trace)

    result = execute(trace)
    print(result.final_digest)
    return 0


def _compare(path_a: Path, path_b: Path) -> int:
    """Out-of-engine comparison. Reports findings, which are not divergences."""
    from tuxghost.compare import first_difference
    from tuxghost.trace import Refused, read

    try:
        a, b = read(path_a), read(path_b)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    for name, trace in (("a", a), ("b", b)):
        for taint in trace.provenance.taints:
            print(f"finding: {name} is tainted: {taint}")
        if trace.provenance.recorder == "human":
            print(
                f"finding: {name} was recorded from a human loop, whose step "
                f"count comes from elapsed real time"
            )

    field = first_difference(a.initial_state, b.initial_state)
    if field is not None:
        print(f"initial_state differs at {field}")
        return 1
    if a.inputs != b.inputs:
        print("input tracks differ")
        return 1
    print("traces agree")
    return 0


def _record(args: argparse.Namespace) -> int:
    """Record a trace by replaying a save forward with no input."""
    import json

    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.loop import run_steps
    from tuxghost.record import Recorder
    from tuxghost.trace import write

    seed_all(args.seed)
    pin_clock(args.clock_epoch)
    save_data = SaveData.model_validate(json.loads(args.from_save.read_text()))
    _client, session = boot_from_save(save_data, seed=args.seed)
    recorder = Recorder(
        session,
        seed=args.seed,
        clock_epoch=args.clock_epoch,
        recorder="human",
    )
    run_steps(session.client, args.steps)
    write(recorder.finish(step_count=args.steps), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add tests for the two remaining subcommands**

```python
# append to tests/test_cli.py
def test_compare_reports_a_human_recorder_as_a_finding(tmp_path, capsys) -> None:
    from tuxghost.record import Recorder
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.trace import write

    seed_all(1234)
    pin_clock(1787694000)
    _c, session = build_client(seed=1234)
    trace = Recorder(session, seed=1234, clock_epoch=1787694000,
                     recorder="human").finish(step_count=10)
    a, b = tmp_path / "a.tuxghost", tmp_path / "b.tuxghost"
    write(trace, a)
    write(trace, b)

    assert main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "finding:" in out and "human loop" in out
    assert "traces agree" in out
```

- [ ] **Step 5: Run the tests**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add tuxghost/cli.py tests/test_cli.py
git commit -m "feat: tuxghost CLI with compare, record, and refused flag combinations"
```

---

### Task 14: Golden traces, round trip, and the long-horizon probe

**Files:**
- Create: `tests/golden/walk_1234.tuxghost`
- Create: `tests/test_golden.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a committed golden trace plus its expected digest.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_golden.py
import os
from pathlib import Path

import pytest

from tuxghost.execute import execute, verify
from tuxghost.trace import read

GOLDEN = Path(__file__).parent / "golden" / "walk_1234.tuxghost"
EXPECTED_DIGEST_FILE = GOLDEN.with_suffix(".digest")


def test_golden_trace_still_reaches_its_recorded_digest() -> None:
    trace = read(GOLDEN)
    assert execute(trace).final_digest == EXPECTED_DIGEST_FILE.read_text().strip()


def test_golden_trace_verifies() -> None:
    assert verify(read(GOLDEN)) == 0


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("TUXGHOST_RUN_SLOW"), reason="minutes per run"
)
def test_long_horizon_trace_is_stable() -> None:
    trace = read(GOLDEN)
    trace.header.step_count = 120_000
    assert execute(trace).final_digest == execute(trace).final_digest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_golden.py -v`
Expected: FAIL — the golden file does not exist.

- [ ] **Step 3: Generate the golden trace**

```bash
PYTHONHASHSEED=0 ./.venv/bin/python - <<'PY'
from pathlib import Path
from tuxghost.boot import build_client
from tuxghost.determinism import pin_clock, seed_all
from tuxghost.record import Recorder
from tuxghost.execute import execute
from tuxghost.trace import write
from tuxemon.platform.const import buttons

seed_all(1234); pin_clock(1787694000)
_c, session = build_client(seed=1234)
rec = Recorder(session, seed=1234, clock_epoch=1787694000, recorder="human")
for i in range(40):
    rec.observe(30 + i * 12, buttons.A, 1.0)
    rec.observe(32 + i * 12, buttons.A, 0.0)
trace = rec.finish(step_count=900)
out = Path("tests/golden/walk_1234.tuxghost")
out.parent.mkdir(parents=True, exist_ok=True)
write(trace, out)
out.with_suffix(".digest").write_text(execute(trace).final_digest + "\n")
print("wrote", out)
PY
```

- [ ] **Step 4: Run the golden tests**

The `slow` marker is already registered in `pyproject.toml` from Task 1.

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_golden.py -v`
Expected: PASS, 2 tests, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add tests/golden tests/test_golden.py
git commit -m "test: golden trace with committed digest and opt-in long-horizon probe"
```

---

### Task 15: Patch 0005 — event actions on the step clock

`EventAction.run()` (`eventaction.py:178`) is a spin loop that computes `dt` from `time.perf_counter()` and never advances the client, so calling a non-instant action normally deadlocks headless forever — its own docstring admits it "may cause the game to hang". The deadlock is observed; the wall-clock `dt` hazard is reasoned, and this patch closes both.

**Files:**
- Create: `patches/0005-action-step-clock.patch`
- Create: `tests/test_action_clock.py`

**Interfaces:**
- Consumes: `run_steps`, `build_client`.
- Produces: no new public API; `EventAction.run` gains a step-driven mode.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_action_clock.py
from tuxghost.boot import build_client
from tuxghost.loop import FIXED_DT, run_steps


def test_action_dt_comes_from_the_step_clock_not_the_wall_clock() -> None:
    """A non-instant action must advance by FIXED_DT per step, so two runs on
    machines of different speed see the same dt sequence."""
    _client, session = build_client(seed=1234)
    seen: list[float] = []

    from tuxemon.event.eventaction import EventAction

    original = EventAction.update

    def spy(self, sess, dt):  # type: ignore[no-untyped-def]
        seen.append(dt)
        return original(self, sess, dt)

    EventAction.update = spy  # type: ignore[method-assign]
    try:
        session.client.event_engine.execute_action(
            "dialog_chain", ("hello",), True
        )
        run_steps(session.client, 30)
    finally:
        EventAction.update = original  # type: ignore[method-assign]

    assert seen, "no action updates observed"
    assert all(d == FIXED_DT for d in seen), f"non-step dt values: {set(seen)}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_action_clock.py -v`
Expected: FAIL — observed `dt` values are wall-clock deltas, not `FIXED_DT`.

- [ ] **Step 3: Write the patch**

```diff
--- a/tuxemon/event/eventaction.py
+++ b/tuxemon/event/eventaction.py
@@
     def run(self, session: Session) -> None:
-        try:
-            last_time = time.perf_counter()
-            while not self.done and not self.cancelled:
-                if self._skip:
-                    self.stop()
-                    return
-
-                now = time.perf_counter()
-                dt = now - last_time
-                last_time = now
-
-                self.update(session, dt)
+        # Actions are advanced by the client's fixed step, not by a wall-clock
+        # spin loop. The old loop never advanced the client, so any action
+        # waiting on game state hung forever, and its dt was machine-speed
+        # dependent.
+        if self._skip:
+            self.stop()
+            return
+        try:
+            session.client.event_engine.defer(self)
         except Exception as e:
             logger.error(f"Error running action: {e}")
             raise
```

And add the deferral queue it calls:

```diff
--- a/tuxemon/event/eventengine.py
+++ b/tuxemon/event/eventengine.py
@@ class EventEngine:
     def __init__(self, ...):
+        self._deferred: list[EventAction] = []
+
+    def defer(self, action: EventAction) -> None:
+        """Queue an action to be advanced by the client's fixed step."""
+        self._deferred.append(action)
+
+    def _update_deferred(self, dt: float) -> None:
+        still_running = []
+        for action in self._deferred:
+            if action.done or action.cancelled:
+                continue
+            action.update(self.session, dt)
+            if not action.done:
+                still_running.append(action)
+        self._deferred = still_running
 
     def update(self, dt: float) -> None:
+        self._update_deferred(dt)
         self.update_running_events(dt)
```

- [ ] **Step 4: Apply and run**

Run: `cd tuxemon && git apply ../patches/0005-action-step-clock.patch && cd .. && PYTHONHASHSEED=0 ./.venv/bin/python -m pytest tests/test_action_clock.py -v && make check-fast`
Expected: PASS, and the full fast suite still green.

- [ ] **Step 5: Mutation gate**

Revert patch 0005 only, rerun. Expected: FAIL. Restore.

- [ ] **Step 6: Update the combat test now that skip=True is unnecessary**

In `tests/test_combat_determinism.py`, change `execute("random_battle", (2, 8, 14), True)` to `execute("random_battle", (2, 8, 14))` and rerun the slow test. Expected: still PASS, no deadlock. If it hangs, patch 0005 is incomplete — say so rather than restoring `skip=True` silently.

- [ ] **Step 7: Commit**

```bash
git add patches/0005-action-step-clock.patch tests/test_action_clock.py tests/test_combat_determinism.py
git commit -m "feat: patch 0005 drives event actions from the step clock"
```

---

### Task 16: Documentation and the project CLAUDE.md

**Files:**
- Create: `CLAUDE.md`
- Create: `docs/STATUS.org`
- Modify: `Makefile`

**Interfaces:**
- Consumes: everything.
- Produces: the entry points a future session reads first.

- [ ] **Step 1: Write `docs/STATUS.org`**

Record: which patches are applied, the exemption register's current size (target zero), the measured determinism results with real digests, what is still unmeasured (capture, evolution, cross-platform), and the next subsystem (S2 CU harness).

- [ ] **Step 2: Write `CLAUDE.md`**

Must state: read `docs/STATUS.org` first; authority order is spec → measurements → plans; docs are org-mode; the trace format is version 1 and rejects any other; traces are step-indexed; never run without dummy SDL drivers; and the non-negotiable that every regression test must be demonstrated to fail against the bug it pins.

- [ ] **Step 3: Add a patch-application target**

```makefile
patch:
	cd tuxemon && for p in ../patches/*.patch; do git apply $$p; done

unpatch:
	cd tuxemon && git checkout -- . && git clean -fd
```

- [ ] **Step 4: Verify the whole gate from a clean tree**

Run: `make unpatch && make patch && make check`
Expected: all gates green, slow tests included.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/STATUS.org Makefile
git commit -m "docs: project entry points and the patch application target"
```
