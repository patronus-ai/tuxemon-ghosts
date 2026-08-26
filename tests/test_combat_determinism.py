"""Closes the spec's open question: does combat stay deterministic through a
*completed* battle, exited back to the world? Every earlier spike/task run
left `CombatState` on the stack, so capture, evolution and the battle exit
itself were never measured.

An exploratory run (see `docs/2026-08-25-battle-exit-measurement.org`) found
`CombatState` leaves the state stack at step 4945 of the post-`random_battle`
mash, reproducibly. 8000 steps leaves a roughly 60% margin over that without
paying for the 200_000-step budget this file started from, which measured at
several minutes per run.

Per the process rule from Task 5: a cross-run *difference* assertion can
never detect "the digest is effectively random" -- only a same-seed
*stability* assertion on the same route can. `test_battle_exit_is_reproducible`
drives the exact same seed and schedule twice and requires `exited` to match.

`test_battle_digest_is_deterministic_through_its_exit` goes one step further
and compares the full state digest. That comparison used to be
`xfail(strict=True)`: the measurement in
`docs/2026-08-25-battle-exit-measurement.org` found it diverged in exactly
four fields, all attributable to unseeded `uuid4()` (`battle.py:29`,
`status/status.py:46`, `event/actions/get_pending_moves.py:62`) or a
wall-clock `time.time()` read (`battle.py:32`) -- not gameplay RNG. Patch
0003 (seeded uuid factory, Task 8) closed the first three; patch 0004
(injectable clock, Task 9) closed the fourth -- `Battle.timestamp`, by
routing it (and the load-bearing second stamp in
`entity/battle.py`'s `BattlesHandler.record_battle`, which otherwise
overwrites `Battle.__init__`'s own timestamp via `Battle.from_save_data`)
through `tuxemon.core.clock.now()`. `_battle_run` pins the clock via
`tuxghost.determinism.pin_clock` before each run so the two runs this test
compares actually share a "now", the same way they already share a seed;
without that call the clock defaults to real wall time and the test would
still fail for a reason that has nothing to do with combat -- the two
`_battle_run` calls simply do not execute at the same wall-clock instant.
With both patches applied and the clock pinned, the test now passes as an
ordinary (non-xfail) test. It is kept separate from the exit-reproducibility
test so a real regression in exit detection is never masked by (what was)
the known, already-diagnosed digest divergence.
"""

from __future__ import annotations

import os

import pytest
from tuxemon.platform.const import buttons

from tuxghost.boot import build_client
from tuxghost.determinism import pin_clock
from tuxghost.digest import digest_of
from tuxghost.loop import install_schedule, run_steps

pytestmark = pytest.mark.slow

BATTLE_STEP_BUDGET = 8_000

# Arbitrary fixed epoch (2026-08-25T14:40:00 local) so both `_battle_run`
# calls in a comparison see the same "now" -- see the module docstring.
BATTLE_EPOCH = 1787694000

_run_slow = pytest.mark.skipif(
    not os.environ.get("TUXGHOST_RUN_SLOW"),
    reason="drives a full battle; seconds to minutes per run",
)


def _mash(button: int, count: int, period: int) -> dict[int, list[tuple[int, float]]]:
    schedule: dict[int, list[tuple[int, float]]] = {}
    for i in range(count):
        schedule.setdefault(i * period, []).append((button, 1.0))
        schedule.setdefault(i * period + 2, []).append((button, 0.0))
    return schedule


def _battle_run(seed: int, steps: int) -> tuple[str, bool]:
    pin_clock(BATTLE_EPOCH)
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


@_run_slow
def test_battle_exit_is_reproducible() -> None:
    """Same-seed stability on whether CombatState leaves the stack. This is
    the substantive result of Task 6: prior work never once saw a battle
    exit across 120000 steps; this route exits reproducibly at step 4945."""
    _digest_a, exited_a = _battle_run(1234, BATTLE_STEP_BUDGET)
    _digest_b, exited_b = _battle_run(1234, BATTLE_STEP_BUDGET)
    assert exited_a == exited_b
    assert exited_a is True


@_run_slow
def test_battle_digest_is_deterministic_through_its_exit() -> None:
    """Was `xfail(strict=True)` pending patches 0003 and 0004 -- see the
    module docstring. The measurement in
    `docs/2026-08-25-battle-exit-measurement.org` found exactly four
    diverging fields: `npc_state.monsters[].status[].instance_id`
    (status/status.py:46), `npc_state.battles[].instance_id`
    (battle.py:29), `npc_state.battles[].timestamp` (battle.py:32), and
    `npc_state.game_variables.chosen_tech`
    (event/actions/get_pending_moves.py:62). Patch 0003 (Task 8) closed
    the first, second and fourth via the seeded id factory; patch 0004
    (Task 9) closed the third -- `Battle.timestamp` -- via the injectable
    clock. All four now closed, this runs as a normal passing test."""
    a, _exited_a = _battle_run(1234, BATTLE_STEP_BUDGET)
    b, _exited_b = _battle_run(1234, BATTLE_STEP_BUDGET)
    assert a == b, "combat diverged; verify needs a divergence-tolerant mode"
