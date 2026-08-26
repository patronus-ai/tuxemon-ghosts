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
*stability* assertion on the same route can. This test drives the exact
same seed and schedule twice and requires the resulting digests to match.
"""

from __future__ import annotations

import os

import pytest
from tuxemon.platform.const import buttons

from tuxghost.boot import build_client
from tuxghost.digest import digest_of
from tuxghost.loop import install_schedule, run_steps

pytestmark = pytest.mark.slow

BATTLE_STEP_BUDGET = 8_000


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
    reason="drives a full battle; seconds to minutes per run",
)
def test_battle_is_deterministic_through_its_exit() -> None:
    a, exited_a = _battle_run(1234, BATTLE_STEP_BUDGET)
    b, exited_b = _battle_run(1234, BATTLE_STEP_BUDGET)
    assert exited_a == exited_b
    assert a == b, "combat diverged; verify needs a divergence-tolerant mode"
