"""Pins patch 0005: `EventAction.run()` must advance on the client's fixed
step, not on `time.perf_counter()`.

The upstream spin loop (`tuxemon/event/eventaction.py:178`, pre-patch)
computed `dt` from the wall clock and never returned control to the
client's frame loop, so a non-instant action either hung headless forever
(the *observed* failure -- see `test_combat_determinism.py`, which needed
`skip=True` to avoid it) or, for an action that self-terminates without
waiting on external state, produced non-uniform, machine-speed `dt`
values instead of the game's fixed 1/60 step (the *reasoned* hazard).

`WaitAction` (`tuxemon/event/actions/wait.py`) is exactly such an action:
its own docstring already claims "measured using accumulated delta time
(dt) from the game loop, not wall clock time" -- a claim the pre-patch
spin loop could not honor, since it fed `update()` a wall-clock `dt`
regardless. This file drives a real `wait` action (not the brief's
`dialog_chain`, which does not exist in this tree) via
`ActionManager.get_action` + `EventAction.execute()` directly -- the same
path `EventEngine.execute_action()` uses internally -- so it gets a handle
on the action instance to instrument, without going through the
plugin-name call and without subclassing `EventAction` (which mypy
--strict rejects: `tuxemon.*` is `follow_imports = "skip"`, so
`EventAction` resolves to `Any`, and strict mode refuses to subclass
`Any`). Instrumentation instance-monkeypatches `update`/`cleanup` on that
one action object, mirroring the brief's own `EventAction.update = spy`
pattern but scoped to a single instance so it can't pick up unrelated
actions that happen to run during `build_client`/`run_steps`.

Because `wait` self-terminates on accumulated `dt`, not on external game
state, the *old* spin loop does not hang on it -- it just busy-spins to
completion using near-zero, back-to-back wall-clock `dt` reads. That is
what makes this a fast, deterministic pinning test rather than a hang:
unpatched, the `dt` assertion fails in well under a second; patched,
`run()` defers the action to `EventEngine`'s per-frame queue after its
first `update()` call, and every subsequent `dt` comes from `run_steps`'
`FIXED_DT`.
"""

from __future__ import annotations

from typing import Any

from tuxghost.boot import build_client
from tuxghost.loop import FIXED_DT, run_steps


def _wait_action(session: Any, seconds: float) -> Any:
    action = session.client.event_engine.action_manager.get_action(
        "wait", (seconds,)
    )
    assert action is not None, "'wait' action failed to load"
    return action


def test_action_dt_comes_from_the_step_clock_not_the_wall_clock() -> None:
    """A non-instant action must advance by FIXED_DT per step, so two runs
    on machines of different speed see the same dt sequence -- not by
    `time.perf_counter()` deltas, which vary run to run."""
    _client, session = build_client(seed=1234)
    action = _wait_action(session, seconds=6 * FIXED_DT)

    dt_log: list[float] = []
    original_update = action.update

    def spy_update(sess: Any, dt: float) -> None:
        dt_log.append(dt)
        original_update(sess, dt)

    action.update = spy_update

    action.execute(session)
    run_steps(session.client, 15)

    assert action.done, "wait action never completed -- looks like a deadlock"
    assert dt_log, "no action updates observed"
    assert all(d == FIXED_DT for d in dt_log), (
        f"non-step dt values: {sorted(set(dt_log))}"
    )


def test_action_execute_does_not_block_the_caller() -> None:
    """`execute()` on a non-instant action must return immediately -- the
    old spin loop blocked the calling frame until the action finished."""
    _client, session = build_client(seed=1234)
    action = _wait_action(session, seconds=1000.0)

    dt_log: list[float] = []
    original_update = action.update

    def spy_update(sess: Any, dt: float) -> None:
        dt_log.append(dt)
        original_update(sess, dt)

    action.update = spy_update

    action.execute(session)

    assert not action.done
    assert dt_log == [FIXED_DT]


def test_deferred_action_cleanup_runs_once_and_only_after_completion() -> None:
    """`ActionContextManager.__exit__` used to call `cleanup()`
    unconditionally right after `run()` returned, which was safe only
    because the old loop never returned until the action was actually
    done. Under the fixed-step queue, `run()` returns as soon as an
    unfinished action is deferred -- if cleanup() still fired there, it
    would run teardown (e.g. `RandomBattleAction.cleanup()` removing the
    NPC it just created for the battle) before the action's own work is
    done. Cleanup must fire exactly once, from the deferred queue, only
    once the action reaches a terminal state."""
    _client, session = build_client(seed=1234)
    action = _wait_action(session, seconds=3 * FIXED_DT)

    cleanup_calls: list[bool] = []
    original_cleanup = action.cleanup

    def spy_cleanup(sess: Any) -> None:
        cleanup_calls.append(action.done)
        original_cleanup(sess)

    action.cleanup = spy_cleanup

    action.execute(session)
    assert cleanup_calls == [], (
        "cleanup() ran before the deferred action finished"
    )

    run_steps(session.client, 1)
    assert cleanup_calls == [], (
        "action should still be running after one step (waits for 3)"
    )

    run_steps(session.client, 10)
    assert cleanup_calls == [True], (
        f"cleanup() should run exactly once, after completion: {cleanup_calls}"
    )
