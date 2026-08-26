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

import pytest

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



def test_deferred_action_error_does_not_wedge_the_queue() -> None:
    """Fix round 1: a previous version of `_update_deferred` let an
    exception from one queued action's update() propagate straight out
    of the for loop. Because `self._deferred` was only reassigned after
    the loop finished, an uncaught exception left it completely
    unchanged -- the failing action stayed at the front of the queue and
    every action queued behind it in that same frame never got updated
    at all. Next frame, the same failing action was retried first, raised
    again, and the cycle repeated: the failing action re-raised on every
    subsequent frame and starved every action behind it, forever.

    A failing action must instead: (a) raise exactly once -- on the frame
    it actually failed, not on every frame after -- (b) still get its
    cleanup() called, and (c) not stop any other queued action from
    advancing, including in the very same frame it failed on."""
    _client, session = build_client(seed=1234)

    bad = _wait_action(session, seconds=100 * FIXED_DT)
    good = _wait_action(session, seconds=5 * FIXED_DT)

    cleanup_calls: list[None] = []
    original_cleanup = bad.cleanup

    def spy_cleanup(sess: Any) -> None:
        cleanup_calls.append(None)
        original_cleanup(sess)

    bad.cleanup = spy_cleanup

    call_count = {"n": 0}
    original_bad_update = bad.update

    def failing_update(sess: Any, dt: float) -> None:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("boom")
        original_bad_update(sess, dt)

    bad.update = failing_update

    good_updates: list[float] = []
    original_good_update = good.update

    def spy_good_update(sess: Any, dt: float) -> None:
        good_updates.append(dt)
        original_good_update(sess, dt)

    good.update = spy_good_update

    # First (synchronous) update happens inside execute() itself --
    # call_count -> 1 for bad, does not raise. Both actions are still
    # running afterwards, so both get deferred; bad is queued first.
    bad.execute(session)
    good.execute(session)
    assert not bad.done and not good.done
    assert good_updates == [FIXED_DT]

    # This frame's deferred pass calls bad.update() a second time (raises)
    # -- (c): good must still get its update in this exact frame, despite
    # coming after bad in the queue.
    with pytest.raises(RuntimeError, match="boom"):
        run_steps(session.client, 1)

    assert call_count["n"] == 2
    assert len(good_updates) == 2, (
        "the action queued behind the failing one must still advance on "
        "the frame the failure happened"
    )
    assert cleanup_calls == [None], "cleanup() must still run on the failing action"

    # (a): no more raises on later frames -- the failing action must not
    # be retried, only removed.
    run_steps(session.client, 10)
    assert call_count["n"] == 2, "the failing action must not be retried"
    assert good.done, "the other queued action must finish normally"
