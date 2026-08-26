"""The agent loop, and the three gates that keep an agent-recorded trace
replayable.

Gate 1 (rendering is inert): recording draws, replay does not. If drawing
perturbed digested state, every agent trace would fail to replay.
Gate 2 (the trace verifies): end to end, through the S1 executor.
Gate 3 (schedule == trace): the dict the runner filled live must equal the
one `execute` rebuilds from the written trace -- the invariant that lets
S2 ship no new format or execution code at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tuxemon.platform.const import buttons
from tuxemon.save_system.save_state import SaveData

from tuxghost.agent.runner import run_agent
from tuxghost.agent.scripted import ScriptedPolicy
from tuxghost.agent.types import Action
from tuxghost.execute import _schedule_of, verify

FIXTURE = Path(__file__).parent / "fixtures" / "paper_town.save"
EPOCH = 1787659200


def _save() -> Any:
    return SaveData.model_validate(json.loads(FIXTURE.read_text()))


def _walk_policy() -> ScriptedPolicy:
    """Three decisions: walk down, walk right, press A. Then stop."""
    return ScriptedPolicy(
        [
            (Action(buttons.DOWN, hold=30, settle=10),),
            (Action(buttons.RIGHT, hold=30, settle=10),),
            (Action(buttons.A, hold=4, settle=20),),
        ]
    )


def test_recorded_trace_verifies() -> None:
    result = run_agent(
        policy=_walk_policy(),
        save_data=_save(),
        seed=1234,
        clock_epoch=EPOCH,
        step_budget=400,
    )
    assert result.trace is not None
    assert verify(result.trace) == 0


def test_runner_schedule_equals_the_traces_schedule() -> None:
    result = run_agent(
        policy=_walk_policy(),
        save_data=_save(),
        seed=1234,
        clock_epoch=EPOCH,
        step_budget=400,
    )
    assert result.trace is not None
    assert result.schedule == _schedule_of(result.trace)
    assert result.schedule, "a walking policy must have scheduled input"


def test_rendering_does_not_change_the_recorded_digest() -> None:
    """Gate 1, end to end: the same scripted run with and without an
    observation pass must agree at EVERY step, not merely at the end.

    Compares per-step digest SEQUENCES, not final digests. Measured in
    task 2: a final-digest comparison cannot detect a render pass that
    advances the game, because Tuxemon's grid movement makes a settled
    tile an attractor -- extra `update()` calls only make movement
    complete sooner, and both runs converge on the same tile by the end
    of the window. Two different mutations (an injected `client.update`
    and an injected `random.random()` draw) both left a final-digest
    comparison green. The sequence form sees the transient at the step it
    happens.
    """
    def kwargs() -> dict[str, Any]:
        # A fresh policy (and save_data) per call: `run_agent(**kwargs,
        # ...)` with a single shared `dict` literal would hand both runs
        # the SAME `ScriptedPolicy` instance, so the `seen` run's three
        # decisions would exhaust `self._index` before the `blind` run
        # ever calls `decide()` -- the second run would see `STOP`
        # immediately and record 0 steps, not a matching walk. Verified:
        # a shared-dict version of this test fails with `assert 104 == 0`.
        return {
            "policy": _walk_policy(), "save_data": _save(), "seed": 1234,
            "clock_epoch": EPOCH, "step_budget": 400, "digest_every": 1,
        }

    seen = run_agent(**kwargs(), observe=True)
    blind = run_agent(**kwargs(), observe=False)

    assert seen.digests, "digest_every=1 must record a digest per step"
    assert len(seen.digests) == len(blind.digests)
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(seen.digests, blind.digests))
         if a != b),
        None,
    )
    assert first_diff is None, (
        f"observed and blind runs diverged at step index {first_diff}: "
        f"{seen.digests[first_diff]} != {blind.digests[first_diff]}"
    )
    # The end state must still agree -- the sequence check subsumes this,
    # but the trace's own field is what `verify()` compares against.
    assert seen.trace is not None
    assert blind.trace is not None
    assert seen.trace.header.final_digest == blind.trace.header.final_digest


def test_observations_carry_a_real_frame_when_observing() -> None:
    """Companion control against a vacuous gate 1: if `observe=True`
    produced no pixels, the test above would pass for the wrong reason."""
    result = run_agent(
        policy=_walk_policy(), save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=400, observe=True,
    )
    first = result.decisions[0]
    assert first.frame_bytes > 1000, "expected a real PNG, not an empty frame"


def test_step_budget_is_never_exceeded() -> None:
    """A policy that always asks for more must be stopped by the budget,
    and the trace's step_count must match what actually ran.

    `step_budget=190` is deliberately NOT a multiple of one action's cost
    (hold=30 + settle=10 = 40): with a budget that divides evenly (200,
    as an earlier version of this test used), `while step < step_budget`
    already halts exactly at the budget on its own, so the guard this
    test exists to pin (`step + cost > step_budget`) never actually gets
    exercised -- a mutated guard (`step > step_budget`, checked before
    that iteration's cost is added) produces the identical `steps=200`
    result, and the test would pass for the wrong reason. Verified: with
    `step_budget=200` the mutated guard still leaves `result.steps==200`,
    a false pass. 190 forces the guard to actually refuse a would-be
    41-step overshoot (160 -> 200) before it happens.
    """
    forever = ScriptedPolicy(
        [(Action(buttons.DOWN, hold=30, settle=10),)] * 100, repeat_last=True
    )
    budget = 190
    result = run_agent(
        policy=forever, save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=budget,
    )
    assert result.trace is not None
    assert result.steps <= budget
    assert result.trace.header.step_count == result.steps


def test_cold_boot_run_records_a_verifiable_trace() -> None:
    """Cold boot is a supported start, not a decorative flag: its trace has
    to replay like any other. `initial_state` is snapshotted at boot, so
    this also covers the case where the run begins on the launcher map with
    the intro's own states still to come."""
    result = run_agent(
        policy=_walk_policy(), cold_boot=True, seed=1234,
        clock_epoch=EPOCH, step_budget=400,
    )
    assert result.trace is not None
    assert verify(result.trace) == 0


def test_run_agent_refuses_both_or_neither_start() -> None:
    """A run has to start from exactly one place."""
    with pytest.raises(ValueError, match="exactly one"):
        run_agent(
            policy=_walk_policy(), seed=1234, clock_epoch=EPOCH,
            step_budget=100,
        )
    with pytest.raises(ValueError, match="exactly one"):
        run_agent(
            policy=_walk_policy(), save_data=_save(), cold_boot=True,
            seed=1234, clock_epoch=EPOCH, step_budget=100,
        )


def test_a_policy_returning_junk_is_refused() -> None:
    class Junk:
        def decide(self, obs: Any) -> Any:
            return [Action(buttons.A, hold=0, settle=0)]

    with pytest.raises(ValueError, match="hold"):
        run_agent(
            policy=Junk(), save_data=_save(), seed=1234,
            clock_epoch=EPOCH, step_budget=100,
        )
