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
    assert verify(result.trace) == 0


def test_runner_schedule_equals_the_traces_schedule() -> None:
    result = run_agent(
        policy=_walk_policy(),
        save_data=_save(),
        seed=1234,
        clock_epoch=EPOCH,
        step_budget=400,
    )
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
    # Anti-vacuity control (task 5 review round 1): if `digest_of` were
    # ever blind to what actually changes along this route -- exactly
    # the historical failure mode in CLAUDE.md's vacuous-test list, where
    # nine runs across three seeds all hashed to the same value because
    # the probed route never touched what the digest covered -- `seen ==
    # blind` would hold trivially, with every entry identical, and the
    # sequence comparison below would pass for no reason at all. Requires
    # at least two distinct digest values across the run so the sequence
    # check is actually discriminating something.
    assert len({d for _, d in seen.digests}) > 1, (
        "every recorded digest was identical -- digest_of is blind to "
        "whatever this route actually changes, so the sequence "
        "comparison below cannot prove anything"
    )
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


def test_observation_tolerates_no_worldstate_and_no_active_map() -> None:
    """M1, whole-branch review: `client.get_map_name()` sat OUTSIDE the
    `except ValueError` guard `_observation` already has to tolerate a
    missing `WorldState` -- but `MapManager.get_map_name` raises its OWN
    `ValueError` ("Name of the map requested when no map is active") in
    exactly the same no-active-map situation, so an observation taken
    between map loads crashed instead of degrading the way `tile_pos`
    already does (`tile_pos=(-1, -1)`). Forces that state directly (pop
    `WorldState`, clear `current_map`) rather than chasing a real
    map-transition window that may or may not land on the right step."""
    from tuxemon.states.world_state import WorldState

    from tuxghost.agent.runner import _observation
    from tuxghost.boot import boot_from_save
    from tuxghost.determinism import seed_all

    seed_all(1234)
    client, _session = boot_from_save(_save(), seed=1234, clock_epoch=EPOCH)
    world = client.get_state_by_name(WorldState)
    client.pop_state(world)
    client.map_manager.current_map = None

    obs, png = _observation(client, 0, None)
    assert obs.map_name == ""
    assert obs.tile_pos == (-1, -1)
    assert png == b""


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
    assert result.steps <= budget
    assert result.trace.header.step_count == result.steps
    # M2+M3, whole-branch review: this exact run ends via the truncation
    # guard `break` -- one of the two paths that used to leave a run
    # directory with no way to say WHY the run ended at all.
    assert result.stop_reason == "step budget"


def test_stop_reason_names_a_policy_that_answered_stop() -> None:
    """M2+M3, whole-branch review, the other of the two silent `break`
    paths: `_walk_policy()` has exactly three decisions and no
    `repeat_last`, so its fourth `decide()` call returns `STOP` well
    before `step_budget=400` is reached -- this run's `stop_reason` must
    say so, not fall back to the budget-shaped default."""
    result = run_agent(
        policy=_walk_policy(), save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=400,
    )
    assert result.stop_reason == "policy returned STOP"


def test_cold_boot_run_records_a_verifiable_trace() -> None:
    """Cold boot is a supported start, not a decorative flag: its trace has
    to replay like any other. `initial_state` is snapshotted at boot, so
    this also covers the case where the run begins on the launcher map with
    the intro's own states still to come."""
    result = run_agent(
        policy=_walk_policy(), cold_boot=True, seed=1234,
        clock_epoch=EPOCH, step_budget=400,
    )
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


def test_a_zero_settle_action_does_not_silently_drop_the_next_press() -> None:
    """Task 5 review round 1: a `settle=0` action's release lands on
    exactly the step the NEXT action's press lands on (`release = step +
    action.hold` equals the following iteration's starting `step`).
    `validate_actions` permits `settle == 0`, so this is reachable from an
    entirely ordinary policy -- `_add_edge`'s predecessor
    (`schedule[step] = [...]`) silently overwrote the first edge with the
    second instead of accumulating both."""
    policy = ScriptedPolicy(
        [
            (Action(buttons.DOWN, hold=30, settle=0),),
            (Action(buttons.RIGHT, hold=30, settle=10),),
        ]
    )
    result = run_agent(
        policy=policy, save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=400,
    )
    assert result.schedule == _schedule_of(result.trace)
    assert verify(result.trace) == 0


def test_a_zero_settle_action_schedules_edges_in_canonical_order() -> None:
    """Companion to the append-only regression above, with the button
    order REVERSED. Task 5 review round 2: the test above collides a
    DOWN release with a RIGHT press, and `buttons.DOWN (2) <
    buttons.RIGHT (8)`, so the live INSERTION order at step 30 is
    ALREADY ascending by (button, value) -- `edges.sort()` is a no-op
    there, and deleting it leaves that test green. This test collides
    RIGHT first, so live insertion order is [(RIGHT, RELEASED),
    (DOWN, PRESSED)] = [(8, 0.0), (2, 1.0)], strictly descending by
    button number, while `_schedule_of` (via `Recorder.finish`'s
    write-time `sorted(self._inputs)`) always reconstructs ascending
    (button, value) order: [(2, 1.0), (8, 0.0)]. Without `_add_edge`'s
    `edges.sort()`, the live schedule and the trace's reconstructed
    schedule would disagree, AND -- the more serious half -- a replay
    would deliver DOWN's press before RIGHT's release, the opposite
    order the live run actually delivered them in."""
    policy = ScriptedPolicy(
        [
            (Action(buttons.RIGHT, hold=30, settle=0),),
            (Action(buttons.DOWN, hold=30, settle=10),),
        ]
    )
    result = run_agent(
        policy=policy, save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=400,
    )
    # The exact list, in order, at the point of collision -- makes the
    # ordering property visible at the point of failure, not just as a
    # dict inequality against `_schedule_of`.
    assert result.schedule[30] == [(buttons.DOWN, 1.0), (buttons.RIGHT, 0.0)]
    assert result.schedule == _schedule_of(result.trace)
    assert verify(result.trace) == 0


def test_taints_and_claimed_outcome_reach_the_trace_provenance() -> None:
    """Task 5 review round 1: nothing pinned `taints=`/`claimed_outcome=`
    actually reaching `trace.provenance` -- the whole suite stayed green
    even with `taints=[]` hardcoded or `claimed_outcome=claimed_outcome`
    deleted from the `Provenance(...)` call. `taints` is an integrity
    signal (how a replayed, not-model-authored trace gets marked so
    `tuxghost.cli`'s `compare` can flag it), so a silently dropped taint
    would make a replayed trace look live."""
    result = run_agent(
        policy=_walk_policy(), save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=400,
        taints=("replayed",), claimed_outcome="won",
    )
    assert result.trace.provenance.taints == ["replayed"]
    assert result.trace.provenance.claimed_outcome == "won"


def test_claimed_outcome_falls_back_to_the_policy_when_not_given() -> None:
    """The `getattr(policy, "claimed_outcome", None)` fallback path,
    exercised separately from the explicit-argument path above."""
    policy = _walk_policy()
    policy.claimed_outcome = "policy-said-so"
    result = run_agent(
        policy=policy, save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=400,
    )
    assert result.trace.provenance.claimed_outcome == "policy-said-so"


def test_claimed_outcome_explicit_empty_string_is_not_treated_as_unset() -> None:
    """Task 5 review round 1, minor: `claimed_outcome or getattr(...)`
    would treat an explicit `claimed_outcome=""` as falsy and silently
    fall back to the policy's own value. Must be an `is None` check."""
    policy = _walk_policy()
    policy.claimed_outcome = "policy-fallback-that-must-not-win"
    result = run_agent(
        policy=policy, save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=400, claimed_outcome="",
    )
    assert result.trace.provenance.claimed_outcome == ""


def test_digest_every_zero_records_nothing() -> None:
    """`digest_every=0` (the default) must cost nothing: `RunResult.digests`
    stays empty. `run_agent`'s own docstring names this as a requirement
    ("Default 0 records nothing and costs nothing"); nothing previously
    asserted the first half."""
    result = run_agent(
        policy=_walk_policy(), save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=400,
    )
    assert result.digests == []


def test_digest_every_never_samples_absolute_step_zero() -> None:
    """M5, whole-branch review: `run_agent`'s digest sampling must match
    `tuxghost.execute.execute`'s checkpoint convention exactly --
    `execute`'s `hook`'s `if checkpoint and i and i % checkpoint == 0`
    never fires at absolute step 0 (`i` is falsy there), but an earlier
    version of `run_agent`'s `sample` had no such guard, so
    `digest_every=1` recorded a step-0 entry `execute` with
    `checkpoint=1` never would. Nothing compares the two sequences today,
    but a future S3 comparison would be off by one entry at the very
    start of every run. `digest_every=1` makes every OTHER absolute step
    sample, so a step-0 entry surviving would be easy to miss among many
    correct ones -- checked explicitly here rather than only inferred
    from a count."""
    result = run_agent(
        policy=_walk_policy(), save_data=_save(), seed=1234,
        clock_epoch=EPOCH, step_budget=400, digest_every=1,
    )
    steps_sampled = [step for step, _digest in result.digests]
    assert 0 not in steps_sampled
    assert steps_sampled == list(range(1, len(steps_sampled) + 1))
