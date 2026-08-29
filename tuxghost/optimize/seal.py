"""Compile a script, run it, and seal the result as a trace.

Composes four primitives that already exist and adds no fifth:
`boot_from_save`, `install_schedule`, `run_steps` (with `execute`'s own
checkpoint hook) and `Recorder`. There is deliberately no new stepping
loop and no new format writer here.

Why not `run_agent`: its scripted path starts the first action at step 0,
so it cannot express a script with a nonzero `lead_in` -- true of 1 of
the 2 traces this project has committed (`walk_1234` waits 30 steps).

`seed_all` and `pin_clock` are called here, mirroring
`tuxghost.execute.execute` and NOT `run_agent` (whose caller does it).
Without them a candidate would differ from its parent for reasons that
have nothing to do with the edit under test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tuxghost.boot import boot_from_save
from tuxghost.determinism import pin_clock, seed_all
from tuxghost.digest import digest_of, state_of
from tuxghost.loop import install_schedule, run_steps
from tuxghost.optimize.schedule import ActionScript, lower
from tuxghost.record import Recorder
from tuxghost.rules import GameRules
from tuxghost.trace import Trace


class OverBudget(ValueError):
    """A script's cost exceeds the caller's `max_cost`.

    A `ValueError` subclass, not a bare `ValueError`, deliberately: `seal`
    also raises plain `ValueError` from `SaveData.model_validate` (a
    malformed `parent.initial_state` -- `pydantic.ValidationError` IS a
    `ValueError` subclass, measured) and, inside `boot_from_save`, from an
    unresolvable `current_map`. Task 7 treats an over-budget candidate as
    a REJECTED ROUND, distinct from an unrunnable parent -- a caller that
    only ever sees `ValueError` cannot tell "this candidate was too
    expensive" from "this parent trace itself does not boot", and since a
    malformed parent fails identically on every round, conflating the two
    would report a whole run as "the editor kept proposing rubbish"
    instead of the real defect. Existing broad `except ValueError`
    handlers keep working unchanged; this type exists so a caller that
    wants precision can have it (see `tuxghost/execute.py`'s own
    reasoning for not conflating distinct refusal causes).
    """


@dataclass
class CandidateResult:
    """One candidate, run. This is everything an `Objective` and an
    `Editor` are allowed to see -- no frames, by design."""

    trace: Trace
    steps: int
    final_state: dict[str, Any]
    #: `(step, digest)` every `checkpoint` steps, on `execute`'s
    #: convention (step 0 excluded). Empty when `checkpoint=0`.
    checkpoints: list[tuple[int, str]] = field(default_factory=list)
    checkpoint_states: dict[int, dict[str, Any]] = field(default_factory=dict)
    #: Populated only when `seal` is given `rules`. `max_progress` is the
    #: high-water mark (progress can plateau but must not decrease);
    #: `goal_step` latches the FIRST step `at_goal` was true, which is why
    #: the observer must run per-step rather than scoring the end state.
    max_progress: int = 0
    goal_step: int | None = None
    died: bool = False


def seal(
    script: ActionScript,
    parent: Trace,
    *,
    checkpoint: int = 0,
    taints: Sequence[str] = (),
    model: str | None = None,
    max_cost: int | None = None,
    rules: GameRules | None = None,
) -> CandidateResult:
    """Run `script` from `parent`'s start and seal what it reached.

    `parent` supplies `initial_state`, `seed` and `clock_epoch` verbatim:
    a candidate that changed any of them would not be a candidate for the
    same problem.

    `max_cost` refuses an over-budget script BEFORE booting anything.
    `HOLD_CAP` and `SETTLE_CAP` are 600 each, so an editor that inserts
    10,000 actions describes a 12,000,000-step run -- hours of engine
    time inside one call, unobservable and uninterruptible, which is the
    same reasoning `tuxghost/agent/types.py` gives for capping a single
    action.

    `rules`, when given, is observed EVERY step (see `tuxghost.rules`)
    to populate `CandidateResult.max_progress`, `.goal_step` and
    `.died`. Death is reported, not acted on: `seal` keeps running the
    full script even after `rules.observe` reports `dead`, since it
    already asserts it ran exactly `cost` steps and stopping early would
    trip that assertion, scoring a clipped candidate as if the clip were
    the edit. It is the caller's `Objective` that decides what a death
    means for the candidate's score.
    """
    from tuxemon.save_system.save_state import SaveData

    cost = script.cost()
    if max_cost is not None and cost > max_cost:
        raise OverBudget(
            f"script costs {cost} steps, over the max_cost of {max_cost}; "
            "refusing before booting rather than running it"
        )

    seed = parent.header.seed
    epoch = parent.header.clock_epoch
    seed_all(seed)
    pin_clock(epoch)

    save_data = SaveData.model_validate(parent.initial_state)
    client, session = boot_from_save(save_data, seed=seed, clock_epoch=epoch)

    schedule = lower(script)
    install_schedule(client, schedule)

    recorder = Recorder(
        session,
        seed=seed,
        clock_epoch=epoch,
        recorder="offline-agent",
        model=model,
        taints=taints,
    )
    for step in sorted(schedule):
        for button, value in schedule[step]:
            recorder.observe(step, button, value)

    # I1 (whole-branch review): `rules` may be the SAME instance across
    # every candidate in one `optimize()` run (round 0 at runner.py:220
    # and every candidate via `_prepare` at runner.py:158), so a `rules`
    # implementation that carries per-run state (e.g. `TuxemonRules
    # ._furthest_map`) must have that state cleared before THIS run
    # steps, or candidate N inherits candidate N-1's high-water mark.
    # `reset` is deliberately not part of the `GameRules` Protocol (see
    # `TuxemonRules.reset`'s docstring), so this is `hasattr`-gated
    # rather than an unconditional call every implementer must support.
    if rules is not None:
        reset = getattr(rules, "reset", None)
        if reset is not None:
            reset()

    checkpoints: list[tuple[int, str]] = []
    states: dict[int, dict[str, Any]] = {}
    ran = 0
    max_progress = 0
    goal_step: int | None = None
    died = False

    def hook(i: int) -> None:
        nonlocal ran, max_progress, goal_step, died
        ran += 1
        if checkpoint and i and i % checkpoint == 0:
            checkpoints.append((i, digest_of(session)))
            states[i] = state_of(session)
        if rules is not None:
            obs = rules.observe(session, i)
            max_progress = max(max_progress, obs["progress"])
            if goal_step is None and obs["at_goal"]:
                goal_step = i
            if obs["dead"]:
                died = True

    run_steps(client, cost, hook=hook)

    if ran != cost:
        raise AssertionError(
            f"seal ran {ran} steps for a {cost}-step script; a clipped "
            "candidate would be scored as if the clip were the edit"
        )

    trace = recorder.finish(step_count=cost)
    return CandidateResult(
        trace=trace,
        steps=cost,
        final_state=state_of(session),
        checkpoints=checkpoints,
        checkpoint_states=states,
        max_progress=max_progress,
        goal_step=goal_step,
        died=died,
    )
