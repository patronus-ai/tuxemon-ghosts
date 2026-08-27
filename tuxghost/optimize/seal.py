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
from tuxghost.trace import Trace


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


def seal(
    script: ActionScript,
    parent: Trace,
    *,
    checkpoint: int = 0,
    taints: Sequence[str] = (),
    model: str | None = None,
    max_cost: int | None = None,
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
    """
    from tuxemon.save_system.save_state import SaveData

    cost = script.cost()
    if max_cost is not None and cost > max_cost:
        raise ValueError(
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

    checkpoints: list[tuple[int, str]] = []
    states: dict[int, dict[str, Any]] = {}
    ran = 0

    def hook(i: int) -> None:
        nonlocal ran
        ran += 1
        if checkpoint and i and i % checkpoint == 0:
            checkpoints.append((i, digest_of(session)))
            states[i] = state_of(session)

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
    )
