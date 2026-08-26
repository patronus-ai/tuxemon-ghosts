"""Offline executor: replays a trace and verifies it reached the same state.

Exit codes (both the module CLI's process exit code and `verify`'s return
value):
  * 0 -- every compared field matched.
  * 1 -- at least one differed.
  * 2 -- refused: a precondition failed (`tuxghost.trace.read`'s refusal
    matrix -- unknown format, missing seed/clock_epoch, a tampered
    `initial_state`), or the trace is unrunnable here (its `initial_state`
    does not validate as a `SaveData`, or lacks what `boot_from_save`
    needs to restore a session -- see `execute`).

`verify` executes a trace EXACTLY ONCE and compares the digest it reaches
against `trace.header.final_digest` -- the state the recorded run actually
reached (see that field's own docstring in `tuxghost.trace`). It
deliberately does NOT execute a trace twice and compare the two runs
against each other: that would only prove replay is internally
reproducible on this build, not that it still reaches what it originally
reached, which is what verification means.

### What a `verify() == 0` result certifies, and what it does not

`digest_of`/`state_of` (`tuxghost.digest`) cover the player's own
`npc_state`, `world_state`, and (as of task 12)
`persistent_npc_state` -- every other NPC the save system considers
persistent. A `verify() == 0` result certifies the replay reached
byte-identical values across ALL of that: player position/party/items/
battles/game_variables, world flags, and every persistent NPC's own
saved state.

It does NOT certify combat internals beyond their effect on the player's
own party: `state_stack` records only state NAMES
(`session.client.state_manager.active_states`), so a `CombatState` with a
different turn count or a different opponent's HP is invisible to
`verify` except indirectly, through the player's own party's HP (which IS
covered). This is a known, documented gap -- see `tuxghost.digest
.state_of`'s own docstring for why it was not widened in this task.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tuxghost.boot import boot_from_save
from tuxghost.compare import describe_divergence, first_divergent_step
from tuxghost.determinism import pin_clock, seed_all
from tuxghost.digest import digest_of, state_of
from tuxghost.loop import InputSchedule, install_schedule, run_steps
from tuxghost.trace import Refused, Trace, read


@dataclass
class ExecutionResult:
    final_digest: str
    final_state: dict[str, Any]
    #: (step, digest) pairs taken every `checkpoint` steps, per `execute`'s
    #: own `checkpoint` argument. Empty when `checkpoint=0` (the default):
    #: nothing to bisect against costs nothing to compute.
    checkpoints: list[tuple[int, str]] = field(default_factory=list)
    #: Full `state_of()` snapshots at the same steps as `checkpoints`,
    #: keyed by step. Only populated when `execute(..., capture_states=
    #: True)` -- a full state tree at every checkpoint is far more
    #: expensive to keep than a digest, and only `bisect_traces` (below)
    #: actually needs it.
    checkpoint_states: dict[int, dict[str, Any]] = field(default_factory=dict)


def _schedule_of(trace: Trace) -> InputSchedule:
    schedule: InputSchedule = {}
    for step, button, value in trace.inputs:
        schedule.setdefault(step, []).append((button, value))
    return schedule


def execute(
    trace: Trace, checkpoint: int = 0, capture_states: bool = False
) -> ExecutionResult:
    """Replay `trace`. Takes no map, seed or save of its own: all of it is
    the header and `initial_state` -- the point of the trace format is
    that this is the ONLY input execution needs.

    Passes `trace.header.clock_epoch` through to `boot_from_save` -- not
    optional for the executor path. Without it, `boot_from_save` leaves
    the session's own elapsed-time bookkeeping
    (`SessionSave.duration`/`total_playtime`/`start_time`) on whatever
    real wall-clock reading was live when the process-wide session
    singleton was first constructed, rather than the pinned epoch the
    trace was recorded against; see `tuxghost.boot.boot_from_save`'s own
    docstring, which names this exact parameter as "the executor's path".

    Raises `Refused` if `trace.initial_state` does not validate as a
    `SaveData`, or lacks what `boot_from_save` needs to restore a session
    (currently: `npc_state.current_map`) -- both are preconditions that
    can only be checked once execution actually starts, unlike
    `tuxghost.trace.read`'s refusal matrix, which is checked on load.
    """
    from tuxemon.save_system.save_state import SaveData

    seed_all(trace.header.seed)
    pin_clock(trace.header.clock_epoch)

    try:
        save_data = SaveData.model_validate(trace.initial_state)
    except ValidationError as exc:
        raise Refused(
            f"trace.initial_state does not validate as a SaveData: {exc}"
        ) from exc

    try:
        _client, session = boot_from_save(
            save_data, seed=trace.header.seed, clock_epoch=trace.header.clock_epoch
        )
    except AssertionError as exc:
        raise Refused(f"trace is unrunnable here: {exc}") from exc

    client = session.client
    install_schedule(client, _schedule_of(trace))

    result = ExecutionResult(final_digest="", final_state={})

    def hook(i: int) -> None:
        if checkpoint and i and i % checkpoint == 0:
            result.checkpoints.append((i, digest_of(session)))
            if capture_states:
                result.checkpoint_states[i] = state_of(session)

    run_steps(client, trace.header.step_count, hook=hook)
    result.final_state = state_of(session)
    result.final_digest = digest_of(session)
    return result


def verify(trace: Trace) -> int:
    """Execute `trace` ONCE and compare the digest it reaches against
    `trace.header.final_digest`. Returns 0 (matched), 1 (diverged), or 2
    (refused) -- see the module docstring."""
    try:
        result = execute(trace)
    except Refused:
        return 2
    return 0 if result.final_digest == trace.header.final_digest else 1


def bisect_traces(a: Trace, b: Trace, checkpoint: int) -> str | None:
    """Run two traces with checkpointing and full-state capture, and
    describe the earliest step and field at which they diverge -- or
    `None` if every checkpoint they share matches.

    Meant for diagnosing a `verify` failure or a cross-run/cross-machine
    reproducibility check: unlike `verify` (which only ever gets
    `trace.header.final_digest`, a single hash for the whole run, to
    compare against), this compares two full EXECUTIONS against each
    other checkpoint by checkpoint, so it can actually narrow down to a
    step -- that is what `first_divergent_step` needs, and what a lone
    final-digest mismatch cannot provide on its own.

    `checkpoint` must be > 0 and `a`/`b` should share a step_count (or at
    least the same checkpoint cadence up to the point they diverge) --
    `first_divergent_step` asserts the two checkpoint lists it is
    comparing report the same step numbers pairwise."""
    assert checkpoint > 0, "bisect_traces needs a checkpoint interval to bisect against"
    result_a = execute(a, checkpoint=checkpoint, capture_states=True)
    result_b = execute(b, checkpoint=checkpoint, capture_states=True)
    step = first_divergent_step(result_a.checkpoints, result_b.checkpoints)
    if step is None:
        return None
    return describe_divergence(
        step, result_a.checkpoint_states[step], result_b.checkpoint_states[step]
    )


def _bootstrap_vendored_tuxemon() -> None:
    """Make the vendored `tuxemon/` package importable and its assets
    findable when this module is run directly (`python -m tuxghost.execute
    ...`) rather than under pytest -- pytest gets this for free from
    `tests/conftest.py`, which every test in this project relies on but
    which never runs for a standalone process. Mirrors that file's fix
    exactly: without both steps, `import tuxemon` resolves to the clone
    directory as an empty namespace package from the repo root, and asset
    loading fails on a relative "mods/tuxemon/mod.yaml" path even once
    imports work."""
    import os
    import sys
    from pathlib import Path

    tuxemon_dir = Path(__file__).resolve().parent.parent / "tuxemon"
    if str(tuxemon_dir) not in sys.path:
        sys.path.insert(0, str(tuxemon_dir))
    os.chdir(tuxemon_dir)


def _main(argv: list[str] | None = None) -> int:
    """CLI entry point: read a trace from disk and verify it, translating
    `tuxghost.trace.read`'s `Refused` (an unknown format, a missing
    seed/clock_epoch, or a tampered `initial_state`) into exit code 2, the
    same code `verify` itself returns for a trace that refuses once
    execution starts. Run as `python -m tuxghost.execute <path>`."""
    _bootstrap_vendored_tuxemon()
    parser = argparse.ArgumentParser(
        prog="tuxghost.execute",
        description="Replay a trace and verify it reaches the same state.",
    )
    parser.add_argument("trace_path", type=Path)
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="downgrade mod_version/step_rate mismatches instead of refusing",
    )
    args = parser.parse_args(argv)

    try:
        trace = read(args.trace_path, allow_mismatch=args.allow_mismatch)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    code = verify(trace)
    if code == 0:
        print(f"verify: OK ({trace.header.final_digest})")
    elif code == 1:
        print("verify: DIVERGED", file=sys.stderr)
    else:
        print("verify: REFUSED", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(_main())
