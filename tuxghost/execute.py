"""Offline executor: replays a trace and verifies it reached the same state.

`verify`'s return value:
  * 0 -- every compared field matched.
  * 1 -- at least one differed.
  * 2 -- refused: a precondition failed (`tuxghost.trace.read`'s refusal
    matrix -- unknown format, missing seed/clock_epoch, a tampered
    `initial_state`), or the trace is unrunnable here (its `initial_state`
    does not validate as a `SaveData`, lacks what `boot_from_save` needs
    to restore a session, or names a `current_map` that resolves to no
    real map asset -- see `execute`).

No CLI lives in this module. Fix round 1 (task 12 review) found a real
exit-code collision in an earlier `python -m tuxghost.execute` entry point
here (`FileNotFoundError` on a missing path exited 1, colliding "diverged"
with "unrunnable") and a broken-by-construction relative-path bug
(`_bootstrap_vendored_tuxemon`'s `os.chdir` ran before argument parsing,
so every relative trace path resolved against the wrong directory). The
next task's deliverable is `tuxghost/cli.py` with `execute`/`verify`/
`compare`/`record`/`info` subcommands, the same `--allow-mismatch` flag,
and the same `Refused` -> 2 mapping, built with path resolution BEFORE
`_bootstrap_vendored_tuxemon`'s chdir and I/O errors mapped to exit 2.
Fixing the old defects in place here would have meant fixing them twice.
`_bootstrap_vendored_tuxemon` itself is kept (below): any non-pytest entry
point needs it.

`verify` executes a trace EXACTLY ONCE and compares the digest it reaches
against `trace.header.final_digest` -- the state the recorded run actually
reached (see that field's own docstring in `tuxghost.trace`). It
deliberately does NOT execute a trace twice and compare the two runs
against each other: that would only prove replay is internally
reproducible on this build, not that it still reaches what it originally
reached, which is what verification means.

### What a `verify() == 0` result certifies, and what it does not

`digest_of`/`state_of` (`tuxghost.digest`) cover the player's own
`npc_state`, `world_state`, and (as of task 12) `persistent_npc_state` --
every other NPC the save system considers persistent. A `verify() == 0`
result certifies the replay reached byte-identical values across ALL of
that: player position/party/items/battles/game_variables, world flags,
and (when any exist) every persistent NPC's own saved state. In practice
that last part currently certifies nothing extra: no NPC in this
project's shipped mod data sets `persistence: true` (see
`tuxghost.digest.state_of`'s own docstring), so `persistent_npc_state` is
`[]` on every trace this project can currently record -- the plumbing is
real and proven to discriminate once a persistent NPC exists, but adds no
practical coverage today.

It does NOT certify combat internals beyond their effect on the player's
own party: `state_stack` records only state NAMES
(`session.client.state_manager.active_states`), so a `CombatState` with a
different turn count or a different opponent's HP is invisible to
`verify` except indirectly, through the player's own party's HP (which IS
covered). This is a known, documented gap -- see `tuxghost.digest
.state_of`'s own docstring for why it was not widened in this task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from tuxghost.boot import boot_from_save, resolve_map_asset
from tuxghost.compare import describe_divergence, first_divergent_step
from tuxghost.determinism import pin_clock, seed_all
from tuxghost.digest import digest_of, state_of
from tuxghost.loop import InputSchedule, install_schedule, run_steps
from tuxghost.trace import Refused, Trace


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
    `SaveData`, lacks what `boot_from_save` needs to restore a session
    (currently: `npc_state.current_map`), or names a `current_map` that
    `tuxghost.boot.resolve_map_asset` cannot resolve to a real map asset
    -- all three are preconditions that can only be checked once
    execution actually starts, unlike `tuxghost.trace.read`'s refusal
    matrix, which is checked on load.

    The second and third checks are done directly here (the second
    mirrors `boot_from_save`'s own `assert npc_state is not None and
    npc_state.current_map is not None`; the third mirrors the `ValueError`
    `boot_from_save` would otherwise raise from an unresolvable
    `current_map`), rather than by calling `boot_from_save` inside a
    `try: ... except (AssertionError, ValueError)`. Fix round 1 flagged
    that catching bare `AssertionError` around `boot_from_save` would ALSO
    catch a genuine internal engine assertion failure -- unrelated to
    whether this particular trace is well-formed -- and silently report it
    as exit code 2 ("this trace is unrunnable"), conflating an engine bug
    with a bad trace; the same reasoning applies to `ValueError`, which
    plenty of genuine engine code can also raise. Checking every
    precondition ourselves, before ever calling `boot_from_save`, means
    the only exceptions that function could still raise are real internal
    invariants, which are left to propagate uncaught.
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

    if save_data.npc_state is None or save_data.npc_state.current_map is None:
        raise Refused(
            "trace is unrunnable here: initial_state.npc_state.current_map "
            "is required to restore a session (see tuxghost.boot"
            ".boot_from_save)"
        )

    if resolve_map_asset(save_data.npc_state.current_map) is None:
        raise Refused(
            "trace is unrunnable here: initial_state.npc_state.current_map="
            f"{save_data.npc_state.current_map!r} resolves to no map asset "
            "(tried it as given and with a .tmx extension). This is a "
            "refusal, not a divergence."
        )

    _client, session = boot_from_save(
        save_data, seed=trace.header.seed, clock_epoch=trace.header.clock_epoch
    )

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
    `first_divergent_step` raises `ValueError` if the two checkpoint lists
    it is comparing report different step numbers pairwise.

    Raises `ValueError` if `checkpoint <= 0` -- a real, caller-triggerable
    contract violation (not an internal invariant), so it is checked with
    a real exception rather than a bare `assert`, which `python -O` would
    silently discard."""
    if checkpoint <= 0:
        raise ValueError("bisect_traces needs a checkpoint interval to bisect against")
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


# No `_main`/CLI here -- see the module docstring. `tuxghost/cli.py` (a
# later task) is where a real entry point belongs; `_bootstrap_vendored_
# tuxemon` above is kept for it to reuse.
