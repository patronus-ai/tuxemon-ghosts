"""Drives a policy against a live session and records the result.

ONE mutable schedule. `tuxghost.loop.install_schedule`'s closure reads
`schedule.get(state["step"])` off the DICT OBJECT on every call, so this
loop installs one empty dict at boot and inserts entries at future step
indices as the agent decides. The step counter inside that closure stays
monotonic, and an agent run and its replay differ only in WHO FILLED THE
DICT: here it is filled live, in `tuxghost.execute` it is filled from the
trace up front, and both then call the same `run_steps`.

Two rejected alternatives, both of which would create a second path by
which input reaches the engine -- the thing `tuxghost/loop.py`'s docstring
exists to prevent:

  * Re-installing a schedule per chunk. `install_schedule` builds a fresh
    closure with `state = {"step": 0}`, so a re-install silently rebases
    every later input to step 0.
  * `run_steps`' per-step `hook`. It exists for checkpoint digests
    (`tuxghost.execute`), and feeding input through it would put two
    mechanisms in the same loop.

Every edge written into the schedule is also handed to `Recorder.observe`,
so the trace IS the schedule -- pinned by
`tests/test_agent_runner.py::test_runner_schedule_equals_the_traces_schedule`.
Edges are APPENDED (via `add_edge`), not assigned: two edges can legally
land on the same step index (a `settle=0` action's release lands exactly
on the next action's press), and `schedule[step] = [...]` would silently
drop one -- see `add_edge`'s own docstring (`tuxghost/loop.py`), and task
5 review round 1, which found this reachable from a perfectly ordinary
policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tuxghost.agent.types import Action, Observation, Policy, validate_actions
from tuxghost.boot import boot_from_save, build_client
from tuxghost.digest import digest_of
from tuxghost.loop import (
    PRESSED,
    RELEASED,
    InputSchedule,
    add_edge,
    install_schedule,
    run_steps,
)
from tuxghost.observe import scaled_context
from tuxghost.record import Recorder
from tuxghost.trace import Trace

if TYPE_CHECKING:
    # Only for the annotations below -- `FrameRenderer` is a real class of
    # OURS (`tuxghost.observe`), not an untyped upstream engine object, so
    # it does not fall under this project's established "bare `Any` for
    # unannotated upstream types" convention and should be spelled out
    # properly. Imported lazily at runtime inside `run_agent` (below,
    # guarded by `observe=`) so constructing a `FrameRenderer` -- which
    # mutates the client by installing a real `MapRenderer` -- stays an
    # opt-in side effect, not a module-import-time one.
    from tuxghost.observe import FrameRenderer

#: Mirrors `tuxghost.trace.Provenance.recorder`. Spelled out here so the
#: runner and the CLI can type this argument without a `type: ignore` at
#: the `Recorder(...)` call -- an ignore there would be a silent licence
#: to pass any string at all into a field the format constrains.
RecorderKind = Literal["cu-agent", "offline-agent", "human"]


@dataclass(frozen=True)
class Decision:
    """One turn of the loop, for the run directory. Records the frame's
    SIZE, not its bytes: a decision log is meant to stay readable and
    small, and the frame itself is written to `frames/` separately."""

    step: int
    actions: tuple[Action, ...]
    frame_bytes: int
    frame_is_blank: bool
    state_stack: tuple[str, ...]
    map_name: str
    tile_pos: tuple[int, int]
    notes: str | None = None
    raw: str | None = None


@dataclass
class RunResult:
    #: Never optional: `run_agent` only ever constructs a `RunResult` once
    #: it has a real `Trace` in hand (`recorder.finish()` has already run).
    #: Everything accumulated while the run is in flight
    #: (`digests`/`decisions`/`frames`/`steps`) is built up in plain local
    #: variables inside `run_agent` instead, precisely so this field never
    #: has to be `Trace | None` -- no dummy `Trace` object, no `# type:
    #: ignore`, and no caller-side `assert result.trace is not None`
    #: needed just to read it back. Task 5 review round 1.
    trace: Trace
    schedule: InputSchedule
    #: `(step, digest)` pairs, one every `digest_every` steps. Empty when
    #: `digest_every=0` (the default).
    digests: list[tuple[int, str]] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    steps: int = 0
    frames: list[bytes] = field(default_factory=list)
    #: WHY the loop ended -- "policy returned STOP" (an empty `actions`
    #: answer) or "step budget" (the loop's own `while step < step_budget`
    #: condition went false, OR the truncation guard refused to start an
    #: action that would run past it). M2+M3, whole-branch review: neither
    #: break path in `run_agent`'s loop reaches `decisions.append`, so
    #: without this field a run directory could not say why a run ended
    #: at all -- see `tuxghost.cli._agent`, which writes this into
    #: `run_dir/run.json`.
    stop_reason: str = "step budget"


def _observation(
    client: Any, step: int, frames: FrameRenderer | None
) -> tuple[Observation, bytes]:
    from tuxemon.states.world_state import WorldState

    png = b"" if frames is None else frames.png()
    blank = False if frames is None else frames.last_frame_was_blank
    try:
        world = client.get_state_by_name(WorldState)
        tile = (int(world.player.tile_pos[0]), int(world.player.tile_pos[1]))
        map_name = client.get_map_name()
    except ValueError:
        # `StateManager.get_state_by_name` raises ValueError("Missing state
        # ...") when nothing on the stack matches -- verified in
        # `tuxemon/state/manager.py`. `MapManager.get_map_name` raises its
        # OWN ValueError ("Name of the map requested when no map is
        # active") in exactly the same no-map moment -- verified in
        # `tuxemon/map/manager.py` -- so it must be covered by the SAME
        # guard, not called after it (parked/whole-branch review M1: an
        # earlier version of this function called `get_map_name()` outside
        # this `try`, so it raised, uncaught, in precisely the situation
        # this guard exists to tolerate). A client with no WorldState/no
        # active map is a legitimate moment (between map loads), and an
        # observation must still be produced or the run would die on a
        # frame the agent could perfectly well have acted on. Caught
        # narrowly, by the exact exception type, so a genuine engine
        # failure still propagates.
        tile = (-1, -1)
        map_name = ""
    return (
        Observation(
            step=step,
            frame_png=png,
            frame_is_blank=blank,
            state_stack=tuple(client.active_state_names),
            map_name=map_name,
            tile_pos=tile,
        ),
        png,
    )


def run_agent(
    *,
    policy: Policy,
    seed: int,
    clock_epoch: int,
    step_budget: int,
    save_data: Any | None = None,
    cold_boot: bool = False,
    recorder_kind: RecorderKind = "cu-agent",
    model: str | None = None,
    observe: bool = True,
    upscale: int = 4,
    taints: Sequence[str] = (),
    claimed_outcome: str | None = None,
    digest_every: int = 0,
    scaled: bool = False,
) -> RunResult:
    """Run `policy` from `save_data` for at most `step_budget` steps.

    `observe=False` skips the render pass entirely and hands the policy
    empty frames. That is not a convenience: it is how
    `tests/test_agent_runner.py` proves the render pass is inert, by
    running the same scripted policy both ways and comparing digests.

    `digest_every=N` records `(step, digest_of(session))` every N steps
    into `RunResult.digests`. Default 0 records nothing and costs
    nothing. This mirrors `tuxghost.execute.execute`'s existing
    `checkpoint`/`capture_states` parameters -- test-support sampling
    living in production code, deliberately, because the alternative is a
    parallel stepping path that could drift from this one. It exists
    because a FINAL-digest comparison is not sufficient to prove the
    render pass is inert: Tuxemon's grid movement makes a settled tile an
    attractor, so a render pass that advanced the game would make
    movement finish sooner and both runs would still converge on the same
    tile (measured in task 2, against two different injected side
    effects). A per-step sequence sees the transient.

    `taints`/`claimed_outcome` pass straight through to `Recorder` --
    see `Recorder.__init__`/`finish()` for what they mean. `claimed_outcome`
    falls back to `policy.claimed_outcome` only when the caller passes
    `None` (the default) -- checked with `is None`, not truthiness, so an
    explicit `claimed_outcome=""` from a caller is honoured as-is rather
    than silently overridden by the policy's own value (task 5 review
    round 1).

    `scaled=True` boots with `tuxghost.observe.scaled_context()` instead
    of the module default `headless_context()`: task 9 measured a
    300-step per-step digest sequence (not just a final digest) identical
    at every index between the two contexts, on a schedule that keeps the
    player moving through the end of the window -- so an agent gets the
    same ~16x9-tile field of view a human player sees (measured scale 5
    on this repo's `~/.tuxemon/tuxemon.yaml`, not the scale 4 / ~20x11
    tiles the task brief assumed before measuring), at no digest cost,
    and `tuxghost.execute` (which always boots unscaled) still reproduces
    a run's digests exactly. See
    `docs/2026-08-26-display-scale-measurement.org` for the raw numbers.

    Default `scaled=False`, deliberately NOT the digest-neutral finding's
    default: `scaled_context()` refuses (a loud `RuntimeError`) unless it
    is the FIRST `DisplayContext`-building call in the whole process --
    `tuxemon.map.view`/`tuxemon.graphics` each bind their own copy of
    `tuxemon.prepare.DISPLAY_CONTEXT` at import time (see
    `scaled_context()`'s docstring), so calling it after ANY earlier
    client has booted in the same process silently would have rendered a
    WORSE frame than the unscaled default (measured: a sparse grid of
    unscaled tiles, player sprite missing). That is true of essentially
    every test process in this repo's suite, which boots many clients
    per process -- only a genuinely fresh process (this project's CLI,
    launched as its own OS process per invocation) can safely pass
    `scaled=True`; `tuxghost/cli.py`'s `agent` subcommand does.
    """
    if step_budget < 1:
        raise ValueError(f"step_budget must be >= 1, got {step_budget!r}")
    if (save_data is None) == (not cold_boot):
        raise ValueError(
            "pass exactly one of save_data= or cold_boot=True; a run has to "
            "start from somewhere, and starting from both is not a thing"
        )

    context = scaled_context() if scaled else None
    if cold_boot:
        # The title screen. Honest computer-use territory -- the intro's
        # name-entry keyboard is exactly the hard case -- and the only
        # route that renders the menu states at all. Probed: mashing A
        # here does NOT get through, so a policy that wants this must
        # actually read the frames.
        client, session = build_client(
            seed=seed, clock_epoch=clock_epoch, context=context
        )
    else:
        client, session = boot_from_save(
            save_data, seed=seed, clock_epoch=clock_epoch, context=context
        )

    frames: FrameRenderer | None = None
    if observe:
        from tuxghost.observe import FrameRenderer

        frames = FrameRenderer(client, upscale=upscale)

    schedule: InputSchedule = {}
    install_schedule(client, schedule)
    recorder = Recorder(
        session,
        seed=seed,
        clock_epoch=clock_epoch,
        recorder=recorder_kind,
        model=model,
        taints=taints,
    )

    digests: list[tuple[int, str]] = []
    decisions: list[Decision] = []
    frame_log: list[bytes] = []
    step = 0
    #: M2+M3, whole-branch review: default covers BOTH ways the loop can
    #: end without the policy ever answering STOP -- the truncation-guard
    #: `break` just below, and the `while step < step_budget` condition
    #: itself going false after the last completed action. Overwritten
    #: below only on the one path that is genuinely a policy decision.
    stop_reason = "step budget"

    while step < step_budget:
        obs, png = _observation(client, step, frames)
        actions = validate_actions(policy.decide(obs))
        if not actions:
            stop_reason = "policy returned STOP"
            break

        cost = sum(a.hold + a.settle for a in actions)
        if step + cost > step_budget:
            # Refuse to truncate an action mid-hold: a half-delivered
            # press would put a button down in the trace and never lift
            # it, and the released edge is what makes a trace replay.
            break

        for action in actions:
            add_edge(schedule, step, action.button, PRESSED)
            recorder.observe(step, action.button, PRESSED)
            release = step + action.hold
            add_edge(schedule, release, action.button, RELEASED)
            recorder.observe(release, action.button, RELEASED)

            base = step

            def sample(i: int, _base: int = base) -> None:
                # `(_base + i) and ...`, not just `% digest_every == 0`,
                # to match `tuxghost.execute.execute`'s own checkpoint
                # convention EXACTLY: `hook`'s `if checkpoint and i and i
                # % checkpoint == 0` there never samples absolute step 0
                # either. Nothing compares the two sequences today (M5,
                # whole-branch review), but a future S3 comparison would
                # otherwise be off by one entry at the start of every run
                # -- `execute` would never have a step-0 digest to line up
                # against one `run_agent` recorded.
                absolute = _base + i
                if digest_every and absolute and absolute % digest_every == 0:
                    digests.append((absolute, digest_of(session)))

            run_steps(
                client,
                action.hold + action.settle,
                hook=sample if digest_every else None,
            )
            step += action.hold + action.settle

        decisions.append(
            Decision(
                step=obs.step,
                actions=actions,
                frame_bytes=len(png),
                frame_is_blank=obs.frame_is_blank,
                state_stack=obs.state_stack,
                map_name=obs.map_name,
                tile_pos=obs.tile_pos,
                notes=getattr(policy, "last_notes", None),
                raw=getattr(policy, "last_raw", None),
            )
        )
        if png:
            frame_log.append(png)

    trace = recorder.finish(
        step_count=step,
        claimed_outcome=(
            claimed_outcome
            if claimed_outcome is not None
            else getattr(policy, "claimed_outcome", None)
        ),
    )
    return RunResult(
        trace=trace,
        schedule=schedule,
        digests=digests,
        decisions=decisions,
        steps=step,
        frames=frame_log,
        stop_reason=stop_reason,
    )
