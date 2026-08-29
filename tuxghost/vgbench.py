"""Export a .tuxghost trace as a videogamebench trajectory.

Their format is RUN-LENGTH HELD STATE -- `{"frame": N, "buttons": [...]}`
means "the held set becomes this at frame N, and stays until the next
segment". Ours is an EDGE list, `(step, button, value)` with
PRESSED/RELEASED. The two convert losslessly.

Export only. Import is out of scope: 51% of their segments hold two or
three buttons at once, and `tuxghost.optimize.schedule.lift` refuses
that outright ("sequential and cannot express two buttons held at
once"). Our single-button traces are a legal SUBSET of their format, so
every .tuxghost exports to a valid trajectory -- the incompatibility
only bites in the direction we are not going.

Two of their schema fields are deliberately never emitted:
  * `started_at` -- wall-clock. CLAUDE.md forbids real timestamps
    entering this format; `PlayerInput.timestamp` was excluded from
    `.tuxghost` for the same reason.
  * `rom` -- meaningless for us; omitted rather than faked.
"""

from __future__ import annotations

from typing import Any

from tuxghost.loop import PRESSED
from tuxghost.trace import Trace


def _button_names() -> dict[int, str]:
    """Int -> their button name.

    Read from `tuxemon.platform.const.buttons`, never written as
    literals: an upstream remap would otherwise leave us confidently
    exporting the wrong button. `tests/test_optimize_claude.py` pins this
    same rule for ClaudeEditor's prompt.
    """
    from tuxemon.platform.const import buttons

    return {
        getattr(buttons, name): name
        for name in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B")
    }


_NAME_TO_BUTTON = {v: k for k, v in _button_names().items()}


def segments_of(inputs: Any) -> list[dict[str, Any]]:
    """Convert our edge list into their run-length held-state segments.

    CONTROLLER RULING R4: two edges landing on the same frame are
    coalesced into a single segment for that frame -- but if the net
    held set after coalescing equals the PREVIOUS segment's held set,
    the coalesced segment is dropped entirely rather than left behind as
    a consecutive duplicate. Without this, e.g. `(5, A, PRESSED)` and
    `(5, A, RELEASED)` landing right after a segment already holding A
    would leave `segments[-1] == segments[-2]`, violating the
    no-consecutive-duplicates invariant this format requires.
    """
    names = _button_names()
    held: set[int] = set()
    segments: list[dict[str, Any]] = []
    last: list[str] | None = None
    for step, button, value in sorted(tuple(i) for i in inputs):
        if value == PRESSED:
            held.add(button)
        else:
            held.discard(button)
        current = sorted(names[b] for b in held)
        if current == last:
            continue
        if segments and segments[-1]["frame"] == step:
            segments[-1]["buttons"] = current  # coalesce same-frame edges
            if len(segments) >= 2 and segments[-1]["buttons"] == segments[-2]["buttons"]:
                segments.pop()
                # Recompute `last` from segments[-1] (NOT from `current`,
                # which is now stale -- it described the popped, now-gone
                # segment). Getting this wrong silently reintroduces the
                # R4 consecutive-duplicate bug
                # `test_same_frame_edges_coalesce_without_leaving_a_
                # duplicate` pins: dropping this line makes `last` stay
                # `current`, so the NEXT same-frame coalesce compares
                # against the value that was just discarded rather than
                # what remains, and a genuine duplicate can slip through.
                last = segments[-1]["buttons"] if segments else None
                continue
        else:
            segments.append({"frame": step, "buttons": current})
        last = current
    return segments


def _stage_of(trace: Trace) -> str:
    """The map a trace's `initial_state` booted on, `.tmx` stripped, or
    `"unknown"` when absent. Pure dict access -- no `SaveData` validation,
    no engine boot -- matching `_export`'s own claim in `tuxghost.cli` to
    be "a pure format conversion, not a replay"."""
    npc_state = trace.initial_state.get("npc_state") or {}
    current_map = npc_state.get("current_map")
    if not current_map:
        return "unknown"
    return str(current_map).removesuffix(".tmx")


def trajectory_filename(trace: Trace, traj: dict[str, Any]) -> str:
    """Their `<role>_<model>_<game>_<stage>_<frames>f.json` filename
    convention (spec's "Surface" section), with the frame count DERIVED
    from `traj["total_frames"]` -- never independently recomputed and
    never taken from a caller-supplied count. 20 of their 21 committed
    result files (`~/Workspace/videogamebench/results/speedrun/*.json`)
    disagree with their own `total_frames` by +10 to +209 frames
    (spec validation item 9); deriving the count from the same field the
    file's OWN `total_frames` holds is what keeps this from reproducing
    that drift.

    `role` = `trace.provenance.recorder` (`cu-agent`/`offline-agent`/
    `human`), `model` = `trace.provenance.model` or `"none"`, `game` =
    `traj["game"]` (== `header.mod_id`), `stage` = the map the trace
    booted on (see `_stage_of`). All four come from data the trace/
    trajectory already carry, so no new required CLI flag is needed to
    name the file.
    """
    role = trace.provenance.recorder
    model = trace.provenance.model or "none"
    game = traj["game"]
    stage = _stage_of(trace)
    frames = traj["total_frames"]
    name = f"{role}_{model}_{game}_{stage}_{frames}f.json"
    # None of the four fields above are free-form user input on any
    # currently-shipped path (`recorder` is a `Literal`, `game` is
    # `header.mod_id`, `stage` comes from a map filename) -- but a path
    # separator slipping into any of them must not escape the containing
    # directory.
    return name.replace("/", "-").replace("\\", "-")


def trajectory_of(trace: Trace, goal_frame: int | None = None) -> dict[str, Any]:
    """Convert a `.tuxghost` trace into a videogamebench trajectory dict.

    `started_at` and `rom` are deliberately omitted -- see the module
    docstring.
    """
    traj: dict[str, Any] = {
        "type": trace.provenance.recorder,
        "game": trace.header.mod_id,
        "fps": trace.header.step_rate,
        "total_frames": trace.header.step_count,
        "segments": segments_of(trace.inputs),
    }
    if goal_frame is not None:
        traj["goal_frame"] = goal_frame
    return traj
