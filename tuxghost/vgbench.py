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
                last = segments[-1]["buttons"] if segments else None
                continue
        else:
            segments.append({"frame": step, "buttons": current})
        last = current
    return segments


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
