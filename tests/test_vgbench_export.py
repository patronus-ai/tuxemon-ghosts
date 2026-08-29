"""Regression tests for `tuxghost.vgbench`, the videogamebench exporter,
and for `tuxghost.cli`'s `export` subcommand (S5 task 6).

Their trajectory format is RUN-LENGTH HELD STATE -- `{"frame": N,
"buttons": [...]}` means "the held set becomes this at frame N and stays
until the next segment". Ours is an EDGE list, `(step, button, value)`
with PRESSED/RELEASED. `segments_of`/`trajectory_of` convert losslessly.

The CLI tests below are driven as a real SUBPROCESS, not through `main()`
in-process -- `tests/test_optimize_cli.py`'s own docstring records why: an
in-process run shares a process with earlier boots, and the exit-code
contract is a property of the process, not of a function's return value.
"""

import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

from tuxghost.trace import read
from tuxghost.vgbench import segments_of, trajectory_of

ROOT = Path(__file__).resolve().parent.parent
PARENT = Path(__file__).parent / "golden" / "scripted_town_1234.tuxghost"


def _run_cli(
    *args: str,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Model: `tests/test_optimize_cli.py`'s `_run`. `cwd` is a real
    parameter, not decoration -- a run launched from somewhere other than
    ROOT is the only way to prove a RELATIVE `--trace` path resolves
    against the CALLER's cwd, not the vendored `tuxemon/` directory the
    process later chdirs into."""
    env = {
        **os.environ,
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "PYTHONHASHSEED": "0",
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "tuxghost.cli", "export", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else ROOT,
        env=env,
        check=False,
    )


def test_segments_are_run_length_held_state() -> None:
    """Their format is 'at frame N the held set becomes X, held until the
    next segment' -- NOT our edge list. So consecutive segments must
    never repeat a held set, and frames must ascend."""
    segs = segments_of(read(PARENT).inputs)
    assert segs, "no segments produced"
    frames = [s["frame"] for s in segs]
    assert frames == sorted(frames)
    assert len(frames) == len(set(frames)), "one segment per frame"
    for a, b in pairwise(segs):
        assert a["buttons"] != b["buttons"], (a, b)


def test_redundant_edges_do_not_produce_duplicate_segments() -> None:
    """The `if current == last: continue` guard in `segments_of` exists
    to suppress a redundant edge (e.g. a re-press of an already-held
    button, on a DIFFERENT frame from the segment that first held it)
    from producing a second, duplicate segment.

    This is deliberately NOT exercised by
    `scripted_town_1234.tuxghost`: that fixture's edges never redundantly
    re-press an already-held button or release an unheld one, so removing
    the guard leaves that fixture's segments byte-for-byte unchanged --
    demonstrating against the guard on the golden fixture alone would be
    a vacuous demonstration. This synthetic case exercises the guard for
    real.
    """
    from tuxemon.platform.const import buttons

    from tuxghost.loop import PRESSED

    inputs = [
        (0, buttons.A, PRESSED),  # frame 0: A held -> segment {0: ["A"]}
        (5, buttons.A, PRESSED),  # frame 5: A re-pressed, already held -> no-op
    ]
    segs = segments_of(inputs)
    assert segs == [{"frame": 0, "buttons": ["A"]}]


def test_edges_round_trip_through_segments() -> None:
    """The conversion must be lossless: rebuilding edges from the
    run-length form reproduces the trace's own inputs."""
    inputs = read(PARENT).inputs
    segs = segments_of(inputs)

    from tuxghost.loop import PRESSED, RELEASED
    from tuxghost.vgbench import _NAME_TO_BUTTON

    rebuilt: list[tuple[int, int, float]] = []
    held: set[int] = set()
    for s in segs:
        now = {_NAME_TO_BUTTON[n] for n in s["buttons"]}
        for b in sorted(now - held):
            rebuilt.append((s["frame"], b, PRESSED))
        for b in sorted(held - now):
            rebuilt.append((s["frame"], b, RELEASED))
        held = now
    assert sorted(rebuilt) == sorted(tuple(i) for i in inputs)


def test_total_frames_matches_the_header() -> None:
    """20 of their 21 committed files disagree with their own
    total_frames by +10 to +209 frames. Ours must not."""
    trace = read(PARENT)
    traj = trajectory_of(trace)
    assert traj["total_frames"] == trace.header.step_count
    assert traj["fps"] == trace.header.step_rate


def test_provenance_is_our_real_value_not_theirs() -> None:
    """Their `type` reads "human" in all 21 files INCLUDING the model
    runs, so copying their values would import a known defect. We emit
    their field name with our honest provenance."""
    traj = trajectory_of(read(PARENT))
    assert traj["type"] == read(PARENT).provenance.recorder


def test_no_wall_clock_reaches_the_export() -> None:
    """CLAUDE.md: real timestamps never enter this format. Their schema
    has `started_at`; we deliberately omit it rather than breach that
    rule for cosmetic parity."""
    blob = json.dumps(trajectory_of(read(PARENT)))
    assert "started_at" not in blob
    assert "timestamp" not in blob


def test_same_frame_edges_coalesce_without_leaving_a_duplicate() -> None:
    """CONTROLLER RULING R4. The brief's own `segments_of` sketch coalesces
    two edges landing on the same frame by overwriting
    `segments[-1]["buttons"]` in place -- but that overwrite can make
    `segments[-1]` equal `segments[-2]`, violating the
    no-consecutive-duplicates invariant this test suite pins elsewhere.

    Case 1: a press-then-release of the SAME button on the SAME frame as
    a prior segment's frame nets back to that prior segment's held set --
    the coalesced segment must be dropped entirely, not merely coalesced.

    Case 2: same-frame edges that net to something genuinely new must
    still produce exactly one segment for that frame, with the net
    value.
    """
    from tuxemon.platform.const import buttons

    from tuxghost.loop import PRESSED, RELEASED

    # Case 1: frame 0 presses A (segment: frame 0 -> ["A"]). At frame 5,
    # A is released and then re-pressed on the SAME frame -- net effect
    # is a no-op: the held set at frame 5 equals the PREVIOUS segment's
    # held set (["A"] persists), so no new segment should appear at all.
    #
    # `sorted(tuple(i) for i in inputs)` orders same-(step, button) edges
    # by VALUE, and RELEASED (0.0) < PRESSED (1.0), so the release is
    # always processed before the re-press here regardless of the order
    # the tuples are written in below -- which is exactly the
    # release-then-repress sequence that nets to a no-op.
    inputs_noop = [
        (0, buttons.A, PRESSED),  # frame 0: A held -> segment {0: ["A"]}
        (5, buttons.A, RELEASED),  # frame 5: A released ...
        (5, buttons.A, PRESSED),  # ... then re-pressed same frame -> back to ["A"]
    ]
    segs = segments_of(inputs_noop)
    frames = [s["frame"] for s in segs]
    assert len(frames) == len(set(frames)), "one segment per frame"
    for a, b in pairwise(segs):
        assert a["buttons"] != b["buttons"], (a, b)
    # The frame-5 segment must not exist: its net held set (["A"]) equals
    # frame 0's, so it must have been dropped, not left behind as a dupe.
    assert segs == [{"frame": 0, "buttons": ["A"]}]

    # Case 2: same-frame edges that net to something new must still
    # collapse into exactly one segment at that frame.
    inputs_new = [
        (0, buttons.A, PRESSED),  # frame 0: A held -> segment {0: ["A"]}
        (5, buttons.A, RELEASED),  # frame 5: A released
        (5, buttons.LEFT, PRESSED),  # frame 5: LEFT pressed same frame -> {5: ["LEFT"]}
    ]
    segs2 = segments_of(inputs_new)
    frames2 = [s["frame"] for s in segs2]
    assert len(frames2) == len(set(frames2)), "one segment per frame"
    for a, b in pairwise(segs2):
        assert a["buttons"] != b["buttons"], (a, b)
    assert segs2 == [
        {"frame": 0, "buttons": ["A"]},
        {"frame": 5, "buttons": ["LEFT"]},
    ]


def test_button_names_track_upstreams_constants() -> None:
    """Swap two VALID button values and confirm the export follows. A
    literal table would emit the same names either way and pass."""
    import importlib
    from unittest import mock

    from tuxemon.platform.const import buttons

    import tuxghost.vgbench as vgb

    assert (buttons.UP, buttons.LEFT) == (1, 4)
    try:
        with (
            mock.patch.object(buttons, "UP", 4),
            mock.patch.object(buttons, "LEFT", 1),
        ):
            remapped = importlib.reload(vgb)
            names = remapped._button_names()
            assert names[4] == "UP"
            assert names[1] == "LEFT"
    finally:
        importlib.reload(vgb)


def test_export_refuses_a_missing_trace(tmp_path: Path) -> None:
    proc = _run_cli(
        "--trace", str(tmp_path / "nope.tuxghost"),
        "--out", str(tmp_path / "o.json"),
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert not (tmp_path / "o.json").exists()


def test_a_relative_trace_path_resolves_against_the_callers_cwd(
    tmp_path: Path,
) -> None:
    """The process chdirs into tuxemon/ before doing work, so a relative
    path must be resolved first. Asserted on the RESOLVED path appearing
    in the refusal, because both the fixed and broken versions exit 2 --
    an exit-code-only assertion would not discriminate."""
    resolved = tmp_path / "nope.tuxghost"
    proc = _run_cli(
        "--trace", "nope.tuxghost",
        "--out", str(tmp_path / "o.json"),
        cwd=tmp_path,
    )
    assert proc.returncode == 2
    assert str(resolved) in proc.stderr


def test_the_export_loads_with_their_own_consumer_expression() -> None:
    """THE CONSISTENCY CHECK. `{s["frame"]: set(s["buttons"]) for s in
    data["segments"]}` is the exact line four of their scripts use to
    read a trajectory. If our file survives it, their consumers can read
    us."""
    traj = trajectory_of(read(PARENT))
    seg_at = {s["frame"]: set(s["buttons"]) for s in traj["segments"]}
    assert seg_at
    assert all(isinstance(k, int) for k in seg_at)
    assert set(traj) <= {
        "type", "game", "rom", "started_at", "boot_frames",
        "fps", "total_frames", "segments", "goal_frame", "rationale",
    }


def test_export_writes_a_trajectory_the_cli_produces_end_to_end(
    tmp_path: Path,
) -> None:
    """The CLI's own output, not just `trajectory_of` called directly --
    proves `_export` actually wires `--trace`/`--out` through
    `tuxghost.vgbench.trajectory_of` and writes valid JSON their own
    consumer expression can read."""
    out = tmp_path / "traj.json"
    proc = _run_cli("--trace", str(PARENT), "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    seg_at = {s["frame"]: set(s["buttons"]) for s in data["segments"]}
    assert seg_at
    assert all(isinstance(k, int) for k in seg_at)
    assert set(data) <= {
        "type", "game", "rom", "started_at", "boot_frames",
        "fps", "total_frames", "segments", "goal_frame", "rationale",
    }
