"""Task 9: the `play` subcommand.

Driven as a real SUBPROCESS, not through `main()` in-process -- the same
rule `tests/test_optimize_cli.py`'s docstring records: an in-process run
shares a process with earlier boots, and the exit-code contract is a
property of the process, not of a function's return value.

`_run_cli` below is modelled directly on `tests/test_optimize_cli.py`'s
`_run` (env vars, cwd, capture, `check=False`) -- there is no shared
helper importable across test files in this project, so each CLI test
module keeps its own copy.

Unlike every other subcommand, `tuxghost play` is the ONE invocation in
this project permitted to run WITHOUT the dummy SDL drivers (see
`tuxghost/play.py`'s module docstring and the "Never run the game
without dummy SDL drivers" carve-out in `CLAUDE.md`). This test file
does not exercise that real-window path at all -- it only pins the
refusal that must happen BEFORE a window is ever opened, so it still
runs under the dummy drivers like every other test in the gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAVE = Path(__file__).parent / "fixtures" / "paper_town.save"
GHOST = Path(__file__).parent / "golden" / "scripted_town_1234.tuxghost"


def _run_cli(
    *args: str,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "PYTHONHASHSEED": "0",
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "tuxghost.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else ROOT,
        env=env,
        # Explicit, per this project's convention (and `make lint`'s
        # PLW1510): this run EXPECTS a non-zero exit.
        check=False,
    )


def test_play_refuses_a_missing_ghost_trace(tmp_path: Path) -> None:
    """Exit 2, the code this CLI uses for every anticipated precondition,
    not the exit 1 an uncaught `FileNotFoundError` deep inside
    `tuxghost.play.play` (which has already called `pygame_init()` by the
    time it would reach its own `read()` call) would otherwise produce.

    The refusal must happen BEFORE `tuxghost.play.play` is ever entered --
    that is itself the property this test pins, not just the exit code:
    proven here by `out` and `run_dir` never coming into existence, since
    both are only ever written at the very end of a real play session.
    """
    out = tmp_path / "o.tuxghost"
    run_dir = tmp_path / "run"
    proc = _run_cli(
        "play",
        "--ghost", str(tmp_path / "nope.tuxghost"),
        "--from-save", str(SAVE),
        "--seed", "1234",
        "--clock-epoch", "1787659200",
        "--out", str(out),
        "--run-dir", str(run_dir),
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "refused" in proc.stderr
    # Nothing downstream of a real play session ever ran.
    assert not out.exists()
    assert not run_dir.exists()


def test_play_refuses_a_malformed_ghost_trace(tmp_path: Path) -> None:
    """Same refusal, for a ghost trace that exists but is not valid JSON --
    a different branch of `_read_trace_or_refuse` than the missing-file
    case above (`json.JSONDecodeError` vs. `OSError`), so both need their
    own coverage rather than assuming one implies the other."""
    bad = tmp_path / "bad.tuxghost"
    bad.write_text("not json")
    out = tmp_path / "o.tuxghost"
    run_dir = tmp_path / "run"
    proc = _run_cli(
        "play",
        "--ghost", str(bad),
        "--from-save", str(SAVE),
        "--seed", "1234",
        "--clock-epoch", "1787659200",
        "--out", str(out),
        "--run-dir", str(run_dir),
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "refused" in proc.stderr
    assert not out.exists()
    assert not run_dir.exists()


def test_play_refuses_a_missing_from_save(tmp_path: Path) -> None:
    """I2, whole-branch review: `_play` never validated `--from-save`
    before this fix -- `--from-save nope.save` reached
    `tuxghost.play.play`'s own `save.read_text()` AFTER `pygame_init()`
    had already opened a window, and raised an uncaught
    `FileNotFoundError` there (exit 1, a real traceback), indistinguishable
    from "diverged". Every other subcommand in this project treats a
    missing `--from-save` as a refused precondition (exit 2); `play` must
    too, and before the window opens -- proven the same way
    `test_play_refuses_a_missing_ghost_trace` above proves it for
    `--ghost`: `out`/`run_dir` never come into existence."""
    out = tmp_path / "o.tuxghost"
    run_dir = tmp_path / "run"
    proc = _run_cli(
        "play",
        "--from-save", str(tmp_path / "nope.save"),
        "--seed", "1234",
        "--clock-epoch", "1787659200",
        "--out", str(out),
        "--run-dir", str(run_dir),
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "refused" in proc.stderr
    assert not out.exists()
    assert not run_dir.exists()


def _write_ghost_trace_with_unresolvable_map(tmp_path: Path) -> Path:
    """A structurally valid, digest-consistent ghost trace whose
    `initial_state.npc_state.current_map` cannot resolve to any map
    asset -- `tuxghost.trace.read`'s own refuse/warn matrix never checks
    this (it only recomputes `initial_state_digest`, which this function
    keeps consistent so the mutated file still reads cleanly), so the
    only thing that can catch it is a map-resolution check downstream,
    same as `tuxghost.execute.execute`'s own preamble."""
    from tuxghost.trace import digest_of_initial_state

    raw = json.loads(GHOST.read_text())
    raw["initial_state"]["npc_state"]["current_map"] = "no_such_map_xyz"
    raw["header"]["initial_state_digest"] = digest_of_initial_state(
        raw["initial_state"]
    )
    path = tmp_path / "unresolvable_map.tuxghost"
    path.write_text(json.dumps(raw))
    return path


def test_play_refuses_a_ghost_trace_whose_map_cannot_be_resolved(
    tmp_path: Path,
) -> None:
    """I3, whole-branch review: an earlier ruling (R1) held that
    `tuxghost.ghost.track.build_track` may skip `execute`'s
    map-resolution refusal "because its caller reads the trace through
    `_read_trace_or_refuse` first" -- false, since that helper never
    calls `resolve_map_asset`. Before this fix, a ghost trace shaped like
    this one reached `boot_from_save` (via `build_track`, inside
    `tuxghost.play.play`, AFTER `pygame_init()`) and raised a bare
    `ValueError` there -- exit 1, a real traceback -- for what
    `tuxghost.execute.execute` already treats as a refusable
    precondition (exit 2) for a played-back trace."""
    ghost = _write_ghost_trace_with_unresolvable_map(tmp_path)
    out = tmp_path / "o.tuxghost"
    run_dir = tmp_path / "run"
    proc = _run_cli(
        "play",
        "--ghost", str(ghost),
        "--from-save", str(SAVE),
        "--seed", "1234",
        "--clock-epoch", "1787659200",
        "--out", str(out),
        "--run-dir", str(run_dir),
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "refused" in proc.stderr
    assert not out.exists()
    assert not run_dir.exists()


def test_a_relative_ghost_path_resolves_against_the_callers_cwd(
    tmp_path: Path,
) -> None:
    """`_resolve_paths` must resolve `--ghost` against the CALLER's cwd
    BEFORE `_bootstrap_vendored_tuxemon` chdirs into the vendored
    `tuxemon/` tree -- the exact defect an earlier, deleted CLI shipped
    with (see `tuxghost/cli.py`'s module docstring).

    The two refusal tests above use only ABSOLUTE `--ghost` paths, so
    they would keep passing even if `_PATH_ARGS["play"]` lost its
    `"ghost"` entry entirely: an unresolved relative path still fails to
    read (from inside `tuxemon/`, where the process has by then chdir'd)
    and still exits 2 either way. What differs is WHICH path the refusal
    names -- resolved against `tmp_path` when the fix is in place, a bare
    unresolved "nope.tuxghost" (silently read against the wrong
    directory) when it is not. This test pins that, not just the exit
    code, exactly the way `tests/test_optimize_cli.py::
    test_a_relative_trace_path_works` pins the same property for
    `optimize --trace`.
    """
    proc = _run_cli(
        "play",
        "--ghost", "nope.tuxghost",
        "--from-save", str(SAVE),
        "--seed", "1234",
        "--clock-epoch", "1787659200",
        "--out", str(tmp_path / "o.tuxghost"),
        "--run-dir", str(tmp_path / "run"),
        cwd=tmp_path,
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    resolved = tmp_path / "nope.tuxghost"
    assert str(resolved) in proc.stderr, proc.stderr
