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

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAVE = Path(__file__).parent / "fixtures" / "paper_town.save"


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
