"""Task 11: the `optimize` subcommand.

Driven as a real SUBPROCESS, not through `main()` in-process. Two
reasons: an in-process run shares a process with earlier boots, and the
exit-code contract is a property of the process, not of a function's
return value. `tests/test_cli.py`'s docstring records the same rule.

Every path a user can get wrong must exit 2, never 1: `optimize`, like
`agent`, has nothing to diverge FROM, so 1 is reserved for a genuine
engine bug.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden" / "claude_town_1234.tuxghost"


def _run(
    *args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "PYTHONHASHSEED": "0",
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "tuxghost.cli", "optimize", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        # Explicit, per this suite's convention (and `make lint`'s
        # PLW1510): every one of these runs EXPECTS a non-zero exit for
        # some case, so raising on it would defeat the test.
        check=False,
    )


def test_a_mutation_run_succeeds_and_writes_its_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "best.tuxghost"
    run_dir = tmp_path / "run"
    proc = _run(
        "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "4", "--patience", "4", "--max-rejections", "3",
        "--max-cost", "4000", "--out", str(out), "--run-dir", str(run_dir),
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()

    info = json.loads((run_dir / "run.json").read_text())
    assert info["editor"] == "mutation" and info["seed"] == 7
    assert info["objective"] == "reach-tile"
    assert info["parent_digest"] == json.loads(GOLDEN.read_text())["header"][
        "final_digest"
    ]
    assert info["stop_reason"]
    assert "best_round" in info

    rows = [
        json.loads(line)
        for line in (run_dir / "optimize.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["index"] == 0 and rows[0]["accepted"] is True
    assert not (run_dir / "frames").exists(), "S3 renders nothing"


def test_the_written_trace_is_a_verifiable_offline_agent_trace(tmp_path: Path) -> None:
    out = tmp_path / "best.tuxghost"
    proc = _run(
        "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "2", "--patience", "2", "--max-rejections", "2",
        "--max-cost", "4000", "--out", str(out),
        "--run-dir", str(tmp_path / "run"),
    )
    assert proc.returncode == 0, proc.stderr
    written = json.loads(out.read_text())
    assert written["provenance"]["recorder"] == "offline-agent"
    taints = written["provenance"]["taints"]
    assert len(taints) == 1 and "parent" in taints[0] and "mutation" in taints[0]

    verify = subprocess.run(
        [sys.executable, "-m", "tuxghost.cli", "verify", str(out)],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy",
             "SDL_AUDIODRIVER": "dummy", "PYTHONHASHSEED": "0"},
        check=False,
    )
    assert verify.returncode == 0, verify.stderr


def test_the_written_trace_is_the_reseal_not_a_patched_copy(tmp_path: Path) -> None:
    """The lineage taint must come from `Recorder`, and the re-sealed
    digest must match what the loop scored. A patched-provenance copy
    would pass every other test in this file."""
    out = tmp_path / "best.tuxghost"
    run_dir = tmp_path / "run"
    proc = _run(
        "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "4", "--patience", "4", "--max-rejections", "3",
        "--max-cost", "4000", "--out", str(out), "--run-dir", str(run_dir),
    )
    assert proc.returncode == 0, proc.stderr
    written = json.loads(out.read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "optimize.jsonl").read_text().splitlines()
        if line.strip()
    ]
    best_index = json.loads((run_dir / "run.json").read_text())["best_round"]
    scored = next(r for r in rows if r["index"] == best_index)
    assert written["header"]["final_digest"] == scored["digest"], (
        "the written trace is not the trace the loop scored"
    )


def test_a_relative_trace_path_works(tmp_path: Path) -> None:
    """The CLI chdirs into the vendored tuxemon/ directory. A relative
    path must resolve against the user's cwd -- the exact defect an
    earlier CLI shipped with, and one that only a relative path finds."""
    proc = _run(
        "--trace", "tests/golden/claude_town_1234.tuxghost",
        "--editor", "mutation", "--seed", "7", "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "4000", "--out", str(tmp_path / "o.tuxghost"),
        "--run-dir", str(tmp_path / "run"),
    )
    assert proc.returncode == 0, proc.stderr


def _base(tmp_path: Path) -> dict[str, str]:
    return {
        "--trace": str(GOLDEN), "--editor": "mutation", "--seed": "7",
        "--objective": "reach-tile",
        "--target": "spyder_paper_town.tmx:11,16", "--rounds": "1",
        "--patience": "1", "--max-rejections": "1", "--max-cost": "4000",
        "--out": str(tmp_path / "o.tuxghost"),
        "--run-dir": str(tmp_path / "run"),
    }


#: (override, drop, expected stderr fragment). `override` replaces one
#: flag's value; `drop` removes a flag entirely. Deliberately explicit --
#: an earlier draft merged these with index arithmetic that was harder to
#: read than the cases it covered, and a test whose own logic needs
#: working out can pass for the wrong reason.
REFUSALS = [
    ({"--trace": "/nonexistent.tuxghost"}, None, "cannot read"),
    ({"--editor": "replay"}, None, "requires --edits"),
    ({"--editor": "scripted"}, None, "requires --edits"),
    ({"--target": "nonsense"}, None, "--target"),
    ({"--target": "map.tmx:a,b"}, None, "--target"),
    ({"--rounds": "0"}, None, "--rounds"),
    ({"--patience": "0"}, None, "--patience"),
    ({"--max-rejections": "0"}, None, "--max-rejections"),
    ({"--max-cost": "0"}, None, "--max-cost"),
    ({"--checkpoint": "-1"}, None, "--checkpoint"),
    ({}, "--seed", "--seed"),
]


@pytest.mark.parametrize("override,drop,needle", REFUSALS)
def test_every_anticipated_precondition_exits_2(
    tmp_path: Path, override: dict[str, str], drop: str | None, needle: str
) -> None:
    flags = _base(tmp_path)
    flags.update(override)
    if drop is not None:
        del flags[drop]
    proc = _run(*[x for k, v in flags.items() for x in (k, v)])
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert needle in proc.stderr


def test_a_trace_file_that_is_not_valid_json_is_refused_not_crashed(
    tmp_path: Path,
) -> None:
    """Added beyond the plan's own refusal table, which had a real hole:
    `tuxghost.trace.read`'s first line is a bare `json.loads`, so a trace
    file that is not valid JSON raises `json.JSONDecodeError` -- and the
    plan's `_optimize` caught only `OSError` and `Refused` around that
    call, so the exception propagated and the process exited 1,
    "diverged", the one code this subcommand must never produce for a bad
    input file. Closed by routing the read through the module's existing
    `_read_trace_or_refuse` boundary, which already catches
    `json.JSONDecodeError` and `pydantic.ValidationError` for exactly
    this reason (see its docstring)."""
    bad = tmp_path / "garbage.tuxghost"
    bad.write_text("this is not JSON at all\n")
    flags = _base(tmp_path)
    flags["--trace"] = str(bad)
    proc = _run(*[x for k, v in flags.items() for x in (k, v)])
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "not valid JSON" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_an_unliftable_parent_is_refused_not_crashed(tmp_path: Path) -> None:
    """A trace with two buttons held at once cannot be lifted. That is a
    refusal (2), not a divergence (1) and not a traceback."""
    parent = json.loads(GOLDEN.read_text())
    parent["inputs"] = [
        [0, 2, 1.0], [4, 8, 1.0], [8, 2, 0.0], [12, 8, 0.0],
    ]
    parent["header"]["step_count"] = 100
    bad = tmp_path / "overlap.tuxghost"
    bad.write_text(json.dumps(parent))
    proc = _run(
        "--trace", str(bad), "--editor", "mutation", "--seed", "7",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "4000", "--out", str(tmp_path / "o.tuxghost"),
        "--run-dir", str(tmp_path / "run"),
    )
    assert proc.returncode == 2, proc.stderr
    assert "overlap" in proc.stderr
    assert "Traceback" not in proc.stderr


#: A driver that stubs `tuxghost.optimize.runner.optimize` so it raises a
#: REAL `AttributeError` -- the class of engine-invariant failure that
#: must stay a visible crash -- and then calls `tuxghost.cli.main`
#: directly. `_optimize` does its `from tuxghost.optimize.runner import
#: optimize` at CALL time, so patching the module attribute before
#: `main()` runs is enough; nothing else about the CLI is touched.
_LAUNDERING_DRIVER = '''\
import sys

sys.path.insert(0, {root!r})

from tuxghost.execute import _bootstrap_vendored_tuxemon

_bootstrap_vendored_tuxemon()

import tuxghost.optimize.runner as runner


def _boom(*args, **kwargs):
    raise AttributeError(
        "'NullRenderer' object has no attribute 'layer'  <-- a REAL "
        "engine invariant failure, not a precondition"
    )


runner.optimize = _boom

from tuxghost.cli import main

print("main() RETURNED", main(sys.argv[1:]))
'''


def test_a_real_engine_bug_out_of_the_loop_is_not_laundered_into_a_refusal(
    tmp_path: Path,
) -> None:
    """The other half of the exit-code contract, and the half the plan
    left to a by-hand check: widening `_optimize`'s `except (ValueError,
    TypeError)` boundary to `except Exception` keeps every other test in
    this file green while turning a genuine engine bug into a tidy
    `refused: ...` and exit 2. Measured: with the boundary widened, this
    file's other 20 tests all still passed.

    So the contract's negative direction gets a real test too. An
    `AttributeError` out of `optimize()` must reach the process boundary
    UNCAUGHT: a traceback and a non-zero exit, not a refusal.
    """
    driver = tmp_path / "driver.py"
    driver.write_text(_LAUNDERING_DRIVER.format(root=str(ROOT)))
    proc = subprocess.run(
        [
            sys.executable, str(driver),
            "optimize",
            "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
            "--objective", "reach-tile",
            "--target", "spyder_paper_town.tmx:11,16",
            "--rounds", "1", "--patience", "1", "--max-rejections", "1",
            "--max-cost", "4000", "--out", str(tmp_path / "o.tuxghost"),
            "--run-dir", str(tmp_path / "run"),
        ],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy",
             "SDL_AUDIODRIVER": "dummy", "PYTHONHASHSEED": "0"},
        check=False,
    )
    assert proc.returncode != 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "Traceback" in proc.stderr
    assert "AttributeError" in proc.stderr
    assert "refused" not in proc.stdout and "refused" not in proc.stderr
    assert "main() RETURNED" not in proc.stdout, (
        "`main` returned instead of crashing: a real engine bug was "
        "laundered into an exit code"
    )


def test_the_claude_editor_is_the_only_one_that_touches_anthropic() -> None:
    """`_optimize` must gate the anthropic import on `--editor claude`,
    like `_agent` does after the 61f63aa residual fix: otherwise a
    genuine engine crash on `--editor mutation` has its traceback
    replaced by ModuleNotFoundError.

    Comment lines are stripped before grepping (Task 11): the plan's
    version of this test grepped the raw file, and it PASSED against a
    deliberately ungated implementation purely because the explanatory
    comment above the import quotes the very string being searched for.
    A source-text test that its own prose can satisfy is the "green but
    proves nothing" trap this project keeps a list of. The real check is
    `test_the_other_editors_run_with_anthropic_unimportable` below --
    behavioural, not textual; this one stays as a cheap, fast statement
    of the same intent."""
    code = "\n".join(
        line
        for line in (ROOT / "tuxghost" / "cli.py").read_text().splitlines()
        if not line.strip().startswith("#")
    )
    assert 'args.editor == "claude"' in code


#: A `sitecustomize` module (auto-imported by `site` for anything on
#: `PYTHONPATH`) that makes `import anthropic` fail no matter what is
#: installed in the venv. `make check`'s own environment deliberately does
#: not install the SDK; this venv DOES (the live capture needed it), so
#: without this shim no test in this repo can observe the gate at all.
_BLOCK_ANTHROPIC = '''\
import sys
from importlib.abc import MetaPathFinder


class _NoAnthropic(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "anthropic" or fullname.startswith("anthropic."):
            raise ModuleNotFoundError(
                "No module named 'anthropic' (blocked by sitecustomize)",
                name=fullname,
            )
        return None


sys.meta_path.insert(0, _NoAnthropic())
'''


def test_the_other_editors_run_with_anthropic_unimportable(tmp_path: Path) -> None:
    """The behavioural form of the gate, and the one that actually fails
    when the import is hoisted out of the `claude` branch.

    `--editor mutation` must complete a real run in an environment where
    `anthropic` cannot be imported at all -- which is `make check`'s
    environment. An ungated import turns this into a refusal (exit 2) or,
    worse, a `ModuleNotFoundError` raised while handling a genuine engine
    crash, replacing the real traceback with the wrong headline
    exception. That is precisely the defect commit 61f63aa fixed in
    `_agent`.
    """
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(_BLOCK_ANTHROPIC)
    blocked = {"PYTHONPATH": str(shim)}

    # Control: without this, the whole test is vacuous -- `anthropic` IS
    # installed in this venv, so a shim that silently failed to load
    # would leave the assertion below passing for the wrong reason.
    control = subprocess.run(
        [sys.executable, "-c", "import anthropic"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, **blocked},
        check=False,
    )
    assert control.returncode != 0, (
        "the sitecustomize shim did not block `import anthropic`, so this "
        "test could not observe the gate at all"
    )
    assert "blocked by sitecustomize" in control.stderr

    proc = _run(
        "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "4000", "--out", str(tmp_path / "o.tuxghost"),
        "--run-dir", str(tmp_path / "run"),
        extra_env=blocked,
    )
    assert proc.returncode == 0, proc.stderr
    assert "anthropic" not in proc.stderr


def test_the_claude_editor_refuses_when_anthropic_is_unimportable(
    tmp_path: Path,
) -> None:
    """The other half of the gate: `--editor claude` without the SDK is a
    REFUSAL (exit 2) with an actionable message, not an uncaught
    `ModuleNotFoundError` deep inside the loop (exit 1 = "diverged")."""
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(_BLOCK_ANTHROPIC)

    proc = _run(
        "--trace", str(GOLDEN), "--editor", "claude",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "4000", "--out", str(tmp_path / "o.tuxghost"),
        "--run-dir", str(tmp_path / "run"),
        extra_env={"PYTHONPATH": str(shim)},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "needs the anthropic SDK" in proc.stderr
    assert "Traceback" not in proc.stderr
