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
        [sys.executable, "-m", "tuxghost.cli", "optimize", *args],
        capture_output=True,
        text=True,
        # `cwd` is a real parameter, not decoration: `_resolve_paths` must
        # resolve every path argument against the CALLER's cwd, and a run
        # launched from somewhere that is neither the repo root nor the
        # vendored tree is the only way to prove a RELATIVE `--edits`
        # path resolves there too (see
        # `test_an_edits_file_drives_both_editors_that_read_one`).
        cwd=str(cwd) if cwd is not None else ROOT,
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

    # Folded in from what used to be a separate test issuing a
    # byte-identical invocation (review round 1): the written trace must
    # be the trace the loop SCORED -- its digest equals the digest logged
    # for `best_round`. What this catches is writing the wrong round's
    # trace, or a re-seal that no longer reaches what the loop recorded
    # (the CLI refuses that case rather than writing it). What it does
    # NOT catch is a patched-`Provenance` copy in place of a real
    # re-seal; nothing in this suite does, and nothing can from the
    # artifact alone -- see the note above `REFUSALS`.
    scored = next(r for r in rows if r["index"] == info["best_round"])
    assert json.loads(out.read_text())["header"]["final_digest"] == (
        scored["digest"]
    ), "the written trace is not the trace the loop scored"


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("TUXGHOST_RUN_SLOW"),
    reason="two full engine passes: the optimization plus a `verify` re-run",
)
def test_the_written_trace_is_a_verifiable_offline_agent_trace(tmp_path: Path) -> None:
    """The end-to-end property: what `optimize` writes is a real
    `offline-agent` trace carrying exactly one lineage taint, and
    `tuxghost verify` replays it to the digest it recorded.

    Slow-tier (review round 1): it was the single most expensive test in
    the module at 17.25s, because it runs the whole optimization AND a
    second full engine pass through `verify`. The fast tier keeps the
    cheaper `--max-cost` regression below, which also ends in a `verify`.
    """
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


def _flags(
    tmp_path: Path,
    *,
    editor: str,
    edits: Path | None = None,
    model: str | None = None,
    seed: str | None = None,
) -> list[str]:
    """A full, valid argument list with the editor-specific flags varied.

    Separate from `_base` because `_base` hardcodes `--editor mutation
    --seed 7`, and every case below is about an editor that takes
    `--edits`/`--model` and must NOT be given a seed.
    """
    flags = [
        "--trace", str(GOLDEN), "--editor", editor,
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "4000", "--out", str(tmp_path / "o.tuxghost"),
        "--run-dir", str(tmp_path / "run"),
    ]
    if edits is not None:
        flags += ["--edits", str(edits)]
    if model is not None:
        flags += ["--model", model]
    if seed is not None:
        flags += ["--seed", seed]
    return flags


# What NOTHING in this file pins, stated plainly rather than implied by
# a test name (review round 1, Important 4): a re-seal through `Recorder`
# is INDISTINGUISHABLE from the loop's own sealed trace with `taints`
# patched in. `Recorder.finish` builds every field deterministically --
# `provenance.recorded_at` is never even set (`tuxghost/record.py`), and
# the header is a pure function of the pinned seed/clock_epoch and the
# state reached -- so two seals of the same script produce byte-identical
# traces. No assertion over the artifact can tell the two apart; the
# reason to re-seal is that `Recorder` stays this project's ONE trace
# writer, plus the digest-agreement re-check the CLI performs. Verified
# by reading `record.py`, not assumed.


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


# A comment-stripped source grep for `args.editor == "claude"` used to
# live here. Deleted in review round 1: it was not vacuous, but it was
# only FULL-LINE-comment-proof -- a trailing inline comment or a
# docstring reflow satisfies it -- and the two behavioural tests below
# are strictly stronger. A weak test next to a strong one covering the
# same property is worse than no weak test, because it invites trusting
# the cheap one.


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


def test_a_parent_over_the_max_cost_still_seals_the_winner(tmp_path: Path) -> None:
    """CRITICAL, review round 1. `optimize()` seals round 0 -- the parent
    -- WITHOUT `max_cost`, by design and pinned by Task 7's
    `test_round_zero_is_sealed_without_max_cost`: refusing the baseline
    would leave the run with nothing to compare against. So the winning
    script can legally be an over-budget PARENT script, and the CLI's
    re-seal sits outside the `except (ValueError, TypeError)` boundary --
    so re-imposing the proposal budget there raised `OverBudget`
    uncaught. Measured before the fix: `--max-cost 400` against the
    442-step parent exited 1 with a traceback, a plain flag value
    producing the one exit code no precondition may produce.

    `max_cost` bounds what an EDITOR MAY PROPOSE, not what the parent
    already is; `_prepare` enforces it on every accepted candidate, which
    is where it means something.
    """
    out = tmp_path / "best.tuxghost"
    proc = _run(
        "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "400", "--out", str(out),
        "--run-dir", str(tmp_path / "run"),
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    written = json.loads(out.read_text())
    # The parent won: nothing cheaper than 442 steps was proposable under
    # a 400-step ceiling, so every candidate was a rejected round.
    assert written["header"]["step_count"] == 442
    assert "Traceback" not in proc.stderr

    verify = subprocess.run(
        [sys.executable, "-m", "tuxghost.cli", "verify", str(out)],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy",
             "SDL_AUDIODRIVER": "dummy", "PYTHONHASHSEED": "0"},
        check=False,
    )
    assert verify.returncode == 0, verify.stderr


#: One round of edits, in the JSONL shape `edits_from_json` accepts:
#: one JSON list per line, each element an `{op, index, action}` object.
_EDITS_LINE = json.dumps(
    [{"op": "insert", "index": 0, "action": {"button": 2, "hold": 8, "settle": 4}}]
)


@pytest.mark.parametrize(
    "editor,extra",
    [
        ("scripted", ()),
        ("replay", ("--model", "a-recorded-transcript")),
    ],
)
def test_an_edits_file_drives_both_editors_that_read_one(
    tmp_path: Path, editor: str, extra: tuple[str, ...]
) -> None:
    """The two happy paths nothing exercised (review round 1, Important
    3). Both `--editor scripted` (a per-line
    `edits_from_json(json.loads(line))` comprehension in `_optimize`) and
    `--editor replay` (`ReplayEditor(args.edits)`) were wired by
    inspection only, so a swapped argument or a wrong exception tuple
    would have shipped green.

    `--edits` is passed RELATIVE, with `cwd` set to `tmp_path`, which is
    also the only exercise of `_PATH_ARGS["optimize"]`'s `"edits"` entry:
    drop that entry and this path resolves against the vendored
    `tuxemon/` directory the process chdirs into, and the run refuses.
    """
    (tmp_path / "edits.jsonl").write_text(_EDITS_LINE + "\n")
    run_dir = tmp_path / "run"
    proc = _run(
        "--trace", str(GOLDEN), "--editor", editor, *extra,
        "--edits", "edits.jsonl",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:11,16",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "4000", "--out", str(tmp_path / "best.tuxghost"),
        "--run-dir", str(run_dir),
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr

    rows = [
        json.loads(line)
        for line in (run_dir / "optimize.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # Round 1 must be a REAL scored candidate, not a rejected round: that
    # is what proves the edits file reached the editor and produced a
    # candidate the engine actually ran.
    assert len(rows) == 2, rows
    assert rows[1]["index"] == 1
    assert rows[1]["rejected_reason"] is None, rows[1]
    assert rows[1]["digest"] and rows[1]["steps"] == 454, rows[1]
    assert len(rows[1]["edits"]) == 1

    info = json.loads((run_dir / "run.json").read_text())
    assert info["editor"] == editor
    assert info["seed"] is None
    assert info["model"] == (extra[1] if extra else None)


def test_replay_without_a_model_is_refused(tmp_path: Path) -> None:
    """A transcript carries no model of its own, so recording a default
    would misattribute provenance -- the same reasoning `--policy replay`
    follows. Refused before the engine boots."""
    (tmp_path / "edits.jsonl").write_text(_EDITS_LINE + "\n")
    proc = _run(*_flags(tmp_path, editor="replay", edits=tmp_path / "edits.jsonl"))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "requires --model" in proc.stderr


@pytest.mark.parametrize("editor", ["scripted", "replay"])
def test_a_missing_edits_file_is_refused(tmp_path: Path, editor: str) -> None:
    extra = {"model": "m"} if editor == "replay" else {}
    proc = _run(
        *_flags(
            tmp_path, editor=editor, edits=tmp_path / "nope.jsonl", **extra
        )
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "cannot read" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("editor", ["scripted", "replay"])
def test_a_malformed_edits_file_is_refused(tmp_path: Path, editor: str) -> None:
    """Both editors validate the WHOLE file before the game boots, and
    both map a bad one to a refusal. `scripted` fails in
    `json.loads`/`edits_from_json`, `replay` inside
    `ReplayEditor.__init__` -- different code, same exit code."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('[{"op": "nope", "index": 0}]\n')
    extra = {"model": "m"} if editor == "replay" else {}
    proc = _run(*_flags(tmp_path, editor=editor, edits=bad, **extra))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "unknown op" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_a_trace_that_is_valid_json_but_not_a_trace_is_refused(
    tmp_path: Path,
) -> None:
    """The `Refused` half of `_read_trace_or_refuse`, as distinct from the
    `json.JSONDecodeError` half above."""
    not_a_trace = tmp_path / "notatrace.tuxghost"
    not_a_trace.write_text('{"hello": 1}')
    flags = _base(tmp_path)
    flags["--trace"] = str(not_a_trace)
    proc = _run(*[x for k, v in flags.items() for x in (k, v)])
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "format_version" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("editor", ["scripted", "replay", "claude"])
def test_a_seed_the_chosen_editor_cannot_use_is_refused(
    tmp_path: Path, editor: str
) -> None:
    """Review round 1, Important 5: only `mutation` draws from a seed, so
    `--editor replay --seed 7` used to write `derived by offline-agent
    (replay, seed 7)` into the winner's lineage taint and `"seed": 7`
    into `run.json` -- a seed no editor ever used, recorded as if it had
    produced the run. Refused rather than silently dropped: dropping the
    taint clause alone would leave the bogus seed in `run.json`.

    `--editor claude` is included and never touches the `anthropic` SDK
    here: this check runs before any editor is constructed.
    """
    (tmp_path / "edits.jsonl").write_text(_EDITS_LINE + "\n")
    extra = {"model": "m"} if editor in ("replay", "claude") else {}
    proc = _run(
        *_flags(
            tmp_path,
            editor=editor,
            edits=tmp_path / "edits.jsonl",
            seed="7",
            **extra,
        )
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "has no use for --seed" in proc.stderr


#: Same shape as `_LAUNDERING_DRIVER`, plus a FAKE `anthropic` module in
#: `sys.modules` so the gated arm has an `AnthropicError` to `isinstance`
#: against without the real SDK (which `make check` does not install).
_ANTHROPIC_ARM_DRIVER = '''\
import sys
import types

sys.path.insert(0, {root!r})

fake = types.ModuleType("anthropic")


class AnthropicError(Exception):
    pass


fake.AnthropicError = AnthropicError
sys.modules["anthropic"] = fake

from tuxghost.execute import _bootstrap_vendored_tuxemon

_bootstrap_vendored_tuxemon()

import tuxghost.optimize.runner as runner


def _boom(*args, **kwargs):
    raise {raiser}


runner.optimize = _boom

from tuxghost.cli import main

print("main() RETURNED", main(sys.argv[1:]))
'''


def _anthropic_arm(tmp_path: Path, raiser: str) -> subprocess.CompletedProcess[str]:
    driver = tmp_path / "driver.py"
    driver.write_text(
        _ANTHROPIC_ARM_DRIVER.format(root=str(ROOT), raiser=raiser)
    )
    return subprocess.run(
        [
            sys.executable, str(driver),
            "optimize",
            "--trace", str(GOLDEN), "--editor", "claude",
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


def test_an_anthropic_api_error_is_a_refusal_not_a_divergence(
    tmp_path: Path,
) -> None:
    """Review round 1, Important 2. `ClaudeEditor.propose` calls the SDK
    from inside `runner._propose`, whose boundary is
    `(ValueError, TypeError)` -- and every error that call can raise
    derives from `anthropic.AnthropicError(Exception)`. So a
    `RateLimitError` is neither a rejected round nor caught by
    `_optimize`'s narrow arm, and without the gated arm it exits 1 =
    "diverged" for what is plainly an API problem."""
    proc = _anthropic_arm(
        tmp_path, 'AnthropicError("429 rate limited (fake SDK)")'
    )
    assert "main() RETURNED 2" in proc.stdout, (proc.stdout, proc.stderr)
    assert "call to the Anthropic API failed" in proc.stderr
    assert "429 rate limited (fake SDK)" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_a_non_anthropic_error_on_the_claude_editor_still_crashes(
    tmp_path: Path,
) -> None:
    """The other side of that arm: it must widen the contract for
    `AnthropicError` ONLY. A plain `RuntimeError` -- a genuine engine or
    programming failure -- has to reach the process boundary uncaught,
    even on `--editor claude`, or the `except Exception` arm becomes the
    laundering the narrow boundary exists to prevent."""
    proc = _anthropic_arm(
        tmp_path, 'RuntimeError("a real bug, not an API problem")'
    )
    assert proc.returncode != 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "Traceback" in proc.stderr
    assert "RuntimeError: a real bug, not an API problem" in proc.stderr
    assert "refused" not in proc.stderr
    assert "main() RETURNED" not in proc.stdout
