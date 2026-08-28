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

from tuxghost.agent.types import Action
from tuxghost.optimize.editors.replay import edits_from_json
from tuxghost.optimize.edits import Insert, Replace
from tuxghost.optimize.objective import ReachTile

ROOT = Path(__file__).resolve().parent.parent
#: The optimizer's parent (handoff item A1). `claude_town_1234` ended
#: inside a `DialogState` that swallowed appended input, so ReachTile's
#: distance term was dead against it; this one ends on a bare
#: `WorldState` at tile [16, 14] after 176 steps.
GOLDEN = ROOT / "tests" / "golden" / "scripted_town_1234.tuxghost"


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


def parent_digest_of_the_taint(taint: str, parent: Path) -> bool:
    """Whether `taint` names `parent`'s own final digest. A lineage that
    named some other trace's digest would be worse than one naming none:
    it would claim a derivation that never happened."""
    return json.loads(parent.read_text())["header"]["final_digest"] in taint


def test_a_mutation_run_succeeds_and_writes_its_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "best.tuxghost"
    run_dir = tmp_path / "run"
    proc = _run(
        "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:19,14",
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

    # The lineage taint, and specifically its SEED clause (whole-branch
    # review, Also-fix 3). It was built and never asserted anywhere in
    # the fast tier, so a lineage that dropped it -- or recorded the
    # wrong seed -- would have left this suite green while the one
    # durable record of HOW this trace was derived lost the only value
    # that makes a `mutation` run reproducible at all.
    taints = json.loads(out.read_text())["provenance"]["taints"]
    assert len(taints) == 1, taints
    assert "mutation" in taints[0] and "seed 7" in taints[0], taints[0]
    assert parent_digest_of_the_taint(taints[0], GOLDEN), taints[0]

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
        "--target", "spyder_paper_town.tmx:19,14",
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
        "--trace", "tests/golden/scripted_town_1234.tuxghost",
        "--editor", "mutation", "--seed", "7", "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:19,14",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "4000", "--out", str(tmp_path / "o.tuxghost"),
        "--run-dir", str(tmp_path / "run"),
    )
    assert proc.returncode == 0, proc.stderr


def _base(tmp_path: Path) -> dict[str, str]:
    return {
        "--trace": str(GOLDEN), "--editor": "mutation", "--seed": "7",
        "--objective": "reach-tile",
        "--target": "spyder_paper_town.tmx:19,14", "--rounds": "1",
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
        "--target", "spyder_paper_town.tmx:19,14",
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
        "--target", "spyder_paper_town.tmx:19,14",
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
            "--target", "spyder_paper_town.tmx:19,14",
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
        "--target", "spyder_paper_town.tmx:19,14",
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
        "--target", "spyder_paper_town.tmx:19,14",
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

    `--max-cost 100` against the 176-step parent, NOT the 400 this test
    used against the retired 442-step one: the whole scenario requires a
    ceiling BELOW the parent's own cost, and 400 now sits above it, which
    would leave the test passing while exercising nothing.
    """
    out = tmp_path / "best.tuxghost"
    proc = _run(
        "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:19,14",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "100", "--out", str(out),
        "--run-dir", str(tmp_path / "run"),
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    written = json.loads(out.read_text())
    # The parent won: nothing was proposable under a 100-step ceiling, so
    # every candidate was a rejected round and round 0 stood.
    assert written["header"]["step_count"] == 176
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
        "--target", "spyder_paper_town.tmx:19,14",
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
    # 188 = the parent's own 176 steps plus the inserted action's cost
    # (hold 8 + settle 4). It was 454 against the retired 442-step parent.
    assert rows[1]["digest"] and rows[1]["steps"] == 188, rows[1]
    assert len(rows[1]["edits"]) == 1

    info = json.loads((run_dir / "run.json").read_text())
    assert info["editor"] == editor
    assert info["seed"] is None
    assert info["model"] == (extra[1] if extra else None)


def test_a_goal_on_an_editor_that_reads_none_is_refused(tmp_path: Path) -> None:
    """The `--seed` refusal's missing twin. Only `--editor claude` reads
    a goal, so `--editor mutation --goal "..."` ran to completion having
    silently discarded the one flag that says what the user wanted.

    `run.json` was already honest about it -- it records `null`, because
    no goal reached any editor -- but a truthful record of a dropped flag
    is not the same as telling the user it was dropped, which is exactly
    the argument `tuxghost/cli.py` already makes for refusing `--seed` on
    a non-mutation editor.

    Exit 2, not 1: this is an anticipated precondition, the code this
    module pins for every other refusal.
    """
    proc = _run(
        "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
        "--goal", "wander northeast",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:19,14",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "4000", "--out", str(tmp_path / "o.tuxghost"),
        "--run-dir", str(tmp_path / "run"),
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "--goal" in proc.stderr, proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    # Refused BEFORE the engine boots, so nothing was written.
    assert not (tmp_path / "o.tuxghost").exists()


def test_a_goal_is_accepted_by_the_editor_that_does_read_one(
    tmp_path: Path,
) -> None:
    """The converse, so the refusal above cannot be over-broad: the same
    flag on `--editor claude` must still work. Without this, a refusal
    that rejected `--goal` unconditionally would pass the test above."""
    proc, calls = _claude_prompt(tmp_path, "--goal", "wander northeast")
    assert "main() RETURNED 0" in proc.stdout, (proc.stdout, proc.stderr)
    assert "wander northeast" in str(calls[0]["messages"])


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
            "--target", "spyder_paper_town.tmx:19,14",
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


#: Two edits IN ONE ROUND -- an insert and a replace -- so a single
#: candidate run produces a logged round containing both ops. Cheaper
#: than two rounds and it is the discrimination that matters, not the
#: round count.
_TWO_OP_EDITS_LINE = json.dumps([
    {"op": "insert", "index": 0, "action": {"button": 2, "hold": 8, "settle": 4}},
    {"op": "replace", "index": 0, "action": {"button": 8, "hold": 16, "settle": 0}},
])


def test_a_logged_round_round_trips_through_edits_from_json(
    tmp_path: Path,
) -> None:
    """WHOLE-BRANCH REVIEW, IMPORTANT 2 -- the auditability contract, end
    to end.

    The spec calls `ReplayEditor` "what makes an LLM-driven optimization
    auditable after the fact", and `--edits FILE` takes exactly the
    `[{"op": ..., "index": ..., "action": {...}}]` shape. That claim is
    only true if what `optimize.jsonl` WRITES can be READ back by
    `edits_from_json` -- and before this fix it could not: rounds were
    logged with bare `dataclasses.asdict`, which recurses by field, and
    `Insert`/`Replace` have identical field sets. Measured on the pre-fix
    checkout, a logged insert came back as

        {"index": 0, "action": {"button": 2, "hold": 8, "settle": 4}}

    indistinguishable from a replace, and fed to the project's own parser
    raised `ValueError: edit 0: unknown op None`.

    Asserted as EQUALITY against the edits that went in, not merely "it
    parsed": a serializer that labelled everything `"delete"` would also
    parse. The `op` sequence is asserted separately so an
    insert-rendered-as-replace fails here rather than passing an
    equality check by luck.
    """
    (tmp_path / "edits.jsonl").write_text(_TWO_OP_EDITS_LINE + "\n")
    run_dir = tmp_path / "run"
    proc = _run(
        "--trace", str(GOLDEN), "--editor", "scripted",
        "--edits", str(tmp_path / "edits.jsonl"),
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:19,14",
        "--rounds", "1", "--patience", "1", "--max-rejections", "1",
        "--max-cost", "4000", "--out", str(tmp_path / "best.tuxghost"),
        "--run-dir", str(run_dir),
    )
    assert proc.returncode == 0, proc.stderr

    rows = [
        json.loads(line)
        for line in (run_dir / "optimize.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # Round 1 must be a real scored candidate, or the edits never reached
    # the log at all and everything below is vacuous.
    assert rows[1]["index"] == 1 and rows[1]["rejected_reason"] is None, rows[1]

    assert [edit["op"] for edit in rows[1]["edits"]] == ["insert", "replace"]
    assert edits_from_json(rows[1]["edits"]) == (
        Insert(0, Action(2, 8, 4)),
        Replace(0, Action(8, 16, 0)),
    )


#: A FAKE `anthropic` module -- like `_ANTHROPIC_ARM_DRIVER`'s, but with a
#: working `Anthropic` client that appends every `messages.create` kwargs
#: dict to a file and answers STOP. STOP (an empty `edits` list) is what
#: keeps this test to ONE prompt and two engine passes: round 0's parent
#: seal, then the winner's re-seal. Braces are doubled throughout because
#: this template goes through `str.format`.
_CLAUDE_PROMPT_DRIVER = '''\
import json
import sys
import types

sys.path.insert(0, {root!r})

CAPTURED = {captured!r}

fake = types.ModuleType("anthropic")


class AnthropicError(Exception):
    pass


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text):
        self.content = [_Block(text)]


class _Messages:
    def create(self, **kwargs):
        with open(CAPTURED, "a") as handle:
            handle.write(json.dumps(kwargs, default=str) + "\\n")
        return _Response('```json\\n{{"edits": []}}\\n```')


class Anthropic:
    def __init__(self, *args, **kwargs):
        self.messages = _Messages()


fake.AnthropicError = AnthropicError
fake.Anthropic = Anthropic
sys.modules["anthropic"] = fake

from tuxghost.execute import _bootstrap_vendored_tuxemon

_bootstrap_vendored_tuxemon()

from tuxghost.cli import main

print("main() RETURNED", main(sys.argv[1:]))
'''


def _claude_prompt(
    tmp_path: Path, *extra: str
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
    """Run `--editor claude` against a stub SDK and return what the CLI
    actually sent."""
    captured = tmp_path / "captured.jsonl"
    driver = tmp_path / "driver.py"
    driver.write_text(
        _CLAUDE_PROMPT_DRIVER.format(root=str(ROOT), captured=str(captured))
    )
    proc = subprocess.run(
        [
            sys.executable, str(driver),
            "optimize",
            "--trace", str(GOLDEN), "--editor", "claude",
            "--objective", "reach-tile",
            "--target", "spyder_paper_town.tmx:19,14",
            "--rounds", "1", "--patience", "1", "--max-rejections", "1",
            "--max-cost", "4000", "--out", str(tmp_path / "o.tuxghost"),
            "--run-dir", str(tmp_path / "run"),
            *extra,
        ],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy",
             "SDL_AUDIODRIVER": "dummy", "PYTHONHASHSEED": "0"},
        check=False,
    )
    calls = [
        json.loads(line)
        for line in (
            captured.read_text().splitlines() if captured.exists() else []
        )
        if line.strip()
    ]
    return proc, calls


def test_an_explicit_goal_reaches_optimizes_run_json(tmp_path: Path) -> None:
    """HANDOFF ITEM A3. `agent`'s `run.json` carries `goal`; `optimize`'s
    did not. A DERIVED goal is reconstructible after the fact from
    `objective` + `target`, which this same file records -- but an
    EXPLICIT `--goal` is not reconstructible from anything, so a
    `--editor claude` run's actual instruction to the model survived
    nowhere once the process exited. The run directory is the only
    durable record of how a run was configured; a field that is not in
    it is gone.

    Asserted with a goal whose text appears in NO other field, so it
    cannot be satisfied by the derived string that `_derived_goal` builds
    out of the objective and target.
    """
    explicit = "wander northeast and do not talk to anyone"
    proc, _ = _claude_prompt(tmp_path, "--goal", explicit)
    assert "main() RETURNED 0" in proc.stdout, (proc.stdout, proc.stderr)

    info = json.loads((tmp_path / "run" / "run.json").read_text())
    assert info["goal"] == explicit, info


def test_a_derived_goal_is_recorded_as_the_one_the_model_was_sent(
    tmp_path: Path,
) -> None:
    """The companion to the test above. With no `--goal`, the CLI derives
    one and sends THAT to the model, so recording `args.goal` verbatim
    would write `""` into `run.json` for a run whose prompt did carry a
    goal -- true to the command line and false about the run. What is
    recorded is the string the editor was actually built with.
    """
    proc, calls = _claude_prompt(tmp_path)
    assert "main() RETURNED 0" in proc.stdout, (proc.stdout, proc.stderr)

    info = json.loads((tmp_path / "run" / "run.json").read_text())
    assert info["goal"], info
    # The same string the model was sent, not merely a non-empty one.
    assert info["goal"] in str(calls[0]["messages"]), info["goal"]


def test_editors_that_read_no_goal_record_none_rather_than_empty_string(
    tmp_path: Path,
) -> None:
    """`--editor mutation` hands no goal to any editor, so `null` is the
    true value here. It is asserted as `None` specifically, not merely
    falsy: `""` would read as "the run had an empty goal" where `null`
    reads as "no goal was involved", and only the second is accurate.
    """
    out = tmp_path / "best.tuxghost"
    run_dir = tmp_path / "run"
    proc = _run(
        "--trace", str(GOLDEN), "--editor", "mutation", "--seed", "7",
        "--objective", "reach-tile",
        "--target", "spyder_paper_town.tmx:19,14",
        "--rounds", "2", "--patience", "2", "--max-rejections", "2",
        "--max-cost", "4000", "--out", str(out), "--run-dir", str(run_dir),
    )
    assert proc.returncode == 0, proc.stderr

    info = json.loads((run_dir / "run.json").read_text())
    assert "goal" in info, info
    assert info["goal"] is None, info


def test_the_target_reaches_the_claude_editors_prompt(tmp_path: Path) -> None:
    """WHOLE-BRANCH REVIEW, IMPORTANT 1. `_optimize` built
    `ClaudeEditor(model=model)` and there was no `--goal` on this parser,
    so `ClaudeEditor.goal` was `""`, `build_prompt` skipped the `Goal:`
    line entirely, and every real `--editor claude` run sent the action
    list, the end tile, the checkpoints and a bare
    `Score (higher is better): [0.0, -3.0, -176.0]` -- no target, no
    legend for the three numbers.

    The mechanism (a goal that IS passed reaches the prompt) was already
    pinned by `tests/test_optimize_claude.py::test_the_goal_reaches_the
    _prompt`. What nothing pinned, and what this asserts, is that the CLI
    passes one. This is the same class of defect the comment at
    `--goal never reached the run` in `tuxghost/cli.py` records for
    `agent`, caught a second time.
    """
    proc, calls = _claude_prompt(tmp_path)
    assert "main() RETURNED 0" in proc.stdout, (proc.stdout, proc.stderr)
    assert len(calls) == 1, calls

    prompt = str(calls[0]["messages"])
    # FINAL RESIDUALS, ITEM 3. `assert "spyder_paper_town.tmx" in prompt`
    # was VACUOUS: `build_prompt` emits an "Ended on map
    # 'spyder_paper_town.tmx'" line unconditionally, from the candidate's
    # own final state, so the map name is in the prompt whether or not a
    # goal ever reached it -- the assertion passed against the exact
    # defect it was written to catch. (The tile half was real: "(19, 14)"
    # appears only in the derived goal.)
    #
    # Fixed by asserting against the GOAL LINE, isolated, rather than
    # against the whole prompt.
    goal_lines = [
        line for line in prompt.replace("\\n", "\n").splitlines()
        if line.lstrip().startswith("Goal:")
    ]
    assert len(goal_lines) == 1, (goal_lines, prompt)
    goal = goal_lines[0]
    assert "spyder_paper_town.tmx" in goal, goal
    assert "(19, 14)" in goal, goal
    # The legend for the score tuple, read from `ReachTile.TERMS` itself
    # rather than written out as literals: this assertion's job is that
    # the CLI PASSES the legend, not that the legend says any particular
    # word, and a literal here silently went stale when term 0 was
    # renamed `-off_target_map` (handoff item A2). Every term is checked,
    # not two of three, so a legend truncated in the middle fails.
    assert ReachTile.TERMS, "an empty legend would make this vacuous"
    for term in ReachTile.TERMS:
        assert term, "an empty term name would make this vacuous"
        assert term in prompt, (term, prompt)

    # Also-fix 2, in the one place it is observable from outside: the
    # button values in the SYSTEM prompt are read from
    # `tuxemon.platform.const.buttons`, not written out as literals, so a
    # remapping upstream cannot leave this editor confidently proposing
    # the wrong direction.
    system = str(calls[0]["system"])
    assert "UP=1" in system and "LEFT=4" in system, system

    # Also-fix 8, and this run is exactly the case that exposed it: the
    # editor answered STOP, so ZERO edits were applied -- and the lineage
    # taint said "over 1 round(s)", because the count was
    # `len(result.rounds) - 1` and the STOP round is deliberately
    # appended to the log. The taint is this branch's sole lineage
    # record, so a count that overstates it by one is worth one line.
    written = json.loads((tmp_path / "o.tuxghost").read_text())
    assert "over 0 round(s)" in written["provenance"]["taints"][0], (
        written["provenance"]["taints"]
    )


def test_an_explicit_goal_overrides_the_derived_one(tmp_path: Path) -> None:
    """`--goal` mirrors `agent`'s and says something `--target` cannot.
    The derived default must not survive alongside it, or a caller who
    typed a goal gets two."""
    proc, calls = _claude_prompt(
        tmp_path, "--goal", "get there without entering the tall grass"
    )
    assert "main() RETURNED 0" in proc.stdout, (proc.stdout, proc.stderr)
    prompt = str(calls[0]["messages"])
    assert "without entering the tall grass" in prompt, prompt
    assert "reach tile" not in prompt, prompt


def test_max_costs_help_states_that_the_parent_is_exempt(tmp_path: Path) -> None:
    """Also-fix 5. `--max-cost 100` against the 176-step parent exits 0
    and writes a 176-step trace -- deliberate (round 0 seals the parent
    without the budget, or the run would have nothing to compare
    against), pinned by
    `test_a_parent_over_the_max_cost_still_seals_the_winner`, and
    documented in docs/STATUS.org. From `--help` alone it was surprising:
    a flag called `--max-cost` that a written artifact openly exceeds.

    A `--help` assertion and not a prose review: the help string is the
    one place a user meets this flag before running it.
    """
    del tmp_path
    proc = _run("--help")
    assert proc.returncode == 0, proc.stderr
    # Collapse argparse's line wrapping before matching.
    helptext = " ".join(proc.stdout.split())
    assert "EDITOR MAY PROPOSE" in helptext, helptext
    assert "parent-cost trace" in helptext, helptext
