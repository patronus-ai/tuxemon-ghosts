"""Pins `tuxghost.cli.main`'s contract.

Two defects proven against an earlier, deleted CLI (`tuxghost/execute.py`'s
`_main`, removed in task-12 fix round 1) must not be reproduced here, and
are pinned by dedicated tests below rather than only exercised in passing:

  * A relative trace path must resolve against the CALLER's cwd, not the
    vendored `tuxemon/` directory `_bootstrap_vendored_tuxemon` chdirs
    into. `main()` called in-process can never actually exercise this --
    `tests/conftest.py` already chdirs the whole pytest session into
    `tuxemon/` before any test runs, so there is no "caller's cwd" left
    to get wrong. `test_relative_paths_resolve_against_the_run_directory`
    below is a real subprocess, launched with its own cwd set to a plain
    `tmp_path` with no relationship to the vendored tree, and every path
    argument passed relative.
  * A missing/unreadable trace file must exit 2 ("refused"), never the
    uncaught-exception exit 1 Python produces for an unhandled
    `FileNotFoundError`. `test_process_level_exit_codes_are_0_1_and_2`
    below checks all three real exit codes by invoking the CLI as an
    actual subprocess and reading `returncode`, not by calling `main()`
    in-process and trusting its return value alone.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tuxghost.cli import main


def test_execute_and_verify_together_are_refused() -> None:
    """`execute --verify` would blend two different operations together;
    refused at parse time, before the (nonexistent) trace path is ever
    touched -- this must return 2 regardless of whether `x.tuxghost`
    exists."""
    assert main(["execute", "x.tuxghost", "--verify"]) == 2


def test_unknown_command_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_execute_does_not_expose_a_flag_that_would_override_the_header(
    tmp_path: Path,
) -> None:
    """The trace's own header (seed, clock_epoch, mod_version, step_rate,
    step_count) is the ONLY input `execute`/`verify` use -- see
    `tuxghost.execute.execute`'s docstring. No flag here may let a caller
    silently override any of it: `execute`/`verify` simply do not define
    such flags, so argparse itself refuses an attempt (an unrecognized
    argument) rather than the value being accepted and quietly ignored."""
    with pytest.raises(SystemExit):
        main(["execute", str(tmp_path / "x.tuxghost"), "--seed", "5"])
    with pytest.raises(SystemExit):
        main(["verify", str(tmp_path / "x.tuxghost"), "--clock-epoch", "1"])


def test_info_refuses_a_missing_clock_epoch(tmp_path: Path) -> None:
    path = tmp_path / "bad.tuxghost"
    path.write_text(json.dumps({"format_version": 1, "header": {"seed": 1}}))
    assert main(["info", str(path)]) == 2


def test_info_refuses_a_missing_trace_file(tmp_path: Path) -> None:
    """The in-process counterpart to the missing-file hard requirement:
    `tuxghost.trace.read`'s bare `Path.read_text()` raises
    `FileNotFoundError` (an `OSError`), which `_read_trace_or_refuse` must
    map to 2, not let propagate as an uncaught exception."""
    assert main(["info", str(tmp_path / "does-not-exist.tuxghost")]) == 2


def _record_trace(
    tmp_path: Path, name: str = "recorded.tuxghost", *, step_count: int = 5
) -> Path:
    """Build a real, self-consistent trace via the recorder and write it
    to `tmp_path / name`. Used by the tests below that need an actual
    trace file rather than a hand-built dict."""
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.loop import run_steps
    from tuxghost.record import Recorder
    from tuxghost.trace import write

    seed_all(1234)
    pin_clock(1787694000)
    client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    )
    run_steps(client, step_count)
    trace = recorder.finish(step_count=step_count)
    path = tmp_path / name
    write(trace, path)
    return path


def test_execute_prints_the_final_digest_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _record_trace(tmp_path)
    assert main(["execute", str(path)]) == 0
    out = capsys.readouterr().out.strip()
    assert out, "execute must print the digest it reached"


def test_checkpoint_flag_prints_intermediate_digests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--checkpoint N` documents "digest every N steps instead of only at
    the end", but until this test existed nothing pinned that
    `tuxghost.execute.execute`'s own `result.checkpoints` (already
    populated and tested at the library layer, see
    `tests/test_execute.py
    ::test_execute_populates_checkpoints_at_the_given_interval`) ever
    reached the CLI's stdout -- `_execute` discarded it and printed only
    `final_digest`. Ten steps at `--checkpoint 3` must yield checkpoints
    at 3, 6, 9, each printed before the final digest line, and the final
    digest must still be the last line for scripts that only read that
    one."""
    path = _record_trace(tmp_path, step_count=10)
    assert main(["execute", str(path), "--checkpoint", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4, lines
    assert lines[0].startswith("checkpoint 3: ")
    assert lines[1].startswith("checkpoint 6: ")
    assert lines[2].startswith("checkpoint 9: ")
    # The final digest, unprefixed, is still the LAST line -- unchanged
    # contract for anything that only reads that one (see
    # `test_execute_prints_the_final_digest_and_returns_zero`).
    assert not lines[3].startswith("checkpoint")
    assert lines[3]


def test_checkpoint_defaults_to_no_intermediate_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Negative control: without `--checkpoint`, `execute` must print
    exactly one line (the final digest) -- otherwise the positive test
    above could pass even if checkpoints were printed unconditionally."""
    path = _record_trace(tmp_path, step_count=10)
    assert main(["execute", str(path)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1, lines


def test_execute_refuses_a_trace_missing_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`tuxghost.trace.Trace.model_validate` raises
    `pydantic.ValidationError` for a structurally malformed trace, e.g.
    one missing its required `provenance` block. Before this fix,
    `_read_trace_or_refuse` did not catch that exception type, so it
    propagated out of `main()` entirely -- an uncaught exception exits 1
    ("diverged"), colliding with the unrelated meaning `verify`/`execute`
    give that code. A malformed trace file must be REFUSED (exit 2), the
    same as a missing file or bad JSON."""
    path = _record_trace(tmp_path)
    data = json.loads(path.read_text())
    del data["provenance"]
    path.write_text(json.dumps(data))

    assert main(["execute", str(path)]) == 2
    assert "refused" in capsys.readouterr().err.lower()


def test_verify_returns_zero_for_a_self_consistent_trace(tmp_path: Path) -> None:
    path = _record_trace(tmp_path)
    assert main(["verify", str(path)]) == 0


def test_verify_returns_one_for_a_tampered_final_digest(tmp_path: Path) -> None:
    from tuxghost.trace import read, write

    path = _record_trace(tmp_path)
    trace = read(path)
    tampered = trace.model_copy(
        update={
            "header": trace.header.model_copy(
                update={"final_digest": "sha256:" + "0" * 64}
            )
        }
    )
    tampered_path = tmp_path / "tampered.tuxghost"
    write(tampered, tampered_path)
    assert main(["verify", str(tampered_path)]) == 1


def test_info_allow_mismatch_downgrades_a_version_mismatch(tmp_path: Path) -> None:
    """`--allow-mismatch` must actually reach `tuxghost.trace.read`: a
    `mod_version` mismatch refuses without the flag and is downgraded
    (tainted, not refused) with it."""
    from tuxghost.trace import read, write

    path = _record_trace(tmp_path, step_count=0)
    trace = read(path)
    mismatched = trace.model_copy(
        update={"header": trace.header.model_copy(update={"mod_version": "not-this-build"})}
    )
    mismatched_path = tmp_path / "mismatched.tuxghost"
    write(mismatched, mismatched_path)

    assert main(["info", str(mismatched_path)]) == 2
    assert main(["info", str(mismatched_path), "--allow-mismatch"]) == 0


def test_compare_reports_a_human_recorder_as_a_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.record import Recorder
    from tuxghost.trace import write

    seed_all(1234)
    pin_clock(1787694000)
    _client, session = build_client(seed=1234, clock_epoch=1787694000)
    trace = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="human"
    ).finish(step_count=10)
    a, b = tmp_path / "a.tuxghost", tmp_path / "b.tuxghost"
    write(trace, a)
    write(trace, b)

    assert main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "finding:" in out and "human loop" in out
    assert "traces agree" in out


def test_compare_reports_a_tainted_trace_as_a_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A trace `read()` had to downgrade (a `mod_version`/`step_rate`
    mismatch, under `--allow-mismatch`) carries `"trace_mismatch"` in its
    `Provenance.taints` -- `compare` must surface that as a finding, not
    silently drop it, and it is NOT a divergence: the two traces here are
    otherwise identical, so the overall result must still be 0."""
    from tuxghost.trace import read, write

    path = _record_trace(tmp_path, name="clean.tuxghost", step_count=0)
    trace = read(path)
    mismatched = trace.model_copy(
        update={"header": trace.header.model_copy(update={"mod_version": "not-this-build"})}
    )
    a = tmp_path / "a.tuxghost"
    write(mismatched, a)
    b = tmp_path / "b.tuxghost"
    write(trace, b)

    assert main(["compare", str(a), str(b), "--allow-mismatch"]) == 0
    out = capsys.readouterr().out
    assert "finding: a is tainted: trace_mismatch" in out
    assert "traces agree" in out


def test_compare_returns_one_when_initial_state_differs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A control for the two `compare` tests above, per this project's own
    process rule: a comparison that only ever reports "agree" would be as
    worthless as a digest that never discriminates. Two traces recorded
    from different seeds diverge in `initial_state` (different seeded ids)
    immediately."""
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.record import Recorder
    from tuxghost.trace import write

    def record(seed: int) -> Path:
        seed_all(seed)
        pin_clock(1787694000)
        _client, session = build_client(seed=seed, clock_epoch=1787694000)
        trace = Recorder(
            session, seed=seed, clock_epoch=1787694000, recorder="offline-agent"
        ).finish(step_count=0)
        path = tmp_path / f"seed-{seed}.tuxghost"
        write(trace, path)
        return path

    a = record(1234)
    b = record(99)

    assert main(["compare", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "initial_state differs at" in out


def test_record_round_trips_through_verify(tmp_path: Path) -> None:
    from tuxghost.boot import build_client, snapshot_save
    from tuxghost.determinism import seed_all

    seed_all(1234)
    _client, session = build_client(seed=1234)
    save_path = tmp_path / "save.json"
    save_path.write_text(snapshot_save(session).model_dump_json())

    out_path = tmp_path / "recorded.tuxghost"
    assert (
        main(
            [
                "record",
                str(out_path),
                "--from-save",
                str(save_path),
                "--seed",
                "1234",
                "--clock-epoch",
                "1787694000",
                "--steps",
                "5",
            ]
        )
        == 0
    )
    assert main(["verify", str(out_path)]) == 0


def test_record_passes_clock_epoch_through_to_boot_from_save(tmp_path: Path) -> None:
    """Regression: a first draft of `_record`, mirroring the task-13
    brief's own sample, called `pin_clock` directly and then
    `boot_from_save(save_data, seed=args.seed)` WITHOUT threading
    `clock_epoch` through. `boot_from_save`'s own `clock_epoch` parameter
    is what resets the session's elapsed-time bookkeeping
    (`SessionSave.duration`/`total_playtime`/`start_time`) onto the pinned
    epoch (see its docstring); omitting it leaves those fields on whatever
    real wall-clock reading was live when the process-wide session
    singleton was first constructed. `tuxghost.digest.state_of` never
    reads `session_state`, so `test_record_round_trips_through_verify`
    above cannot catch this -- confirmed by deliberately reintroducing the
    bug and observing this test fail (see the task report). Mirrors
    `tests/test_execute.py
    ::test_execute_passes_clock_epoch_through_to_boot_from_save`."""
    from datetime import UTC, datetime

    from tuxemon.core.clock import set_epoch
    from tuxemon.session import local_session

    from tuxghost.boot import build_client, snapshot_save
    from tuxghost.determinism import seed_all

    epoch = 1787694000
    expected_start_time = (
        datetime.fromtimestamp(epoch, tz=UTC)
        .replace(tzinfo=None)
        .strftime("%Y-%m-%d %H:%M")
    )

    seed_all(4321)
    _client, session = build_client(seed=4321, clock_epoch=epoch)
    save_path = tmp_path / "save.json"
    save_path.write_text(snapshot_save(session).model_dump_json())

    # Dirty the singleton's clock bookkeeping between snapshotting the
    # save and running `record` -- otherwise the assertion below would
    # pass even if `record` never pinned anything, since it might already
    # happen to hold the right values from the `build_client` call above.
    set_epoch(None)
    snapshot_save(session)

    out_path = tmp_path / "out.tuxghost"
    assert (
        main(
            [
                "record",
                str(out_path),
                "--from-save",
                str(save_path),
                "--seed",
                "4321",
                "--clock-epoch",
                str(epoch),
                "--steps",
                "0",
            ]
        )
        == 0
    )

    restored_state = json.loads(snapshot_save(local_session).model_dump_json())[
        "session_state"
    ]
    assert restored_state["duration"] == 0.0
    assert restored_state["total_playtime"] == 0.0
    assert restored_state["start_time"] == expected_start_time


def test_record_refuses_a_save_missing_current_map(tmp_path: Path) -> None:
    save_path = tmp_path / "save.json"
    save_path.write_text(json.dumps({}))
    out_path = tmp_path / "out.tuxghost"

    assert (
        main(
            [
                "record",
                str(out_path),
                "--from-save",
                str(save_path),
                "--seed",
                "1",
                "--clock-epoch",
                "1787694000",
                "--steps",
                "0",
            ]
        )
        == 2
    )
    assert not out_path.exists()


# --- Process-level tests -----------------------------------------------
#
# `tests/conftest.py` chdirs the entire pytest session into `tuxemon/`
# before any test runs, so calling `main()` in-process can never actually
# exercise `_bootstrap_vendored_tuxemon`'s own chdir the way a fresh
# process does. These launch the real CLI as a subprocess and check its
# real exit status -- the only honest way to test cwd-sensitive path
# handling, per this project's own process notes.


def test_relative_paths_resolve_against_the_run_directory(tmp_path: Path) -> None:
    """Hard requirement (a). Every path argument below is RELATIVE, and
    the subprocess's cwd is `tmp_path` -- a directory with no relationship
    to the vendored `tuxemon/` tree `_bootstrap_vendored_tuxemon` chdirs
    into. The earlier, deleted CLI resolved paths AFTER that chdir, so a
    relative path that existed fine in the caller's own shell was
    reported as not found. `record` (two path args: `out`, `--from-save`)
    followed by `verify` (one more: `trace`) exercises three separate
    relative-path arguments across two separate subprocess invocations."""
    from tuxghost.boot import build_client, snapshot_save
    from tuxghost.determinism import seed_all

    seed_all(1234)
    _client, session = build_client(seed=1234)
    (tmp_path / "save.json").write_text(snapshot_save(session).model_dump_json())

    record = subprocess.run(
        [
            sys.executable,
            "-m",
            "tuxghost.cli",
            "record",
            "out.tuxghost",
            "--from-save",
            "save.json",
            "--seed",
            "1234",
            "--clock-epoch",
            "1787694000",
            "--steps",
            "5",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert record.returncode == 0, (record.stdout, record.stderr)
    assert (tmp_path / "out.tuxghost").exists()

    verify = subprocess.run(
        [sys.executable, "-m", "tuxghost.cli", "verify", "out.tuxghost"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify.returncode == 0, (verify.stdout, verify.stderr)


def test_process_level_exit_codes_are_0_1_and_2(tmp_path: Path) -> None:
    """Hard requirements (b) and (c): all three exit codes, checked by
    actually invoking the CLI as a subprocess and reading its real
    `returncode` -- not only via `main()` called in-process."""
    from tuxghost.boot import build_client
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.loop import run_steps
    from tuxghost.record import Recorder
    from tuxghost.trace import write

    seed_all(1234)
    pin_clock(1787694000)
    client, session = build_client(seed=1234, clock_epoch=1787694000)
    recorder = Recorder(
        session, seed=1234, clock_epoch=1787694000, recorder="offline-agent"
    )
    # `finish()` records the state actually reached -- it must be called
    # AFTER `run_steps` advances the declared `step_count`, or the sealed
    # `final_digest` is captured at step 0 while `execute`/`verify` (which
    # genuinely advance `step_count` steps) reach somewhere else. See
    # `tests/test_execute.py
    # ::test_verify_returns_zero_for_a_self_consistent_trace`'s docstring
    # for the exact same defect, caught the same way.
    run_steps(client, 5)
    trace = recorder.finish(step_count=5)

    good_path = tmp_path / "good.tuxghost"
    write(trace, good_path)

    tampered = trace.model_copy(
        update={
            "header": trace.header.model_copy(
                update={"final_digest": "sha256:" + "0" * 64}
            )
        }
    )
    bad_path = tmp_path / "bad.tuxghost"
    write(tampered, bad_path)

    def run(path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "tuxghost.cli", "verify", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

    ok = run(good_path)
    assert ok.returncode == 0, (ok.stdout, ok.stderr)

    diverged = run(bad_path)
    assert diverged.returncode == 1, (diverged.stdout, diverged.stderr)

    missing = run(tmp_path / "does-not-exist.tuxghost")
    assert missing.returncode == 2, (missing.stdout, missing.stderr)
    assert "refused" in missing.stderr.lower()


# --- `agent` subcommand -------------------------------------------------

FIXTURE = Path(__file__).parent / "fixtures" / "paper_town.save"


def _write_actions(tmp_path: Path) -> Path:
    """A scripted policy's decisions as JSONL -- the same shape
    ReplayPolicy reads, so --policy scripted and --policy replay take the
    same file."""
    path = tmp_path / "actions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"actions": [{"button": 2, "hold": 30, "settle": 10}]})
            for _ in range(3)
        )
    )
    return path


def test_agent_writes_a_trace_and_a_run_directory(tmp_path: Path) -> None:
    out = tmp_path / "run.tuxghost"
    run_dir = tmp_path / "rundir"
    code = main(
        [
            "agent", "--policy", "scripted",
            "--actions", str(_write_actions(tmp_path)),
            "--seed", "1234", "--clock-epoch", "1787659200",
            "--from-save", str(FIXTURE), "--steps", "300",
            "--out", str(out), "--run-dir", str(run_dir),
        ]
    )
    assert code == 0
    assert out.exists()
    assert (run_dir / "decisions.jsonl").exists()
    assert list((run_dir / "frames").glob("*.png"))


def test_agent_refuses_an_unreadable_save(tmp_path: Path) -> None:
    """Refusal is exit 2, never exit 1 -- exit 1 means 'diverged', and a
    recording has nothing to diverge from. See this module's docstring."""
    code = main(
        [
            "agent", "--policy", "scripted",
            "--actions", str(_write_actions(tmp_path)),
            "--seed", "1234", "--clock-epoch", "1787659200",
            "--from-save", str(tmp_path / "nope.save"), "--steps", "60",
            "--out", str(tmp_path / "o.tuxghost"),
            "--run-dir", str(tmp_path / "d"),
        ]
    )
    assert code == 2


def test_agent_accepts_a_relative_out_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vacuous-test lesson from CLAUDE.md: every path in the old CLI
    test was an absolute tmp_path, so a real bug -- chdir into tuxemon/
    BEFORE parsing args, breaking every relative path a user would type --
    went undetected. This test types a relative path on purpose.

    `--from-save` is ALSO typed relative here (review round 1, minor):
    the first draft of this test left `--from-save` as the absolute
    `FIXTURE` constant, covering only `--out`/`--run-dir` and repeating
    the exact shape of blind spot this test exists to close, just for two
    of `_PATH_ARGS["agent"]`'s four entries instead of all of them.

    `--actions` is ALSO typed relative here (review round 2, minor): the
    round-1 fix closed `--from-save` but left `--actions` absolute via
    `_write_actions(tmp_path)`'s own absolute return value -- the last of
    `_PATH_ARGS["agent"]`'s four entries never exercised as a relative
    path. Written directly here, relative, rather than through that
    helper, to close it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "save.json").write_text(FIXTURE.read_text())
    (tmp_path / "actions.jsonl").write_text(
        json.dumps({"actions": [{"button": 2, "hold": 30, "settle": 10}]})
    )
    code = main(
        [
            "agent", "--policy", "scripted",
            "--actions", "actions.jsonl",
            "--seed", "1234", "--clock-epoch", "1787659200",
            "--from-save", "save.json", "--steps", "120",
            "--out", "relative.tuxghost", "--run-dir", "reldir",
        ]
    )
    assert code == 0
    assert (tmp_path / "relative.tuxghost").exists()


def test_agent_cold_boot_needs_no_save(tmp_path: Path) -> None:
    code = main(
        [
            "agent", "--policy", "scripted",
            "--actions", str(_write_actions(tmp_path)),
            "--cold-boot",
            "--seed", "1234", "--clock-epoch", "1787659200",
            "--steps", "200",
            "--out", str(tmp_path / "cold.tuxghost"),
            "--run-dir", str(tmp_path / "colddir"),
        ]
    )
    assert code == 0


# --- Review round 1, Critical: `run_agent(...)` had no exception
# boundary at all, and three separate inputs each reached uncaught
# ValueError/TypeError/AttributeError deep inside it, exiting 1 instead
# of the required 2. `--policy replay` had NO committed coverage before
# this round -- the positive-path test below closes that gap; the
# malformed-transcript test below it pins the exact reproduced defect.


def _write_malformed_replay_transcript(tmp_path: Path) -> Path:
    """Two well-formed records, then a third whose action is missing
    `settle` -- `actions_from_json` (called LAZILY inside
    `ReplayPolicy.decide()`, not eagerly when `ReplayPolicy` is
    constructed) raises `ValueError` for this only once the run reaches
    that record, well after every precondition check in `_agent` already
    succeeded on the first two records' account. No eager,
    construction-time check can catch this -- only a boundary around the
    `run_agent(...)` call itself can."""
    path = tmp_path / "malformed_transcript.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"actions": [{"button": 2, "hold": 30, "settle": 10}]}
                ),
                json.dumps(
                    {"actions": [{"button": 2, "hold": 30, "settle": 10}]}
                ),
                json.dumps({"actions": [{"button": 2, "hold": 30}]}),
            ]
        )
    )
    return path


def test_agent_replay_writes_a_trace(tmp_path: Path) -> None:
    """The positive-path counterpart the review flagged as entirely
    missing: before this round, no committed test ever exercised
    `--policy replay` at all, which is exactly what let the malformed-
    transcript defect below through undetected."""
    out = tmp_path / "replay.tuxghost"
    run_dir = tmp_path / "replaydir"
    code = main(
        [
            "agent", "--policy", "replay",
            "--actions", str(_write_actions(tmp_path)),
            "--cold-boot",
            "--seed", "1234", "--clock-epoch", "1787659200", "--steps", "300",
            "--out", str(out), "--run-dir", str(run_dir),
        ]
    )
    assert code == 0
    assert out.exists()
    assert (run_dir / "decisions.jsonl").exists()


def test_agent_refuses_a_transcript_that_fails_mid_run(tmp_path: Path) -> None:
    """Review round 1, Critical, case 1 of 3. Reproduced by the reviewer
    as a real subprocess: `--policy replay` with a third transcript
    record missing `settle` raised `ValueError` from
    `ReplayPolicy.decide()` (called from `run_agent`'s loop), uncaught,
    exiting 1. Fixed with a `try/except (ValueError, TypeError)` boundary
    around the `run_agent(...)` call in `_agent`."""
    code = main(
        [
            "agent", "--policy", "replay",
            "--actions", str(_write_malformed_replay_transcript(tmp_path)),
            "--cold-boot",
            "--seed", "1234", "--clock-epoch", "1787659200", "--steps", "300",
            "--out", str(tmp_path / "o.tuxghost"),
            "--run-dir", str(tmp_path / "d"),
        ]
    )
    assert code == 2


def test_agent_refuses_a_non_object_actions_line(tmp_path: Path) -> None:
    """Review round 1, Critical, case 2 of 3. Reproduced by the reviewer:
    a `--actions` line that is valid JSON but not an object (`[1, 2,
    3]`) made `record.get("actions", [])` raise `AttributeError` -- a
    bare Python builtin, not one of this project's own
    `ValueError`/`TypeError` input-validation exceptions, so it fell
    through the `except (OSError, json.JSONDecodeError, ValueError,
    TypeError)` tuple guarding the scripted actions-file parse and
    propagated uncaught, exiting 1. Fixed with an explicit `isinstance`
    check naming the offending line number."""
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps([1, 2, 3]))
    code = main(
        [
            "agent", "--policy", "scripted",
            "--actions", str(path),
            "--cold-boot",
            "--seed", "1234", "--clock-epoch", "1787659200", "--steps", "60",
            "--out", str(tmp_path / "o.tuxghost"),
            "--run-dir", str(tmp_path / "d"),
        ]
    )
    assert code == 2


def test_agent_refuses_invalid_upscale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review round 1, Critical, case 3 of 3. Reproduced by the reviewer:
    `--upscale 0` raised `ValueError` from `FrameRenderer.__init__`
    (called from `run_agent`), uncaught, exiting 1. `--upscale` is a bad
    ARGUMENT with a knowable valid range (>= 1) -- it should never reach
    `run_agent` in the first place, so it is refused eagerly in `_agent`'s
    precondition block, before `seed_all`/anything else runs.

    Review round 2: `--upscale 0` is ALSO caught by the `run_agent(...)`
    boundary below (`FrameRenderer.__init__` raises the identical
    `ValueError` if this ever reached it), so `assert code == 2` alone
    cannot tell which of the two guards actually fired -- confirmed in
    round 1's report by removing only the precondition and watching this
    test still pass. The two guards' messages are textually distinct
    (the precondition never says "run failed"; the boundary always does
    -- see `_agent`'s `except (ValueError, TypeError)` handler), so
    asserting on the exact wording is what isolates the precondition
    specifically, the same way an earlier task in this plan used a
    message match to distinguish two exception paths that shared an
    exception type."""
    code = main(
        [
            "agent", "--policy", "scripted",
            "--actions", str(_write_actions(tmp_path)),
            "--cold-boot",
            "--seed", "1234", "--clock-epoch", "1787659200", "--steps", "60",
            "--upscale", "0",
            "--out", str(tmp_path / "o.tuxghost"),
            "--run-dir", str(tmp_path / "d"),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "refused: --upscale must be >= 1, got 0" in err
    assert "run failed" not in err


# --- Review round 2, Critical (still open): the SAME non-object-record
# defect fixed for `--policy scripted` in round 1 was left live for
# `--policy replay`. `ReplayPolicy.__init__` never checked record shape,
# so a transcript line that is valid JSON but not an object constructed
# cleanly and `decide()`'s unconditional `record.get("notes")` raised
# `AttributeError` lazily, mid-run -- not caught by the (deliberately
# narrow) `run_agent(...)` boundary, exiting 1. Fixed in
# `ReplayPolicy.__init__` itself (Ruling X): validated at construction,
# not lazily inside `decide()`, so a malformed transcript is refused
# before the game ever boots.


def test_agent_replay_refuses_a_non_object_record(tmp_path: Path) -> None:
    path = tmp_path / "bad_replay.jsonl"
    path.write_text(json.dumps([1, 2, 3]))
    code = main(
        [
            "agent", "--policy", "replay",
            "--actions", str(path),
            "--cold-boot",
            "--seed", "1234", "--clock-epoch", "1787659200", "--steps", "60",
            "--out", str(tmp_path / "o.tuxghost"),
            "--run-dir", str(tmp_path / "d"),
        ]
    )
    assert code == 2
