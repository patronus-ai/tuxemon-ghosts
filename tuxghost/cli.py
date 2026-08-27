"""`tuxghost` command line: `execute`, `verify`, `compare`, `record`, `info`.

Two proven defects from an earlier, deleted CLI (`tuxghost/execute.py`'s
`_main`, removed in task-12 fix round 1 -- see that module's docstring)
must not be reproduced here:

  * Relative trace paths must resolve against the CALLER's cwd, not the
    vendored `tuxemon/` directory `_bootstrap_vendored_tuxemon` chdirs
    into. Every path argument is `.resolve()`d immediately after
    `parse_args`, before anything below ever calls
    `_bootstrap_vendored_tuxemon`. Getting this order backwards is exactly
    the bug the old CLI shipped with: it chdir'd first, so a trace file
    that existed relative to the user's own shell silently "did not
    exist" once the process had already moved.
  * A missing or unreadable trace file is a REFUSED precondition (exit 2),
    not a divergence (exit 1). The old CLI let `tuxghost.trace.read`'s
    bare `Path(path).read_text()` raise an uncaught `FileNotFoundError`,
    which Python reports as exit 1 -- indistinguishable from "the trace
    diverged". `_read_trace_or_refuse` below catches every `OSError`
    (covers `FileNotFoundError`, `PermissionError`, `IsADirectoryError`,
    ...) and `json.JSONDecodeError` alongside `tuxghost.trace.Refused`,
    mapping all three to exit 2 uniformly.

Exit codes, for `verify` (and, transitively, `execute --verify` -- which
is refused, see below -- and `compare`):
  * 0 -- every compared field matched.
  * 1 -- at least one differed.
  * 2 -- refused: an unreadable/missing trace file, or a precondition
    `tuxghost.trace.read`/`tuxghost.execute.execute` failed.

`execute` alone (no comparison target) can only ever return 0 (ran to
completion) or 2 (refused) -- there is nothing for it to diverge FROM.
Running `execute --verify` would blend those two different operations
back together, so that combination is refused at parse time rather than
silently reinterpreted as `verify`. No flag on `execute`/`verify` lets a
caller override a value the trace's own header already commits to (seed,
clock_epoch, mod_version, step_rate, step_count): the whole point of the
trace format is that the header and `initial_state` are the ONLY input
execution needs (see `tuxghost.execute.execute`'s own docstring), so such
a flag would let a caller silently execute something other than what the
trace claims to be. None is exposed here, by design.

`compare` is deliberately NOT an in-engine execution: it reads two traces'
provenance (`tuxghost.trace.Provenance`) and their recorded `inputs`/
`initial_state`, and reports FINDINGS -- any taint either trace already
carries (a tainted replay, a replay-of-a-replay, a wall-clock source: any
string `tuxghost.trace.read`/a recorder chose to put in
`Provenance.taints`), plus a specific check for a human-recorded trace,
whose `step_count` came from elapsed real time rather than a deterministic
schedule. Findings are informational, not divergences: only a difference
in `initial_state` or `inputs` moves the exit code to 1.

`agent` (drives a policy against a live session and records what it did,
via `tuxghost.agent.runner.run_agent`) returns 0 or 2 ONLY, like `execute`
above and for the identical reason: a recording has nothing to diverge
FROM. Conflating "refused" with "diverged" here would be exactly the same
defect the old CLI shipped -- an uncaught exception (e.g. a missing
`--from-save` file) reports exit 1, indistinguishable from a real
divergence, so every precondition `_agent`/`_agent_save_or_refuse` can
detect is checked explicitly and mapped to exit 2 before anything can
raise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tuxghost.trace import Refused, Trace, read

#: Which of each subcommand's parsed arguments are paths that must be
#: `.resolve()`d against the caller's cwd before any chdir happens.
_PATH_ARGS: dict[str, tuple[str, ...]] = {
    "execute": ("trace",),
    "verify": ("trace",),
    "info": ("trace",),
    "compare": ("a", "b"),
    "record": ("out", "from_save"),
    "agent": ("actions", "from_save", "out", "run_dir"),
}

#: Subcommands that need the vendored `tuxemon/` package importable and
#: cwd-relative asset loading working -- i.e. anything that actually boots
#: a session. `info` and `compare` are pure trace-file inspection and need
#: neither.
_NEEDS_BOOTSTRAP = frozenset({"execute", "verify", "record", "agent"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tuxghost")
    sub = parser.add_subparsers(dest="command", required=True)

    p_exec = sub.add_parser(
        "execute", help="Replay a trace and print the digest it reaches."
    )
    p_exec.add_argument("trace", type=Path)
    p_exec.add_argument(
        "--verify",
        action="store_true",
        help="refused: run `tuxghost verify` instead",
    )
    p_exec.add_argument("--allow-mismatch", action="store_true")
    p_exec.add_argument(
        "--checkpoint",
        type=int,
        default=0,
        help="digest every N steps instead of only at the end",
    )

    p_verify = sub.add_parser(
        "verify",
        help="Replay a trace and compare against its recorded final_digest.",
    )
    p_verify.add_argument("trace", type=Path)
    p_verify.add_argument("--allow-mismatch", action="store_true")

    p_info = sub.add_parser("info", help="Print a trace's header and provenance.")
    p_info.add_argument("trace", type=Path)
    p_info.add_argument("--allow-mismatch", action="store_true")

    p_cmp = sub.add_parser(
        "compare",
        help="Out-of-engine comparison of two traces' provenance/inputs.",
    )
    p_cmp.add_argument("a", type=Path)
    p_cmp.add_argument("b", type=Path)
    p_cmp.add_argument("--allow-mismatch", action="store_true")

    p_rec = sub.add_parser(
        "record", help="Record a new trace by replaying a save forward with no input."
    )
    p_rec.add_argument("out", type=Path)
    p_rec.add_argument("--from-save", type=Path, required=True, dest="from_save")
    p_rec.add_argument("--seed", type=int, required=True)
    p_rec.add_argument("--clock-epoch", type=int, required=True, dest="clock_epoch")
    p_rec.add_argument("--steps", type=int, required=True)

    p_agent = sub.add_parser(
        "agent",
        help="Record a trace by driving a policy against a live session.",
    )
    p_agent.add_argument(
        "--policy", choices=["scripted", "replay", "claude"], required=True
    )
    p_agent.add_argument(
        "--actions", type=Path, help="JSONL decisions for scripted/replay"
    )
    p_agent.add_argument("--from-save", type=Path, dest="from_save")
    p_agent.add_argument("--cold-boot", action="store_true", dest="cold_boot")
    p_agent.add_argument("--seed", type=int, required=True)
    p_agent.add_argument("--clock-epoch", type=int, required=True, dest="clock_epoch")
    p_agent.add_argument("--steps", type=int, required=True)
    p_agent.add_argument("--goal", default="")
    p_agent.add_argument("--model", default="claude-sonnet-5")
    p_agent.add_argument("--upscale", type=int, default=4)
    p_agent.add_argument("--out", type=Path, required=True)
    p_agent.add_argument("--run-dir", type=Path, required=True, dest="run_dir")

    return parser


def _resolve_paths(args: argparse.Namespace) -> None:
    """Resolve every path argument against the CALLER's cwd. Must run
    before `_bootstrap_vendored_tuxemon` (called later, only for
    subcommands that need it) ever chdirs -- see the module docstring."""
    for name in _PATH_ARGS.get(args.command, ()):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).resolve())


def _read_trace_or_refuse(path: Path, allow_mismatch: bool) -> Trace | None:
    """Read a trace, mapping every way that can fail -- a missing or
    unreadable file, malformed JSON, a structurally malformed trace, or
    `tuxghost.trace.read`'s own refusal matrix -- to a single outcome:
    print `refused: ...` to stderr and return `None`. Callers return 2
    immediately when this returns `None`. `OSError` (covering
    `FileNotFoundError`, `PermissionError`, `IsADirectoryError`, ...) is
    caught explicitly here so a missing trace file maps to exit 2, never
    the uncaught-exception exit 1 Python would otherwise produce -- see
    the module docstring. `pydantic.ValidationError` is caught for the
    same reason: `tuxghost.trace.read`'s final `Trace.model_validate`
    raises it for any structurally malformed trace that slips past the
    refuse/warn matrix's own explicit checks (e.g. a missing
    `provenance` block); left uncaught it would propagate past this
    function entirely and exit 1, colliding "unrunnable trace file" with
    "diverged" -- exactly the ambiguity `_record` below already avoids by
    catching it there."""
    try:
        return read(path, allow_mismatch=allow_mismatch)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"refused: cannot read {path}: {exc}", file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(f"refused: {path} is not valid JSON: {exc}", file=sys.stderr)
        return None
    except ValidationError as exc:
        print(f"refused: {path} does not validate as a trace: {exc}", file=sys.stderr)
        return None


def _execute(path: Path, allow_mismatch: bool, checkpoint: int) -> int:
    """Replay `path` and print its digest(s). With `--checkpoint N` (N >
    0), also prints one `checkpoint <step>: <digest>` line per interval
    BEFORE the final digest -- `execute` has no comparison target of its
    own (see the module docstring: it can only ever return 0 or 2), so
    there is nothing here to bisect against yet, but the digests are
    exactly what a later `tuxghost compare`/`bisect_traces` run against a
    second execution needs to locate a divergence without re-running from
    scratch. Printing nothing for a flag the help text promises would let
    `--checkpoint` silently rot -- see `tests/test_cli.py
    ::test_checkpoint_flag_prints_intermediate_digests`."""
    from tuxghost.execute import execute

    trace = _read_trace_or_refuse(path, allow_mismatch)
    if trace is None:
        return 2
    try:
        result = execute(trace, checkpoint=checkpoint)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    for step, digest in result.checkpoints:
        print(f"checkpoint {step}: {digest}")
    print(result.final_digest)
    return 0


def _verify(path: Path, allow_mismatch: bool) -> int:
    from tuxghost.execute import verify

    trace = _read_trace_or_refuse(path, allow_mismatch)
    if trace is None:
        return 2
    code = verify(trace)
    if code == 0:
        print(f"verify: OK ({trace.header.final_digest})")
    elif code == 1:
        print("verify: DIVERGED", file=sys.stderr)
    else:
        print("verify: REFUSED", file=sys.stderr)
    return code


def _info(path: Path, allow_mismatch: bool) -> int:
    trace = _read_trace_or_refuse(path, allow_mismatch)
    if trace is None:
        return 2
    print(trace.header.model_dump_json(indent=2))
    print(trace.provenance.model_dump_json(indent=2))
    return 0


def _compare(path_a: Path, path_b: Path, allow_mismatch: bool) -> int:
    """Out-of-engine comparison: reads provenance and reports FINDINGS,
    which are not divergences -- see the module docstring."""
    from tuxghost.compare import first_difference

    a = _read_trace_or_refuse(path_a, allow_mismatch)
    if a is None:
        return 2
    b = _read_trace_or_refuse(path_b, allow_mismatch)
    if b is None:
        return 2

    for name, trace in (("a", a), ("b", b)):
        for taint in trace.provenance.taints:
            print(f"finding: {name} is tainted: {taint}")
        if trace.provenance.recorder == "human":
            print(
                f"finding: {name} was recorded from a human loop, whose "
                f"step count comes from elapsed real time, not a "
                f"deterministic schedule"
            )

    field = first_difference(a.initial_state, b.initial_state)
    if field is not None:
        print(f"initial_state differs at {field}")
        return 1
    if a.inputs != b.inputs:
        print("input tracks differ")
        return 1
    print("traces agree")
    return 0


def _record(args: argparse.Namespace) -> int:
    """Record a trace by replaying a save forward with no scheduled input
    -- a bare "walk N steps from this save" recording. Mirrors
    `tuxghost.boot.boot_from_save`'s own precondition check (`npc_state
    .current_map` required) directly, rather than letting its internal
    `assert` propagate, for the same reason `tuxghost.execute.execute`
    checks it itself: catching a bare `AssertionError` around
    `boot_from_save` would also catch a genuine internal engine invariant
    failure and misreport it as "this save is unrunnable"."""
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save
    from tuxghost.determinism import seed_all
    from tuxghost.loop import run_steps
    from tuxghost.record import Recorder
    from tuxghost.trace import write

    seed_all(args.seed)

    try:
        raw = json.loads(args.from_save.read_text())
    except OSError as exc:
        print(f"refused: cannot read {args.from_save}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"refused: {args.from_save} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    try:
        save_data = SaveData.model_validate(raw)
    except ValidationError as exc:
        print(
            f"refused: {args.from_save} does not validate as a SaveData: {exc}",
            file=sys.stderr,
        )
        return 2

    if save_data.npc_state is None or save_data.npc_state.current_map is None:
        print(
            "refused: from-save is missing npc_state.current_map; cannot "
            "restore a session (see tuxghost.boot.boot_from_save)",
            file=sys.stderr,
        )
        return 2

    client, session = boot_from_save(
        save_data, seed=args.seed, clock_epoch=args.clock_epoch
    )
    recorder = Recorder(
        session,
        seed=args.seed,
        clock_epoch=args.clock_epoch,
        recorder="human",
    )
    run_steps(client, args.steps)
    write(recorder.finish(step_count=args.steps), args.out)
    print(f"wrote {args.out}")
    return 0


def _agent_save_or_refuse(args: argparse.Namespace) -> Any | None:
    """Read, parse, validate and map-check `--from-save`, or print a
    refusal and return None.

    Every check mirrors `_record`'s, verbatim and in the same order, plus
    Task 1's `resolve_map_asset`. Factored out so `_agent` reads as the
    control flow it is rather than 60 lines of guard clauses.
    """
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import resolve_map_asset

    try:
        raw = json.loads(args.from_save.read_text())
    except OSError as exc:
        print(f"refused: cannot read {args.from_save}: {exc}", file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(
            f"refused: {args.from_save} is not valid JSON: {exc}",
            file=sys.stderr,
        )
        return None

    try:
        save_data = SaveData.model_validate(raw)
    except ValidationError as exc:
        print(
            f"refused: {args.from_save} does not validate as a SaveData: {exc}",
            file=sys.stderr,
        )
        return None

    if save_data.npc_state is None or save_data.npc_state.current_map is None:
        print(
            "refused: from-save is missing npc_state.current_map; cannot "
            "restore a session (see tuxghost.boot.boot_from_save)",
            file=sys.stderr,
        )
        return None

    if resolve_map_asset(save_data.npc_state.current_map) is None:
        print(
            "refused: from-save's npc_state.current_map="
            f"{save_data.npc_state.current_map!r} resolves to no map asset "
            "(tried it as given and with a .tmx extension)",
            file=sys.stderr,
        )
        return None

    return save_data


def _agent(args: argparse.Namespace) -> int:
    """Record a trace by driving a policy against a live session.

    Returns 0 or 2 only. Exit 1 means "diverged", and a recording has
    nothing to diverge FROM -- see this module's docstring, which records
    the earlier CLI defect that conflated the two.
    """
    import dataclasses

    from tuxghost.agent.replay import ReplayPolicy, actions_from_json
    from tuxghost.agent.runner import RecorderKind, run_agent
    from tuxghost.agent.scripted import ScriptedPolicy
    from tuxghost.determinism import seed_all
    from tuxghost.trace import write

    if bool(args.from_save) == bool(args.cold_boot):
        print(
            "refused: pass exactly one of --from-save or --cold-boot",
            file=sys.stderr,
        )
        return 2
    if args.policy in ("scripted", "replay") and args.actions is None:
        print(
            f"refused: --policy {args.policy} requires --actions "
            "(a JSONL file of decisions)",
            file=sys.stderr,
        )
        return 2

    seed_all(args.seed)

    save_data = None
    if not args.cold_boot:
        save_data = _agent_save_or_refuse(args)
        if save_data is None:
            return 2

    # Policy, plus the provenance each one honestly implies (see the
    # spec's Provenance section): a scripted list is a person's, a replayed
    # transcript is the model's decisions taken from a recording rather
    # than live, and only a live call is an untainted cu-agent trace.
    kind: RecorderKind
    taints: tuple[str, ...] = ()
    model: str | None = None
    policy: Any
    if args.policy == "scripted":
        try:
            decisions = [
                actions_from_json(json.loads(line).get("actions", []))
                for line in args.actions.read_text().splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"refused: bad --actions file: {exc}", file=sys.stderr)
            return 2
        policy = ScriptedPolicy(decisions)
        kind = "human"
    elif args.policy == "replay":
        try:
            policy = ReplayPolicy(args.actions)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"refused: bad --actions file: {exc}", file=sys.stderr)
            return 2
        kind = "cu-agent"
        model = args.model
        taints = (
            "decisions replayed from a captured transcript, not taken live",
        )
    else:
        # `ClaudePolicy.__init__` never imports `anthropic` -- only its
        # `_ensure_client`, called lazily on the first `decide()` inside
        # `run_agent`'s loop, does (see `tuxghost/agent/claude.py`'s
        # module docstring). Wrapping `ClaudePolicy(...)` construction in
        # `except ImportError` (an earlier draft of this branch did) is
        # dead code: the import never happens there, so a missing
        # `anthropic` package would instead raise deep inside `run_agent`,
        # UNCAUGHT -- which Python reports as exit 1, indistinguishable
        # from "diverged", exactly the collision this module's docstring
        # says `agent` must never produce. Importing `anthropic` here,
        # explicitly, before any policy or `run_agent` call, is a real
        # precondition check: it fails fast, on the same branch the old
        # `except ImportError` was meant to guard, without ever touching
        # the network (a missing package fails before any client is
        # constructed).
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            print(
                f"refused: --policy claude needs the anthropic SDK "
                f"(pip install -e '.[agent]'): {exc}",
                file=sys.stderr,
            )
            return 2

        from tuxghost.agent.claude import ClaudePolicy

        policy = ClaudePolicy(model=args.model, goal=args.goal)
        kind = "cu-agent"
        model = args.model

    result = run_agent(
        policy=policy,
        save_data=save_data,
        cold_boot=bool(args.cold_boot),
        seed=args.seed,
        clock_epoch=args.clock_epoch,
        step_budget=args.steps,
        recorder_kind=kind,
        model=model,
        upscale=args.upscale,
        taints=taints,
    )
    write(result.trace, args.out)

    frames_dir = args.run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, png in enumerate(result.frames):
        (frames_dir / f"{index:05d}.png").write_bytes(png)
    with (args.run_dir / "decisions.jsonl").open("w") as handle:
        for decision in result.decisions:
            record = dataclasses.asdict(decision)
            record["actions"] = [
                dataclasses.asdict(a) for a in decision.actions
            ]
            handle.write(json.dumps(record) + "\n")

    print(f"wrote {args.out} ({result.steps} steps) and {args.run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve every path argument BEFORE any chdir. `_bootstrap_vendored_
    # tuxemon` (called below, only when a subcommand needs it) chdirs into
    # the vendored tuxemon/ directory; resolving paths after that would
    # resolve a relative trace path against the WRONG directory -- the
    # exact defect the earlier, deleted CLI shipped with. See the module
    # docstring.
    _resolve_paths(args)

    if args.command == "execute" and args.verify:
        print(
            "refused: --verify on `execute` would blend two different "
            "operations together; run `tuxghost verify TRACE` instead",
            file=sys.stderr,
        )
        return 2

    if args.command in _NEEDS_BOOTSTRAP:
        from tuxghost.execute import _bootstrap_vendored_tuxemon

        _bootstrap_vendored_tuxemon()

    command: Any = args.command
    if command == "info":
        return _info(args.trace, args.allow_mismatch)
    if command == "compare":
        return _compare(args.a, args.b, args.allow_mismatch)
    if command == "record":
        return _record(args)
    if command == "agent":
        return _agent(args)
    if command == "verify":
        return _verify(args.trace, args.allow_mismatch)
    if command == "execute":
        return _execute(args.trace, args.allow_mismatch, args.checkpoint)

    raise AssertionError(f"unhandled command {command!r}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
