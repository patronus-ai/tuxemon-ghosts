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
`Provenance.taints`), plus a specific check for `recorder="human"`. That
kind covers TWO things this format does not otherwise distinguish: a real
human play session (`step_count` came from elapsed real time, not a
deterministic schedule) and a `--policy scripted` agent run
(`tuxghost.cli._agent` deliberately reuses this same kind for a fixed,
already-deterministic action list rather than adding a fourth
`RecorderKind`) -- M7, whole-branch review, on an earlier version of this
finding that named only the human-loop half unconditionally, which is
simply wrong for a scripted-agent trace. Findings are informational, not
divergences: only a difference in `initial_state` or `inputs` moves the
exit code to 1.

`agent` (drives a policy against a live session and records what it did,
via `tuxghost.agent.runner.run_agent`) maps every precondition it can
anticipate to 0 or 2, like `execute` above and for the identical reason:
a recording has nothing to diverge FROM, so "refused" and "diverged" must
never collide on the same exit code. Conflating them would be exactly the
same defect the old CLI shipped -- an uncaught exception (e.g. a missing
`--from-save` file) reports exit 1, indistinguishable from a real
divergence, so every precondition `_agent`/`_agent_save_or_refuse` can
detect is checked explicitly and mapped to exit 2 before anything can
raise. This ALSO covers `--policy claude`: `ClaudePolicy.decide` calls the
`anthropic` SDK's `messages.create` from inside `run_agent`'s loop, and
every error that call can raise (`RateLimitError`, `APIConnectionError`,
`AuthenticationError`, a missing/invalid API key, ...) derives from
`anthropic.AnthropicError(Exception)`, not `ValueError`/`TypeError` --
`_agent`'s `except (ValueError, TypeError)` boundary alone does NOT catch
any of them. A second `except Exception` arm, reachable only when
`--policy claude` (final residuals, item 1), maps a real `AnthropicError`
to exit 2 with its own distinct message and re-raises anything else. Exit
1 remains possible, deliberately: a genuine internal engine invariant
failure -- anything that is not one of the specific, anticipated
precondition failures above -- must stay a visible crash rather than
being laundered into a tidy refusal. "0 or 2 only" describes every path
this module checks for, not a guarantee that nothing else can go wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from tuxghost.trace import Refused, Trace, read

if TYPE_CHECKING:
    # Both are real, project-owned (or, for `SaveData`, project-adjacent)
    # types, not stand-ins for an unannotated upstream shape -- named here
    # only for static analysis. Neither is imported at RUNTIME at module
    # scope: `tuxghost.agent.types` imports `tuxemon.platform.const` at
    # its own module level, and `tuxemon.save_system.save_state` is
    # vendored `tuxemon` -- both need `tuxemon/` on `sys.path` first,
    # which only happens once `_bootstrap_vendored_tuxemon` runs inside
    # `main()`. An eager top-level import here would break `python -m
    # tuxghost.cli ...` (the module is imported before that bootstrap
    # ever runs). `from __future__ import annotations` (above) means
    # every annotation below is a string at runtime, never evaluated, so
    # this TYPE_CHECKING-only import is all mypy needs.
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.agent.types import Action, Policy

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
    # No default here (M8, whole-branch review): `--policy claude` falls
    # back to `tuxghost.agent.claude.DEFAULT_MODEL` explicitly in `_agent`
    # below when this is `None`, but `--policy replay` must NOT silently
    # record whatever that default happens to be -- a replayed transcript
    # was produced by some specific model (or none at all, for a
    # synthetic test fixture), and recording an unrelated default as
    # `model` would misattribute provenance the trace format claims to be
    # honest about. `--policy replay` refuses instead when this is
    # `None`, requiring the caller to say which model the transcript is
    # actually of.
    p_agent.add_argument("--model", default=None)
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
            # M7, whole-branch review: `recorder="human"` covers TWO
            # different things this format does not distinguish in any
            # other field -- a real human play session (`record`
            # subcommand), whose step count genuinely comes from elapsed
            # real time and is NOT a deterministic schedule, and a
            # `--policy scripted` agent run (`tuxghost.cli._agent`), which
            # deliberately reuses this same recorder kind for a fixed,
            # already-deterministic action list rather than adding a
            # fourth `RecorderKind` (an S1 trace-format change). The
            # earlier wording here asserted the human-loop half
            # unconditionally, which is simply wrong for a scripted-agent
            # trace. Reworded to name the ambiguity honestly instead of
            # guessing which one produced this trace.
            print(
                f"finding: {name} is recorded as \"human\" -- either a "
                f"real human play session (step count from elapsed real "
                f"time, not a deterministic schedule) or a `--policy "
                f"scripted` agent run (a fixed, already-deterministic "
                f"action list recorded under the same recorder kind); "
                f"this format does not distinguish the two in any other "
                f"field"
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


def _agent_save_or_refuse(args: argparse.Namespace) -> SaveData | None:
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

    Maps every anticipated precondition to 0 or 2. Exit 1 means
    "diverged", and a recording has nothing to diverge FROM, so no
    anticipated precondition failure may report it -- see this module's
    docstring, which records the earlier CLI defect that conflated the
    two. A genuine internal engine invariant failure still exits 1,
    deliberately: that is a visible crash, not a precondition this
    function can refuse.
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
    # Both are knowable-range arguments -- `run_agent` (via
    # `FrameRenderer.__init__`/its own `step_budget < 1` check) would
    # raise `ValueError` for either deep inside the run, but a bad CLI
    # argument is a REFUSED precondition (exit 2), not something that
    # should ever reach the run_agent boundary below at all. Checked here,
    # eagerly, before anything else can fail on this run's behalf.
    if args.upscale < 1:
        print(
            f"refused: --upscale must be >= 1, got {args.upscale!r}",
            file=sys.stderr,
        )
        return 2
    if args.steps < 1:
        print(
            f"refused: --steps must be >= 1, got {args.steps!r}",
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
    policy: Policy
    if args.policy == "scripted":
        try:
            lines = args.actions.read_text().splitlines()
        except OSError as exc:
            print(f"refused: cannot read {args.actions}: {exc}", file=sys.stderr)
            return 2
        decisions: list[tuple[Action, ...]] = []
        try:
            for lineno, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"line {lineno} is not valid JSON: {exc}"
                    ) from exc
                # `record.get(...)` below would raise `AttributeError` (a
                # bare Python builtin, not one of this project's own
                # input-validation exception types) for a line that is
                # valid JSON but not an object -- e.g. `[1, 2, 3]`. Caught
                # here, explicitly, and named by line number: a malformed
                # actions file is bad INPUT, and a user fixing it needs to
                # know which line, not just that "bad --actions file"
                # happened somewhere in it.
                if not isinstance(record, dict):
                    raise TypeError(
                        f"line {lineno} must be a JSON object (with an "
                        f"'actions' key), got {type(record).__name__}"
                    )
                try:
                    decisions.append(
                        actions_from_json(record.get("actions", []))
                    )
                except (ValueError, TypeError) as exc:
                    raise type(exc)(f"line {lineno}: {exc}") from exc
        except (ValueError, TypeError) as exc:
            print(
                f"refused: bad --actions file ({args.actions}): {exc}",
                file=sys.stderr,
            )
            return 2
        policy = ScriptedPolicy(decisions)
        kind = "human"
    elif args.policy == "replay":
        # M8, whole-branch review: `--model` has NO default (see
        # `_build_parser`) specifically so this cannot silently record
        # some unrelated default as the model that produced this
        # transcript. A transcript carries no `model` field of its own to
        # read it back from (records are `{"actions": ..., "notes": ...,
        # "raw": ..., "claimed_outcome": ...}` -- see
        # `tuxghost/agent/replay.py`), so the honest fix is requiring the
        # caller to say which model this is, not guessing.
        if args.model is None:
            print(
                "refused: --policy replay requires --model (the id of "
                "the model whose transcript this is; the transcript "
                "itself carries no model field to read it back from)",
                file=sys.stderr,
            )
            return 2
        try:
            policy = ReplayPolicy(args.actions)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            # `TypeError` here is `ReplayPolicy.__init__`'s own
            # record-shape check (task 8 review round 2): every parsed
            # record must be a dict, checked at CONSTRUCTION so a
            # malformed transcript is refused before the game ever boots,
            # not caught by the narrower `run_agent(...)` boundary below
            # (which deliberately does NOT catch `AttributeError`).
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
            # No unused-import suppression comment needed here (unlike an
            # earlier version of this line): this `anthropic` binding is
            # a real precondition check on its own, AND the name is
            # genuinely used again later in this same function scope, in
            # the `except Exception` arm around `run_agent(...)` below
            # (`isinstance(exc, anthropic.AnthropicError)`) -- ruff's
            # unused-import check is scope-wide, not per-binding, so it
            # no longer considers this import unused at all.
            import anthropic
        except ImportError as exc:
            print(
                f"refused: --policy claude needs the anthropic SDK "
                f"(pip install -e '.[agent]'): {exc}",
                file=sys.stderr,
            )
            return 2

        from tuxghost.agent.claude import DEFAULT_MODEL, ClaudePolicy

        # `--model` has no argparse default (M8, whole-branch review, see
        # `_build_parser`) so `--policy replay` cannot silently borrow it;
        # `--policy claude` still gets a sensible default here instead,
        # resolved explicitly rather than via `ClaudePolicy`'s own
        # `model: str = DEFAULT_MODEL` parameter default -- `args.model`
        # would otherwise pass `None` through it, which that parameter's
        # own type does not accept.
        model = args.model if args.model is not None else DEFAULT_MODEL
        policy = ClaudePolicy(model=model, goal=args.goal)
        kind = "cu-agent"

    try:
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
            # `scaled=True`: the real CLI is always launched as its own
            # fresh OS process (one `agent` invocation per process), which
            # is exactly the precondition `tuxghost.observe.
            # scaled_context()` requires to render at human field-of-view
            # rather than refuse -- see task 9's review round 1. Anything
            # that calls `main()` in-process (sharing a process with
            # earlier boots -- most of this repo's own test suite) is NOT
            # that precondition and must launch this subcommand as a real
            # subprocess instead; see tests/test_cli.py's own docstring.
            scaled=True,
        )
    except (ValueError, TypeError) as exc:
        # The structural fix, not a patch for one call site: bad input
        # can surface AFTER this point, once the run is already under
        # way, not just before it. `ReplayPolicy` parses each transcript
        # record LAZILY inside `decide()` (`tuxghost/agent/replay.py`),
        # called from deep inside `run_agent`'s loop -- a transcript whose
        # Nth record is malformed raises there, well past every
        # precondition check above. `ValueError`/`TypeError` are exactly
        # the two exception types this project's OWN input validation
        # raises for bad input (`validate_actions`, `actions_from_json`,
        # `FrameRenderer.__init__`) -- the same two-exception taxonomy
        # already caught above for the scripted/replay `--actions` file
        # itself. Deliberately NOT a bare `except Exception`: a genuine
        # internal engine invariant failure must stay a visible crash
        # rather than being laundered into a tidy "refused" -- see
        # `tuxghost.execute`'s own comments on the identical tradeoff
        # around `boot_from_save`'s `AssertionError`.
        print(
            f"refused: --policy {args.policy} run failed: {exc}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        # Important #2, whole-branch review: `--policy claude`'s
        # `ClaudePolicy.decide` calls `messages.create(...)` from inside
        # `run_agent`'s loop, and every error the `anthropic` SDK can
        # raise for that call -- `RateLimitError`, `APIConnectionError`,
        # `AuthenticationError`, a missing/invalid API key, ... -- derives
        # from `anthropic.AnthropicError(Exception)`, none of them
        # `ValueError`/`TypeError`, so the boundary above never catches
        # any of them. Left uncaught, any of them exits 1 = "diverged",
        # the one code this module's docstring says `agent` must never
        # produce.
        #
        # Gated on `args.policy == "claude"` (final residuals, item 1):
        # this `try` is shared by all three policies, but
        # `anthropic.AnthropicError` can only ever originate from
        # `ClaudePolicy.decide`, reachable only when `--policy claude`. An
        # earlier version imported `anthropic` unconditionally inside this
        # arm (to build the `isinstance` check below) with a comment
        # claiming that import "cannot fail in practice" -- true only on
        # the claude branch, but the arm is reachable from ALL three
        # policies, and `make check`'s environment deliberately does not
        # install `anthropic` at all. The result: any genuine, non-
        # `ValueError`/`TypeError` crash out of `run_agent` on
        # `--policy scripted`/`replay` (e.g. a real `AttributeError`
        # engine-invariant failure, the same class this branch hit for
        # real during Task 8) got its bare `raise` immediately superseded
        # by `ModuleNotFoundError: No module named 'anthropic'` raised
        # while handling it -- the wrong headline exception on exactly the
        # path whose whole design intent is "a genuine engine invariant
        # failure must stay a visible crash". Gating on the policy means
        # `--policy scripted`/`replay` never enter this arm at all, so
        # `anthropic` need not be importable for them, and the real
        # exception re-raises unmodified via the bare `raise` below.
        if args.policy != "claude":
            raise
        import anthropic

        if not isinstance(exc, anthropic.AnthropicError):
            raise
        print(
            f"refused: --policy {args.policy} call to the Anthropic API "
            f"failed: {exc}",
            file=sys.stderr,
        )
        return 2
    write(result.trace, args.out)

    frames_dir = args.run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, png in enumerate(result.frames):
        (frames_dir / f"{index:05d}.png").write_bytes(png)
    with (args.run_dir / "decisions.jsonl").open("w") as handle:
        for decision in result.decisions:
            # `dataclasses.asdict` is recursive: it already converts each
            # nested `Action` dataclass in `decision.actions` on its own,
            # so a second, manual `record["actions"] = [...]` pass (an
            # earlier draft of this loop had one) is dead code that
            # reassigns the exact list `asdict` already built.
            record = dataclasses.asdict(decision)
            handle.write(json.dumps(record) + "\n")

    # M2+M3, whole-branch review: `--goal` never reached the run
    # directory despite the spec/STATUS claiming it did, and neither
    # `break` path in `run_agent`'s loop ever reaches `decisions.append`,
    # so nothing in the run directory could say WHY a run ended. One
    # small artifact closes both: `result.stop_reason` (see
    # `tuxghost.agent.runner.RunResult`) names the reason, and everything
    # else here is exactly what a person re-reading this run later would
    # need and could not otherwise reconstruct from `decisions.jsonl`
    # alone (the goal was never IN any decision; the seed/clock_epoch/
    # policy/model are process arguments, not part of the trace's own
    # per-step record).
    run_info = {
        "goal": args.goal,
        "seed": args.seed,
        "clock_epoch": args.clock_epoch,
        "step_budget": args.steps,
        "policy": args.policy,
        "model": model,
        "steps": result.steps,
        "stop_reason": result.stop_reason,
    }
    (args.run_dir / "run.json").write_text(json.dumps(run_info, indent=2))

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
