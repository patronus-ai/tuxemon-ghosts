"""`tuxghost` command line: `execute`, `verify`, `compare`, `record`, `info`,
`agent`, `optimize`, `play`.

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

`optimize` (edits a recorded trace offline, re-runs every candidate
through `tuxghost.optimize.seal.seal`, and writes the winner --
`tuxghost.optimize.runner.optimize`) has the SAME contract as `agent`,
for the same reason: a candidate search has nothing to diverge FROM, so
exit 1 must never mean "a precondition was wrong". What it refuses, all
mapped to exit 2 before the engine boots:

  * `--rounds`, `--patience`, `--max-rejections`, `--max-cost` below 1.
    `--checkpoint` is checked separately, as `>= 0`: 0 is a LEGAL value
    meaning "do not sample", so folding it into the `>= 1` loop by adding
    one would be a lie about what the bound is.
  * a `--target` that is not `MAP:X,Y`.
  * `--editor scripted`/`replay` without `--edits`; `--editor replay`
    without `--model` (a transcript carries no model of its own, so
    recording a default would misattribute provenance -- the same M8
    reasoning `--policy replay` above already follows); `--editor
    mutation` without `--seed` (a default would make an unreproducible
    run look reproducible); and any OTHER editor WITH a `--seed`, since
    only `mutation` draws from one and recording an unused seed in the
    winner's lineage taint and `run.json` would misattribute the run.
  * an unreadable/malformed `--trace`, or a malformed `--edits` file.
    The parent goes through `_read_trace_or_refuse`, not a bare `read`:
    `tuxghost.trace.read` starts with an unguarded `json.loads` and ends
    with `Trace.model_validate`, so a non-JSON or structurally malformed
    trace raises `json.JSONDecodeError`/`pydantic.ValidationError` --
    neither an `OSError` nor a `Refused` -- and would otherwise exit 1
    for what is plainly a bad input file.
  * anything `ValueError`/`TypeError` out of `optimize(...)`: an
    unliftable parent (two buttons held at once -- `optimize.schedule
    .lift` refuses rather than reinterprets) and a malformed
    `parent.initial_state` (`pydantic.ValidationError` IS a `ValueError`
    subclass) are both statements about the PARENT, and both are refusals
    about input. Note what is NOT here: a bad EDITOR proposal never
    reaches this boundary at all -- `tuxghost.optimize.runner` contains
    it as a rejected round and the run continues.
  * a re-seal of the winning script that reaches a different digest than
    the loop scored (see `_optimize`: the winner is re-sealed through
    `Recorder`, this project's one trace writer, rather than having its
    sealed provenance patched). That re-seal deliberately does NOT pass
    `max_cost`: `optimize()` seals round 0 -- the parent -- without it by
    design, so the winning script may legally be an over-budget parent,
    and enforcing a proposal budget against a script the loop already
    accepted turned a plain `--max-cost 400` into an uncaught
    `OverBudget` and exit 1 (measured, task 11 review round 1).
  * an `anthropic.AnthropicError` out of a `--editor claude` run
    (`RateLimitError`, `AuthenticationError`, ...). These derive from
    `Exception`, not `ValueError`/`TypeError`, so neither
    `tuxghost.optimize.runner._propose`'s rejected-round boundary nor
    the `(ValueError, TypeError)` arm below sees them; a second arm,
    gated on `args.editor == "claude"` and re-raising anything else,
    maps them to 2. Exactly `_agent`'s shape, for exactly its reason.

That `optimize(...)` boundary is `except (ValueError, TypeError)` and
deliberately not `except Exception`: measured during task 11, widening it
laundered a real `AttributeError` engine-invariant failure into
`refused: ...` + exit 2 while every other CLI test stayed green
(`tests/test_optimize_cli.py
::test_a_real_engine_bug_out_of_the_loop_is_not_laundered_into_a_refusal`
pins it). And the `anthropic` import is gated on `args.editor ==
"claude"`, not import-guarded at module scope, for exactly the reason
commit 61f63aa gives for `_agent`: `make check` does not install the SDK,
and an unconditional import on a path all four editors reach replaces a
genuine crash's traceback with `ModuleNotFoundError` raised while
handling it. `optimize` renders nothing -- no frames directory, ever.
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
    from tuxghost.optimize.runner import Round

#: Which of each subcommand's parsed arguments are paths that must be
#: `.resolve()`d against the caller's cwd before any chdir happens.
_PATH_ARGS: dict[str, tuple[str, ...]] = {
    "execute": ("trace",),
    "verify": ("trace",),
    "info": ("trace",),
    "compare": ("a", "b"),
    "record": ("out", "from_save"),
    "agent": ("actions", "from_save", "out", "run_dir"),
    "optimize": ("trace", "edits", "out", "run_dir"),
    "export": ("trace", "out"),
    # "ghost" is OPTIONAL (plain windowed play without a ghost passes
    # `--ghost` as None) -- `_resolve_paths` already skips any argument
    # whose value is `None`, so listing it here is safe either way.
    "play": ("ghost", "from_save", "out", "run_dir"),
}

#: Subcommands that need the vendored `tuxemon/` package importable and
#: cwd-relative asset loading working -- i.e. anything that actually boots
#: a session. `info` and `compare` are pure trace-file inspection and need
#: neither.
_NEEDS_BOOTSTRAP = frozenset(
    {"execute", "verify", "record", "agent", "optimize", "play", "export"}
)


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

    p_opt = sub.add_parser(
        "optimize",
        help="edit a trace offline, re-run each candidate, keep the best",
    )
    p_opt.add_argument("--trace", type=Path, required=True)
    p_opt.add_argument(
        "--editor",
        choices=("scripted", "mutation", "replay", "claude"),
        required=True,
    )
    # No argparse default for --seed: `mutation` REQUIRES one (a default
    # would make an unreproducible run look reproducible) and the other
    # three editors have no use for one.
    p_opt.add_argument("--seed", type=int, default=None)
    p_opt.add_argument("--edits", type=Path, default=None)
    p_opt.add_argument("--model", default=None)
    p_opt.add_argument("--objective", choices=("reach-tile",), required=True)
    p_opt.add_argument("--target", required=True)
    # Mirrors `agent`'s `--goal`, and for the same reason: `--editor
    # claude` is the only editor that reads it, and without it the model
    # was sent the action list, the end tile and a bare score tuple with
    # NO statement of what it was optimizing toward (whole-branch review,
    # Important 1 -- and a repeat of the defect the comment at the
    # `--goal`-never-reached-the-run site below records). Unlike
    # `agent`'s, this one is NOT left empty when unset: `--objective`
    # plus `--target` already say what the goal is, so `_optimize`
    # derives a default from them (`_derived_goal`) rather than sending
    # nothing. Set it explicitly to say something the target cannot,
    # e.g. "get there without entering the tall grass".
    p_opt.add_argument("--goal", default="")
    p_opt.add_argument("--rounds", type=int, required=True)
    p_opt.add_argument("--patience", type=int, required=True)
    p_opt.add_argument(
        "--max-rejections", type=int, required=True, dest="max_rejections"
    )
    p_opt.add_argument(
        "--max-cost", type=int, required=True, dest="max_cost",
        help=(
            "Step ceiling on what an EDITOR MAY PROPOSE, not on the run: "
            "round 0 seals the parent without it (refusing the baseline "
            "would leave nothing to compare against), so a budget below "
            "the parent's own cost is not an error and still writes a "
            "parent-cost trace."
        ),
    )
    p_opt.add_argument("--checkpoint", type=int, default=64)
    p_opt.add_argument("--out", type=Path, required=True)
    p_opt.add_argument("--run-dir", type=Path, required=True, dest="run_dir")

    p_exp = sub.add_parser(
        "export", help="Write a trace as a videogamebench trajectory JSON."
    )
    p_exp.add_argument("--trace", type=Path, required=True)
    p_exp.add_argument("--out", type=Path, required=True)

    p_play = sub.add_parser(
        "play", help="Play in a window with a ghost, recording the session."
    )
    # Optional, unlike every other trace-path argument in this parser:
    # plain windowed play with no ghost at all is a legitimate mode
    # (`tuxghost.play.play` treats `None` as "no ghost installed").
    p_play.add_argument("--ghost", type=Path, default=None)
    p_play.add_argument(
        "--from-save", type=Path, required=True, dest="from_save"
    )
    p_play.add_argument("--seed", type=int, required=True)
    p_play.add_argument(
        "--clock-epoch", type=int, required=True, dest="clock_epoch"
    )
    p_play.add_argument("--out", type=Path, required=True)
    p_play.add_argument("--run-dir", type=Path, required=True, dest="run_dir")

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


def _parse_target(raw: str) -> tuple[str, tuple[int, int]] | None:
    """`MAP:X,Y` -> `(map, (x, y))`, or None if it does not parse."""
    map_name, _, coords = raw.partition(":")
    x, _, y = coords.partition(",")
    if not map_name or not x or not y:
        return None
    try:
        return map_name, (int(x), int(y))
    except ValueError:
        return None


def _derived_goal(objective: str, target: tuple[str, tuple[int, int]]) -> str:
    """The default `--goal` when the caller gives none.

    `--objective`/`--target` already say what the run is optimizing
    toward, so an unset `--goal` derives from them rather than leaving
    `ClaudeEditor` blindfolded (whole-branch review, Important 1). Only
    `reach-tile` exists; a new objective must extend this rather than
    fall through to a stale sentence about tiles, so the `else` raises.
    """
    map_name, (x, y) = target
    if objective == "reach-tile":
        return f"reach tile ({x}, {y}) on map {map_name}"
    raise AssertionError(  # pragma: no cover -- argparse `choices` refuses this
        f"unhandled objective {objective!r}"
    )


def _round_json(entry: Round) -> dict[str, Any]:
    """One `runner.Round` as a JSON object, with its edits serialized
    through `Edit.to_json` rather than `dataclasses.asdict`.

    Whole-branch review, Important 2. `asdict` recurses by FIELD, and
    `Insert` and `Replace` have identical field sets, so an
    `asdict`-logged round could not be told apart from the other and fed
    back through `edits_from_json` raised `edit 0: unknown op None`
    (measured). The spec's claim that `ReplayEditor` "is what makes an
    LLM-driven optimization auditable after the fact" only holds if
    `optimize.jsonl`'s rows can be read by `--edits`, and after an
    `--editor claude` run -- the one editor the spec calls unreproducible
    by construction -- that log is the only record of what was proposed.
    """
    import dataclasses

    row: dict[str, Any] = dataclasses.asdict(entry)
    row["edits"] = [edit.to_json() for edit in entry.edits]
    return row


def _optimize(args: argparse.Namespace) -> int:
    """Optimize a trace offline. Exits 0 or 2 for everything it can
    anticipate; 1 is reserved for a genuine engine bug, exactly as
    `_agent` documents -- there is nothing here to diverge FROM."""
    from tuxghost.optimize.editors.mutation import MutationEditor
    from tuxghost.optimize.editors.replay import ReplayEditor, edits_from_json
    from tuxghost.optimize.editors.scripted import ScriptedEditor
    from tuxghost.optimize.objective import ReachTile
    from tuxghost.optimize.runner import Editor, optimize
    from tuxghost.optimize.seal import seal
    from tuxghost.trace import write

    for name, value in (
        ("--rounds", args.rounds),
        ("--patience", args.patience),
        ("--max-rejections", args.max_rejections),
        ("--max-cost", args.max_cost),
    ):
        if value < 1:
            print(f"refused: {name} must be >= 1, got {value!r}", file=sys.stderr)
            return 2
    # Checked separately, not folded into the loop above with an
    # `args.checkpoint + 1` trick: 0 is a legal checkpoint (it means "do
    # not sample"), so this bound is >= 0 and saying so plainly beats
    # reusing a >= 1 loop by adding one.
    if args.checkpoint < 0:
        print(
            f"refused: --checkpoint must be >= 0, got {args.checkpoint!r} "
            "(0 means do not sample)",
            file=sys.stderr,
        )
        return 2

    target = _parse_target(args.target)
    if target is None:
        print(
            f"refused: --target {args.target!r} does not parse; expected "
            "MAP:X,Y (e.g. spyder_paper_town.tmx:11,16)",
            file=sys.stderr,
        )
        return 2

    if args.editor in ("scripted", "replay") and args.edits is None:
        print(
            f"refused: --editor {args.editor} requires --edits "
            "(a JSONL file, one round of edits per line)",
            file=sys.stderr,
        )
        return 2
    if args.editor == "mutation" and args.seed is None:
        print(
            "refused: --editor mutation requires --seed; without one the "
            "run is unreproducible and would still look reproducible",
            file=sys.stderr,
        )
        return 2
    # The converse, and a REFUSAL rather than a silent drop (Task 11
    # review round 1, Important 5): `mutation` is the only editor that
    # draws from a seed, so `--editor replay --seed 7` used to write
    # `derived by offline-agent (replay, seed 7)` into the winner's
    # lineage taint and `"seed": 7` into `run.json` -- a seed no editor
    # ever used, recorded as if it had produced the run. That is the same
    # misattribution this module already refuses `--editor replay`
    # without `--model` to avoid, so it gets the same answer. Refusing
    # rather than just dropping the taint clause: dropping it would still
    # leave the bogus seed in `run.json`, and silently ignoring a flag
    # the user deliberately typed teaches them it did something.
    if args.editor != "mutation" and args.seed is not None:
        print(
            f"refused: --editor {args.editor} has no use for --seed "
            f"(got {args.seed!r}); only --editor mutation draws from one, "
            "and recording an unused seed would misattribute the run",
            file=sys.stderr,
        )
        return 2
    # The same rule for `--goal`, and for the same stated reason: only
    # `--editor claude` reads one, so `--editor mutation --goal "..."`
    # used to run to completion having silently discarded the single flag
    # that says what the user wanted. `run.json` records `null` there,
    # which is TRUE -- no goal reached any editor -- but a true record of
    # a dropped flag is not the same as telling the user it was dropped.
    # "Silently ignoring a flag the user deliberately typed teaches them
    # it did something", as the comment above puts it.
    if args.editor != "claude" and args.goal:
        print(
            f"refused: --editor {args.editor} has no use for --goal "
            f"(got {args.goal!r}); only --editor claude reads one, and "
            "running as if it had been applied would silently discard it",
            file=sys.stderr,
        )
        return 2

    # Task 11 review of the plan's own code: the plan read the parent with
    # a bare `read(args.trace)` under `except (OSError, Refused)`, which
    # leaves a real hole in this subcommand's whole reason for existing.
    # `tuxghost.trace.read`'s FIRST statement is an unguarded
    # `json.loads(...)`, and its last is `Trace.model_validate(...)`: a
    # trace file that is not valid JSON raises `json.JSONDecodeError` and
    # a structurally malformed one raises `pydantic.ValidationError`,
    # neither of them `OSError` or `Refused`. Both would have propagated
    # uncaught and exited 1 = "diverged" for what is plainly a bad input
    # file. `_read_trace_or_refuse` is this module's existing, single
    # trace-reading boundary and already maps all four failure modes to
    # one refusal (see its docstring), so `optimize` uses it rather than
    # growing a second, weaker copy. `allow_mismatch=False`: `optimize`
    # exposes no `--allow-mismatch` flag, and silently downgrading a
    # header mismatch on the trace every candidate is derived FROM is not
    # a decision this subcommand should make on the caller's behalf.
    parent = _read_trace_or_refuse(args.trace, False)
    if parent is None:
        return 2

    editor: Editor
    model: str | None = None
    #: The goal string actually handed to an editor, or `None` when no
    #: editor read one. Declared here beside `model`, and for the same
    #: reason: it is set only in the branch that genuinely uses it, so
    #: `run.json` records what influenced the run rather than what
    #: happened to be on the command line. `--editor claude` is the only
    #: consumer, so `None` for the other three is a true statement, not
    #: a dropped field (handoff item A3).
    goal: str | None = None
    if args.editor == "mutation":
        editor = MutationEditor(seed=args.seed)
    elif args.editor == "replay":
        if args.model is None:
            print(
                "refused: --editor replay requires --model; a transcript "
                "carries no model of its own and recording a default "
                "would misattribute provenance",
                file=sys.stderr,
            )
            return 2
        try:
            editor = ReplayEditor(args.edits)
        except OSError as exc:
            print(f"refused: cannot read {args.edits}: {exc}", file=sys.stderr)
            return 2
        except (ValueError, TypeError) as exc:
            print(f"refused: {args.edits} is malformed: {exc}", file=sys.stderr)
            return 2
        model = args.model
    elif args.editor == "scripted":
        try:
            rounds_of_edits = [
                edits_from_json(json.loads(line))
                for line in args.edits.read_text().splitlines()
                if line.strip()
            ]
        except OSError as exc:
            print(f"refused: cannot read {args.edits}: {exc}", file=sys.stderr)
            return 2
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"refused: {args.edits} is malformed: {exc}", file=sys.stderr)
            return 2
        editor = ScriptedEditor(rounds_of_edits)
    elif args.editor == "claude":
        # Gated on the policy, NOT import-guarded: this arm is reachable
        # only for `--editor claude`, so the other three never touch an
        # optional dependency `make check` does not install (the 61f63aa
        # residual fix, same reasoning).
        #
        # Spelled as an explicit `elif args.editor == "claude"` rather
        # than the plan's bare `else` (Task 11: the plan's own code and
        # its own test contradicted each other here). The plan's `else`
        # arm never mentions `claude`, so its
        # `test_the_claude_editor_is_the_only_one_that_touches_anthropic`
        # -- which greps this file for exactly `args.editor == "claude"`
        # -- failed against the plan's implementation. Naming the branch
        # is also the honest form: the gate IS a policy check, and the
        # unreachable fourth case below says so instead of quietly
        # treating "anything else" as claude.
        try:
            # No unused-import suppression here any more (review round
            # 1): this binding is a real precondition check AND the name
            # is genuinely used again later in this same function scope,
            # in the gated `except Exception` arm's
            # `isinstance(exc, anthropic.AnthropicError)` below -- ruff's
            # unused-import check is scope-wide, not per-binding, so it
            # no longer considers this import unused at all. Exactly the
            # arrangement `_agent` arrived at. `ClaudeEditor` imports
            # `anthropic` lazily inside `_ensure_client`, deep inside the
            # loop, so without this check a missing SDK would raise
            # there, uncaught, and exit 1 = "diverged".
            import anthropic
        except ImportError as exc:
            print(
                f"refused: --editor claude needs the anthropic SDK "
                f"(pip install -e '.[agent]'): {exc}",
                file=sys.stderr,
            )
            return 2
        from tuxghost.agent.claude import DEFAULT_MODEL
        from tuxghost.optimize.editors.claude import ClaudeEditor

        model = args.model if args.model is not None else DEFAULT_MODEL
        # `goal` and `score_legend` are BOTH required for this editor to
        # be told what it is optimizing toward (whole-branch review,
        # Important 1). Without the first the prompt named no target at
        # all; without the second the prompt's only quantitative feedback
        # was three unlabelled numbers (`[0.0, -2.0, -442.0]`) whose
        # order and sign the model had to guess. `ReachTile.TERMS` is the
        # objective's own statement of its terms, so the legend cannot
        # drift out of step with `ReachTile.score`.
        # Computed once and reused for `run.json` below rather than
        # recomputed there: an explicit `--goal` is NOT reconstructible
        # from anything else the run directory records, and a derived one
        # recomputed at the writer could silently disagree with the one
        # the model was actually sent.
        goal = args.goal or _derived_goal(args.objective, target)
        editor = ClaudeEditor(
            model=model,
            goal=goal,
            score_legend=", ".join(ReachTile.TERMS),
        )
    else:  # pragma: no cover -- argparse `choices` already refuses this
        raise AssertionError(f"unhandled editor {args.editor!r}")

    try:
        result = optimize(
            parent,
            editor,
            ReachTile(target[0], target[1]),
            rounds=args.rounds,
            patience=args.patience,
            max_rejections=args.max_rejections,
            max_cost=args.max_cost,
            checkpoint=args.checkpoint,
            model=model,
        )
    except (ValueError, TypeError) as exc:
        # Deliberately narrow, like `_agent`'s boundary: these are the two
        # exceptions this project's own input validation raises (an
        # unliftable parent among them, via `lift`, and a malformed
        # `parent.initial_state` via `SaveData.model_validate` inside
        # `seal` -- `pydantic.ValidationError` IS a `ValueError`
        # subclass). An AttributeError here is a programming error and
        # must stay a visible crash.
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Task 11 review round 1, Important 2 -- the same arm, with the
        # same gate, that `_agent` grew in the 61f63aa residual fix, and
        # for the identical reason. `ClaudeEditor.propose` calls the
        # `anthropic` SDK's `messages.create` from inside
        # `tuxghost.optimize.runner._propose`, whose own boundary is
        # `(ValueError, TypeError)` -- and every error that call can
        # raise (`RateLimitError`, `APIConnectionError`,
        # `AuthenticationError`, a missing/invalid API key, ...) derives
        # from `anthropic.AnthropicError(Exception)`, none of them a
        # `ValueError` or `TypeError`. So none of them is a rejected
        # round, none is caught by the arm above, and left alone each one
        # exits 1 = "diverged" for what is plainly an API problem.
        #
        # Gated on `args.editor == "claude"`, NOT import-guarded: this
        # `try` is shared by all four editors, `anthropic.AnthropicError`
        # can only originate from `ClaudeEditor`, and `make check` does
        # not install the SDK -- so an unconditional import here would
        # replace a genuine engine crash's traceback on `--editor
        # mutation` with `ModuleNotFoundError` raised while handling it.
        # That is exactly the defect 61f63aa fixed. The bare `raise`
        # keeps every other editor's real exception unmodified.
        if args.editor != "claude":
            raise
        import anthropic

        if not isinstance(exc, anthropic.AnthropicError):
            raise
        print(
            f"refused: --editor claude call to the Anthropic API failed: {exc}",
            file=sys.stderr,
        )
        return 2

    # The seed clause is reachable only for `--editor mutation` now (see
    # the refusal above); the `is not None` guard stays as the local,
    # readable statement of that rather than an assumption about a check
    # 80 lines up.
    # The count is rounds that ACTUALLY PROPOSED EDITS, not
    # `len(result.rounds) - 1` (whole-branch review, Also-fix 8). Round 0
    # is the parent and the STOP round is deliberately appended to the
    # log, so an editor that stopped immediately -- applying zero edits
    # -- was recorded as "over 1 round(s)". Cosmetic, but the taint is
    # this branch's sole lineage record. A round whose `propose` RAISED
    # also has no edits and is also not counted: it proposed nothing.
    edited_rounds = sum(1 for entry in result.rounds if entry.edits)
    lineage = (
        f"derived by offline-agent ({args.editor}"
        + (f", seed {args.seed}" if args.seed is not None else "")
        + f") from parent {parent.header.final_digest} over "
        f"{edited_rounds} round(s)"
    )
    # RE-SEAL the winner with its lineage rather than rewriting the trace
    # the loop already sealed. The round count is only known now, and
    # `Recorder` is this project's one trace writer -- editing a sealed
    # `Provenance` here would make this a second one. The extra engine
    # run is also a free re-confirmation that the winning script still
    # reaches the score it won with; the check below is that
    # confirmation.
    # NO `max_cost=` here, deliberately -- do not "restore consistency"
    # by adding it back (Task 11 review round 1, Critical). `max_cost`
    # bounds what an EDITOR MAY PROPOSE, not what the parent already IS:
    # `optimize()`'s round 0 seals the parent WITHOUT it by design
    # (`tuxghost/optimize/runner.py`, pinned by
    # `tests/test_optimize_runner.py::test_round_zero_is_sealed_without_
    # max_cost`), because refusing the baseline would leave the run with
    # nothing to compare against. So `best_script` can legally be an
    # over-budget PARENT script, and this re-seal sits OUTSIDE the
    # `except (ValueError, TypeError)` boundary above -- passing
    # `max_cost` here raised `OverBudget` uncaught and exited 1 for a
    # plain `--max-cost 400` against the 442-step parent, measured. This
    # call is not a proposal; it is a re-run of a script the loop already
    # accepted, and `_prepare` enforced the budget on every accepted
    # candidate at the point where the budget means something.
    final = seal(
        result.best_script,
        parent,
        checkpoint=args.checkpoint,
        taints=(lineage,),
        model=model,
    )
    if final.trace.header.final_digest != result.best.trace.header.final_digest:
        print(
            "refused: re-sealing the winning script reached a different "
            f"digest ({final.trace.header.final_digest}) than the loop "
            f"recorded ({result.best.trace.header.final_digest}); this is "
            "a determinism failure, not a bad edit",
            file=sys.stderr,
        )
        return 2
    write(final.trace, args.out)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "optimize.jsonl").open("w") as handle:
        for entry in result.rounds:
            handle.write(json.dumps(_round_json(entry), default=str) + "\n")
    # `answers.jsonl`: the model's OWN replies, verbatim, one per line --
    # the counterpart to `agent`'s `decisions.jsonl` and its `raw` field,
    # which S2 added for exactly this reason and `optimize` has lacked
    # ever since.
    #
    # `optimize.jsonl` records what was PARSED out of each reply, so it
    # can say nothing about a reply that failed to parse -- and
    # `parse_response` raising on a missing json fence is the one live
    # failure this project has already been bitten by. Without this file,
    # `ClaudeEditor.parse_response` cannot be tested against real model
    # text at all, because no real model text survives a run.
    #
    # `getattr`, not an isinstance check: `answers` is a `ClaudeEditor`
    # detail and the other three editors have no equivalent, so this
    # writes the file only when there is something to put in it rather
    # than leaving an empty artifact that implies a capture happened.
    answers = getattr(editor, "answers", None)
    if answers:
        with (args.run_dir / "answers.jsonl").open("w") as handle:
            for answer in answers:
                handle.write(json.dumps(answer) + "\n")
    (args.run_dir / "run.json").write_text(
        json.dumps(
            {
                "parent_digest": parent.header.final_digest,
                "goal": goal,
                "objective": args.objective,
                "target": args.target,
                "editor": args.editor,
                "seed": args.seed,
                "model": model,
                "rounds": args.rounds,
                "patience": args.patience,
                "max_rejections": args.max_rejections,
                "max_cost": args.max_cost,
                "best_round": result.best_round,
                "stop_reason": result.stop_reason,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} (best round {result.best_round}) and {args.run_dir}")
    return 0


def _export(path: Path, out: Path) -> int:
    """Write `path` (a `.tuxghost` trace) as a videogamebench trajectory
    JSON at `out`. Reads through `_read_trace_or_refuse`, this module's one
    trace-reading boundary, so a missing or malformed `--trace` is a
    REFUSED precondition (exit 2), exactly like every other subcommand
    here -- never the uncaught traceback (exit 1) that a bare
    `tuxghost.trace.read` call would produce. `export` has nothing to
    diverge FROM (it is a pure format conversion, not a replay), so like
    `agent`/`optimize` it can only ever return 0 or 2."""
    from tuxghost.vgbench import trajectory_of

    trace = _read_trace_or_refuse(path, False)
    if trace is None:
        return 2
    out.write_text(json.dumps(trajectory_of(trace), indent=2))
    print(f"wrote {out}")
    return 0


def _ghost_trace_map_or_refuse(trace: Trace) -> bool:
    """`True` if `trace.initial_state` is a bootable `SaveData` whose
    `npc_state.current_map` resolves to a real map asset; otherwise prints
    a `refused: ...` line and returns `False`.

    Whole-branch review, I3: `tuxghost.ghost.track.build_track`'s own
    docstring claims it may skip this check "because its caller reads the
    trace through `_read_trace_or_refuse` first" -- that claim was false
    until this function existed. `_read_trace_or_refuse` only maps
    `Refused`/`OSError`/JSON/`ValidationError` on the trace FILE itself; it
    never calls `resolve_map_asset` on `trace.initial_state`, so a ghost
    trace whose `npc_state.current_map` cannot be resolved reached
    `tuxghost.boot.boot_from_save` (via `build_track`, inside
    `tuxghost.play.play`, AFTER `pygame_init()` had already opened a
    window) and raised a bare `ValueError` there -- an exit-1 traceback for
    what is a refusable precondition, not a divergence. Mirrors
    `tuxghost.execute.execute`'s own preamble checks (same order, same
    messages) so a ghost trace and a played-back trace are refused
    identically."""
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import resolve_map_asset

    try:
        save_data = SaveData.model_validate(trace.initial_state)
    except ValidationError as exc:
        print(
            f"refused: --ghost trace's initial_state does not validate as "
            f"a SaveData: {exc}",
            file=sys.stderr,
        )
        return False

    if save_data.npc_state is None or save_data.npc_state.current_map is None:
        print(
            "refused: --ghost trace's initial_state.npc_state.current_map "
            "is required to build a ghost track (see tuxghost.boot"
            ".boot_from_save)",
            file=sys.stderr,
        )
        return False

    if resolve_map_asset(save_data.npc_state.current_map) is None:
        print(
            "refused: --ghost trace's initial_state.npc_state.current_map="
            f"{save_data.npc_state.current_map!r} resolves to no map asset "
            "(tried it as given and with a .tmx extension)",
            file=sys.stderr,
        )
        return False

    return True


def _play(args: argparse.Namespace) -> int:
    """Wire the one windowed entry point (`tuxghost.play.play`) behind this
    module's usual refusal boundary.

    Two preconditions are checked here, both BEFORE `tuxghost.play.play`
    (and so `pygame_init()`) is ever called:

    * `--from-save`, through `_agent_save_or_refuse` -- the same read,
      parse, validate and map-resolve sequence `agent` already uses.
      Whole-branch review, I2: an earlier version of this function never
      validated `--from-save` at all, so `--from-save nope.save` opened a
      window via `pygame_init()` and then raised an uncaught
      `FileNotFoundError` (exit 1) out of `tuxghost.play.play`'s own
      `save.read_text()` -- indistinguishable from "diverged", for what is
      a refusable precondition like every other subcommand's.
    * `--ghost`, when given, is optional -- `None` means plain windowed
      play with no ghost installed -- but when it IS given, it must be
      read through `_read_trace_or_refuse` here, THEN map-checked through
      `_ghost_trace_map_or_refuse` (I3, see that function's docstring): a
      missing, malformed, or map-unresolvable ghost trace is a REFUSED
      precondition (exit 2), not the uncaught traceback (exit 1) that
      would result if `tuxghost.play.play`'s own internal `read()`/
      `build_track()` calls were left to hit it first, deep inside a
      function that has already called `pygame_init()`.

    See the module docstring's refuse/exit-1 distinction, and
    `tuxghost.play`'s own docstring for why this is the sole subcommand
    exempt from the dummy-SDL rule."""
    if _agent_save_or_refuse(args) is None:
        return 2

    if args.ghost is not None:
        ghost_trace = _read_trace_or_refuse(args.ghost, False)
        if ghost_trace is None:
            return 2
        if not _ghost_trace_map_or_refuse(ghost_trace):
            return 2

    from tuxghost.play import play

    return play(
        ghost_trace=args.ghost,
        save=args.from_save,
        seed=args.seed,
        clock_epoch=args.clock_epoch,
        out=args.out,
        run_dir=args.run_dir,
    )


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
    if command == "optimize":
        return _optimize(args)
    if command == "play":
        return _play(args)
    if command == "export":
        return _export(args.trace, args.out)
    if command == "verify":
        return _verify(args.trace, args.allow_mismatch)
    if command == "execute":
        return _execute(args.trace, args.allow_mismatch, args.checkpoint)

    raise AssertionError(f"unhandled command {command!r}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
