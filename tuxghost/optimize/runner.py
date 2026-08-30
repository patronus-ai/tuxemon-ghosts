"""The edit -> seal -> score loop.

Round 0 is the parent, sealed and scored, so improvement is measured
against something this process actually ran rather than against a digest
recorded by another one.

An editor is UNTRUSTED INPUT -- `ClaudeEditor` parses a language model's
JSON. A malformed answer is a rejected round: logged with its reason,
patience and the consecutive-rejection counter incremented, loop
continues, because one malformed answer must not kill a 20-round run
(the spec states this as a requirement, not a nicety).

That containment covers BOTH ways an editor can be malformed, and the
distinction is not academic:

  * `propose` itself raising -- `_propose` below. `json.JSONDecodeError`
    IS a `ValueError` subclass, and Task 10's `ClaudeEditor` raises
    `ValueError` for a model reply with no JSON fence, so an uncontained
    `propose` would let one bad reply end a whole run.
  * a proposal that parses but does not validate -- `_prepare` below,
    out of `apply_edits`.

A rejected `propose` is NOT routed through the STOP path: an empty
proposal is a deliberate "stop now" answer, a raise is a broken answer,
and reporting the second as the first would make a crashing editor look
like a satisfied one.

WHICH exceptions mean "the editor is at fault" is the whole subtlety
here, and it is why `_prepare` has TWO `try` blocks rather than one (see
its docstring). Both editor boundaries catch exactly
`(ValueError, TypeError)` and NOT `Exception`: an `AttributeError` out
of an editor is a programming error, and laundering it into a tidy
rejected round would hide it for the whole run. `seal`'s boundary stays
narrower still -- `OverBudget` alone. `objective.score` is deliberately
called outside every handler: a malformed `final_state` is not the
editor's fault either, and `ReachTile.score` refuses such a state with a
`ValueError` rather than inventing a position.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from tuxghost.optimize.edits import Edit, apply_edits
from tuxghost.optimize.objective import Objective, RulesObjective
from tuxghost.optimize.schedule import ActionScript, lift
from tuxghost.optimize.seal import CandidateResult, OverBudget, seal
from tuxghost.rules import GameRules
from tuxghost.trace import Trace


class Editor(Protocol):
    def propose(
        self,
        script: ActionScript,
        candidate: CandidateResult,
        score: tuple[float, ...],
    ) -> Sequence[Edit]:
        """Edits to apply to `script`. An EMPTY sequence means STOP -- a
        valid answer, not an error (the same convention
        `tuxghost.agent.types.STOP` sets for policies)."""
        ...


@dataclass
class Round:
    index: int
    edits: tuple[Edit, ...]
    score: tuple[float, ...] | None = None
    steps: int | None = None
    digest: str | None = None
    accepted: bool = False
    rejected_reason: str | None = None


@dataclass
class OptimizeResult:
    best: CandidateResult
    best_script: ActionScript
    best_round: int
    rounds: list[Round] = field(default_factory=list)
    stop_reason: str = "rounds exhausted"


@dataclass(frozen=True)
class _Rejection:
    """Why a proposal never became a scored candidate. The reason is the
    EDITOR's fault by construction -- see `_prepare`."""

    reason: str


def _propose(
    editor: Editor,
    best_script: ActionScript,
    best: CandidateResult,
    best_score: tuple[float, ...],
) -> tuple[Edit, ...] | _Rejection:
    """Ask the editor, or say why asking failed.

    `ClaudeEditor` parses a language model's reply here, and
    `json.JSONDecodeError` is a `ValueError` subclass -- so without this
    boundary one un-fenced model reply ends a 20-round run, which the
    spec explicitly forbids.

    `(ValueError, TypeError)` only, deliberately NOT `Exception`: an
    `AttributeError` from a buggy editor is a programming error and must
    stay a visible crash. An EMPTY tuple is a valid answer (STOP) and is
    handled by the caller, not here -- a raise and a deliberate stop are
    different events and must not be reported as the same one.
    """
    try:
        return tuple(editor.propose(best_script, best, best_score))
    except (ValueError, TypeError) as exc:
        return _Rejection(
            f"editor.propose raised {type(exc).__name__}: {exc}"
        )


def _prepare(
    best_script: ActionScript,
    proposal: tuple[Edit, ...],
    parent: Trace,
    *,
    checkpoint: int,
    model: str | None,
    max_cost: int,
    rules: GameRules | None,
) -> tuple[ActionScript, CandidateResult] | _Rejection:
    """Apply one proposal and run it, or say why the EDITOR is at fault.

    Two `try` blocks, never one, and the second catches only
    `OverBudget`. `pydantic.ValidationError` subclasses `ValueError`
    (measured, pydantic 2.13.4) and `seal` has three `ValueError`
    sources: its own `max_cost` refusal (`OverBudget`),
    `SaveData.model_validate` on a malformed `parent.initial_state`, and
    `boot_from_save`'s unresolvable-`current_map` raise. Only the first
    is a statement about the PROPOSAL; the other two are statements
    about the parent trace, which fails identically every round -- so a
    single broad `except ValueError` around both calls would report an
    unrunnable parent as "the editor kept proposing rubbish" and bury
    the real defect under a tidy rejected-round log. That conflation is
    exactly what `tuxghost/execute.py`'s refusal taxonomy exists to
    avoid.

    `apply_edits` keeps the wider `(ValueError, TypeError)` boundary --
    the two exceptions this project's own input validation raises
    (`validate_actions` raises both) -- and deliberately NOT
    `Exception`: an `AttributeError` out of an edit object is a
    programming error and must stay a visible crash.
    """
    try:
        script = apply_edits(best_script, proposal)
    except (ValueError, TypeError) as exc:
        return _Rejection(str(exc))

    try:
        candidate = seal(
            script,
            parent,
            checkpoint=checkpoint,
            model=model,
            max_cost=max_cost,
            rules=rules,
        )
    except OverBudget as exc:
        return _Rejection(str(exc))

    return script, candidate


def optimize(
    parent: Trace,
    editor: Editor,
    objective: Objective,
    *,
    rounds: int,
    patience: int,
    max_rejections: int,
    max_cost: int,
    checkpoint: int = 0,
    model: str | None = None,
    rules: GameRules | None = None,
) -> OptimizeResult:
    """Edit, seal, score, keep if better.

    Every bound is required rather than defaulted: no realistic value has
    been measured yet (see docs/2026-08-27-optimizer-tuning.org), and a
    default invented here would be mistaken for one that had.
    """
    # I2 (whole-branch review): `RulesObjective` scores
    # `(goal_state, max_progress, -steps)` entirely from `CandidateResult
    # .max_progress`/`.goal_step`/`.died`, which are populated ONLY when
    # `seal` is given `rules=...` (both call sites below). Combine
    # `RulesObjective` with no `rules` and every candidate silently scores
    # `(0.0, 0.0, -steps)` -- goal never reached, no progress ever seen --
    # which degenerates the whole search into "shortest wins" with no
    # error and no warning. Refusing the combination here, at the one
    # place both are given together, makes it structurally impossible to
    # reach that vacuity by accident.
    if isinstance(objective, RulesObjective) and rules is None:
        raise ValueError(
            "RulesObjective requires seal(rules=...); without it every "
            "candidate scores (0.0, 0.0, -steps), silently degenerating "
            "to 'shortest wins' (whole-branch review, I2)"
        )
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds!r}")
    for name, value in (
        ("patience", patience),
        ("max_rejections", max_rejections),
        ("max_cost", max_cost),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value!r}")
    # `>= 0`, not `>= 1`: 0 is the LEGAL "do not sample" value, so this
    # bound cannot be folded into the loop above. Checked here and not
    # only in `tuxghost.cli` (whole-branch review, Also-fix 1) because
    # `optimize()` is a boundary too, and a negative value is silently
    # catastrophic rather than merely wrong: `tuxghost.optimize.seal`'s
    # hook tests `i % checkpoint == 0`, and `i % -1` is 0 for EVERY step,
    # so `checkpoint=-1` takes a `digest_of` and a `state_of` snapshot on
    # all 442 of them without saying so.
    if checkpoint < 0:
        raise ValueError(
            f"checkpoint must be >= 0 (0 means do not sample), got "
            f"{checkpoint!r}"
        )

    base_script = lift(parent.inputs, parent.header.step_count)
    # Round 0 seals the parent WITHOUT `max_cost`: the parent is the
    # baseline, not a proposal, and refusing it would leave the run with
    # nothing to compare against.
    best = seal(
        base_script, parent, checkpoint=checkpoint, model=model, rules=rules
    )
    best_score = tuple(objective.score(best))
    score_length = len(best_score)
    best_script = base_script
    best_round = 0

    log = [
        Round(
            index=0,
            edits=(),
            score=best_score,
            steps=best.steps,
            digest=best.trace.header.final_digest,
            accepted=True,
        )
    ]

    stop_reason = "rounds exhausted"
    stale = 0
    consecutive_rejections = 0

    for index in range(1, rounds + 1):
        proposed = _propose(editor, best_script, best, best_score)
        outcome: tuple[ActionScript, CandidateResult] | _Rejection
        if isinstance(proposed, _Rejection):
            # A raise is a broken answer, not a STOP: it is logged as a
            # rejected round with no edits and the run continues.
            entry = Round(index=index, edits=())
            log.append(entry)
            outcome = proposed
        else:
            if not proposed:
                stop_reason = "editor returned STOP"
                log.append(
                    Round(index=index, edits=(), rejected_reason="STOP")
                )
                break

            entry = Round(index=index, edits=proposed)
            log.append(entry)
            outcome = _prepare(
                best_script,
                proposed,
                parent,
                checkpoint=checkpoint,
                model=model,
                max_cost=max_cost,
                rules=rules,
            )

        if isinstance(outcome, _Rejection):
            entry.rejected_reason = outcome.reason
            consecutive_rejections += 1
            stale += 1
            if consecutive_rejections >= max_rejections:
                stop_reason = (
                    f"editor rejected {consecutive_rejections} rounds "
                    "consecutively"
                )
                break
            if stale >= patience:
                stop_reason = "patience exhausted"
                break
            continue

        script, candidate = outcome
        consecutive_rejections = 0
        # Outside both `try` blocks in `_prepare`, deliberately: a
        # `ValueError` from here means the candidate's `final_state` is
        # malformed, which is not something the editor proposed.
        score = tuple(objective.score(candidate))
        if len(score) != score_length:
            raise ValueError(
                f"objective returned a score length of {len(score)} after "
                f"{score_length}; a varying score length is a bug, not a "
                "tie-break"
            )

        entry.score = score
        entry.steps = candidate.steps
        entry.digest = candidate.trace.header.final_digest

        improved = score > best_score
        # A TIE is ADOPTED while the goal is unreached, and this is the
        # other half of `RulesObjective`'s `-steps` fix rather than a
        # separate idea. Neutralising `-steps` before the goal stops the
        # objective PUNISHING an approach to it, but strict `>` then
        # rejects the approach anyway for merely tying -- measured: an
        # editor appending one UP action per round against
        # `hearthrock_idle_600` proposes on a 1-action script eight
        # times running, because every 2-action candidate ties at
        # `(0.0, 1020.0, 0.0)` and never displaces it. The script cannot
        # grow, so the thirteen tiles to the gym leader can never be
        # walked, so the goal that would finally break the tie is
        # unreachable.
        #
        # Scoped tightly: only for `RulesObjective`, and only while
        # `goal_step` is None. `ReachTile` scores `-steps`
        # unconditionally and is untouched. Once the goal IS reached,
        # `-steps` applies again and strict `>` resumes, so shortening a
        # winning run still requires a real improvement.
        explores = (
            score == best_score
            and isinstance(objective, RulesObjective)
            and candidate.goal_step is None
        )
        if improved or explores:
            best, best_score, best_script, best_round = (
                candidate,
                score,
                script,
                index,
            )
            entry.accepted = True
        if improved:
            stale = 0
        else:
            # A tie is adopted but is NOT progress: it still counts
            # toward patience, so an exploring run terminates on the same
            # bound as any other rather than wandering for `rounds`.
            stale += 1
            if stale >= patience:
                stop_reason = "patience exhausted"
                break

    return OptimizeResult(
        best=best,
        best_script=best_script,
        best_round=best_round,
        rounds=log,
        stop_reason=stop_reason,
    )
