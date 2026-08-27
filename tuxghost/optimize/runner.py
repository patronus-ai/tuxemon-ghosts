"""The edit -> seal -> score loop.

Round 0 is the parent, sealed and scored, so improvement is measured
against something this process actually ran rather than against a digest
recorded by another one.

An editor is UNTRUSTED INPUT -- `ClaudeEditor` parses a language model's
JSON. A proposal that fails validation is a rejected round: logged with
its reason, patience incremented, loop continues, because one malformed
answer must not kill a 20-round run.

WHICH exceptions mean "the editor is at fault" is the whole subtlety
here, and it is why `_prepare` below has TWO `try` blocks rather than one
(see its docstring). `objective.score` is deliberately called OUTSIDE
both of them: a malformed `final_state` is not the editor's fault
either, and `ReachTile.score` refuses such a state with a `ValueError`
rather than inventing a position.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from tuxghost.optimize.edits import Edit, apply_edits
from tuxghost.optimize.objective import Objective
from tuxghost.optimize.schedule import ActionScript, lift
from tuxghost.optimize.seal import CandidateResult, OverBudget, seal
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


def _prepare(
    best_script: ActionScript,
    proposal: tuple[Edit, ...],
    parent: Trace,
    *,
    checkpoint: int,
    model: str | None,
    max_cost: int,
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
) -> OptimizeResult:
    """Edit, seal, score, keep if better.

    Every bound is required rather than defaulted: no realistic value has
    been measured yet (see docs/2026-08-27-optimizer-tuning.org), and a
    default invented here would be mistaken for one that had.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds!r}")
    for name, value in (
        ("patience", patience),
        ("max_rejections", max_rejections),
        ("max_cost", max_cost),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value!r}")

    base_script = lift(parent.inputs, parent.header.step_count)
    # Round 0 seals the parent WITHOUT `max_cost`: the parent is the
    # baseline, not a proposal, and refusing it would leave the run with
    # nothing to compare against.
    best = seal(base_script, parent, checkpoint=checkpoint, model=model)
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
        proposal = tuple(editor.propose(best_script, best, best_score))
        if not proposal:
            stop_reason = "editor returned STOP"
            log.append(Round(index=index, edits=(), rejected_reason="STOP"))
            break

        entry = Round(index=index, edits=proposal)
        log.append(entry)

        outcome = _prepare(
            best_script,
            proposal,
            parent,
            checkpoint=checkpoint,
            model=model,
            max_cost=max_cost,
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

        if score > best_score:
            best, best_score, best_script, best_round = (
                candidate,
                score,
                script,
                index,
            )
            entry.accepted = True
            stale = 0
        else:
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
