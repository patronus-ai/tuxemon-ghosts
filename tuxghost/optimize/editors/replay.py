"""Replays a recorded optimization's edits, round by round.

`edits_from_json` is shared with `ClaudeEditor`: a model's answer and a
transcript of that answer must be validated identically, or the replay
would accept what the live path rejected. Same arrangement
`actions_from_json` gives `ClaudePolicy`/`ReplayPolicy`.

The whole transcript is validated at CONSTRUCTION. A malformed round is
knowable before the game boots, so failing now saves a pointless run and
matches how `--policy scripted` already pre-parses its actions file
(S2 task 8 review, Ruling X).

It deliberately does NOT loop when exhausted: a transcript that silently
restarted would make a replayed run diverge from the run it claims to
reproduce, which is the one thing a replay must not do.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tuxghost.agent.replay import actions_from_json
from tuxghost.optimize.edits import Delete, Edit, Insert, Replace
from tuxghost.optimize.schedule import ActionScript
from tuxghost.optimize.seal import CandidateResult


def _one(raw: Any, position: int) -> Edit:
    if not isinstance(raw, dict):
        raise TypeError(
            f"edit {position} must be an object, got {type(raw).__name__}"
        )
    op = raw.get("op")
    # Op validity is checked BEFORE index/action presence: a round
    # bearing an unrecognized op (e.g. `{"op": "nope"}`, no `index` at
    # all) must be refused as "unknown op", not "missing index" -- the
    # more specific and more actionable diagnosis of the two.
    if op not in ("delete", "insert", "replace"):
        raise ValueError(
            f"edit {position}: unknown op {op!r}; expected one of "
            "'insert', 'delete', 'replace'"
        )
    if "index" not in raw:
        raise ValueError(f"edit {position} is missing 'index'")
    index = int(raw["index"])
    if op == "delete":
        return Delete(index)
    if "action" not in raw:
        raise ValueError(f"edit {position} ({op}) is missing 'action'")
    # Reuses S2's validation so an edit's action is checked by exactly
    # the rules a policy's action is.
    (action,) = actions_from_json([raw["action"]])
    return Insert(index, action) if op == "insert" else Replace(index, action)


def edits_from_json(raw: Any) -> tuple[Edit, ...]:
    """Build edits from a decoded JSON list, refusing anything partial."""
    if not isinstance(raw, list):
        raise TypeError(f"edits must be a JSON list, got {type(raw).__name__}")
    return tuple(_one(item, i) for i, item in enumerate(raw))


class ReplayEditor:
    def __init__(self, transcript: Path) -> None:
        lines = Path(transcript).read_text().splitlines()
        self._rounds = [
            edits_from_json(json.loads(line)) for line in lines if line.strip()
        ]
        self._index = 0

    def propose(
        self,
        script: ActionScript,
        candidate: CandidateResult,
        score: tuple[float, ...],
    ) -> Sequence[Edit]:
        del script, candidate, score
        if self._index >= len(self._rounds):
            return ()
        proposal = self._rounds[self._index]
        self._index += 1
        return proposal
