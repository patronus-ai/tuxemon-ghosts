"""An editor that answers from a fixed list.

Drives every loop test in the gate: no network, no model, no key, no
randomness. Straight from `tuxghost.agent.scripted.ScriptedPolicy`'s
precedent, including "returns STOP when exhausted" -- an empty proposal
is a valid answer, not an error.
"""

from __future__ import annotations

from collections.abc import Sequence

from tuxghost.optimize.edits import Edit
from tuxghost.optimize.schedule import ActionScript
from tuxghost.optimize.seal import CandidateResult


class ScriptedEditor:
    def __init__(self, rounds: Sequence[Sequence[Edit]]) -> None:
        self._rounds = [tuple(r) for r in rounds]
        self._index = 0

    def propose(
        self,
        script: ActionScript,
        candidate: CandidateResult,
        score: tuple[float, ...],
    ) -> Sequence[Edit]:
        del script, candidate, score  # a fixed list ignores the state
        if self._index >= len(self._rounds):
            return ()
        proposal = self._rounds[self._index]
        self._index += 1
        return proposal
