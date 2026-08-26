"""A policy that answers from a fixed list. Drives every harness test in
the gate: it needs no network, no model, and no pixels, so a test can
isolate the harness from the agent entirely."""

from __future__ import annotations

from collections.abc import Sequence

from tuxghost.agent.types import STOP, Action, Observation


class ScriptedPolicy:
    def __init__(
        self,
        decisions: Sequence[Sequence[Action]],
        repeat_last: bool = False,
    ) -> None:
        self._decisions = [tuple(d) for d in decisions]
        self._repeat_last = repeat_last
        self._index = 0
        self.last_notes: str | None = None
        self.last_raw: str | None = None
        self.claimed_outcome: str | None = None

    def decide(self, obs: Observation) -> Sequence[Action]:
        if self._index < len(self._decisions):
            decision = self._decisions[self._index]
            self._index += 1
            return decision
        if self._repeat_last and self._decisions:
            return self._decisions[-1]
        return STOP
