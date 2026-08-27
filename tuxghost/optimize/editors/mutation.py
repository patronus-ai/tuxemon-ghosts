"""A deliberately dumb seeded mutator: the baseline, not the point.

It exists so that any claim about `ClaudeEditor` has something to be
compared against. An LLM editor with no control to beat is the
"green but proves nothing" trap this project has already been burned by.

It draws from its OWN `random.Random(seed)` instance and never the global
module RNG. `tuxghost.determinism.seed_all` pins the global stream for
the GAME; drawing from it here would perturb the engine's own randomness
and change the very thing being scored. Patch 0002 exists because
instance-vs-global matters in this codebase.

Every value it emits is in range by construction, so
`validate_actions` never has to catch it -- a mutator that proposed
invalid actions would spend the whole budget on rejected rounds while
still looking like it ran.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from tuxemon.platform.const import buttons

from tuxghost.agent.types import HOLD_CAP, SETTLE_CAP, VALID_BUTTONS, Action
from tuxghost.optimize.edits import Delete, Edit, Insert, Replace
from tuxghost.optimize.schedule import ActionScript
from tuxghost.optimize.seal import CandidateResult

#: Only the overworld buttons an edit can usefully reach. `VALID_BUTTONS`
#: includes mouse and finger events the overworld ignores; proposing them
#: would spend rounds on no-ops that still cost a full engine run.
#:
#: Read from `tuxemon.platform.const.buttons`, not written out as
#: `(1, 2, 4, 8, 64, 128)` (whole-branch review, Also-fix 2). The
#: `assert` below catches a value LEAVING `VALID_BUTTONS` but not a
#: REMAPPING: if upstream swapped `UP` and `LEFT`, every literal here
#: would still be a valid button, this editor would silently propose the
#: wrong direction, and the assert would stay green. The prompt in
#: `tuxghost.optimize.editors.claude` states the same six and now reads
#: them from the same place, for the same reason.
_USEFUL = (
    buttons.UP,
    buttons.DOWN,
    buttons.LEFT,
    buttons.RIGHT,
    buttons.A,
    buttons.B,
)


class MutationEditor:
    def __init__(self, seed: int, nudge: int = 4) -> None:
        assert set(_USEFUL) <= VALID_BUTTONS, "a useful button went away"
        self._rng = random.Random(seed)
        self._nudge = nudge

    def _action(self) -> Action:
        return Action(
            button=self._rng.choice(_USEFUL),
            hold=self._rng.randint(1, 32),
            settle=self._rng.randint(0, 16),
        )

    def _clamp(self, value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    def propose(
        self,
        script: ActionScript,
        candidate: CandidateResult,
        score: tuple[float, ...],
    ) -> Sequence[Edit]:
        del candidate, score  # blind hill-climbing, by design
        n = len(script.actions)
        if n == 0:
            return [Insert(0, self._action())]

        choice = self._rng.choice(("hold", "settle", "button", "insert", "delete"))
        index = self._rng.randrange(n)
        current = script.actions[index]
        delta = self._rng.choice((-self._nudge, self._nudge))

        if choice == "hold":
            return [
                Replace(
                    index,
                    Action(
                        current.button,
                        self._clamp(current.hold + delta, 1, HOLD_CAP),
                        current.settle,
                    ),
                )
            ]
        if choice == "settle":
            return [
                Replace(
                    index,
                    Action(
                        current.button,
                        current.hold,
                        self._clamp(current.settle + delta, 0, SETTLE_CAP),
                    ),
                )
            ]
        if choice == "button":
            return [
                Replace(
                    index,
                    Action(self._rng.choice(_USEFUL), current.hold, current.settle),
                )
            ]
        if choice == "insert":
            return [Insert(self._rng.randint(0, n), self._action())]
        return [Delete(index)]
