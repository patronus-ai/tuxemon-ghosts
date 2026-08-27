"""The LLM editor: telemetry in, edits out. No pixels, ever.

`anthropic` is an OPTIONAL extra, imported lazily inside
`_ensure_client` and never at module scope -- `make check` installs no
network SDK and must still import this module to test the two pure
functions below.

Two deliberate differences from `tuxghost.agent.claude.ClaudePolicy`:

  * NO images. S3 renders nothing, so a round is text-only and cheap.
  * `EDITOR_MAX_TOKENS` is far above `ClaudePolicy`'s 1024. The S2 live
    capture measured 1024 sufficient FOR ONE ACTION; a list of edits plus
    reasoning is a different size of answer and that measurement is not
    evidence for it.

The fence regex carries `re.DOTALL`, and that is load-bearing rather than
incidental: every stub answer in `tests/test_agent_claude.py` has a
single-line json payload while every real model answer measured so far is
multi-line, so a regex without it passes all 16 stub tests and fails only
against real text (measured -- see docs/STATUS.org, "The live-model
capture", finding 1).

No comparison against `tuxghost.optimize.editors.mutation.MutationEditor`
has been run: this editor is not asserted or implied to perform better
(or worse) than that seeded baseline anywhere in this module. That
comparison is unmeasured future work.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from tuxghost.agent.claude import DEFAULT_MODEL
from tuxghost.optimize.editors.replay import edits_from_json
from tuxghost.optimize.edits import Edit
from tuxghost.optimize.schedule import ActionScript
from tuxghost.optimize.seal import CandidateResult

EDITOR_MAX_TOKENS = 4096

_JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)

SYSTEM = """You are improving a recorded Tuxemon play trace, offline.

The trace is a list of ACTIONS. Each action presses one button, holds it
`hold` steps, releases it, then waits `settle` steps. The game runs at 60
steps per second and one tile of walking takes 16 steps of a direction
held. Buttons: UP=1, DOWN=2, LEFT=4, RIGHT=8, A=64, B=128.

You never see the game. You see the action list, where the run ended, and
a sequence of checkpoints showing where it was along the way. Propose
EDITS to the action list; the trace is then re-run and scored for you.

Reply with exactly one fenced json block:

```json
{"edits": [{"op": "replace", "index": 0,
            "action": {"button": 2, "hold": 32, "settle": 8}}],
 "notes": "what you want your future self to remember"}
```

`op` is one of "insert", "delete", "replace". "delete" takes only an
`index`. Edits apply IN ORDER, each seeing the list the previous one left.
An empty `edits` list ends the run. `notes` replaces your previous notes.
"""


class ClaudeEditor:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        goal: str = "",
        # The injected `anthropic.Anthropic` instance (real or test stub).
        # Bare `Any` for the same reason `ClaudePolicy` uses it: naming
        # the type would need an unconditional import of an optional
        # dependency, and this project's mypy config resolves
        # `anthropic.*` to `Any` regardless.
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.goal = goal
        self.notes = ""
        self.last_raw: str | None = None
        self._client = client

    def build_prompt(
        self,
        script: ActionScript,
        candidate: CandidateResult,
        score: tuple[float, ...],
    ) -> str:
        lines = [
            f"lead_in: {script.lead_in} steps, then {len(script.actions)} "
            + "actions:",
        ]
        for i, a in enumerate(script.actions):
            lines.append(
                f"  [{i}] button={a.button} hold={a.hold} settle={a.settle}"
            )
        lines.append(f"\nTotal cost: {script.cost()} steps.")
        if self.goal:
            lines.append(f"\nGoal: {self.goal}")
        lines.append(f"\nScore (higher is better): {list(score)}")
        state = candidate.final_state
        lines.append(
            f"\nEnded on map {state.get('map')!r}, tile_pos "
            f"{state.get('tile_pos')}, after {candidate.steps} steps."
        )
        if candidate.checkpoint_states:
            lines.append("\nAlong the way:")
            for step in sorted(candidate.checkpoint_states):
                snap = candidate.checkpoint_states[step]
                lines.append(
                    f"  step {step}: map {snap.get('map')!r} tile "
                    f"{snap.get('tile_pos')} facing {snap.get('facing')!r}"
                )
        if self.notes:
            lines.append(f"\nYour notes:\n{self.notes}")
        lines.append("\nWhat do you change?")
        return "\n".join(lines)

    def parse_response(self, text: str) -> tuple[tuple[Edit, ...], str | None]:
        blocks = _JSON_BLOCK.findall(text)
        if not blocks:
            raise ValueError(
                "response contains no fenced json block; expected one "
                '```json { "edits": [...] } ``` block'
            )
        try:
            payload = json.loads(blocks[-1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"the last fenced json block is not valid json: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise TypeError(
                f"the json block must be an object, got "
                f"{type(payload).__name__}"
            )
        return edits_from_json(payload.get("edits", [])), payload.get("notes")

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def propose(
        self,
        script: ActionScript,
        candidate: CandidateResult,
        score: tuple[float, ...],
    ) -> Sequence[Edit]:
        response = self._ensure_client().messages.create(
            model=self.model,
            max_tokens=EDITOR_MAX_TOKENS,
            system=SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": self.build_prompt(script, candidate, score),
                }
            ],
        )
        raw = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        edits, notes = self.parse_response(raw)
        self.last_raw = raw
        if notes:
            self.notes = notes
        return edits
