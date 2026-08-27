"""The computer-use policy: a frame in, actions out.

`anthropic` is an OPTIONAL extra and is imported lazily inside
`_ensure_client`, never at module import -- `make check` installs no
network SDK and must still be able to import this module to test the two
pure functions below.

`build_messages` and `parse_response` are deliberately pure and separately
tested: they are the parts most likely to rot silently as prompts and
response shapes change, and `ReplayPolicy` re-covers `parse_response` end
to end against a captured transcript. Context is a sliding window of the
last `window` turns plus a notes block carried forward verbatim -- no
summarisation call, whose output would silently steer the run and be a
second prompt to maintain.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tuxghost.agent.replay import actions_from_json
from tuxghost.agent.types import (
    HOLD_CAP,
    SETTLE_CAP,
    STOP,
    Action,
    Observation,
)

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024

#: Buttons worth telling the model about. The full set in
#: `tuxemon.platform.const.buttons` includes mouse/finger events and
#: shoulder buttons this game does not use in the overworld; naming them
#: all would spend tokens inviting actions that do nothing.
SYSTEM = f"""You are playing Tuxemon, a tile-based creature-collecting RPG.
You see one rendered frame of the game at a time and reply with button
presses. The game does not advance while you think.

Buttons: UP=1, DOWN=2, LEFT=4, RIGHT=8, A=64, B=128, START=32.
A confirms and talks; B cancels and backs out.

The game runs at 60 steps per second. Every action holds a button for
`hold` steps, releases it, then lets `settle` steps pass before you see
the next frame. Holding a direction for too few steps does not move you a
whole tile; a dialog needs A pressed and released. `hold` must be between
1 and {HOLD_CAP}; `settle` between 0 and {SETTLE_CAP}.

Reply with exactly one fenced json block:

```json
{{"actions": [{{"button": 2, "hold": 30, "settle": 10}}],
  "notes": "what you want your future self to remember",
  "claimed_outcome": null}}
```

`notes` replaces your previous notes -- carry forward anything still
useful, since you will not see older frames. Set `claimed_outcome` to a
short sentence only when you believe you are done. An empty `actions` list
ends the run.
"""

#: Deliberately captures ANY fenced content, not `\{.*?\}` -- a regex
#: requiring the capture to itself start with `{` and end with `}` would
#: make `parse_response`'s "the json block must be an object" check below
#: unreachable dead code: any string of the form `{...}` that survives
#: `json.loads` is, by JSON's own grammar, guaranteed to decode to a
#: `dict`, so that branch could never fire (found while writing
#: `tests/test_agent_claude.py::test_parse_response_raises_type_error_for_a_non_object_json_block`,
#: which failed against the narrower regex with "no fenced json block"
#: rather than the intended `TypeError` -- the regex, not the test, was
#: wrong). This form still requires a model to answer with a bare
#: `[...]`/`"..."`/number etc. inside the fence for that branch to fire,
#: which the system prompt never asks for, but a wrong-shaped answer is
#: exactly the case this check exists to refuse rather than crash on.
_JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


@dataclass
class _Turn:
    frame_png: bytes
    raw: str


class ClaudePolicy:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        goal: str = "",
        window: int = 4,
        # `client` is the injected `anthropic.Anthropic` instance (real or
        # test stub, see `tests/test_agent_claude.py`'s `_StubClient`).
        # `anthropic` is an optional dependency imported lazily inside
        # `_ensure_client`, never at module scope (see the module
        # docstring), so this type cannot be named as `anthropic.Anthropic`
        # without either importing it unconditionally (defeats the point
        # of the optional extra) or a `TYPE_CHECKING`-only import -- and
        # even that would resolve to `Any` anyway under this project's
        # `[[tool.mypy.overrides]] module = "anthropic.*"
        # ignore_missing_imports = true` (pyproject.toml), which is what
        # lets the gate typecheck this module without the package
        # installed. Bare `Any` here is that deliberate tradeoff, not an
        # omission.
        client: Any | None = None,
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window!r}")
        self.model = model
        self.goal = goal
        self.notes = ""
        self.last_notes: str | None = None
        self.last_raw: str | None = None
        self.claimed_outcome: str | None = None
        self._window = window
        self._client = client
        self._turns: list[_Turn] = []

    # -- prompt construction (pure) ------------------------------------

    def _image_block(self, png: bytes) -> dict[str, Any]:
        # Returns one `anthropic` SDK message-content block: an untyped
        # JSON dict, not a class -- see the constructor's `client`
        # comment for why `anthropic`'s own types are not named here.
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png).decode(),
            },
        }

    def _prompt_text(self, obs: Observation) -> str:
        """The text half of the current turn.

        Deliberately carries NO `state_stack` and no `map_name`: the frame
        is the observation. A policy told which states are on the stack is
        a state-machine agent wearing a computer-use agent's clothes, and
        `tests/test_agent_claude.py` asserts that leak cannot happen.
        """
        parts = [f"Step {obs.step}."]
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        if self.notes:
            parts.append(f"Your notes:\n{self.notes}")
        if obs.frame_is_blank:
            parts.append(
                "The frame is blank -- the game is mid-transition. Waiting "
                "is usually the right move."
            )
        parts.append("What do you do next?")
        return "\n\n".join(parts)

    def build_messages(self, obs: Observation) -> list[dict[str, Any]]:
        # `list[dict[str, Any]]` throughout this method is the `anthropic`
        # SDK's untyped message-list wire shape (same `Any` as
        # `_image_block`'s return type), not a shortcut around a type we
        # could otherwise have named.
        messages: list[dict[str, Any]] = []
        for turn in self._turns[-self._window :]:
            content: list[dict[str, Any]] = (
                [self._image_block(turn.frame_png)]
                if turn.frame_png
                else [{"type": "text", "text": "(frame not retained)"}]
            )
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": turn.raw})
        messages.append(
            {
                "role": "user",
                "content": [
                    self._image_block(obs.frame_png),
                    {"type": "text", "text": self._prompt_text(obs)},
                ],
            }
        )
        return messages

    # -- response parsing (pure) ---------------------------------------

    def parse_response(
        self, text: str
    ) -> tuple[tuple[Action, ...], str | None, str | None]:
        """Actions, notes, claimed outcome -- or an exception naming what
        was wrong: `ValueError` for a missing block, invalid json, or a
        value out of range; `TypeError` for a wrong shape (matching
        `actions_from_json`'s own two-exception taxonomy -- see
        `tuxghost/agent/replay.py`).

        Routes actions through `actions_from_json`, the SAME validation
        `ReplayPolicy` uses, so a captured transcript can never be
        accepted where the live answer that produced it would have been
        rejected.
        """
        blocks = _JSON_BLOCK.findall(text)
        if not blocks:
            raise ValueError(
                "response contains no fenced json block; expected one "
                '```json { "actions": [...] } ``` block'
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
        actions = actions_from_json(payload.get("actions", []))
        notes = payload.get("notes")
        outcome = payload.get("claimed_outcome")
        return actions, notes, outcome

    # -- history -------------------------------------------------------

    def record_turn(
        self,
        actions: Sequence[Action],
        notes: str | None = None,
        raw: str = "",
        frame_png: bytes = b"",
    ) -> None:
        self._turns.append(_Turn(frame_png=frame_png, raw=raw))
        del actions  # kept in the signature for callers/tests; history is
        # the raw answer, which already contains them verbatim.
        del notes  # same: kept for callers/tests (`decide()` already
        # applies `notes` to `self.notes` before calling this), but turn
        # HISTORY only ever stores the raw answer, never notes on its own.

    # -- the policy ----------------------------------------------------

    def _ensure_client(self) -> Any:
        # Returns the untyped injected/lazily-constructed `anthropic`
        # client -- see the constructor's `client` comment for why `Any`.
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def decide(self, obs: Observation) -> Sequence[Action]:
        response = self._ensure_client().messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=self.build_messages(obs),
        )
        raw = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        actions, notes, outcome = self.parse_response(raw)
        self.last_raw = raw
        self.last_notes = notes
        if notes:
            self.notes = notes
        if outcome:
            self.claimed_outcome = outcome
        self.record_turn(actions, notes=notes, raw=raw, frame_png=obs.frame_png)
        return actions if actions else STOP
