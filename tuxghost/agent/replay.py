"""Replays a captured model transcript.

This is what gives `ClaudePolicy`'s response PARSING real gate coverage
without a network call: the records are raw model answers captured from a
live run, so a parsing regression fails here rather than in production.
It deliberately does NOT loop when exhausted -- a transcript that silently
restarted would make a replayed run diverge from the run it claims to
reproduce, which is the one thing a replay must not do.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tuxghost.agent.types import STOP, Action, Observation, validate_actions


def actions_from_json(raw: Any) -> tuple[Action, ...]:
    """Build actions from a decoded JSON list, refusing anything partial.

    Shared with `ClaudePolicy`: both parse the same shape, and a model's
    output and a transcript of that output must be validated identically
    or the replay would accept what the live path rejected.
    """
    if not isinstance(raw, list):
        raise TypeError(f"actions must be a JSON list, got {type(raw).__name__}")
    built = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"actions[{index}] must be an object")
        for key in ("button", "hold", "settle"):
            if key not in item:
                raise ValueError(f"actions[{index}] is missing {key!r}")
        built.append(
            Action(
                button=int(item["button"]),
                hold=int(item["hold"]),
                settle=int(item["settle"]),
            )
        )
    return validate_actions(built)


class ReplayPolicy:
    def __init__(self, transcript: Path) -> None:
        self._records = [
            json.loads(line)
            for line in Path(transcript).read_text().splitlines()
            if line.strip()
        ]
        self._index = 0
        self.last_notes: str | None = None
        self.last_raw: str | None = None
        self.claimed_outcome: str | None = None

    def decide(self, obs: Observation) -> Sequence[Action]:
        if self._index >= len(self._records):
            return STOP
        record = self._records[self._index]
        self._index += 1
        self.last_notes = record.get("notes")
        self.last_raw = record.get("raw")
        if record.get("claimed_outcome"):
            self.claimed_outcome = record["claimed_outcome"]
        return actions_from_json(record.get("actions", []))
