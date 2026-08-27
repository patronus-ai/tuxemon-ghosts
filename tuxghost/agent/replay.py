"""Replays a captured model transcript.

`ReplayPolicy.decide` returns `actions_from_json(record.get("actions",
[]))` directly -- it never calls `ClaudePolicy.parse_response`. So this
module re-covers `actions_from_json` (the validation the two policies
share) end to end without a network call, but NOT `parse_response`'s own
fence parsing (`_JSON_BLOCK`, last-block selection, the json-decode and
non-object-payload checks) or prompt building -- those are covered only
by the stub-client tests in `tests/test_agent_claude.py`. The
transcripts this project commits are SYNTHETIC test fixtures, not
captured from a live `ClaudePolicy` run.

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
        records = [
            json.loads(line)
            for line in Path(transcript).read_text().splitlines()
            if line.strip()
        ]
        # Validated HERE, at construction, not lazily inside `decide()`
        # (task 8 review round 2's Ruling X): a malformed transcript is
        # knowable at load time, so failing now -- before the game ever
        # boots -- saves a pointless run and matches how `--policy
        # scripted` already behaves (the CLI pre-parses its whole
        # `--actions` file up front). Without this, a record that is
        # valid JSON but not an object (e.g. `[1, 2, 3]`) constructs
        # cleanly and `decide()`'s unconditional `record.get("notes")`
        # raises `AttributeError` lazily, mid-run -- not one of this
        # project's own `ValueError`/`TypeError` input-validation
        # exceptions, so it is not (and must not be) caught by
        # `tuxghost.cli._agent`'s `run_agent(...)` boundary, which is
        # deliberately narrow: an `AttributeError` there almost always
        # signals a programming error, not bad user input, and widening
        # the boundary to swallow it would launder a genuine bug into a
        # tidy "refused".
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(
                    f"record {index} must be a JSON object, got "
                    f"{type(record).__name__}"
                )
        self._records = records
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
