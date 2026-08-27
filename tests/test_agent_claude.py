"""ClaudePolicy's prompt building and response parsing, with no network.

These two functions are the parts most likely to rot silently, so they are
pure and tested directly; `ReplayPolicy` covers them again end to end
against a captured transcript.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from tuxemon.platform.const import buttons

from tuxghost.agent.claude import MAX_TOKENS, SYSTEM, ClaudePolicy
from tuxghost.agent.types import Action, Observation

OBS = Observation(
    step=42,
    frame_png=b"\x89PNG\r\n\x1a\nfake",
    frame_is_blank=False,
    state_stack=("WorldState",),
    map_name="spyder_paper_town.tmx",
    tile_pos=(10, 10),
)


def test_messages_carry_the_frame_as_an_image_block() -> None:
    policy = ClaudePolicy(goal="leave town")
    messages = policy.build_messages(OBS)
    blocks = messages[-1]["content"]
    kinds = [b["type"] for b in blocks]
    assert "image" in kinds, "the frame IS the observation; it must be sent"
    assert any("leave town" in b.get("text", "") for b in blocks)


def test_messages_do_not_leak_the_state_stack_to_the_model() -> None:
    """The frame is the observation. A policy handed `state_stack` or
    `map_name` in its prompt would be a state-machine agent wearing a CU
    agent's clothes -- both are named in the binding constraint, so both
    are pinned here."""
    policy = ClaudePolicy()
    text = str(policy.build_messages(OBS))
    assert "WorldState" not in text
    assert "spyder_paper_town.tmx" not in text


def test_parse_response_reads_actions_notes_and_outcome() -> None:
    policy = ClaudePolicy()
    actions, notes, outcome = policy.parse_response(
        'thinking...\n```json\n{"actions": '
        f'[{{"button": {buttons.DOWN}, "hold": 12, "settle": 6}}], '
        '"notes": "south past the sign", "claimed_outcome": null}\n```'
    )
    assert actions == (Action(buttons.DOWN, 12, 6),)
    assert notes == "south past the sign"
    assert outcome is None


def test_parse_response_refuses_a_response_with_no_json_block() -> None:
    policy = ClaudePolicy()
    with pytest.raises(ValueError, match="json"):
        policy.parse_response("I will walk south.")


def test_parse_response_refuses_an_out_of_range_hold() -> None:
    policy = ClaudePolicy()
    with pytest.raises(ValueError, match="hold"):
        policy.parse_response(
            '```json\n{"actions": [{"button": 64, "hold": 999999, '
            '"settle": 0}]}\n```'
        )


def test_parse_response_raises_type_error_when_actions_is_not_a_list() -> None:
    """`actions_from_json` raises `TypeError` (not `ValueError`) when
    `actions` is the wrong shape entirely -- a model answering with
    `{"actions": "up"}` must be refused loudly, not silently misparsed,
    and a future refactor that wraps this call in a blanket `except
    ValueError` must be caught rather than let the wrong-shaped answer
    crash the run some other way."""
    policy = ClaudePolicy()
    with pytest.raises(TypeError, match="list"):
        policy.parse_response('```json\n{"actions": "up"}\n```')


def test_parse_response_raises_type_error_for_a_non_object_json_block() -> None:
    """The fenced block itself, not just `actions` inside it, can be the
    wrong shape: `[1, 2, 3]` is valid json but not an object. This is
    `parse_response`'s OWN shape check (not routed through
    `actions_from_json`), and must raise `TypeError` to match
    `actions_from_json`'s shape-vs-value taxonomy."""
    policy = ClaudePolicy()
    with pytest.raises(TypeError, match="object"):
        policy.parse_response("```json\n[1, 2, 3]\n```")


def test_window_bounds_the_message_history() -> None:
    policy = ClaudePolicy(window=2)
    for _ in range(6):
        policy.build_messages(OBS)
        policy.record_turn(
            (Action(buttons.A, 2, 2),), notes="n", raw="```json\n{}\n```"
        )
    # window=2 turns, each contributing at most one user + one assistant
    # message, plus the current observation.
    assert len(policy.build_messages(OBS)) <= 2 * 2 + 1


# -- `decide()` end to end, against a STUB client (no network) ------------
#
# This project is deliberately never allowed to make a live Anthropic API
# call from the gate (see the task instructions for `ClaudePolicy`): a real
# call is outward-facing, costs money, and is a decision for a human, not
# an autonomous test suite. `ClaudePolicy(client=...)` accepts an injected
# client for exactly this reason -- a stub object shaped like the real
# `anthropic.Anthropic().messages` lets these tests exercise the ENTIRE
# `decide()` path (prompt built, response parsed, `last_raw`/`last_notes`/
# `claimed_outcome` updated, `record_turn` called, actions returned)
# without ever importing `anthropic` or touching the network. What this
# does NOT cover is the real SDK's request/response wire shape -- see the
# task report for what has no live-model evidence.


@dataclass
class _StubTextBlock:
    text: str
    type: str = "text"


@dataclass
class _StubResponse:
    # A real multi-block response can mix `_StubTextBlock` and
    # `_StubThinkingBlock` (see `test_decide_ignores_non_text_blocks_...`
    # below) -- `_StubThinkingBlock` is defined further down, but
    # `from __future__ import annotations` makes this forward reference
    # fine for both mypy and runtime.
    content: list[_StubTextBlock | _StubThinkingBlock]


class _StubMessages:
    def __init__(self, response: _StubResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _StubResponse:
        self.calls.append(kwargs)
        return self._response


class _StubClient:
    def __init__(self, response: _StubResponse) -> None:
        self.messages = _StubMessages(response)


def _stub_client(raw_text: str) -> _StubClient:
    return _StubClient(_StubResponse(content=[_StubTextBlock(text=raw_text)]))


def test_decide_returns_actions_parsed_from_the_stub_response() -> None:
    client = _stub_client(
        '```json\n{"actions": [{"button": '
        f'{buttons.DOWN}, "hold": 12, "settle": 6}}], '
        '"notes": "south past the sign", "claimed_outcome": null}\n```'
    )
    policy = ClaudePolicy(client=client)
    actions = policy.decide(OBS)
    assert actions == (Action(buttons.DOWN, 12, 6),)


def test_decide_updates_last_raw_last_notes_and_claimed_outcome() -> None:
    raw = (
        '```json\n{"actions": [], "notes": "reached the gate", '
        '"claimed_outcome": "left town"}\n```'
    )
    client = _stub_client(raw)
    policy = ClaudePolicy(client=client)
    actions = policy.decide(OBS)
    assert actions == ()  # empty actions means STOP
    assert policy.last_raw == raw
    assert policy.last_notes == "reached the gate"
    assert policy.claimed_outcome == "left town"


def test_decide_calls_the_stub_client_with_the_built_messages() -> None:
    client = _stub_client('```json\n{"actions": []}\n```')
    policy = ClaudePolicy(model="claude-sonnet-5", client=client)
    policy.decide(OBS)
    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["system"] == SYSTEM
    assert call["max_tokens"] == MAX_TOKENS
    assert isinstance(call["messages"], list)
    last_message = call["messages"][-1]
    assert last_message["role"] == "user"
    kinds = [block["type"] for block in last_message["content"]]
    assert "image" in kinds


@dataclass
class _StubThinkingBlock:
    """Shaped like the real SDK's thinking-block content: `type` is not
    `"text"`, and there is no `.text` attribute at all -- `thinking`
    blocks carry a `.thinking` field instead. This is the shape `decide()`
    must skip over, with no live model call ever having exercised it."""

    type: str = "thinking"
    thinking: str = "reasoning that must not leak into raw"


def test_decide_ignores_non_text_blocks_in_a_multi_block_response() -> None:
    """The real Anthropic API can return a `thinking` block alongside the
    text block in one response; `decide()`'s `getattr(block, "type", None)
    == "text"` filter is the ONLY thing keeping a thinking block out of
    `raw`, and nothing exercised that filter's false branch before this
    test -- it is exactly the kind of thing with no live-model evidence
    that is still testable with no network at all."""
    client = _StubClient(
        _StubResponse(
            content=[
                _StubThinkingBlock(),
                _StubTextBlock(text='```json\n{"actions": []}\n```'),
            ]
        )
    )
    policy = ClaudePolicy(client=client)
    policy.decide(OBS)
    assert policy.last_raw == '```json\n{"actions": []}\n```'
    assert "reasoning that must not leak" not in (policy.last_raw or "")


def test_decide_records_the_turn_so_the_next_prompt_carries_history() -> None:
    client = _stub_client(
        '```json\n{"actions": [{"button": '
        f'{buttons.A}, "hold": 2, "settle": 2}}], "notes": "n"}}\n```'
    )
    policy = ClaudePolicy(client=client)
    assert len(policy.build_messages(OBS)) == 1  # only the current turn, so far
    policy.decide(OBS)
    # After one decision, one prior turn (user image + assistant raw) is
    # carried into the next prompt, ahead of the current observation.
    messages = policy.build_messages(OBS)
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
