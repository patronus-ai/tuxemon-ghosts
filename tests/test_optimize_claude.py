"""Task 10: the LLM editor's prompt and parse paths, against stubs only.

No live call in the gate: `make check` must need no network and no key,
and must pass with `anthropic` absent entirely (Step 6 verifies that).

Two things this file pins that the S2 capture taught. The fence regex
must span NEWLINES -- every stub answer's payload in
`tests/test_agent_claude.py` is single-line while every real model answer
is multi-line, so a regex without `re.DOTALL` passed all 16 of those
tests and failed only against real text. And the prompt must carry the
telemetry, because a blind editor is the no-gradient failure in a
different costume.
"""

from __future__ import annotations

from typing import Any

import pytest

from tuxghost.agent.types import Action
from tuxghost.optimize.editors.claude import (
    EDITOR_MAX_TOKENS,
    ClaudeEditor,
)
from tuxghost.optimize.edits import Delete, Replace
from tuxghost.optimize.schedule import ActionScript
from tuxghost.optimize.seal import CandidateResult

SCRIPT = ActionScript(lead_in=30, actions=(Action(2, 16, 8), Action(8, 16, 8)))


def _candidate() -> CandidateResult:
    trace: Any = None
    return CandidateResult(
        trace=trace,
        steps=48,
        final_state={"map": "spyder_paper_town.tmx", "tile_pos": [12, 13]},
        checkpoints=[(16, "aa" * 32), (32, "bb" * 32)],
        checkpoint_states={
            16: {"map": "spyder_paper_town.tmx", "tile_pos": [12, 12]},
            32: {"map": "spyder_paper_town.tmx", "tile_pos": [12, 13]},
        },
    )


class _StubTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _StubResponse:
    def __init__(self, text: str) -> None:
        self.content = [_StubTextBlock(text)]


class _StubClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        return _StubResponse(self._text)


ANSWER = """Looking at the checkpoints the player stalls after step 32.

```json
{"edits": [{"op": "replace", "index": 0,
            "action": {"button": 2, "hold": 32, "settle": 8}}],
 "notes": "hold DOWN twice as long"}
```
"""


def test_a_real_shaped_multiline_answer_parses() -> None:
    editor = ClaudeEditor(client=_StubClient(ANSWER))
    assert editor.propose(SCRIPT, _candidate(), (0.0, -1.0, -48.0)) == (
        Replace(0, Action(2, 32, 8)),
    )


def test_the_prompt_carries_the_telemetry_but_no_image() -> None:
    client = _StubClient(ANSWER)
    ClaudeEditor(client=client).propose(SCRIPT, _candidate(), (0.0,))
    sent = client.calls[0]
    body = str(sent["messages"])
    assert "tile_pos" in body and "12, 13" in body.replace("[", "").replace("]", "")
    assert "image" not in body, "S3 renders nothing; no frame may be sent"
    assert sent["max_tokens"] == EDITOR_MAX_TOKENS > 1024


def test_the_prompt_describes_the_current_script() -> None:
    client = _StubClient(ANSWER)
    ClaudeEditor(client=client).propose(SCRIPT, _candidate(), (0.0,))
    body = str(client.calls[0]["messages"])
    assert "lead_in" in body and "30" in body


def test_an_empty_edit_list_is_a_stop() -> None:
    editor = ClaudeEditor(client=_StubClient('```json\n{"edits": []}\n```'))
    assert editor.propose(SCRIPT, _candidate(), (0.0,)) == ()


def test_a_missing_fenced_block_raises_value_error() -> None:
    editor = ClaudeEditor(client=_StubClient("I would delete the third one."))
    with pytest.raises(ValueError, match="no fenced json block"):
        editor.propose(SCRIPT, _candidate(), (0.0,))


def test_malformed_json_in_the_fence_raises_value_error() -> None:
    editor = ClaudeEditor(client=_StubClient("```json\n{not json\n```"))
    with pytest.raises(ValueError, match="not valid json"):
        editor.propose(SCRIPT, _candidate(), (0.0,))


def test_a_non_object_payload_raises_type_error() -> None:
    editor = ClaudeEditor(client=_StubClient("```json\n[1, 2, 3]\n```"))
    with pytest.raises(TypeError, match="must be an object"):
        editor.propose(SCRIPT, _candidate(), (0.0,))


def test_an_unknown_op_raises_value_error() -> None:
    editor = ClaudeEditor(
        client=_StubClient('```json\n{"edits": [{"op": "x", "index": 0}]}\n```')
    )
    with pytest.raises(ValueError, match="unknown op"):
        editor.propose(SCRIPT, _candidate(), (0.0,))


def test_the_last_fenced_block_wins() -> None:
    text = (
        '```json\n{"edits": [{"op": "delete", "index": 0}]}\n```\n'
        'on reflection:\n'
        '```json\n{"edits": [{"op": "delete", "index": 1}]}\n```'
    )
    editor = ClaudeEditor(client=_StubClient(text))
    assert editor.propose(SCRIPT, _candidate(), (0.0,)) == (Delete(1),)


def test_notes_are_carried_forward_between_rounds() -> None:
    client = _StubClient(ANSWER)
    editor = ClaudeEditor(client=client)
    editor.propose(SCRIPT, _candidate(), (0.0,))
    editor.propose(SCRIPT, _candidate(), (0.0,))
    assert "hold DOWN twice as long" in str(client.calls[1]["messages"])


def test_the_goal_reaches_the_prompt() -> None:
    client = _StubClient(ANSWER)
    ClaudeEditor(client=client, goal="reach the shop").propose(
        SCRIPT, _candidate(), (0.0,)
    )
    assert "reach the shop" in str(client.calls[0]["messages"])


def test_construction_and_propose_never_reach_the_lazy_import() -> None:
    """`make check` installs no network SDK. The import is lazy, inside
    `_ensure_client`, reached only from a real `propose` with NO injected
    client -- so both of these must work with `anthropic` unavailable.

    Asserted by BLOCKING the import for the duration, not by inspecting
    `sys.modules`: an earlier draft of this test wrote
    `assert "anthropic" not in sys.modules or True`, which can never
    fail. Step 6 runs the same block across the whole suite."""
    import sys
    from importlib.abc import MetaPathFinder

    class Block(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
            if fullname == "anthropic" or fullname.startswith("anthropic."):
                raise ModuleNotFoundError(f"blocked: {fullname}")

    blocker = Block()
    sys.meta_path.insert(0, blocker)
    try:
        ClaudeEditor()
        ClaudeEditor(client=_StubClient(ANSWER)).propose(
            SCRIPT, _candidate(), (0.0,)
        )
    finally:
        sys.meta_path.remove(blocker)


def test_a_missing_client_reaches_the_lazy_import_and_says_so() -> None:
    """The companion control: with no client injected, `propose` MUST
    reach the import. If it did not, the test above would pass for the
    wrong reason -- because nothing ever tried."""
    import sys
    from importlib.abc import MetaPathFinder

    class Block(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
            if fullname == "anthropic" or fullname.startswith("anthropic."):
                raise ModuleNotFoundError(f"blocked: {fullname}")

    blocker = Block()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(ModuleNotFoundError, match="anthropic"):
            ClaudeEditor().propose(SCRIPT, _candidate(), (0.0,))
    finally:
        sys.meta_path.remove(blocker)


def test_an_insert_index_beyond_the_script_is_the_loops_problem() -> None:
    """`ClaudeEditor` parses; it does not know the script's length. An
    out-of-range index becomes a REJECTED ROUND in the loop, not a parse
    error here -- pinned so nobody 'helpfully' adds bounds checking in two
    places that can disagree."""
    editor = ClaudeEditor(
        client=_StubClient('```json\n{"edits": [{"op": "delete", "index": 99}]}\n```')
    )
    assert editor.propose(SCRIPT, _candidate(), (0.0,)) == (Delete(99),)
