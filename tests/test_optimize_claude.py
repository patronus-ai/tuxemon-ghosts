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
    # The final-state line alone would satisfy the assertion above even
    # with `checkpoint_states` dropped entirely from the prompt -- pin
    # the CHECKPOINT telemetry too, on data only that block renders.
    # Checkpoint 16's tile is [12, 12], distinct from both the final
    # state's and checkpoint 32's shared [12, 13], so this fails on its
    # own if `checkpoint_states` is dropped, independently of the
    # assertion above.
    assert "step 16" in body
    assert "12, 12" in body.replace("[", "").replace("]", "")
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
    from collections.abc import Sequence
    from importlib.abc import MetaPathFinder
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    class Block(MetaPathFinder):
        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None = None,
            target: ModuleType | None = None,
        ) -> ModuleSpec | None:
            if fullname == "anthropic" or fullname.startswith("anthropic."):
                raise ModuleNotFoundError(f"blocked: {fullname}")
            return None

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
    from collections.abc import Sequence
    from importlib.abc import MetaPathFinder
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    class Block(MetaPathFinder):
        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None = None,
            target: ModuleType | None = None,
        ) -> ModuleSpec | None:
            if fullname == "anthropic" or fullname.startswith("anthropic."):
                raise ModuleNotFoundError(f"blocked: {fullname}")
            return None

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


def test_the_system_prompts_button_values_track_upstreams_constants() -> None:
    """FINAL RESIDUALS, ITEM 4. Whole-branch review Also-fix 2 replaced
    hardcoded button numbers in `SYSTEM` with values read from
    `tuxemon.platform.const.buttons`, and a later fix wave concluded no
    test could pin that. One can.

    The existing coverage (`tests/test_optimize_cli.py`, `assert "UP=1"
    in system and "LEFT=4" in system`) cannot: those ARE the real values,
    so a legend built from hardcoded literals emits exactly the same
    string and passes. What discriminates is remapping upstream and
    checking the legend follows.

    UP and LEFT are SWAPPED rather than given novel values -- 4 and 1 are
    both still valid buttons, so the six-name legend stays well-formed
    and any assertion about its shape stays green. Only a legend that
    actually reads the constants changes its content:

        reads the constants -> "UP=4, ..., LEFT=1"   (passes)
        hardcoded literals  -> "UP=1, ..., LEFT=4"   (fails)

    The module is reloaded because `_BUTTON_LEGEND` and `SYSTEM` are
    built once at import time, and reloaded again in `finally` so a
    failure here cannot leave a remapped module behind for later tests.
    """
    import importlib
    from unittest import mock

    from tuxemon.platform.const import buttons

    import tuxghost.optimize.editors.claude as claude_module

    assert (buttons.UP, buttons.LEFT) == (1, 4), (
        "this test swaps UP and LEFT; if upstream's real values are no "
        "longer 1 and 4 the swap below is not a swap"
    )
    try:
        with (
            mock.patch.object(buttons, "UP", 4),
            mock.patch.object(buttons, "LEFT", 1),
        ):
            remapped = importlib.reload(claude_module)
            legend = remapped._BUTTON_LEGEND
            system = remapped.SYSTEM
    finally:
        importlib.reload(claude_module)

    assert "UP=4" in legend, legend
    assert "LEFT=1" in legend, legend
    # And in the assembled system prompt, which is what the model sees.
    assert "UP=4" in system and "LEFT=1" in system, system
    # The un-remapped values must be GONE, not merely joined by the new
    # ones -- a legend that appended rather than replaced would satisfy
    # the assertions above while still telling the model UP=1.
    assert "UP=1" not in legend, legend
    assert "LEFT=4" not in legend, legend


def test_an_unparseable_reply_is_still_recorded() -> None:
    """The ordering inside `propose` is the whole point of `answers`.

    `parse_response` raises `ValueError` on a reply carrying no fenced
    json block. The append that records the reply sits BEFORE that call,
    so the text survives the raise; with it after -- where `last_raw`'s
    assignment still is, deliberately, because `last_raw` means "the last
    reply that PARSED" -- the single most diagnostic answer a live run
    can produce would be discarded and no artifact would remember it.

    That failure mode is not hypothetical for this project: S2's missing
    `re.DOTALL` made every real multi-line model answer unparseable while
    sixteen single-line stub tests stayed green.
    """
    editor = ClaudeEditor(client=_StubClient("no fence here, just prose"))
    with pytest.raises(ValueError, match="no fenced json block"):
        editor.propose(SCRIPT, _candidate(), (0.0, -2.0, -48.0))

    # Field-by-field rather than whole-dict equality: the record grew
    # `stop_reason` and `block_types` when a live run produced four
    # consecutive EMPTY replies the answers file could not explain. What
    # this test pins is the ORDERING -- that the text survives the raise
    # -- not the record's exact shape, and an equality assertion made it
    # fail for a reason it does not care about.
    assert len(editor.answers) == 1
    assert editor.answers[0]["call"] == 1
    assert editor.answers[0]["raw"] == "no fence here, just prose"
    # And `last_raw` is untouched, because nothing parsed. The two fields
    # mean different things and this pins that they are not merged.
    assert editor.last_raw is None


def test_answers_accumulate_across_calls_and_number_themselves() -> None:
    """One record per `propose`, in order, each carrying the call number
    it came from rather than relying on its position in the list.

    The number is taken from a counter incremented at the TOP of
    `propose`, so a call that dies before any text arrives (an API error,
    say) leaves a visible gap -- `call` jumping 1, 3 -- instead of
    silently shifting every later record down by one and making the file
    quietly disagree with `optimize.jsonl`'s round numbering.
    """
    reply = '```json\n{"edits": [{"op": "delete", "index": 0}]}\n```'
    editor = ClaudeEditor(client=_StubClient(reply))
    for _ in range(3):
        editor.propose(SCRIPT, _candidate(), (0.0, -2.0, -48.0))

    assert [a["call"] for a in editor.answers] == [1, 2, 3]
    assert all(a["raw"] == reply for a in editor.answers)
    assert editor.last_raw == reply


def test_the_prompt_explains_what_progress_rewards() -> None:
    """Score term NAMES were not enough.

    The prompt has always carried the map, the tile and a checkpoint
    trail -- the editor was never blind to where it stood. What nothing
    told it was that walking onto a new map raised the number it was
    asked to maximise; `score_legend` supplies only the term's name,
    `max_progress`. Across four runs and 37 rounds from
    `hearthrock_city.save` that number never moved off 20.

    Pinned against `build_prompt` rendering `progress_legend`: drop that
    branch and the weights vanish from the prompt.
    """
    from tuxghost.optimize.editors.claude import ClaudeEditor
    from tuxghost.optimize.schedule import ActionScript
    from tuxghost.rules import MAP_POINTS, TuxemonRules

    editor = ClaudeEditor(
        goal="win a battle",
        score_legend="goal_state, max_progress, -steps",
        progress_legend=TuxemonRules.describe_progress(),
        client=object(),
    )
    prompt = editor.build_prompt(
        ActionScript(lead_in=10, actions=()),
        _candidate(),
        (0.0, 20.0, -600.0),
    )
    assert f"{MAP_POINTS:,}" in prompt
    assert "per new map reached" in prompt


def test_the_prompt_omits_the_explanation_when_none_is_given() -> None:
    """The control. `reach-tile`'s terms describe themselves from their
    names, so `tuxghost.cli` passes no `progress_legend` there, and the
    prompt must not grow a stray empty section.
    """
    from tuxghost.optimize.editors.claude import ClaudeEditor
    from tuxghost.optimize.schedule import ActionScript

    editor = ClaudeEditor(
        goal="reach a tile",
        score_legend="-distance_to_target, -steps",
        client=object(),
    )
    prompt = editor.build_prompt(
        ActionScript(lead_in=10, actions=()),
        _candidate(),
        (0.0, -2.0, -442.0),
    )
    assert "per new map reached" not in prompt
    assert "max_progress is a sum" not in prompt


# --- Failure modes measured in a live run, not imagined ------------------
#
# Round 8 from `hearthrock_city.save` was the first time this project's
# optimizer ever moved `max_progress` (20 -> 1020). Rounds 9-13 then
# ended the run: four replies of ZERO characters and one truncated
# mid-JSON on `{"`. The answers file recorded only `raw`, so it could
# say the replies were empty but not why.


class _BlockResponse:
    """A response whose blocks and stop_reason are both controllable --
    the real API returns non-text block types and a stop_reason, and
    `_StubResponse` models neither."""

    def __init__(self, blocks: list[Any], stop_reason: str | None) -> None:
        self.content = blocks
        self.stop_reason = stop_reason


class _BlockClient:
    def __init__(self, blocks: list[Any], stop_reason: str | None) -> None:
        self._blocks = blocks
        self._stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> _BlockResponse:
        self.calls.append(kwargs)
        return _BlockResponse(self._blocks, self._stop_reason)


class _NonTextBlock:
    type = "thinking"


def test_an_empty_reply_is_reported_as_empty_not_as_a_missing_fence() -> None:
    """The message a reader gets has to point at the real defect.

    A reply with no TEXT block and a reply with text but no json fence
    are different failures with different causes, and the generic "no
    fenced json block" sent a reader hunting for a malformed fence when
    there was no text to fence. Four consecutive empty replies ended a
    live run and the error never said so.

    Pinned against the `if not raw` guard: remove it and this raises the
    fence error instead, which is the misdiagnosis being fixed.
    """
    editor = ClaudeEditor(
        client=_BlockClient([_NonTextBlock()], stop_reason="max_tokens")
    )
    with pytest.raises(ValueError, match="no text"):
        editor.propose(SCRIPT, _candidate(), (0.0, 20.0, -600.0))


def test_a_failed_reply_records_why_it_failed() -> None:
    """`raw` cannot explain its own absence.

    Whether the model emitted only non-text blocks or stopped before
    writing any looks identical in a file that records the text alone,
    and the two need different fixes. Pinned against the extra fields:
    drop them and the answers file goes back to being unable to explain
    the failure it just recorded.
    """
    editor = ClaudeEditor(
        client=_BlockClient([_NonTextBlock()], stop_reason="max_tokens")
    )
    with pytest.raises(ValueError):
        editor.propose(SCRIPT, _candidate(), (0.0, 20.0, -600.0))

    assert editor.answers[0]["raw"] == ""
    assert editor.answers[0]["stop_reason"] == "max_tokens"
    assert editor.answers[0]["block_types"] == ["thinking"]


def test_the_token_cap_leaves_room_for_a_grown_script() -> None:
    """The cap must not shrink the model's answer as the search succeeds.

    Edits do not persist between rounds, so the model restates the whole
    action list every time; an ACCEPTED edit therefore makes every later
    reply longer. Round 8's winning script was 13 actions and the next
    reply was cut off mid-JSON at 4096. This asserts headroom against
    that measured size rather than a number someone liked.
    """
    from tuxghost.optimize.editors.claude import EDITOR_MAX_TOKENS

    # ~35 tokens per restated action is generous for
    # `{"op": "insert", "index": N, "action": {...}}` plus notes and
    # reasoning; 13 actions truncated at 4096, so the floor must clear it
    # by a wide margin, not by one action.
    assert EDITOR_MAX_TOKENS >= 4 * 4096
