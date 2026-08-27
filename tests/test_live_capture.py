"""The project's one LIVE-model capture, and what it is evidence of.

Every other `ClaudePolicy` test in this repo injects a stub client
(`tests/test_agent_claude.py`'s `_StubClient`) and every other transcript
this repo commits is synthetic. These three artifacts are not:

  * `tests/golden/claude_town_1234.tuxghost` -- a `provenance.recorder =
    "cu-agent"` trace recorded by a real `tuxghost agent --policy claude`
    invocation against the real Anthropic API on 2026-08-27.
  * `tests/fixtures/claude_town_1234.decisions.jsonl` -- that run's
    `decisions.jsonl`, i.e. the model's own answers, `raw` field included.
  * `tests/fixtures/claude_town_1234.run.json` -- the run's own record of
    the goal, seed, epoch, model id and stop reason, so the transcript is
    not an orphan whose provenance lives only in a commit message.

The invocation, verbatim (model id: `claude-sonnet-5`, the
`tuxghost.agent.claude.DEFAULT_MODEL` the CLI resolves when `--model` is
omitted -- so this covers the shipped default path, and NO other model id
has live evidence behind it):

    tuxghost agent --policy claude \\
      --from-save tests/fixtures/paper_town.save \\
      --seed 1234 --clock-epoch 1787659200 --steps 600 --upscale 1 \\
      --goal "Explore this town on foot and find another person to talk
              to. Keep moving and keep trying; do not end the run early."

It returned 0 after 442 steps and five decisions, stopping on the step
budget. `--upscale 1` because at `scaled_context()`'s measured scale 5 the
drawn region already fills 1280x720 (see `tuxghost/observe.py`'s `png`);
the default `--upscale 4` would have sent 5120x2880 PNGs the API
downscales anyway.

WHAT THESE TESTS ADD over the stub-client suite:

  * The trace exercises DIRECTIONAL input and a pushed `DialogState`. The
    older golden, `walk_1234.tuxghost`, presses button 64 (`A`) and
    nothing else for all 80 of its inputs -- despite its name it never
    walks -- so before this fixture existed, "the suite is green" said
    nothing about replaying movement or a dialog. That is not a
    hypothetical: it is how two S1 defects hid (see `docs/STATUS.org`).
  * `ClaudePolicy.parse_response` gets exercised against REAL model text.
    `tuxghost/agent/replay.py`'s docstring records that a replayed
    transcript covers `actions_from_json` but never `parse_response`'s
    fence parsing, and that the committed transcripts were synthetic;
    `test_transcript_raw_answers_reparse_to_their_recorded_actions` below
    closes exactly that gap with answers a model actually wrote.

WHAT THEY DO NOT ADD: nothing here re-runs the model, so the *request*
shape and the live response object's `.content`/`.type`/`.text` shape are
evidenced by the capture having happened at all (and by this file's
provenance record), not by anything a later `make check` re-verifies. A
live run is also not reproducible from seed alone -- see `docs/STATUS.org`'s
"Model sampling is not determinism".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tuxghost.agent.claude import ClaudePolicy
from tuxghost.execute import execute, verify
from tuxghost.trace import read

_HERE = Path(__file__).parent
TRACE = _HERE / "golden" / "claude_town_1234.tuxghost"
TRANSCRIPT = _HERE / "fixtures" / "claude_town_1234.decisions.jsonl"
RUN_INFO = _HERE / "fixtures" / "claude_town_1234.run.json"
SAVE = _HERE / "fixtures" / "paper_town.save"

#: `tuxemon.platform.const.buttons.A`, spelled out rather than imported so
#: this module's assertions do not depend on the vendored engine's own
#: constant staying put -- the recorded trace pins the integer, not a name.
A_BUTTON = 64


def _records() -> list[dict[str, Any]]:
    lines = TRANSCRIPT.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_live_captured_trace_verifies() -> None:
    assert verify(read(TRACE)) == 0


def test_live_captured_trace_still_reaches_its_recorded_digest() -> None:
    trace = read(TRACE)
    assert execute(trace).final_digest == trace.header.final_digest


def test_live_captured_trace_presses_directions_not_only_a() -> None:
    """The anti-vacuity control, and the whole reason this fixture earns
    its place next to `walk_1234.tuxghost`.

    `walk_1234`'s 80 inputs are button 64 (`A`) and nothing else, so the
    two tests above would pass identically against a trace that never
    moved the player. Assert on the recorded input stream that this one
    presses real directions -- and that a dialog was actually on the
    state stack while it did, which is what makes the replay cross a
    state push and not just a walk.
    """
    trace = read(TRACE)
    buttons = {button for _step, button, _value in trace.inputs}
    assert buttons - {A_BUTTON}, (
        f"every recorded input is button {A_BUTTON} (A); this fixture "
        "exists to cover directional input and adds nothing if it does not"
    )
    assert A_BUTTON in buttons, "no A press; the dialog could not have advanced"

    stacks = [tuple(record["state_stack"]) for record in _records()]
    assert any("DialogState" in stack for stack in stacks), (
        f"no decision was taken with a DialogState on the stack: {stacks}"
    )


def test_transcript_raw_answers_reparse_to_their_recorded_actions() -> None:
    """Real model text through `ClaudePolicy.parse_response`.

    Each transcript record carries both the model's `raw` answer and the
    `actions` the live run derived from it. Re-parsing `raw` must yield
    exactly those actions, which puts `parse_response`'s OWN fence
    handling (`_JSON_BLOCK`, last-block selection, the json-decode and
    non-object-payload checks) under a real model's formatting rather than
    a stub's. `ReplayPolicy` cannot do this: it calls `actions_from_json`
    on the already-parsed `actions` field and never touches
    `parse_response` (see `tuxghost/agent/replay.py`).

    Note this asserts against the recorded `actions`, NOT against
    `parse_response(raw)` twice: the recorded field was frozen by a
    different process, months of code changes ago from any future
    reader's point of view, so this is not the "compared an object to
    itself" trap `tests/test_golden.py`'s docstring describes.
    """
    # No client: `_ensure_client` (and with it the lazy `import
    # anthropic`) is only reached from `decide()`, which this test never
    # calls -- so this runs in `make check`'s SDK-less environment.
    policy = ClaudePolicy()
    records = _records()
    assert records, "the committed transcript is empty"
    for index, record in enumerate(records):
        raw = record["raw"]
        assert raw, f"record {index} has no raw answer"
        actions, notes, _outcome = policy.parse_response(raw)
        assert [
            {"button": a.button, "hold": a.hold, "settle": a.settle} for a in actions
        ] == record["actions"], f"record {index} reparsed to different actions"
        assert notes == record["notes"], f"record {index} reparsed to different notes"


def test_live_transcript_replays_to_the_captured_trace() -> None:
    """`ReplayPolicy` over the REAL transcript reproduces the live run's
    input schedule byte-for-byte and lands on the same digest.

    This is the pairing that makes a model-driven run auditable at all:
    the model is not reproducible, but its recorded decisions are, and a
    replay that silently drifted from them would make the whole
    `--policy replay` path a lie.

    Two deliberate differences from the live invocation, neither of which
    may change the digest:
      * `scaled=False`. `scaled_context()` refuses unless it is the first
        `DisplayContext` built in the process (`tuxghost/observe.py`), and
        this suite boots many clients per process. Task 9 measured the two
        contexts digest-identical at every index of a 300-step sequence;
        this test re-confirms it end to end on a real recorded run.
      * `stop_reason`. The live run ran out of step budget at 442 of 600
        steps; the replay runs out of TRANSCRIPT at the same step, so it
        stops for the other reason. Asserted, so the difference stays a
        known one.
    """
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.agent.replay import ReplayPolicy
    from tuxghost.agent.runner import run_agent
    from tuxghost.determinism import seed_all

    live = read(TRACE)
    run_info = json.loads(RUN_INFO.read_text())
    save_data = SaveData.model_validate(json.loads(SAVE.read_text()))

    seed_all(live.header.seed)
    result = run_agent(
        policy=ReplayPolicy(TRANSCRIPT),
        save_data=save_data,
        seed=live.header.seed,
        clock_epoch=live.header.clock_epoch,
        step_budget=run_info["step_budget"],
        recorder_kind="cu-agent",
        model=run_info["model"],
        upscale=1,
        scaled=False,
    )

    assert result.steps == live.header.step_count
    assert result.trace.inputs == live.inputs
    assert result.trace.header.final_digest == live.header.final_digest
    assert result.stop_reason == "policy returned STOP"
    assert run_info["stop_reason"] == "step budget"
