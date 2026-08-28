"""S3's live-model capture: a real `--editor claude` optimization.

Every other `ClaudeEditor` test in this repo injects a stub client. These
two artifacts are not stubbed:

  * `tests/fixtures/claude_optimize_town.edits.jsonl` -- the 32 rounds of
    edits a real `tuxghost optimize --editor claude` run proposed against
    the real Anthropic API on 2026-08-27, one JSON array per line, in the
    order proposed. This is exactly the shape `--edits` reads, which is
    the property `tuxghost/cli.py`'s `_round_json` docstring says makes
    an LLM-driven optimization auditable at all.
  * `tests/fixtures/claude_optimize_town.run.json` -- that run's own
    record of parent digest, goal, objective, target, model id, the four
    bounds and its stop reason.

THE INVOCATION, verbatim (model `claude-sonnet-5`, the
`tuxghost.agent.claude.DEFAULT_MODEL` the CLI resolves when `--model` is
omitted -- so no other model id has live evidence behind it):

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYTHONHASHSEED=0 \\
    python -m tuxghost.cli optimize \\
      --trace tests/golden/scripted_town_1234.tuxghost \\
      --editor claude \\
      --objective reach-tile --target spyder_paper_town.tmx:19,14 \\
      --rounds 32 --patience 32 --max-rejections 32 --max-cost 4000 \\
      --out ... --run-dir ...

Three such runs were made. This fixture is the second, the only one that
improved across more than a single round; the other two are described in
`docs/2026-08-27-editor-comparison.org` and are not committed, because
their proposals are byte-identical to this one's round 1.

WHAT THE RUN DID, measured:

    round 0  (0.0, -3.0, -176.0)   the parent
    round 1  (0.0, -0.0, -230.0)   ARRIVED -- one inserted RIGHT, hold 48
    round 2  (0.0, -0.0, -188.0)   9 edits, script compressed
    round 3  (0.0, -0.0, -178.0)   2 edits, the winner
    rounds 4-32                    the SAME two edits, 29 times, each
                                   scoring (0.0, -1.0, -176.0) -- worse
                                   than its own best, never accepted

`hold 48` in round 1 is three tiles at this game's 16 steps per tile,
against a target exactly three tiles east. The model computed the
mechanics rather than searching for them: `MutationEditor` needed 19-32
rounds to arrive on the same parent, and arrived in only two of three
seeds.

WHAT THESE TESTS DO NOT COVER, and it is a real gap rather than an
oversight. `ClaudeEditor.parse_response` is NOT exercised against real
model text here. `ClaudeEditor` keeps the raw reply on `self.last_raw`
but nothing persists it: `optimize`'s run directory has no equivalent of
`agent`'s `decisions.jsonl` and its `raw` field, so a live optimize run's
actual model answers are unrecoverable once the process exits. S2 closed
exactly this hole for `agent` (see `tests/test_live_capture.py`, whose
`test_transcript_raw_answers_reparse_to_their_recorded_actions` has no
counterpart here, and cannot until `optimize` records raw replies). What
IS covered is everything downstream of parsing: the edits the model
actually produced, validated by the same `edits_from_json` the live path
used, and replayed to the same scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tuxghost.optimize.editors.replay import ReplayEditor, edits_from_json
from tuxghost.optimize.edits import Insert
from tuxghost.optimize.objective import ReachTile
from tuxghost.optimize.runner import optimize
from tuxghost.trace import Trace, read

_HERE = Path(__file__).parent
PARENT = _HERE / "golden" / "scripted_town_1234.tuxghost"
EDITS = _HERE / "fixtures" / "claude_optimize_town.edits.jsonl"
RUN_INFO = _HERE / "fixtures" / "claude_optimize_town.run.json"

TARGET = ReachTile("spyder_paper_town.tmx", (19, 14))

#: The live run's own scores, round by round, copied from its
#: `optimize.jsonl`. Only the first four are replayed here -- see
#: `test_replaying_the_transcript_reproduces_the_live_scores`.
LIVE_SCORES: list[tuple[float, ...]] = [
    (0.0, -3.0, -176.0),
    (0.0, -0.0, -230.0),
    (0.0, -0.0, -188.0),
    (0.0, -0.0, -178.0),
]


def _rounds() -> list[list[dict[str, Any]]]:
    lines = EDITS.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _parent() -> Trace:
    return read(PARENT)


def test_the_run_info_describes_the_run_these_edits_came_from() -> None:
    """The transcript is not an orphan: its provenance is on disk, not
    only in a commit message. Asserted against the committed parent's own
    digest, so a transcript paired with the wrong trace is caught."""
    info = json.loads(RUN_INFO.read_text())
    assert info["editor"] == "claude"
    assert info["model"] == "claude-sonnet-5"
    assert info["objective"] == "reach-tile"
    assert info["target"] == "spyder_paper_town.tmx:19,14"
    assert info["parent_digest"] == _parent().header.final_digest
    # Handoff item A3: without this key the goal a live `--editor claude`
    # run was given survived nowhere. This fixture is the first committed
    # artifact that carries one.
    assert info["goal"] == "reach tile (19, 14) on map spyder_paper_town.tmx"
    assert info["seed"] is None, "claude draws from no seed"


def test_every_recorded_round_reparses_through_the_shared_validator() -> None:
    """`edits_from_json` is the SAME function the live run validated
    these edits with, and the same one `--edits` would validate them with
    on replay. A round that only round-tripped through `Edit.to_json` but
    not back would make `optimize.jsonl` unreadable by the very flag its
    docstring says makes an LLM run auditable.

    Pure, no engine: this is the cheap half of the capture's value.
    """
    rounds = _rounds()
    assert len(rounds) == 32, len(rounds)
    for index, raw in enumerate(rounds):
        edits = edits_from_json(raw)
        assert len(edits) == len(raw), f"round {index} lost an edit"
    # Not a vacuous pass over empty rounds: the run's first proposal is
    # one insert and its second is nine edits.
    assert len(rounds[0]) == 1
    assert len(rounds[1]) == 9


def test_the_first_proposal_is_the_arrival_the_model_computed() -> None:
    """Round 1 is a single inserted RIGHT with `hold 48` -- three tiles at
    16 steps per tile, against a target exactly three tiles east.

    Pinned because it is the capture's most specific finding: the model
    did not stumble onto the arrival, it computed the hold. A future
    prompt change that stopped conveying the tile geometry would still
    produce SOME edit and still pass a "the run improved" assertion; it
    would not reproduce this one.
    """
    (edit,) = edits_from_json(_rounds()[0])
    # `isinstance`, not a duck-typed `.action` lookup: `edits_from_json`
    # returns the `Edit` union and `Delete` carries no action at all, so
    # this both narrows the type for mypy and asserts the op.
    assert isinstance(edit, Insert), edit
    assert edit.index == 8, "appended after the parent's eight actions"
    assert edit.action.button == 8, "tuxemon.platform.const.buttons.RIGHT"
    assert edit.action.hold == 48, "three tiles at 16 steps each"


def test_replaying_the_transcript_reproduces_the_live_scores() -> None:
    """THE DETERMINISM CONTROL, and the reason the transcript is worth
    committing at all.

    A live model run is not reproducible from a seed -- `docs/STATUS.org`,
    "Model sampling is not determinism". What IS reproducible is the
    ENGINE's response to the edits that run produced: feeding them back
    through `ReplayEditor` against the same parent, target and bounds must
    reach the same scores it reached live. If it does not, either the
    engine drifted or the recorded transcript does not describe the run
    it claims to.

    It seals four candidates of 176-230 steps each, ~18 s, and stays in
    the FAST tier deliberately: a `@pytest.mark.slow` here without the
    `skipif(not TUXGHOST_RUN_SLOW)` this repo pairs it with would make
    the test run in BOTH tiers, and adding the skipif would remove the
    capture's only engine-level control from every ordinary `make
    check-fast`.

    Only the first four rounds are replayed. The live run's rounds 4-32
    are 29 repetitions of one non-improving proposal (see the module
    docstring); replaying them would add minutes of engine time to
    re-derive the same rejected score 29 times.
    """
    result = optimize(
        _parent(),
        ReplayEditor(EDITS),
        TARGET,
        rounds=3,
        patience=3,
        max_rejections=3,
        max_cost=4_000,
    )
    replayed = [r.score for r in result.rounds if r.score is not None]
    assert replayed == LIVE_SCORES, (
        f"replay diverged from the live run.\n  live:     {LIVE_SCORES}\n"
        f"  replayed: {replayed}"
    )
    assert result.best_round == 3
