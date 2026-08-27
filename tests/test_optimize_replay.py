"""Task 9: replaying a recorded optimization.

`edits_from_json` is shared with `ClaudeEditor`, so a transcript record
is checked by exactly the rules a live model answer is -- the same
arrangement `actions_from_json` gives `ClaudePolicy`/`ReplayPolicy`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tuxghost.agent.types import Action
from tuxghost.optimize.editors.replay import ReplayEditor, edits_from_json
from tuxghost.optimize.edits import Delete, Insert, Replace
from tuxghost.optimize.schedule import ActionScript
from tuxghost.optimize.seal import CandidateResult

SCRIPT = ActionScript(lead_in=0, actions=(Action(2, 16, 8),))
#: `propose`'s `candidate` parameter is typed `CandidateResult`, but
#: `ReplayEditor` ignores it by design (see the module docstring) and
#: these tests never construct a real one. `cast` rather than `object`,
#: so `mypy --strict` sees a value of the declared type instead of
#: rejecting the call outright.
FAKE = cast(CandidateResult, None)


def _write(tmp_path: Path, rows: list[object]) -> Path:
    path = tmp_path / "edits.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_edits_from_json_builds_all_three_operations() -> None:
    raw = [
        {"op": "insert", "index": 0, "action": {"button": 2, "hold": 4, "settle": 1}},
        {"op": "delete", "index": 1},
        {"op": "replace", "index": 0, "action": {"button": 8, "hold": 4, "settle": 1}},
    ]
    assert edits_from_json(raw) == (
        Insert(0, Action(2, 4, 1)),
        Delete(1),
        Replace(0, Action(8, 4, 1)),
    )


def test_edits_from_json_refuses_a_non_list() -> None:
    with pytest.raises(TypeError, match="list"):
        edits_from_json({"op": "delete", "index": 0})


def test_edits_from_json_refuses_an_unknown_op() -> None:
    with pytest.raises(ValueError, match="unknown op"):
        edits_from_json([{"op": "reverse", "index": 0}])


def test_edits_from_json_refuses_a_missing_index() -> None:
    with pytest.raises(ValueError, match="index"):
        edits_from_json([{"op": "delete"}])


def test_edits_from_json_refuses_an_insert_without_an_action() -> None:
    with pytest.raises(ValueError, match="action"):
        edits_from_json([{"op": "insert", "index": 0}])


def test_replay_returns_each_round_in_order(tmp_path: Path) -> None:
    path = _write(tmp_path, [
        [{"op": "delete", "index": 0}],
        [{"op": "insert", "index": 0,
          "action": {"button": 2, "hold": 4, "settle": 1}}],
    ])
    editor = ReplayEditor(path)
    assert editor.propose(SCRIPT, FAKE, (0.0,)) == (Delete(0),)
    assert editor.propose(SCRIPT, FAKE, (0.0,)) == (Insert(0, Action(2, 4, 1)),)


def test_replay_stops_rather_than_looping(tmp_path: Path) -> None:
    """A transcript that silently restarted would make a replayed run
    diverge from the run it claims to reproduce."""
    path = _write(tmp_path, [[{"op": "delete", "index": 0}]])
    editor = ReplayEditor(path)
    assert editor.propose(SCRIPT, FAKE, (0.0,)) == (Delete(0),)
    assert editor.propose(SCRIPT, FAKE, (0.0,)) == ()
    assert editor.propose(SCRIPT, FAKE, (0.0,)) == ()


def test_a_malformed_transcript_is_refused_at_construction(tmp_path: Path) -> None:
    """Not lazily, mid-run: a malformed transcript is knowable before the
    game boots, so failing now saves a pointless run (S2 Ruling X)."""
    path = _write(tmp_path, [[{"op": "delete", "index": 0}], [{"op": "nope"}]])
    with pytest.raises(ValueError, match="unknown op"):
        ReplayEditor(path)


def test_a_non_list_round_is_refused_at_construction(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"op": "delete", "index": 0}])
    with pytest.raises(TypeError, match="list"):
        ReplayEditor(path)


def test_an_empty_round_is_a_stop_not_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path, [[]])
    editor = ReplayEditor(path)
    assert editor.propose(SCRIPT, FAKE, (0.0,)) == ()
