from __future__ import annotations

import json
from pathlib import Path

import pytest
from tuxemon.platform.const import buttons

from tuxghost.agent.replay import ReplayPolicy
from tuxghost.agent.scripted import ScriptedPolicy
from tuxghost.agent.types import Action, Observation

OBS = Observation(
    step=0, frame_png=b"", frame_is_blank=True,
    state_stack=("WorldState",), map_name="m.tmx", tile_pos=(1, 1),
)


def test_scripted_policy_yields_decisions_in_order_then_stops() -> None:
    policy = ScriptedPolicy(
        [(Action(buttons.UP, 2, 2),), (Action(buttons.A, 3, 3),)]
    )
    assert policy.decide(OBS) == (Action(buttons.UP, 2, 2),)
    assert policy.decide(OBS) == (Action(buttons.A, 3, 3),)
    assert policy.decide(OBS) == ()  # exhausted means stop


def test_scripted_policy_can_repeat_its_last_decision() -> None:
    policy = ScriptedPolicy([(Action(buttons.UP, 2, 2),)], repeat_last=True)
    assert policy.decide(OBS) == policy.decide(OBS) != ()


def test_replay_policy_replays_a_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "actions": [{"button": buttons.DOWN, "hold": 12, "settle": 6}],
                    "notes": "heading south",
                    "raw": '{"actions": [...]}',
                },
                {"actions": [], "notes": "done", "raw": "", "claimed_outcome": "left town"},
            ]
        )
    )
    policy = ReplayPolicy(transcript)
    assert policy.decide(OBS) == (Action(buttons.DOWN, 12, 6),)
    assert policy.last_notes == "heading south"
    assert policy.decide(OBS) == ()
    assert policy.claimed_outcome == "left town"


def test_replay_policy_refuses_a_malformed_record(tmp_path: Path) -> None:
    transcript = tmp_path / "bad.jsonl"
    transcript.write_text(json.dumps({"actions": [{"button": 64}]}))
    policy = ReplayPolicy(transcript)
    with pytest.raises(ValueError, match="hold"):
        policy.decide(OBS)


def test_replay_policy_does_not_loop_when_exhausted(tmp_path: Path) -> None:
    """A transcript that silently restarted would make a replayed run
    diverge from the run it claims to reproduce."""
    transcript = tmp_path / "one.jsonl"
    transcript.write_text(
        json.dumps({"actions": [{"button": buttons.A, "hold": 2, "settle": 2}]})
    )
    policy = ReplayPolicy(transcript)
    assert policy.decide(OBS) != ()
    assert policy.decide(OBS) == ()
    assert policy.decide(OBS) == ()
