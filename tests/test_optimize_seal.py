"""Task 5: compile a script, run it, seal it.

THE IDENTITY CONTROL is the point of this file: sealing a script lifted
straight out of a trace, with no edits, must reproduce that trace's
`inputs`, `final_digest` AND `step_count`. All three are asserted because
a probe of this exact path once matched inputs and digest while dropping
the parent's last 10 steps -- the digest could not see it, because those
steps changed nothing that had not already settled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tuxghost.execute import execute
from tuxghost.optimize.edits import Delete, apply_edits
from tuxghost.optimize.schedule import lift
from tuxghost.optimize.seal import seal
from tuxghost.trace import read

GOLDEN = Path(__file__).parent / "golden"
LIVE = GOLDEN / "claude_town_1234.tuxghost"


def test_sealing_an_unedited_script_reproduces_its_parent_exactly() -> None:
    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    result = seal(script, parent)

    assert result.trace.inputs == parent.inputs
    assert result.trace.header.step_count == parent.header.step_count
    assert result.steps == parent.header.step_count
    assert result.trace.header.final_digest == parent.header.final_digest


def test_a_sealed_candidate_is_an_offline_agent_trace_that_verifies() -> None:
    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    result = seal(script, parent, taints=("derived in a test",))

    assert result.trace.provenance.recorder == "offline-agent"
    assert result.trace.provenance.taints == ["derived in a test"]
    assert execute(result.trace).final_digest == result.trace.header.final_digest


def test_an_edited_script_changes_the_input_stream() -> None:
    """A control against `seal` ignoring its script: deleting an action
    must produce fewer inputs and a shorter run."""
    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    edited = apply_edits(script, [Delete(0)])
    result = seal(edited, parent)

    assert len(result.trace.inputs) == len(parent.inputs) - 2
    assert result.steps == edited.cost() < parent.header.step_count


def test_seal_samples_checkpoints_on_executes_convention() -> None:
    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    result = seal(script, parent, checkpoint=64, taints=())

    assert [s for s, _ in result.checkpoints] == [64, 128, 192, 256, 320, 384]
    assert result.checkpoints == execute(parent, checkpoint=64).checkpoints
    assert sorted(result.checkpoint_states) == [s for s, _ in result.checkpoints]
    assert "tile_pos" in result.checkpoint_states[64]


def test_seal_captures_nothing_when_checkpoint_is_zero() -> None:
    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    result = seal(script, parent)
    assert result.checkpoints == [] and result.checkpoint_states == {}


def test_seal_runs_exactly_the_scripts_cost() -> None:
    """`seal` counts the steps it actually ran and refuses to seal a trace
    claiming a different number. Not tautological: the count comes from
    `run_steps`' own per-step hook, so a `run_steps(client, cost - 1)`
    fires it. See this task's demonstration step."""
    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    result = seal(script, parent)
    assert result.steps == script.cost()


def test_seal_refuses_a_script_that_would_outlive_its_budget() -> None:
    parent = read(LIVE)
    script = lift(parent.inputs, parent.header.step_count)
    with pytest.raises(ValueError, match="max_cost"):
        seal(script, parent, max_cost=10)


def test_an_action_free_script_still_runs_its_lead_in() -> None:
    parent = read(LIVE)
    result = seal(lift([], 120), parent)
    assert result.steps == 120
    assert result.trace.inputs == []
    assert result.trace.header.step_count == 120
