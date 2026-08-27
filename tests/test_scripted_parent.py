"""The optimizer's parent fixture, and the one property it exists for.

`tests/golden/scripted_town_1234.tuxghost` is the trace every
`tuxghost.optimize` test now treats as its parent. It replaced
`claude_town_1234.tuxghost` in that role for a measured reason, and this
module pins the reason rather than assuming it.

WHY THE OLD PARENT WAS RETIRED. `claude_town_1234` ends with
`state_stack == ['DialogState', 'WorldState']`, and that trailing dialog
swallows every input: appended `UP`/`DOWN`/`LEFT`/`RIGHT`, `A` then
`DOWN`, and `B` then `DOWN` all leave the tile at `(11, 14)`. So
`ReachTile`'s `distance` term could never carry an acceptance against it
and "optimize" could only ever mean SHORTEN, never RE-ROUTE. It remains
committed and is still exercised -- `tests/test_live_capture.py` is
built on it, and `tests/test_optimize_runner.py` keeps a parametrized
case on it precisely because an input-swallowing parent is a real
adversarial scenario worth not losing.

THE INVOCATION THAT PRODUCED THIS FIXTURE, verbatim:

    python -m tuxghost.cli agent --policy scripted \\
      --actions tests/fixtures/scripted_town_1234.actions.jsonl \\
      --from-save tests/fixtures/paper_town.save \\
      --seed 1234 --clock-epoch 1787659200 --steps 200 \\
      --out tests/golden/scripted_town_1234.tuxghost --run-dir <run>

The actions file and the run's own `run.json` are committed beside it
(`tests/fixtures/scripted_town_1234.{actions,run}.jsonl/json`), so the
fixture is re-derivable and its provenance is not confined to a commit
message -- the arrangement `test_live_capture.py` established. The run
directory's `frames/` (580 KB of PNGs) is deliberately NOT committed;
an early trace in this project's history carried embedded screenshots
and still accounts for ~75% of the repository's object size.

MEASURED, on this tree, at the time of committing:

    step_count 176   lead_in 0   16 inputs   8 actions
    ends spyder_paper_town.tmx tile [16, 14] facing RIGHT
    state_stack ['WorldState']          <- bare, no dialog
    verify() == 0
    patch_series_id matches this build  <- warns ZERO times

`recorder` is `"human"`, not `"offline-agent"`: a `--policy scripted`
run replays a canned input list rather than making decisions, so it
records the same way `walk_1234` does. That is upstream behaviour this
module observes, not a claim this fixture makes about itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tuxghost.agent.types import Action
from tuxghost.execute import execute, verify
from tuxghost.optimize.edits import Insert, apply_edits
from tuxghost.optimize.objective import ReachTile
from tuxghost.optimize.schedule import lift
from tuxghost.optimize.seal import seal
from tuxghost.trace import Trace, read

PARENT = Path(__file__).parent / "golden" / "scripted_town_1234.tuxghost"

#: `tuxemon.platform.const.buttons.RIGHT`/`LEFT`, spelled out rather than
#: imported for the reason `test_live_capture.py` gives: the recorded
#: trace pins the integer, not a name.
RIGHT = 8
LEFT = 4

MAP = "spyder_paper_town.tmx"
#: Where the parent ends, measured.
PARENT_TILE = [16, 14]
#: Three tiles east of `PARENT_TILE`: reachable by appending `RIGHT`, and
#: NOT already satisfied by the parent, so the distance term is a real
#: unmet term rather than one relocated onto the parent's own endpoint.
TARGET = ReachTile(MAP, (19, 14))


@pytest.fixture(scope="module")
def parent() -> Trace:
    return read(PARENT)


def test_the_parent_verifies(parent: Trace) -> None:
    assert verify(parent) == 0


def test_the_parent_ends_on_a_bare_world_state(parent: Trace) -> None:
    """The property that made this fixture necessary. If a future change
    to the save, the map or the engine leaves this trace ending inside a
    `DialogState` again, it stops being usable as an optimizer parent --
    and would do so SILENTLY, because every optimize test would still be
    green while measuring step count alone."""
    state = execute(parent).final_state
    assert state["map"] == MAP
    assert state["tile_pos"] == PARENT_TILE
    assert state["state_stack"] == ["WorldState"], (
        "the parent no longer ends on a bare WorldState; a pushed state "
        "swallows appended input and makes ReachTile's distance term "
        "dead against this fixture -- exactly why claude_town_1234 was "
        "retired from this role"
    )


def test_the_distance_term_is_live_against_this_parent(parent: Trace) -> None:
    """THE REASON THIS FIXTURE EXISTS, asserted rather than assumed.

    Against the retired parent every candidate ended on the same tile, so
    `ReachTile`'s middle term was constant and only `steps` could ever
    carry an acceptance. Here an appended `RIGHT` must strictly IMPROVE
    the distance term and an appended `LEFT` must strictly WORSEN it.

    Asserted on the distance term specifically, not on the score tuple:
    appending steps always makes the third term worse, so a whole-tuple
    comparison would pass on lexicographic priority alone and would still
    pass against an immovable parent. Measured values at the time of
    writing: parent -3, +RIGHT -2, +LEFT -4.
    """
    script = lift(parent.inputs, parent.header.step_count)
    tail = len(script.actions)

    def distance_after(button: int) -> float:
        edited = apply_edits(script, [Insert(tail, Action(button, 16, 6))])
        return TARGET.score(seal(edited, parent))[1]

    baseline = TARGET.score(seal(script, parent))[1]
    assert distance_after(RIGHT) > baseline, (
        "appending RIGHT did not get closer to the target; this parent's "
        "input is being swallowed and the distance term is dead"
    )
    assert distance_after(LEFT) < baseline, (
        "appending LEFT did not get further from the target; the tile is "
        "not responding to input in either direction"
    )
