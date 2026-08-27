"""Task 1: `run_agent(digest_every=N)` and `execute(checkpoint=N)` must
agree, step for step, on which steps get sampled.

Both skip step 0 and both count absolutely, but they compute that
independently -- `run_agent` samples inside a per-ACTION `run_steps` call
with a `_base` offset added back, `execute` samples inside one whole-run
call. S3 reads the first and verifies with the second, so this file is
the assertion that the two conventions are one convention.
"""

from __future__ import annotations

import json
from pathlib import Path

from tuxemon.platform.const import buttons
from tuxemon.save_system.save_state import SaveData

from tuxghost.agent.runner import run_agent
from tuxghost.agent.scripted import ScriptedPolicy
from tuxghost.agent.types import Action
from tuxghost.determinism import seed_all
from tuxghost.execute import execute

FIXTURE = Path(__file__).parent / "fixtures" / "paper_town.save"
EPOCH = 1787659200
SEED = 1234

#: A MOVING route, not an idle one. 600 idle ticks leave `state_of`
#: byte-identical (measured, see docs/STATUS.org "The digest's measured
#: blind spot"), so an idle schedule would make every sampled digest
#: equal and this test would pass against two conventions that disagree.
#: (12, 12) is leavable DOWN/UP/RIGHT only -- LEFT is blocked.
ROUTE = (
    Action(buttons.DOWN, 16, 8),
    Action(buttons.RIGHT, 16, 8),
    Action(buttons.DOWN, 16, 8),
    Action(buttons.UP, 16, 8),
)
COST = sum(a.hold + a.settle for a in ROUTE)
EVERY = 8


def _save() -> SaveData:
    return SaveData.model_validate(json.loads(FIXTURE.read_text()))


def test_run_agent_and_execute_sample_the_same_steps() -> None:
    seed_all(SEED)
    recorded = run_agent(
        policy=ScriptedPolicy([ROUTE]),
        save_data=_save(),
        seed=SEED,
        clock_epoch=EPOCH,
        step_budget=COST,
        observe=False,
        digest_every=EVERY,
        recorder_kind="offline-agent",
    )
    replayed = execute(recorded.trace, checkpoint=EVERY)

    assert [s for s, _ in recorded.digests] == [s for s, _ in replayed.checkpoints]
    assert recorded.digests == replayed.checkpoints


def test_the_sampled_route_actually_moves() -> None:
    """Anti-vacuity control. If the route never changed the digest, the
    test above would pass against conventions that sample DIFFERENT steps,
    because every sample would be the same hash."""
    seed_all(SEED)
    recorded = run_agent(
        policy=ScriptedPolicy([ROUTE]),
        save_data=_save(),
        seed=SEED,
        clock_epoch=EPOCH,
        step_budget=COST,
        observe=False,
        digest_every=EVERY,
        recorder_kind="offline-agent",
    )
    assert len(recorded.digests) >= 8, recorded.digests
    assert 0 not in [s for s, _ in recorded.digests], "step 0 must be skipped"
    assert len({d for _, d in recorded.digests}) > 1, (
        "every sampled digest is identical -- this route proves nothing "
        "about sampling alignment"
    )
