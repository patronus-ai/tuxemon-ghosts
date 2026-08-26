"""Task 14: a committed golden trace, pinned to the digest it reached when
recorded.

`walk_1234.tuxghost` is not a synthetic/empty run: `initial_state` already
carries a two-monster party (rockitten, budaye) and an item (2x potion),
added directly to the session before `Recorder` captured it -- real
`SaveData`, restorable by `boot_from_save` the same way any other trace's
`initial_state` is. The recorded inputs are a 40x A-mash schedule over 900
steps (the same shape `tests/test_digest.py`'s `_run` and
`tests/test_combat_determinism.py`'s intro-clearing walk use), which
leaves the player away from the fresh spawn point (`spyder_bedroom.tmx`
tile (4, 4)) with `DialogState`/`ImageState` still layered over
`WorldState` -- genuine progress through the intro dialogue, not a no-op.
See `tests/golden/` generation notes in the task-14 report for the exact
script and why the brief's own literal Step 3 script (missing
`install_schedule`/`run_steps` before `finish()`) would have pinned a
vacuous digest instead.

`EXPECTED_DIGEST_FILE` (`walk_1234.digest`) is a redundant, human-readable/
diffable sidecar: `trace.header.final_digest` is the field `verify()`
actually compares against and is authoritative. Both are asserted here so
a corrupted sidecar OR a corrupted header both fail loudly rather than
silently agreeing with each other.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tuxghost.execute import execute, verify
from tuxghost.trace import read

GOLDEN = Path(__file__).parent / "golden" / "walk_1234.tuxghost"
EXPECTED_DIGEST_FILE = GOLDEN.with_suffix(".digest")


def test_golden_trace_still_reaches_its_recorded_digest() -> None:
    trace = read(GOLDEN)
    assert execute(trace).final_digest == EXPECTED_DIGEST_FILE.read_text().strip()
    # The sidecar is redundant with the header by construction (both were
    # stamped from the same recording run) -- assert they agree, so a hand
    # edit to only one of the two is caught here rather than silently
    # trusted.
    assert trace.header.final_digest == EXPECTED_DIGEST_FILE.read_text().strip()


def test_golden_trace_verifies() -> None:
    assert verify(read(GOLDEN)) == 0


def test_golden_trace_reaches_a_nonempty_party() -> None:
    """Companion control against a vacuous golden trace: a route that adds
    nothing to the party and pins the digest of an empty run would still
    pass the two tests above. Assert directly on `initial_state` (the
    party is baked in before recording starts -- see the module
    docstring) that this route actually did something."""
    trace = read(GOLDEN)
    monsters = trace.initial_state["npc_state"]["monsters"]
    items = trace.initial_state["npc_state"]["items"]
    assert [m["slug"] for m in monsters] == ["rockitten", "budaye"]
    assert [(i["slug"], i["quantity"]) for i in items] == [("potion", 2)]


def test_golden_trace_round_trips_through_record_execute_record() -> None:
    """record -> execute -> record -> compare. Replays the golden trace
    exactly the way `execute()` does (`boot_from_save` + replay its
    recorded inputs), then wraps a FRESH `Recorder` around the resulting
    live session and seals it with `step_count=0` -- i.e. records what was
    just reached. That independently-driven re-recording must land on
    exactly the state `execute()` itself reports for the same trace,
    compared field-by-field with `tuxghost.compare.first_difference` (not
    only by digest, so a same-digest-different-tree bug -- impossible
    today, but exactly the kind of thing a bare hash comparison could
    hide -- would still be caught) as well as by digest.

    This is a stronger check than `test_recorded_trace_round_trips_
    without_warning_or_refusal` in `tests/test_record.py`, which only
    round-trips through JSON (write -> read). This test round-trips
    through the actual replay pipeline `execute()` uses internally
    (`boot_from_save` + `install_schedule` + `run_steps`), then through
    `Recorder` again -- proving record and execute agree on what state a
    trace's inputs reach, not just that serialisation is lossless.
    """
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save
    from tuxghost.compare import first_difference
    from tuxghost.determinism import pin_clock, seed_all
    from tuxghost.digest import state_of
    from tuxghost.loop import InputSchedule, install_schedule, run_steps
    from tuxghost.record import Recorder

    trace_a = read(GOLDEN)
    result = execute(trace_a)

    seed_all(trace_a.header.seed)
    pin_clock(trace_a.header.clock_epoch)
    save_data = SaveData.model_validate(trace_a.initial_state)
    client, session = boot_from_save(
        save_data, seed=trace_a.header.seed, clock_epoch=trace_a.header.clock_epoch
    )
    schedule: InputSchedule = {}
    for step, button, value in trace_a.inputs:
        schedule.setdefault(step, []).append((button, value))
    install_schedule(client, schedule)
    run_steps(client, trace_a.header.step_count)

    trace_b = Recorder(
        session,
        seed=trace_a.header.seed,
        clock_epoch=trace_a.header.clock_epoch,
        recorder="offline-agent",
    ).finish(step_count=0)

    assert trace_b.header.final_digest == result.final_digest
    assert first_difference(state_of(session), result.final_state) is None


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("TUXGHOST_RUN_SLOW"), reason="minutes per run")
def test_long_horizon_trace_is_stable() -> None:
    trace = read(GOLDEN)
    trace.header.step_count = 120_000
    assert execute(trace).final_digest == execute(trace).final_digest
