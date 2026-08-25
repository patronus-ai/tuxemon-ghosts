import time
from typing import Any

from tuxghost.loop import (
    FIXED_DT,
    STEP_RATE,
    InputSchedule,
    install_schedule,
    run_steps,
)


def test_step_rate_matches_upstream_fixed_dt() -> None:
    assert STEP_RATE == 60
    assert FIXED_DT == 1.0 / 60.0


def test_schedule_delivers_events_on_exact_steps() -> None:
    """Timing is preserved: an event scheduled for step 5 arrives at step 5,
    not on the next frame the way upstream's playback does."""
    from tuxghost.boot import build_client

    client, _session = build_client(seed=1234)
    seen: list[tuple[int, int]] = []
    install_schedule(client, {5: [(64, 1.0)], 9: [(64, 0.0)]})

    original = client.input_manager.process_events

    def spy() -> object:
        for event in original():
            seen.append((step_counter[0], event.button))
            yield event

    step_counter = [0]
    client.input_manager.process_events = spy
    run_steps(client, 12, hook=lambda i: step_counter.__setitem__(0, i))

    assert [s for s, _b in seen] == [5, 9]


def test_synthetic_events_serialize_deterministically() -> None:
    """Upstream's `PlayerInput.timestamp` defaults to `time.time()`.
    Nothing consumes it today, but `PlayerInput.to_dict()` serializes it,
    so the moment a later task snapshots/hashes input events, a wall-clock
    reading leaking in here would reintroduce nondeterminism -- the same
    class of bug as the `Battle.timestamp` seam. Pin the actual property
    we need (identical schedule -> byte-identical serialized events) by
    running the same schedule through `install_schedule` twice, with a
    real wall-clock gap between the runs, and requiring the serialized
    output to match exactly. This is stronger than asserting a single
    field's value: it fails on *any* nondeterministic field, not just the
    one we already know about."""
    from tuxghost.boot import build_client

    client, _session = build_client(seed=1234)
    schedule: InputSchedule = {5: [(64, 1.0)], 9: [(64, 0.0)]}

    def capture_once() -> list[dict[str, Any]]:
        install_schedule(client, schedule)
        original = client.input_manager.process_events
        captured: list[dict[str, Any]] = []

        def spy() -> object:
            for event in original():
                captured.append(event.to_dict())
                yield event

        client.input_manager.process_events = spy
        run_steps(client, 12)
        return captured

    first = capture_once()
    time.sleep(1.1)  # real wall-clock gap a `time.time()` default would see
    second = capture_once()

    assert first  # two events were actually captured, not an empty no-op
    assert first == second
