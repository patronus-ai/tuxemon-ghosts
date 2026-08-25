from tuxghost.loop import FIXED_DT, STEP_RATE, install_schedule, run_steps


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
