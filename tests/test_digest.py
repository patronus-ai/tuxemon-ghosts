from tuxemon.platform.const import buttons
from tuxemon.session import local_session

from tuxghost.boot import build_client
from tuxghost.digest import EXEMPTIONS, digest_of, state_of
from tuxghost.loop import InputSchedule, install_schedule, run_steps


def _run(seed: int, steps: int = 900) -> str:
    # `local_session` is a module-level singleton (see the note on
    # `test_digest_discriminates_between_seeds`). Calling `build_client`
    # twice in one process WITHOUT resetting first leaves the previous
    # run's state stack (e.g. `DialogState`/`ImageState` from the intro)
    # on the new client, so the second run's gameplay genuinely diverges
    # from the first -- measured: `player_steps` came back 1.0 on the
    # first in-process call and 3.0 on the second, for byte-identical
    # seed and schedule, purely from that leftover contamination. Reset
    # before every build, exactly as `boot_from_save` already does.
    local_session.reset()
    client, session = build_client(seed=seed)
    schedule: InputSchedule = {}
    for i in range(40):
        schedule[30 + i * 12] = [(buttons.A, 1.0)]
        schedule[32 + i * 12] = [(buttons.A, 0.0)]
    install_schedule(client, schedule)
    run_steps(client, steps)
    return digest_of(session)


def test_digest_is_stable_for_one_seed() -> None:
    assert _run(1234) == _run(1234)


def test_digest_discriminates_between_seeds() -> None:
    """A digest that cannot tell seeds apart makes every determinism test
    vacuous. This is the control that caught a worthless probe in the spike.

    `tuxemon.session.local_session` is a module-level singleton:
    `build_client()` returns that same object every call, so `session_a`
    and `session_b` below are literally the same object once the second
    `build_client` call has run -- `GameLauncher.launch` re-creates the
    player *in place* on that shared session. Computing `digest_of(session_a)`
    only after `session_b` exists would silently digest the same
    already-mutated session twice (an assertion of x != x on identical
    state, which reads as "discrimination" but proves nothing). Digest A is
    therefore captured as a string BEFORE the second client is built.
    """
    local_session.reset()
    _client_a, session_a = build_client(seed=1234)
    session_a.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    digest_a = digest_of(session_a)

    local_session.reset()
    _client_b, session_b = build_client(seed=99)
    session_b.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    digest_b = digest_of(session_b)

    assert digest_a != digest_b


def test_digest_sees_gameplay_state() -> None:
    _client, session = build_client(seed=1234)
    before = digest_of(session)
    session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    assert digest_of(session) != before


def test_exemptions_each_name_a_defect() -> None:
    for field, reason in EXEMPTIONS.items():
        assert reason.strip(), f"exemption {field!r} has no stated reason"
    assert "timestamp" not in EXEMPTIONS, (
        "Battle.timestamp becomes deterministic under patch 0004; exempting "
        "it would re-hide the bug the spike spent a bisect finding"
    )


def test_state_of_is_not_blind_to_the_state_stack() -> None:
    """The spike's worthless digest never looked at `state_stack`, so it
    could not see the one thing that was actually changing. Pin its
    presence directly rather than only through an end-to-end hash
    comparison."""
    _client, session = build_client(seed=1234)
    state = state_of(session)
    assert state["state_stack"], "state_stack must be present and non-empty"
