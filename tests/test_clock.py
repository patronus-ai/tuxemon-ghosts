"""Pins every hunk of patch 0004, the injectable clock.

`tuxemon.core.clock` is the single choke point patch 0004 introduces;
`tuxghost.determinism.pin_clock` is the public entry point. Each test below
pins one call site the patch rewrote from a raw wall-clock read
(`datetime.now()` / `date.today()` / `time.time()`) to `now()`/
`now_datetime()`. Per the project's process rule, a cross-run *difference*
assertion alone can never prove a value came from the pinned epoch rather
than from chance -- only a same-seed *equality* assertion against a value
independently computed from that epoch can. Every discriminating
(different-epoch) test here is paired with an equality/stability test on
the same route.
"""

from __future__ import annotations

import os
import time as _time
from datetime import UTC, datetime

from tuxemon.db import OutputBattle
from tuxemon.save_system.save_state import TIME_FORMAT

from tuxghost.boot import build_client, snapshot_save
from tuxghost.determinism import pin_clock, seed_all

EPOCH = 1787694000  # 2026-08-25T21:20:00 UTC (via now_datetime's UTC interpretation)


def _expected_datetime(epoch: float) -> datetime:
    """The exact conversion `tuxemon.core.clock.now_datetime()` performs.

    Naive on purpose, matching production: `now_datetime()` builds an
    aware UTC datetime via `datetime.fromtimestamp(epoch, tz=UTC)`
    -- so the pinned epoch alone determines the result, independent of the
    host's `TZ` (see `test_pinned_epoch_is_independent_of_host_timezone`
    below and the docstring on `tuxemon.core.clock.now_datetime`) -- then
    strips the tzinfo, since `TimeHandler` (hemisphere season logic,
    `daytime`/`stage_of_day` cutoffs, weekday names, ...) and everything
    else that consumes it was already naive-datetime throughout upstream
    before this patch; comparing against a tz-aware value here would raise
    `TypeError` on any comparison with the naive value this test is
    checking, not make the test more correct.
    """
    return datetime.fromtimestamp(epoch, tz=UTC).replace(tzinfo=None)


def test_time_variables_come_from_the_pinned_epoch() -> None:
    """Pins `time_handler.py`'s `get_current_time()` hunk: both
    `get_ordinal()` and `get_time_variables()` -- and therefore
    `day_of_year`/`hour` game variables -- route through it."""
    seed_all(1234)
    pin_clock(EPOCH)
    _client, session = build_client(seed=1234)
    session.client.event_engine.execute_action("update_time", ("player",))
    variables = dict(session.player.game_variables.items())
    first = (variables.get("day_of_year"), variables.get("hour"))
    assert first != (None, None)

    seed_all(1234)
    pin_clock(EPOCH)
    _client2, session2 = build_client(seed=1234)
    session2.client.event_engine.execute_action("update_time", ("player",))
    variables2 = dict(session2.player.game_variables.items())
    assert (variables2.get("day_of_year"), variables2.get("hour")) == first


def test_a_different_epoch_changes_the_day() -> None:
    """Discriminating companion to the stability test above, per the
    process rule: a route that never varies with the epoch cannot be
    trusted to prove the epoch is what set the value."""
    seed_all(1234)
    pin_clock(EPOCH)
    _c, session = build_client(seed=1234)
    session.client.event_engine.execute_action("update_time", ("player",))
    day_a = dict(session.player.game_variables.items()).get("day_of_year")

    seed_all(1234)
    pin_clock(EPOCH + 86400 * 30)
    _c2, session2 = build_client(seed=1234)
    session2.client.event_engine.execute_action("update_time", ("player",))
    day_b = dict(session2.player.game_variables.items()).get("day_of_year")
    assert day_a is not None and day_b is not None
    assert day_a != day_b


def test_battle_timestamps_are_pinned() -> None:
    """Pins `battle.py`'s `Battle.__init__` hunk directly."""
    from tuxemon.battle import Battle

    pin_clock(EPOCH)
    assert Battle().timestamp == Battle().timestamp == 1787694000.0


def test_battle_timestamp_is_pinned_through_record_battle() -> None:
    """Pins `entity/battle.py`'s `BattlesHandler.record_battle` hunk -- the
    load-bearing call site the brief did not list. `Battle.__init__`
    stamps its own `timestamp` via `now()`, but `record_battle` builds a
    *second*, independent timestamp and hands it to
    `Battle.from_save_data()`, which overwrites the first via `setattr`
    (`"timestamp"` is in `SIMPLE_PERSISTANCE_ATTRIBUTES`). Patching only
    `battle.py` leaves this path -- the one `combat/utils.py` actually
    calls when a battle completes -- still wall-clocked. Without this
    hunk, `test_battle_digest_is_deterministic_through_its_exit` in
    `tests/test_combat_determinism.py` would still diverge on
    `npc_state.battles[].timestamp`."""
    pin_clock(EPOCH)
    _client, session = build_client(seed=1234)
    battle = session.player.battle_handler.record_battle(
        opponent="rival", outcome=OutputBattle.WON
    )
    assert battle.timestamp == 1787694000.0


def test_capture_date_is_pinned_to_the_epoch() -> None:
    """Pins `time_handler.py`'s `today_month_day()` hunk. `Monster.
    spawn_base()` (reached by the `add_monster` event action, exercised
    by `tests/test_combat_determinism.py`) stamps `capture_date` from it,
    and `capture_date` is part of `_persist_simple`, i.e. reachable in the
    digested `npc_state.monsters[]` tree. An unpinned `date.today()` here
    would not show up in same-day test runs (the divergence this patch
    closes is invisible unless the two runs cross a real midnight), which
    is exactly why it was not among the four divergences the exploratory
    measurement in `docs/2026-08-25-battle-exit-measurement.org` found --
    it is still a real leak."""
    expected = _expected_datetime(float(EPOCH))

    pin_clock(EPOCH)
    _client, session = build_client(seed=1234)
    session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    capture_date_a = session.player.monsters[-1].capture_date
    assert capture_date_a == (expected.month, expected.day)

    # Same-seed stability companion.
    pin_clock(EPOCH)
    _client2, session2 = build_client(seed=1234)
    session2.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    capture_date_b = session2.player.monsters[-1].capture_date
    assert capture_date_b == capture_date_a


def test_a_different_epoch_changes_the_capture_date() -> None:
    """Discriminating companion to the stability test above."""
    pin_clock(EPOCH)
    _client, session = build_client(seed=1234)
    session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    date_a = session.player.monsters[-1].capture_date

    pin_clock(EPOCH + 86400 * 30)
    _client2, session2 = build_client(seed=1234)
    session2.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    date_b = session2.player.monsters[-1].capture_date
    assert date_a != date_b


def test_session_start_time_is_pinned() -> None:
    """Pins the `AbstractSession.__init__` hunk in `session.py`.
    `local_session` is a process-lifetime singleton whose `__init__` runs
    exactly once, so a fresh `Session()` -- not `build_client`, whose
    `reset()` never redraws `_start_time`/`_start_timestamp` -- is the
    only way to observe this hunk directly."""
    from tuxemon.session import Session

    pin_clock(EPOCH)
    session = Session()
    assert session._start_time == _expected_datetime(float(EPOCH))
    assert session._start_timestamp == float(EPOCH)


def test_session_reset_time_is_pinned() -> None:
    """Pins the `reset_time()` hunk in `session.py`."""
    from tuxemon.session import Session

    pin_clock(EPOCH)
    session = Session()

    pin_clock(EPOCH + 3600)
    session.reset_time()
    assert session._start_time == _expected_datetime(float(EPOCH + 3600))
    assert session._start_timestamp == float(EPOCH + 3600)


def test_session_get_state_duration_is_pinned() -> None:
    """Pins the `get_state()` hunk in `session.py`: `current_duration` and
    the timestamp it re-stamps `_start_timestamp` with must both come from
    the pinned clock, not `time.time()`."""
    from tuxemon.session import Session

    pin_clock(EPOCH)
    session = Session()

    pin_clock(EPOCH + 10)
    state = session.get_state()
    assert state.duration == 10.0
    assert state.start_time == _expected_datetime(float(EPOCH)).strftime(TIME_FORMAT)

    # `_start_timestamp` is re-stamped to the same pinned "now" too.
    pin_clock(EPOCH + 10)
    state2 = session.get_state()
    assert state2.duration == 0.0


def test_save_data_time_is_pinned() -> None:
    """Pins the `save_system/save.py` hunk: `SaveData.time` is written by
    `snapshot_save`/`get_save_data`, not by anything `digest_of` reaches,
    but it is still part of what a recorded save/trace round-trips."""
    pin_clock(EPOCH)
    _client, session = build_client(seed=1234)
    saved = snapshot_save(session)
    assert saved.time == _expected_datetime(float(EPOCH)).strftime(TIME_FORMAT)


def test_pinned_epoch_is_independent_of_host_timezone() -> None:
    """Fix round 1: the pinned epoch alone must determine the resulting
    wall-clock fields, independent of the host machine's `TZ` -- a trace
    recorded on one machine must replay byte-identically on another.

    A first cut of `now_datetime()` used `datetime.fromtimestamp(now())`
    (local time), exactly as the task-9 brief's own code sketch
    specified. Measured with `EPOCH` fixed and only `TZ` varying:
    `TZ=UTC` gave `(hour, day_of_year) = (21, 237)`,
    `TZ=Asia/Tokyo` gave `(6, 238)` -- a different calendar day for the
    identical pinned epoch, and `Monster.capture_date` (which IS in the
    digested tree) shifted the same way. `now_datetime()` now interprets
    the epoch in UTC before stripping tzinfo (see its docstring), so this
    test drives the two most different zones available (UTC and
    Asia/Tokyo, +9h with no overlapping daylight-saving complications)
    and requires every wall-clock-derived, digest-reachable field to
    match. `os.environ["TZ"]`/`time.tzset()` is Unix-only; fine here,
    since this suite already assumes a Unix-like host (`SDL_VIDEODRIVER
    =dummy`, headless pygame, ...). The original `TZ` is restored in
    `finally` so this test cannot contaminate any test that runs after
    it in the same process.
    """
    original_tz = os.environ.get("TZ")

    def _measure(tz: str) -> tuple[object, object, object]:
        os.environ["TZ"] = tz
        _time.tzset()
        seed_all(1234)
        pin_clock(EPOCH)
        _client, session = build_client(seed=1234)
        session.client.event_engine.execute_action("update_time", ("player",))
        session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
        variables = dict(session.player.game_variables.items())
        return (
            variables.get("day_of_year"),
            variables.get("hour"),
            session.player.monsters[-1].capture_date,
        )

    try:
        utc = _measure("UTC")
        tokyo = _measure("Asia/Tokyo")
        assert utc != (None, None, None)
        assert utc == tokyo
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        _time.tzset()
