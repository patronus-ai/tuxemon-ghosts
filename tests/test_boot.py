import pytest


def test_headless_context_can_convert_surfaces() -> None:
    """A converted surface is what sprite loading needs; without set_mode
    pygame raises 'No convert format has been set'."""
    import pygame as pg

    from tuxghost.boot import headless_context

    ctx = headless_context()
    assert ctx.resolution[0] > 0
    surface = pg.Surface((8, 8), pg.SRCALPHA)
    converted = surface.convert_alpha()
    assert converted.get_size() == (8, 8)


def test_headless_context_leaves_pygame_globally_initialized() -> None:
    """pygame_menu asserts pygame.get_init() before pushing a menu state; if
    headless_init() only calls pg.display.init()/pg.font.init()/set_mode()
    without pg.init(), pygame.get_init() stays False and that assertion
    fails. Empirically, none of display.init(), font.init(), or set_mode()
    flip pygame.get_init() to True -- only pg.init() does."""
    import pygame as pg

    from tuxghost.boot import headless_context

    headless_context()
    assert pg.get_init() is True


def test_boot_from_save_restores_position_and_party() -> None:
    """`session2 is session` (the module-level `local_session` singleton),
    so comparing `session2.player.tile_pos == session.player.tile_pos`
    directly is `x == x` and would pass even if restoration did nothing.
    The mod's starting position is also (0, 0), same as a freshly created
    NPC's default, so even a pre-captured comparison would pass vacuously
    at the default position. Move the player to a distinct, non-default
    tile and capture that position as a plain tuple before restoring, so
    the assertion can only pass if `load_state` actually restored it."""
    from tuxghost.boot import boot_from_save, build_client, snapshot_save

    _client, session = build_client(seed=1234)
    session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    session.player.set_position((5, 7))
    expected_tile_pos = tuple(session.player.tile_pos)
    assert expected_tile_pos == (5, 7)
    saved = snapshot_save(session)

    _client2, session2 = boot_from_save(saved, seed=1234)
    assert [m.slug for m in session2.player.monsters] == ["rockitten"]
    assert tuple(session2.player.tile_pos) == expected_tile_pos


def test_boot_from_save_ids_are_reproducible_regardless_of_prior_build_seed() -> None:
    """Pins `boot_from_save`'s own `seed_ids(seed)` call (`tuxghost/boot.py`).
    Without it, `boot_from_save`'s `seed` argument is not authoritative for
    entity ids: `local_session.reset()` (called inside `boot_from_save`)
    draws from whatever `_rng` state a *prior, unrelated* `build_client`
    call in the same process happened to leave behind, so restoring the
    same save with the same `restore_seed` gives different ids depending on
    what seed some earlier, unrelated build used -- a disagreement between
    `boot_from_save`'s own explicit `seed` parameter and the ids it
    actually produces. (Restored monsters keep their saved
    `instance_id`s -- `load_state` round-trips them -- so the divergence is
    not visible there; it shows up in the freshly-created player NPC and
    in `session._uuid`, both drawn fresh by `boot_from_save` itself.)

    Sequence: build with seed 1234, snapshot, restore with seed 99 --
    then build again with an unrelated seed 5555 (to perturb `_rng`), and
    restore the *same* save with the *same* restore seed 99 again. Every
    comparable field must match between the two restores, since only the
    restore seed (99, unchanged) should be authoritative."""
    from tuxghost.boot import boot_from_save, build_client, snapshot_save
    from tuxghost.digest import digest_of

    _client, session = build_client(seed=1234)
    saved = snapshot_save(session)

    _client_r1, restored_1 = boot_from_save(saved, seed=99)
    digest_1 = digest_of(restored_1)
    uuid_1 = str(restored_1._uuid)
    player_iid_1 = str(restored_1.player.instance_id)

    _client_unrelated, _session_unrelated = build_client(seed=5555)

    _client_r2, restored_2 = boot_from_save(saved, seed=99)
    digest_2 = digest_of(restored_2)
    uuid_2 = str(restored_2._uuid)
    player_iid_2 = str(restored_2.player.instance_id)

    assert digest_1 == digest_2
    assert uuid_1 == uuid_2
    assert player_iid_1 == player_iid_2


def test_boot_from_save_session_time_is_reproducible_with_clock_epoch() -> None:
    """The offline executor (a later task) calls `boot_from_save` directly
    and never constructs a `tuxghost.record.Recorder` -- so it must get
    session-time reproducibility from `boot_from_save` itself, not from the
    recorder. Without `clock_epoch`, `AbstractSession._start_timestamp`
    (`tuxemon/session.py`) is set once, at process start, from whatever
    wall clock was in effect when the module-level `local_session`
    singleton was first constructed -- almost always before any caller has
    pinned the clock -- and `get_state()` (what `snapshot_save` drives)
    mutates it as a side effect on every call. That leaks real, unpinned,
    call-order-dependent time into `SessionSave.duration`/`total_playtime`/
    `start_time`, invisible to `tuxghost.digest.state_of`/`digest_of`
    (which never read `session_state` at all) but directly reachable from
    a full `SaveData` snapshot -- exactly what `boot_from_save` produces
    when a caller `snapshot_save`s the restored session.

    `saved` is built WITH the clock pinned (`build_client(..., clock_epoch
    =epoch)`) so its own `session_state` is the deterministic ground truth
    a properly-recorded trace's `initial_state` would show: `duration ==
    0.0`, `total_playtime == 0.0`, `start_time` == the epoch's own UTC
    string. Asserting those exact values -- not merely that the two
    restores agree with each other -- matters because a same-vs-same
    comparison can pass vacuously once `_start_timestamp` has already
    settled to the pinned epoch from an earlier call in the same process.

    That vacuous-pass risk is not hypothetical: an earlier version of this
    test built `saved` via `build_client(..., clock_epoch=epoch)` and then
    called `boot_from_save` with NO further isolation. `build_client`'s own
    `pin_clock`+`reset_time()` (run to produce `saved`) already settles the
    shared `local_session` singleton's `_start_timestamp` to `epoch`, and
    `local_session.reset()` (which `boot_from_save` DOES call) never clears
    it -- only `reset_time()` does. `get_state()` then self-corrects to a
    fixed point once the clock is pinned, so `boot_from_save`'s OWN
    clock-pinning rode for free on `build_client`'s earlier call: deleting
    `pin_clock`/`reset_time()` from `boot_from_save` entirely did not fail
    this test. Fixed by deliberately DIRTYING the singleton's clock
    bookkeeping between producing `saved` and calling `boot_from_save`:
    unpin the clock and force one more `get_state()` call (via
    `snapshot_save`) while unpinned, settling `_start_timestamp` back to
    real wall-clock time. `boot_from_save` must then redo the pin-and-reset
    itself to reach the expected epoch-derived values -- it can no longer
    inherit them for free. An unrelated build still runs between the two
    restores, to rule out THAT accounting for the match instead."""
    import json
    from datetime import UTC, datetime

    from tuxemon.core.clock import set_epoch

    from tuxghost.boot import boot_from_save, build_client, snapshot_save

    epoch = 1787694000
    expected_start_time = (
        datetime.fromtimestamp(epoch, tz=UTC)
        .replace(tzinfo=None)
        .strftime("%Y-%m-%d %H:%M")
    )

    _client, session = build_client(seed=1234, clock_epoch=epoch)
    saved = snapshot_save(session)
    source_session_state = json.loads(saved.model_dump_json())["session_state"]
    assert source_session_state["duration"] == 0.0
    assert source_session_state["total_playtime"] == 0.0
    assert source_session_state["start_time"] == expected_start_time

    # Dirty the singleton's clock bookkeeping: unpin the clock, then force
    # another `get_state()` call while unpinned so `_start_timestamp`/
    # `_start_time` settle back to real wall-clock time. Without this,
    # `boot_from_save` inherits `build_client`'s earlier pin for free (see
    # this test's own docstring), and the assertions below would pass even
    # with `boot_from_save`'s own pin/reset entirely deleted.
    set_epoch(None)
    snapshot_save(session)

    def restore_and_snapshot() -> dict[str, object]:
        _client_r, restored = boot_from_save(saved, seed=99, clock_epoch=epoch)
        state: dict[str, object] = json.loads(
            snapshot_save(restored).model_dump_json()
        )["session_state"]
        return state

    session_state_1 = restore_and_snapshot()
    _client_unrelated, _session_unrelated = build_client(seed=5555)
    session_state_2 = restore_and_snapshot()

    assert session_state_1 == session_state_2
    assert session_state_1["duration"] == 0.0
    assert session_state_1["total_playtime"] == 0.0
    assert session_state_1["start_time"] == expected_start_time


def test_build_client_resets_the_singleton_between_builds() -> None:
    """`local_session` is a module-level singleton reused by every
    `build_client` call in a process. Without an internal reset, a second
    build starts from whatever the first build's session already
    accumulated (its player, its added monsters, ...) instead of a
    genuinely fresh session -- silent contamination, not an error. Add a
    monster to the first build's session, build again with the same seed,
    and require the second session's player to start with none: this can
    only pass if `build_client` actually resets `local_session` before
    creating the new player, not merely by both builds coincidentally
    agreeing."""
    from tuxghost.boot import build_client

    _client1, session1 = build_client(seed=1234)
    session1.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    assert [m.slug for m in session1.player.monsters] == ["rockitten"]

    _client2, session2 = build_client(seed=1234)
    assert [m.slug for m in session2.player.monsters] == []


def test_build_client_does_not_leak_the_network_port() -> None:
    """Two headless clients built in the same process must not fight over
    the websocket server's fixed port (40081/0.0.0.0). Upstream starts a
    TuxemonServer as a side effect of BaseClient.__init__; without patch
    0006's `start_networking=False` for HeadlessClient, the second
    client's bind fails in a background thread (an unhandled `OSError`)
    while `TuxemonServer.__init__` unconditionally sets `listening = True`
    regardless of whether the bind succeeded, so `is_host()` reports
    `True` even though nothing is listening -- a process-external race
    (what else is bound to that port) leaking into supposedly
    deterministic client state. (A harness-side fix that starts the
    server and then calls `network_manager.shutdown()` does not work:
    upstream's shutdown never calls `stop_listening()`, so the port stays
    bound; reaching past it to call `stop_listening()` directly does free
    the port but trips a second, independent upstream bug -- it raises an
    unhandled `RuntimeError` in the same background thread. Not starting
    the server at all for headless clients was the only fix that produced
    zero thread exceptions.)"""
    import threading
    import time

    from tuxghost.boot import build_client

    thread_exceptions: list[threading.ExceptHookArgs] = []
    previous_hook = threading.excepthook
    threading.excepthook = thread_exceptions.append
    try:
        client1, _session1 = build_client(seed=1)
        assert client1.network_manager.server is None
        assert client1.network_manager.is_host() is False

        client2, _session2 = build_client(seed=2)
        assert client2.network_manager.server is None
        assert client2.network_manager.is_host() is False

        # If a server thread were spawned, a failed bind would be raised
        # asynchronously here, a moment after construction returns rather
        # than synchronously.
        time.sleep(0.5)
    finally:
        threading.excepthook = previous_hook

    assert thread_exceptions == []


def test_null_renderer_survives_set_bubble() -> None:
    """Fix round 3 (patch 0001): `HeadlessClient` never pushes a real
    `MapRenderer` (there is nothing to draw to), so `session.client
    .map_renderer` stays a `NullRenderer` for the life of any headless
    client -- confirmed below by `isinstance`, not assumed. `set_bubble`
    (a common map-script action) reads and writes `map_renderer
    .bubble_manager` unconditionally; before this fix `NullRenderer` never
    set it, so this raised `AttributeError` the first time any headless
    route touched a speech bubble. `check_world`'s `"bubble"` branch reads
    `bubble_manager.has_bubble(...)` too, so the assertions below exercise
    both call sites, not just the mutator."""
    from tuxemon.event.conditions.check_world import CheckWorldCondition
    from tuxemon.map.view import NullRenderer

    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234)
    assert isinstance(session.client.map_renderer, NullRenderer)

    execute = session.client.event_engine.execute_action
    execute("set_bubble", ("player", "note"), True)
    has_bubble = CheckWorldCondition("bubble", "player").test(session)
    assert has_bubble is True

    execute("set_bubble", ("player",), True)
    has_bubble_after_removal = CheckWorldCondition("bubble", "player").test(
        session
    )
    assert has_bubble_after_removal is False


def test_null_renderer_survives_map_and_tile_animations() -> None:
    """Fix round 3 (patch 0001): `play_map_animation`/`play_tile_animation`
    read `map_renderer.map_animations` unconditionally and call
    `.setup_and_play(...)`, which loads and caches a real animation from
    the mod's asset data (`grass`, used by several maps' own scripts, e.g.
    `mods/tuxemon/maps/eclipse_park_south.yaml`) -- not a stub call.
    Before this fix `NullRenderer` never set `map_animations`, so either
    action raised `AttributeError` the first time a headless route reached
    one."""
    from tuxghost.boot import build_client

    _client, session = build_client(seed=1234)

    execute = session.client.event_engine.execute_action
    execute("play_map_animation", ("grass", 0.1, "noloop", "player"), True)
    execute("play_tile_animation", (0, 0, "grass", 0.1, "noloop"), True)

    assert "grass" in session.client.map_renderer.map_animations._cache



def test_assert_fps_matches_step_rate_accepts_the_harness_default() -> None:
    """`build_client`'s default config (`display.fps: 60.0`) must not be
    rejected -- every other test in this suite already exercises this
    path implicitly (they'd all be failing at `build_client()` otherwise),
    but this pins the guard's accept branch directly against the values
    it actually compares."""
    from types import SimpleNamespace

    from tuxghost.boot import _assert_fps_matches_step_rate

    client = SimpleNamespace(config=SimpleNamespace(fps=60.0))
    _assert_fps_matches_step_rate(client)  # must not raise


def test_assert_fps_matches_step_rate_rejects_a_mismatched_fps() -> None:
    """Fix round 1: `EventAction.run()`'s synchronous first update() call
    takes `dt` from `client.config.fps` -- read from
    `~/.tuxemon/tuxemon.yaml`, outside this repo -- while every later,
    deferred frame comes from `run_steps`' `FIXED_DT`
    (`tuxghost/loop.py`), which never reads `config.fps` at all. Nothing
    enforced the two agreeing; on a machine where `display.fps` was set
    to, say, 30, a deferred action would silently see a different dt on
    its first frame than on every frame after it, and traces recorded on
    that machine would silently encode a value nothing in this repo
    controls. `build_client`/`boot_from_save` must refuse to boot instead
    of letting that happen quietly."""
    from types import SimpleNamespace

    from tuxghost.boot import _assert_fps_matches_step_rate

    client = SimpleNamespace(config=SimpleNamespace(fps=30.0))
    with pytest.raises(ValueError, match="FIXED_DT"):
        _assert_fps_matches_step_rate(client)


def test_assert_fps_matches_step_rate_rejects_non_positive_fps() -> None:
    """`display.fps: 0` must not be left to blow up as a bare
    `ZeroDivisionError` deep inside an unrelated deferred action's error
    handling (`eventaction.py`'s `1.0 / session.client.config.fps`) --
    caught here, at boot, with a message that says why."""
    from types import SimpleNamespace

    from tuxghost.boot import _assert_fps_matches_step_rate

    client = SimpleNamespace(config=SimpleNamespace(fps=0.0))
    with pytest.raises(ValueError, match="positive"):
        _assert_fps_matches_step_rate(client)


def test_resolve_map_asset_accepts_a_name_with_or_without_the_extension() -> None:
    """`spyder_*` maps script teleports as `teleport player,spyder_route1`
    (no extension) while `eclipse_*` maps use `...bank1.tmx`, so a real
    save's `current_map` is legitimately either form."""
    from tuxghost.boot import headless_context, resolve_map_asset

    headless_context()  # fetch_asset needs the mod db loaded
    assert resolve_map_asset("start_tuxemon.tmx") is not None
    assert resolve_map_asset("start_tuxemon") is not None
    assert resolve_map_asset("no_such_map_xyz") is None
