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
