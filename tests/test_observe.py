"""The renderer install is the whole difference between a blank frame and
a usable one: probed, `NullRenderer` draws 1 distinct colour where a real
`MapRenderer` draws 48 on the same state. A test that merely asserts a
frame is "not empty" would pass against a black rectangle, which is the
vacuous shape this file exists to avoid."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from tuxemon.platform.const import buttons
from tuxemon.save_system.save_state import SaveData

from tuxghost.boot import boot_from_save
from tuxghost.determinism import pin_clock
from tuxghost.digest import digest_of
from tuxghost.loop import InputSchedule, install_schedule, run_steps
from tuxghost.observe import BACKGROUND, FrameRenderer, distinct_colors

FIXTURE = Path(__file__).parent / "fixtures" / "paper_town.save"
EPOCH = 1787659200  # daytime, matches tests/test_digest.py's DIGEST_EPOCH


def _boot_fixture() -> tuple[Any, Any]:
    import json

    pin_clock(EPOCH)
    save = SaveData.model_validate(json.loads(FIXTURE.read_text()))
    return boot_from_save(save, seed=1234, clock_epoch=EPOCH)


def test_null_renderer_draws_nothing_and_map_renderer_draws_the_world() -> None:
    import pygame as pg
    from tuxemon.state.draw import StateDrawer

    client, _session = _boot_fixture()
    schedule: InputSchedule = {}
    install_schedule(client, schedule)
    run_steps(client, 10)

    # Baseline: the stock headless renderer, drawn the same way.
    bare = pg.Surface(client.context.resolution)
    bare.fill(BACKGROUND)
    StateDrawer(bare, client.state_manager, client.config).draw()
    assert len(distinct_colors(bare)) == 1, "NullRenderer should draw nothing"

    # FrameRenderer installs a real MapRenderer through the public seam.
    frames = FrameRenderer(client)
    assert len(distinct_colors(frames.surface())) > 1


def test_frame_png_is_cropped_upscaled_and_stable() -> None:
    """Geometry, not a byte-count proxy: a `len(png) < N` check measured to
    discriminate NOTHING (review round 1) -- black padding compresses away
    to almost nothing, so a PNG with the crop skipped entirely (full
    5120x2880 frame) and a PNG with the upscale skipped entirely (a bare
    640x320 crop) both still satisfied the old `< 400_000` bound. Decoding
    the actual PNG dimensions is the only check that can tell "cropped and
    upscaled" apart from "wasn't."

    UNSCOPED FINDING, found while fixing the above: this file's stability
    claim ("two encodes of an unchanged frame are byte-identical") is only
    true at a FIXED wall-clock instant. `pyscroll`'s tile data adapter
    (`.venv/.../pyscroll/data.py:132`, `_update_time`) reads real
    `time.time()` on every `draw()` (via `process_animation_queue`,
    `orthographic.py:414`) to pick an animated tile's current frame --
    `tuxghost.determinism.pin_clock` does NOT cover it (it only pins
    `tuxemon.core.clock`, patch 0004's call sites). Measured: calling
    `frames.png()` twice back-to-back with no `time.time` pin flaked
    (~1 run in 3) on a byte mismatch, because the two calls straddled an
    animated tile's frame boundary. This is a real wall-clock leak into
    RENDERED PIXELS -- it does not reach `state_of()`'s digested game
    state, so it cannot break replay determinism, but it does mean a
    recorded PNG is not a pure function of (schedule, step) the way this
    project's core guarantee is for game state. Pinning `time.time` here
    is a targeted fix for this test's own flakiness, not a fix for the
    underlying leak, which is out of this task's scope -- see the task
    report.
    """
    from unittest.mock import patch

    import pygame as pg

    client, _session = _boot_fixture()
    install_schedule(client, {})
    run_steps(client, 10)
    frames = FrameRenderer(client, upscale=4)

    with patch("time.time", return_value=12345.0):
        png = frames.png()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert not frames.last_frame_was_blank
        # Two encodes of an unchanged frame, AT THE SAME INSTANT, are
        # byte-identical. (Without the pin: flaky, see finding above.)
        assert frames.png() == png

    width4, height4 = pg.image.load(BytesIO(png)).get_size()
    # Cropped: strictly smaller than the full headless resolution upscaled.
    assert (width4, height4) < (1280 * 4, 720 * 4)

    # A second renderer on the SAME client, upscale=1, draws the identical
    # frame uncropped-relative-to-itself -- its dimensions are exactly a
    # quarter of the 4x renderer's, which is only true if both actually
    # cropped to the same region and the first actually upscaled by 4.
    bare = FrameRenderer(client, upscale=1)
    width1, height1 = pg.image.load(BytesIO(bare.png())).get_size()
    assert (width4, height4) == (width1 * 4, height1 * 4)


def test_a_frame_with_nothing_drawn_flags_last_frame_was_blank() -> None:
    """`last_frame_was_blank` is documented to distinguish "genuinely
    nothing to see" (legitimate between map loads) from ordinary content,
    but no other test in this file ever exercises that branch -- every
    other assertion is `not last_frame_was_blank`. Forcing the drawer to
    draw nothing is the only way to reach it without an actual empty
    active-state stack, which nothing in this fixture produces."""
    import pygame as pg

    client, _session = _boot_fixture()
    install_schedule(client, {})
    run_steps(client, 10)
    frames = FrameRenderer(client)
    # White-box: force "nothing drawn" without needing an actual
    # empty active-state stack, which no route in this fixture produces.
    frames._drawer.draw = lambda: None

    png = frames.png()
    assert frames.last_frame_was_blank
    # A blank frame is encoded WHOLE, not cropped to nothing.
    assert pg.image.load(BytesIO(png)).get_size() == client.context.resolution


def test_frame_renderer_rejects_upscale_below_one() -> None:
    client, _session = _boot_fixture()
    with pytest.raises(ValueError, match="upscale"):
        FrameRenderer(client, upscale=0)


def test_frame_renderer_refuses_a_collision_map_enabled_config() -> None:
    """`MapRenderer.draw()` gates its `DebugRenderer` on exactly
    `config.collision_map` (`tuxemon/map/view.py:522`) -- wiring one into
    every `FrameRenderer` is inert only by accident of the default config
    being False. If it were ever True, every frame this class produces
    would silently gain collision boxes and a red centre line: a frame
    that quietly differs from what a human player sees. Refusing
    construction is cheaper than a policy trained on a debug overlay."""
    client, _session = _boot_fixture()
    client.config.config_model.display.collision_map = True
    with pytest.raises(ValueError, match="collision_map"):
        FrameRenderer(client)


def test_rendering_does_not_change_the_digest() -> None:
    """The structural hazard: recording draws, replay does not. If drawing
    perturbs digested state, every agent-recorded trace would fail to
    replay.

    Uses a schedule with a PENDING input due a few steps AFTER the digest
    is captured, rather than an empty schedule: measured, an idle client
    with no press due -- `client.update` called on nothing but empty
    frames -- never perturbs this digest at all, because `state_of()`
    only covers `npc_state`/`world_state`/`persistent_npc_state`, and
    idle ticks touch none of them (`world_state.get_state` returns only
    `factions_manager` and `menu_flags`; `npc_state` is the player's own
    state, unmoved with no input). An empty-schedule version of this test
    would pass even against a `FrameRenderer.surface()` that silently
    called `client.update` on every draw, for the same reason six earlier
    tests in this project passed while proving nothing: the digest simply
    isn't looking at anything an idle tick touches. A pending `DOWN`
    press due at step 12 (well after the digest snapshot at step 10)
    turns "does drawing silently drain/advance the input queue and move
    the player" into a question this test can actually catch -- confirmed
    by mutation, see the task report.

    The trailing `run_steps(client, 30)` + inequality assertion is a
    POSITIVE CONTROL (review round 1, Ruling K): the equality assertion
    above is only meaningful if the pending press is real and reachable,
    which depends on the step cursor living inside `install_schedule`'s
    closure (`tuxghost/loop.py:44,57`) rather than, say, `run_steps`. If
    that cursor's home ever moved, the equality assertion could revert to
    silently vacuous exactly like the six historic examples; this makes
    the test prove its own instrument is live on every run.
    """
    client, session = _boot_fixture()
    schedule: InputSchedule = {
        12: [(buttons.DOWN, 1.0)],
        20: [(buttons.DOWN, 0.0)],
    }
    install_schedule(client, schedule)
    run_steps(client, 10)

    before = digest_of(session)
    frames = FrameRenderer(client)
    for _ in range(60):
        frames.png()
    assert digest_of(session) == before

    # Positive control: stepping for real past the still-pending press
    # DOES move the digest -- the instrument above is live, not inert.
    run_steps(client, 30)
    assert digest_of(session) != before


def test_recording_renders_matches_a_render_free_replay() -> None:
    """The axis that actually differs between recording and replay
    (review round 1, Important #2): constructing `FrameRenderer` swaps
    `NullRenderer` -- whose `update` is `pass`
    (`tuxemon/map/view.py:476-477`) -- for a `MapRenderer` whose
    `update(dt)` runs `camera_manager.update(dt)` and
    `map_animations.update_all(dt)` (`tuxemon/map/view.py:525-528`), and
    `WorldState.update` calls it on EVERY step
    (`states/world_state.py:152`). A recorded run therefore executes
    strictly more per-step engine code than a render-free replay of the
    identical schedule. `test_rendering_does_not_change_the_digest` above
    only checks a snapshot taken before/after idle drawing; it never
    drives rendering through the stepping loop itself, so it cannot see
    this. This test drives `png()` through `run_steps`'s `hook` on every
    step of a real multi-step walk and requires the same final digest a
    render-free run of the identical schedule reaches.

    INFERENCE-TO-KNOWLEDGE, NOT a demonstrated-mutation regression test
    (same caveat as `test_installed_renderer_survives_a_map_change`
    below; if this ever fails, that is a real finding, report it and
    stop rather than patching the test). Measured, THREE separate times,
    that it does NOT catch the review's suggested mutations on this
    fixture/route:

      1. `self._client.update(1.0 / 60)` added to `FrameRenderer.surface()`
         (the exact mutation that DOES fail
         `test_rendering_does_not_change_the_digest`, see the task
         report) -- final digest UNCHANGED on this 150-step DOWN+RIGHT
         walk, and also unchanged on an open-ended (never-released) hold.
      2. A bare `random.random()` call added to `surface()` -- also
         unchanged.

    Mechanism, read from `tuxghost/loop.py`: `install_schedule`'s step
    counter (`state["step"]`) increments once per call to
    `client.input_manager.process_events`, which fires from EVERY
    `client.update()` call regardless of who calls it -- monotonically,
    by exactly 1, with no gaps. An extra render-induced `update()` call
    therefore never skips or double-fires a scheduled input; it only
    reaches each scheduled counter value sooner, then idles past it,
    and idling is invisible to this digest (see
    `test_rendering_does_not_change_the_digest`'s own docstring). Tuxemon's
    grid movement is triggered by discrete counter-keyed events, not
    accumulated continuously from `dt` (measured: holding `DOWN` open-ended
    for 600 steps reaches the exact same tile as 75), so doubling the
    total `update()` call count does not move the player any further,
    either. Nothing on this route draws from `random` unconditionally per
    tick (no `WorldWeatherManager` is wired into `WorldState` for this
    map). Net effect: for schedule-driven grid movement replayed through
    `tuxghost.loop`, "recording renders, replay doesn't" is SAFE by
    construction, given the counter design above -- not because nothing
    could go wrong, but because this specific mechanism can't produce a
    divergence here. A live per-step recorder (Task 5) that keys its own
    schedule by an INDEPENDENT loop counter while ALSO rendering each
    step is a different, not-yet-built hazard this test cannot speak to.
    """
    schedule: InputSchedule = {
        5: [(buttons.DOWN, 1.0)],
        65: [(buttons.DOWN, 0.0)],
        75: [(buttons.RIGHT, 1.0)],
        135: [(buttons.RIGHT, 0.0)],
    }

    client_replay, session_replay = _boot_fixture()
    start = tuple(session_replay.player.tile_pos)
    install_schedule(client_replay, dict(schedule))
    run_steps(client_replay, 150)
    replay_digest = digest_of(session_replay)

    client_recorded, session_recorded = _boot_fixture()
    install_schedule(client_recorded, dict(schedule))
    frames = FrameRenderer(client_recorded)

    def _render_every_step(_i: int) -> None:
        frames.png()

    run_steps(client_recorded, 150, hook=_render_every_step)
    recorded_digest = digest_of(session_recorded)

    # The schedule must have actually moved the player in BOTH runs, or
    # this proves nothing about the recording/replay axis at all.
    assert tuple(session_replay.player.tile_pos) != start
    assert tuple(session_recorded.player.tile_pos) != start
    assert recorded_digest == replay_digest


def test_installed_renderer_survives_a_map_change() -> None:
    """Unmeasured before this test: no probe crossed a map boundary with a
    real renderer installed. `reset_renderer()` is a no-op on `BaseClient`
    and `MapRenderer` holds managers rather than a map, so it SHOULD
    survive -- this is what turns that inference into knowledge."""
    client, _session = _boot_fixture()
    schedule: InputSchedule = {}
    install_schedule(client, schedule)
    run_steps(client, 10)
    frames = FrameRenderer(client)
    before_map = client.get_map_name()

    client.event_engine.execute_action(
        "teleport", ("npc_red", "spyder_paper_scoop", 5, 5)
    )
    run_steps(client, 90)
    assert client.get_map_name() != before_map
    assert len(distinct_colors(frames.surface())) > 1
