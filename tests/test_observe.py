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
    identical schedule.

    REVIEW ROUND 2, RULING L: a FINAL-digest comparison (round 1's first
    attempt) cannot see this, and it is not because the render pass is
    inert -- it is because a grid-movement endpoint is an ATTRACTOR. Extra
    `update()` calls make movement complete SOONER, but by the end of a
    fully-drained schedule both runs settle onto the same tile, so the
    transient gets erased before the final snapshot is ever taken. This
    version compares the digest SEQUENCE, captured every step via
    `run_steps`'s `hook`, which catches the recorded run being AHEAD of
    the replay while a move is still in flight, and the schedule below
    schedules a walk that is still resolving a move within the last few
    steps of the window (not settled well before it) specifically so the
    tail of the sequence is not itself an attractor.

    Confirmed both ways on this exact schedule/window: unmutated, the two
    sequences are identical at all 60 indices (see the task report for
    the real command output). With `self._client.update(1.0 / 60)` added
    to `FrameRenderer.surface()` -- the same mutation this file's
    `test_rendering_does_not_change_the_digest` already uses -- the
    sequences diverge at index 1 and stay diverged through index 56 of
    59, re-converging only in the last 3 steps (the final `RIGHT` move
    settles in both by then). Round 1's final-digest-only version would
    have missed all of this.
    """
    schedule: InputSchedule = {
        0: [(buttons.DOWN, 1.0)],
        8: [(buttons.DOWN, 0.0)],
        12: [(buttons.DOWN, 1.0)],
        20: [(buttons.DOWN, 0.0)],
        # Scheduled late enough that this move is still resolving at the
        # end of the window, not settled well before it -- see the
        # docstring above (Ruling L's "no attractor at the tail").
        40: [(buttons.RIGHT, 1.0)],
        48: [(buttons.RIGHT, 0.0)],
    }
    steps = 60

    def _digest_sequence(client: Any, session: Any, *, render: bool) -> list[str]:
        install_schedule(client, dict(schedule))
        digests: list[str] = []
        if render:
            frames = FrameRenderer(client)

            def hook(_i: int) -> None:
                frames.png()
                digests.append(digest_of(session))

        else:

            def hook(_i: int) -> None:
                digests.append(digest_of(session))

        run_steps(client, steps, hook=hook)
        return digests

    client_replay, session_replay = _boot_fixture()
    start = tuple(session_replay.player.tile_pos)
    replay_digests = _digest_sequence(client_replay, session_replay, render=False)

    client_recorded, session_recorded = _boot_fixture()
    recorded_digests = _digest_sequence(client_recorded, session_recorded, render=True)

    # The schedule must have actually moved the player in BOTH runs, or
    # this proves nothing about the recording/replay axis at all.
    assert tuple(session_replay.player.tile_pos) != start
    assert tuple(session_recorded.player.tile_pos) != start
    assert len(replay_digests) == len(recorded_digests) == steps

    first_diff = next(
        (i for i in range(steps) if replay_digests[i] != recorded_digests[i]),
        None,
    )
    assert first_diff is None, (
        f"recorded (rendered) and replay (render-free) digest sequences "
        f"first diverge at step {first_diff} of {steps - 1}: "
        f"replay={replay_digests[first_diff]!r} "
        f"recorded={recorded_digests[first_diff]!r}"
    )


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
