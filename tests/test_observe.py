"""The renderer install is the whole difference between a blank frame and
a usable one: probed, `NullRenderer` draws 1 distinct colour where a real
`MapRenderer` draws 48 on the same state. A test that merely asserts a
frame is "not empty" would pass against a black rectangle, which is the
vacuous shape this file exists to avoid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    client, _session = _boot_fixture()
    install_schedule(client, {})
    run_steps(client, 10)
    frames = FrameRenderer(client, upscale=4)

    png = frames.png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert not frames.last_frame_was_blank
    # Cropped, so far smaller than a full 1280x720 frame's worth of pixels.
    assert len(png) < 400_000
    # Two encodes of an unchanged frame are byte-identical.
    assert frames.png() == png


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
