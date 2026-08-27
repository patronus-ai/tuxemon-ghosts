"""Task 9: does `DisplayContext.scale` ever reach `digest_of(session)`?

MEASURED, not assumed -- see
`docs/2026-08-26-display-scale-measurement.org` for the full numbers.
A 300-step PER-STEP digest sequence (`tuxghost.digest.digest_of`, not
just a final digest -- task 2 measured that a final digest is blind to a
transient divergence, because Tuxemon's grid movement makes a settled
tile an ATTRACTOR that erases the transient by the time a final snapshot
is taken) is IDENTICAL AT EVERY INDEX between a client booted with the
unscaled `tuxghost.boot.headless_context()` (`scale=1`, `tile_size=(16,
16)`, ~80x45 visible tiles) and one booted with
`tuxghost.observe.scaled_context()` (measured `scale=5` on this repo's
`~/.tuxemon/tuxemon.yaml`, `tile_size=(80, 80)`, ~16x9 visible tiles --
NOT the scale 4 / ~20x11 tiles the task's own brief assumed before this
was measured). `DisplayContext` only ever feeds rendering geometry;
nothing `tuxghost.digest.state_of` reads is derived from it.

This is the DIGEST-NEUTRAL outcome of the three the brief described (the
other two being "digest-affecting" and "converging" -- sequences differ
mid-window but re-converge by the final step, which this measurement did
NOT find: the two sequences never differ at all, not even transiently).
`tuxghost.agent.run_agent` now boots with `scaled_context()` for human
field-of-view; `tuxghost.execute.execute` keeps the unscaled default --
safe by this measurement, not by construction.

A digest-equality test taken alone is a real vacuousness risk of exactly
the shape this project has been burned by before: if `boot_from_save`
silently ignored its `context` parameter, BOTH "runs" below would boot
identically and the sequences would trivially match, proving nothing
about scale at all. `test_scaled_context_actually_renders_a_different_
geometry` is the POSITIVE CONTROL that catches that -- demonstrated by
mutation in the task report, not merely asserted here.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pygame as pg
from tuxemon.platform.const import buttons
from tuxemon.platform.const.sizes import NATIVE_RESOLUTION
from tuxemon.platform.const.sizes import TILE_SIZE as NATIVE_TILE_SIZE
from tuxemon.save_system.save_state import SaveData
from tuxemon.scaling import make_default_scaling
from tuxemon.user_config import CONFIG

from tuxghost.boot import boot_from_save, headless_context
from tuxghost.determinism import pin_clock
from tuxghost.digest import digest_of
from tuxghost.loop import InputSchedule, install_schedule, run_steps
from tuxghost.observe import FrameRenderer, scaled_context

FIXTURE = Path(__file__).parent / "fixtures" / "paper_town.save"
EPOCH = 1787659200  # daytime, matches tests/test_digest.py's DIGEST_EPOCH
SEED = 1234
STEPS = 300

# Keeps the player moving through the end of the window. Only DOWN, UP,
# RIGHT move the player from this fixture's spawn tile (12, 12) -- LEFT
# is blocked by map collision (measured in
# tests/test_agent_fixture.py::test_player_can_walk_from_the_fixture_spawn).
# The final press (step 264) is never released within the window, so a
# move is still resolving at step 299 rather than settled into an
# attractor tile well before the end -- see test_observe.py's Ruling L
# for why a settled tail would hide a transient divergence.
SCHEDULE: InputSchedule = {
    0: [(buttons.DOWN, 1.0)],
    20: [(buttons.DOWN, 0.0)],
    24: [(buttons.UP, 1.0)],
    44: [(buttons.UP, 0.0)],
    48: [(buttons.RIGHT, 1.0)],
    68: [(buttons.RIGHT, 0.0)],
    72: [(buttons.DOWN, 1.0)],
    92: [(buttons.DOWN, 0.0)],
    96: [(buttons.UP, 1.0)],
    116: [(buttons.UP, 0.0)],
    120: [(buttons.RIGHT, 1.0)],
    140: [(buttons.RIGHT, 0.0)],
    144: [(buttons.DOWN, 1.0)],
    164: [(buttons.DOWN, 0.0)],
    168: [(buttons.UP, 1.0)],
    188: [(buttons.UP, 0.0)],
    192: [(buttons.RIGHT, 1.0)],
    212: [(buttons.RIGHT, 0.0)],
    216: [(buttons.DOWN, 1.0)],
    236: [(buttons.DOWN, 0.0)],
    240: [(buttons.UP, 1.0)],
    260: [(buttons.UP, 0.0)],
    264: [(buttons.DOWN, 1.0)],
    # No release scheduled after this -- a move is still in flight when
    # the 300-step window ends.
}


def _boot(context: Any) -> tuple[Any, Any]:
    pin_clock(EPOCH)
    save = SaveData.model_validate(json.loads(FIXTURE.read_text()))
    return boot_from_save(save, seed=SEED, clock_epoch=EPOCH, context=context)


def _digest_sequence(context: Any) -> tuple[list[str], tuple[int, int], tuple[int, int]]:
    client, session = _boot(context)
    start = tuple(session.player.tile_pos)
    install_schedule(client, dict(SCHEDULE))
    digests: list[str] = []
    run_steps(client, STEPS, hook=lambda _i: digests.append(digest_of(session)))
    end = tuple(session.player.tile_pos)
    return digests, start, end


def test_display_scale_is_digest_neutral_across_a_300_step_moving_window() -> None:
    """The measurement this whole task turns on. See this module's
    docstring for the full context; this test pins the DIGEST-NEUTRAL
    result exactly as measured: not "no divergence found at the end" but
    "no divergence found at ANY of the 300 indices."""
    default_digests, default_start, default_end = _digest_sequence(headless_context())
    scaled_digests, scaled_start, scaled_end = _digest_sequence(scaled_context())

    # Positive control: the schedule actually moved the player in BOTH
    # runs. Six earlier tests in this project passed vacuously in six
    # different ways (see CLAUDE.md); an equality assertion below is
    # worthless if the player never moved in the first place.
    assert default_end != default_start, (
        f"schedule never moved the player away from {default_start} in "
        "the unscaled run -- this test would prove nothing"
    )
    assert scaled_end != scaled_start, (
        f"schedule never moved the player away from {scaled_start} in "
        "the scaled run -- this test would prove nothing"
    )
    assert len(default_digests) == len(scaled_digests) == STEPS

    first_diff = next(
        (i for i in range(STEPS) if default_digests[i] != scaled_digests[i]),
        None,
    )
    assert first_diff is None, (
        f"digest sequences first diverge at step {first_diff} of "
        f"{STEPS - 1}: unscaled={default_digests[first_diff]!r} "
        f"scaled={scaled_digests[first_diff]!r} -- scale is "
        "digest-affecting after all; see this task's report/doc note, "
        "this file's docstring is now WRONG and must be rewritten for "
        "the digest-affecting (or converging) outcome instead"
    )


def test_scaled_context_actually_renders_a_different_geometry() -> None:
    """Positive control for the digest-neutral test above (demonstrated
    by mutation in the task report, not merely asserted here): if
    `boot_from_save` silently ignored its `context` parameter, the
    "scaled" run above would actually boot with the unscaled default,
    its digest sequence would trivially match the unscaled run's, and
    the digest-neutral test would pass having proven nothing. This test
    fails loudly in exactly that scenario, because `client.context`
    would then report `scale=1`/`tile_size=(16, 16)` for BOTH clients.

    Geometry is asserted two ways, per the task brief: directly off
    `client.context` (the actual object `scaled_context()` built and
    that `FrameRenderer`'s crop/draw geometry is computed from -- not a
    value this test reimplements independently), and by DECODING the
    rendered PNG rather than trusting its byte count (task 2 measured a
    byte-count threshold discriminates nothing here: removing the crop
    entirely, 118,218 bytes, and removing the upscale entirely, 35,385
    bytes, both sat comfortably under the same 400,000-byte bound).

    The crop-region PNG dimensions are NOT asserted to be an exact
    multiple of each other between the two contexts (unlike task 2's
    same-client upscale=4-vs-upscale=1 comparison, which IS exact): more
    of the 1280x720 canvas is genuinely non-background at the larger
    tile size (fewer, bigger tiles fit on screen, so less of the frame
    is background to crop away), so the crop sizes scale by a smaller,
    content-dependent factor -- measured 640x320 (unscaled) versus
    1216x656 (scaled), a ~1.9x/2.05x change, not 5x. Decoding still
    matters here: it proves each PNG is a real, differently-sized,
    non-blank image, which a byte-count check could not.
    """
    expected_scale = make_default_scaling(CONFIG, NATIVE_RESOLUTION)._scale
    # Guard against vacuity if ~/.tuxemon/tuxemon.yaml ever disables
    # scaling (display.scaling: false) or sets a resolution matching
    # NATIVE_RESOLUTION: scale 1 would make scaled_context() identical
    # to the unscaled default and every assertion below pass trivially.
    assert expected_scale > 1, (
        f"make_default_scaling(CONFIG, NATIVE_RESOLUTION) computed "
        f"scale={expected_scale!r} -- this test needs the two contexts "
        "to genuinely differ; check ~/.tuxemon/tuxemon.yaml's "
        "display.scaling/resolution"
    )

    default_client, _s1 = _boot(headless_context())
    install_schedule(default_client, {})
    run_steps(default_client, 10)

    scaled_client, _s2 = _boot(scaled_context())
    install_schedule(scaled_client, {})
    run_steps(scaled_client, 10)

    default_ctx = default_client.context
    scaled_ctx = scaled_client.context

    assert default_ctx.scale == 1
    assert default_ctx.tile_size == NATIVE_TILE_SIZE
    assert scaled_ctx.scale == expected_scale
    assert scaled_ctx.tile_size == (
        NATIVE_TILE_SIZE[0] * expected_scale,
        NATIVE_TILE_SIZE[1] * expected_scale,
    )
    # Resolution itself is untouched by scale -- only tile density
    # changes (more, smaller tiles vs fewer, bigger ones in the same
    # window).
    assert default_ctx.resolution == scaled_ctx.resolution == CONFIG.resolution

    default_frames = FrameRenderer(default_client, upscale=1)
    default_png = default_frames.png()
    scaled_frames = FrameRenderer(scaled_client, upscale=1)
    scaled_png = scaled_frames.png()

    assert not default_frames.last_frame_was_blank
    assert not scaled_frames.last_frame_was_blank

    default_size = pg.image.load(BytesIO(default_png)).get_size()
    scaled_size = pg.image.load(BytesIO(scaled_png)).get_size()
    assert default_size != scaled_size, (
        f"unscaled and scaled PNGs decoded to the SAME size "
        f"{default_size!r} -- the two contexts are not actually "
        "rendering differently"
    )
    # Both must be real, positive-area images, well within the
    # (scale-independent) full-canvas bound.
    assert 0 < default_size[0] <= CONFIG.resolution[0]
    assert 0 < default_size[1] <= CONFIG.resolution[1]
    assert 0 < scaled_size[0] <= CONFIG.resolution[0]
    assert 0 < scaled_size[1] <= CONFIG.resolution[1]
