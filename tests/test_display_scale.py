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
was measured). `DisplayContext` only ever feeds rendering geometry that
`tuxghost.digest.state_of` reads through -- one contingent exception is
known (see `tuxghost/boot.py`'s `build_client` docstring): grid
coordinates stay scale-invariant because `tuxemon/map/loader.py` captures
a map's NATIVE tile size before overwriting it with `context.tile_size`,
not because nothing reads `context.tile_size` at all.

This is the DIGEST-NEUTRAL outcome of the three the brief described (the
other two being "digest-affecting" and "converging" -- sequences differ
mid-window but re-converge by the final step, which this measurement did
NOT find: the two sequences never differ at all, not even transiently).
`tuxghost.agent.run_agent(scaled=True)` boots with `scaled_context()` for
human field-of-view; `tuxghost.execute.execute` always boots unscaled --
safe by this measurement, not by construction.

EVERY MEASUREMENT BELOW RUNS EACH CONTEXT IN ITS OWN FRESH SUBPROCESS
(task 9 review round 1, Important #1 and #3). `tuxemon.map.view` and
`tuxemon.graphics` each bind their own copy of `tuxemon.prepare
.DISPLAY_CONTEXT` at MODULE IMPORT TIME -- a `tuxemon.prepare
.DISPLAY_CONTEXT = ...` reassignment inside `scaled_context()` cannot
reach either already-imported copy. Call the unscaled context first and
the scaled one second IN THE SAME PROCESS (this file's own first,
now-corrected draft did exactly that) and the "scaled" run silently
renders at the STALE scale=1 instead -- `scaled_context()` now refuses
that loudly (see its docstring and
`test_scaled_context_refuses_when_it_is_not_the_first_context_built`
below), but refusing is still not the same as measuring the real thing.
Only a genuinely fresh process is both first-and-only, so that is what
every context in this file boots in.

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
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from tuxemon.platform.const import buttons

REPO_ROOT = Path(__file__).resolve().parent.parent
TUXEMON_DIR = REPO_ROOT / "tuxemon"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "paper_town.save"
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
#
# Plain ints, not `buttons.DOWN` directly: this schedule is JSON-encoded
# below to cross into a subprocess, and `buttons` entries are an IntEnum
# whose `repr` is not valid Python literal syntax.
SCHEDULE: dict[int, list[tuple[int, float]]] = {
    0: [(int(buttons.DOWN), 1.0)],
    20: [(int(buttons.DOWN), 0.0)],
    24: [(int(buttons.UP), 1.0)],
    44: [(int(buttons.UP), 0.0)],
    48: [(int(buttons.RIGHT), 1.0)],
    68: [(int(buttons.RIGHT), 0.0)],
    72: [(int(buttons.DOWN), 1.0)],
    92: [(int(buttons.DOWN), 0.0)],
    96: [(int(buttons.UP), 1.0)],
    116: [(int(buttons.UP), 0.0)],
    120: [(int(buttons.RIGHT), 1.0)],
    140: [(int(buttons.RIGHT), 0.0)],
    144: [(int(buttons.DOWN), 1.0)],
    164: [(int(buttons.DOWN), 0.0)],
    168: [(int(buttons.UP), 1.0)],
    188: [(int(buttons.UP), 0.0)],
    192: [(int(buttons.RIGHT), 1.0)],
    212: [(int(buttons.RIGHT), 0.0)],
    216: [(int(buttons.DOWN), 1.0)],
    236: [(int(buttons.DOWN), 0.0)],
    240: [(int(buttons.UP), 1.0)],
    260: [(int(buttons.UP), 0.0)],
    264: [(int(buttons.DOWN), 1.0)],
    # No release scheduled after this -- a move is still in flight when
    # the 300-step window ends.
}

# Renders one frame at `upscale=1` after a short warm-up, in addition to
# the digest sequence -- built into the SAME subprocess run so the
# geometry test does not need a second boot.
_SCRIPT = textwrap.dedent(
    """
    import json
    from io import BytesIO

    import pygame as pg
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save, headless_context
    from tuxghost.determinism import pin_clock
    from tuxghost.digest import digest_of
    from tuxghost.loop import install_schedule, run_steps
    from tuxghost.observe import FrameRenderer, scaled_context

    which = {which!r}
    fixture = {fixture!r}
    epoch = {epoch!r}
    seed = {seed!r}
    steps = {steps!r}
    schedule = {{int(k): [tuple(p) for p in v] for k, v in json.loads({schedule_json!r}).items()}}

    context = scaled_context() if which == "scaled" else headless_context()

    pin_clock(epoch)
    save = SaveData.model_validate(json.loads(open(fixture).read()))
    client, session = boot_from_save(save, seed=seed, clock_epoch=epoch, context=context)
    start = tuple(session.player.tile_pos)
    install_schedule(client, dict(schedule))
    digests = []
    run_steps(client, steps, hook=lambda i: digests.append(digest_of(session)))
    end = tuple(session.player.tile_pos)

    frames = FrameRenderer(client, upscale=1)
    png = frames.png()
    png_size = pg.image.load(BytesIO(png)).get_size()

    print(json.dumps({{
        "scale": context.scale,
        "tile_size": list(context.tile_size),
        "resolution": list(context.resolution),
        "start": list(start),
        "end": list(end),
        "digests": digests,
        "png_size": list(png_size),
        "blank": frames.last_frame_was_blank,
    }}))
    """
)


def _measure(which: str) -> dict[str, Any]:
    """Boot `which` ("default" or "scaled") as the ONLY context built in
    a brand-new subprocess, run the 300-step schedule, and return the
    digest sequence plus geometry -- see this module's docstring for why
    a fresh process per arm is required, not merely convenient."""
    script = _SCRIPT.format(
        which=which,
        fixture=str(FIXTURE),
        epoch=EPOCH,
        seed=SEED,
        steps=STEPS,
        schedule_json=json.dumps({str(k): v for k, v in SCHEDULE.items()}),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=TUXEMON_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"measurement subprocess ({which}) failed: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # The subprocess may log pygame/pygame-menu banners and engine
    # warnings to stdout before its one JSON line -- take the last line.
    parsed: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    return parsed


def test_display_scale_is_digest_neutral_across_a_300_step_moving_window() -> None:
    """The measurement this whole task turns on. See this module's
    docstring for the full context; this test pins the DIGEST-NEUTRAL
    result exactly as measured: not "no divergence found at the end" but
    "no divergence found at ANY of the 300 indices." -- and each arm is a
    genuinely fresh, correctly-scaled process (task 9 review round 1)."""
    default = _measure("default")
    scaled = _measure("scaled")

    # Positive control: the schedule actually moved the player in BOTH
    # runs. Six earlier tests in this project passed vacuously in six
    # different ways (see CLAUDE.md); an equality assertion below is
    # worthless if the player never moved in the first place.
    assert default["end"] != default["start"], (
        f"schedule never moved the player away from {default['start']} in "
        "the unscaled run -- this test would prove nothing"
    )
    assert scaled["end"] != scaled["start"], (
        f"schedule never moved the player away from {scaled['start']} in "
        "the scaled run -- this test would prove nothing"
    )
    # Positive control: the arms really did boot at different scales --
    # otherwise sequence equality below would be unsurprising for the
    # wrong reason (see the geometry test's own anti-vacuity guard).
    assert default["scale"] == 1
    assert scaled["scale"] > 1

    default_digests = default["digests"]
    scaled_digests = scaled["digests"]
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
    fails loudly in exactly that scenario, because the reported
    `scale`/`tile_size` would then be identical for both arms.

    Geometry is asserted two ways, per the task brief: directly off the
    booted context's own `scale`/`tile_size` (what `FrameRenderer`'s
    crop/draw geometry is actually computed from), and by DECODING the
    rendered PNG rather than trusting its byte count (task 2 measured a
    byte-count threshold discriminates nothing here: removing the crop
    entirely, 118,218 bytes, and removing the upscale entirely, 35,385
    bytes, both sat comfortably under the same 400,000-byte bound).

    Correctly rendered (task 9 review round 1 -- this file's first draft
    reported 1216x656 here and explained it as "content-dependent",
    which was FABRICATED: that number was the SIGNATURE of the
    stale-import bug, measured with the unscaled arm booted first in the
    same process, not a property of scale at all -- see
    `docs/2026-08-26-display-scale-measurement.org`'s corrected
    arithmetic), a genuinely scaled, freshly-booted frame's drawn region
    fills the ENTIRE 1280x720 canvas -- the crop is a near no-op at this
    scale, not a smaller island like the unscaled render's.
    """
    default = _measure("default")
    scaled = _measure("scaled")

    # Guard against vacuity if ~/.tuxemon/tuxemon.yaml ever disables
    # scaling (display.scaling: false) or sets a resolution matching
    # NATIVE_RESOLUTION: scale 1 would make scaled_context() identical
    # to the unscaled default and every assertion below pass trivially.
    assert scaled["scale"] > 1, (
        f"scaled_context() computed scale={scaled['scale']!r} -- this "
        "test needs the two contexts to genuinely differ; check "
        "~/.tuxemon/tuxemon.yaml's display.scaling/resolution"
    )

    assert default["scale"] == 1
    assert default["tile_size"] == [16, 16]
    assert scaled["tile_size"] == [
        16 * scaled["scale"],
        16 * scaled["scale"],
    ]
    # Resolution itself is untouched by scale -- only tile density
    # changes (more, smaller tiles vs fewer, bigger ones in the same
    # window).
    assert default["resolution"] == scaled["resolution"] == [1280, 720]

    assert not default["blank"]
    assert not scaled["blank"]

    assert tuple(default["png_size"]) != tuple(scaled["png_size"]), (
        f"unscaled and scaled PNGs decoded to the SAME size "
        f"{default['png_size']!r} -- the two contexts are not actually "
        "rendering differently"
    )
    # Both must be real, positive-area images, well within the
    # (scale-independent) full-canvas bound.
    assert 0 < default["png_size"][0] <= 1280
    assert 0 < default["png_size"][1] <= 720
    assert 0 < scaled["png_size"][0] <= 1280
    assert 0 < scaled["png_size"][1] <= 720
    # The scaled frame's drawn region fills the entire canvas -- measured
    # (see this test's docstring), not merely "differs from the unscaled
    # one".
    assert tuple(scaled["png_size"]) == (1280, 720)


def test_scaled_context_refuses_when_it_is_not_the_first_context_built() -> None:
    """The guard task 9 review round 1 added: `tuxemon.map.view` and
    `tuxemon.graphics` each bind their own copy of `tuxemon.prepare
    .DISPLAY_CONTEXT` at import time, so `scaled_context()` reassigning
    that name later cannot reach an already-imported copy. Booting
    unscaled FIRST in a process, then calling `scaled_context()` in the
    SAME process, must refuse loudly instead of silently rendering at
    the stale (unscaled) geometry -- this is the exact bug this file's
    own first draft shipped with (see the geometry test's docstring)."""
    script = textwrap.dedent(
        """
        from tuxghost.boot import build_client
        from tuxghost.observe import scaled_context

        build_client(seed=1234)
        try:
            scaled_context()
        except RuntimeError as exc:
            print("RAISED", exc)
        else:
            print("NOT RAISED")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=TUXEMON_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line.startswith("RAISED"), (
        f"scaled_context() did not refuse when tuxemon.map.view/"
        f"tuxemon.graphics were already imported at a different scale -- "
        f"got {last_line!r}. This is exactly the silent-degradation bug "
        "the guard exists to prevent; see stdout/stderr for the full "
        f"subprocess output: {result.stdout!r} {result.stderr!r}"
    )
    assert "DisplayContext-building call" in last_line
