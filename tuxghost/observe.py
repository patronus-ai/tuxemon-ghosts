"""Renders what the agent sees.

No engine patch: a real `MapRenderer` is installed through the PUBLIC
`BaseClient.set_renderer` seam, and upstream's `StateDrawer` composites
`state_manager.active_states` bottom-to-top with the same occlusion rule
the real game uses -- so a frame rendered here is the frame a human would
see, including the case where an opaque full-screen state (a cutscene
`ImageState`, a menu) legitimately hides the world.

Why not upstream's `Renderer`: it adds an fps window caption, a debug
overlay, and `save_frame`-to-disk, none of which this harness wants (the
first is wall-clock flavoured). `StateDrawer` is the part that matters.

Where animation time comes from: `WorldState.update(dt)` already calls
`client.map_renderer.update(dt)` every step, so once a real renderer is
installed the engine's own per-step update advances its animations at
this harness's fixed dt. Observation frequency cannot leak into animation
state -- drawing here never calls `update`.
"""

from __future__ import annotations

import os
import sys
from io import BytesIO
from typing import Any

#: The colour a frame is cleared to before drawing. Doubles as the
#: "nothing was drawn here" key for cropping, via `set_colorkey`.
BACKGROUND = (0, 0, 0)

#: Pixel step used when sampling a surface's colours. 7 is coprime with
#: the tile size (16) and every power of two in the resolution, so the
#: sample does not land on a single repeating column of a tiled floor.
SAMPLE_STRIDE = 7


def scaled_context() -> Any:
    """A headless `DisplayContext` scaled the way the REAL game scales,
    instead of `tuxghost.boot.headless_context()`'s hardcoded `scale=1`.

    `tuxemon/prepare.py`'s `headless_init` builds `DisplayContext` with
    `DefaultScaling(1)` unconditionally -- fine for the fast, render-free
    tests this project mostly runs, but it means a headless camera shows
    roughly 80x45 tiles at 1280x720 (`tile_size=(16, 16)`) where a real,
    windowed boot (`pygame_init`, `make_default_scaling(CONFIG,
    NATIVE_RESOLUTION)`) shows far fewer, larger tiles -- measured on
    this machine's `~/.tuxemon/tuxemon.yaml`, scale 5 (`tile_size=(80,
    80)`, ~16x9 tiles), not the scale 4 this task's brief assumed; see
    `docs/2026-08-26-display-scale-measurement.org` for the real numbers
    and why the assumption was off. An agent that only ever sees the
    unscaled view sees far more of the map, at far smaller sprites, than
    a human player ever would.

    This function mirrors `pygame_init`'s scaling arithmetic exactly
    (same `make_default_scaling` call, same `NATIVE_TILE_SIZE`), but
    keeps `headless_init`'s headless plumbing: both dummy SDL drivers,
    `platform.init()`, `pg.init()`/`display.init()`/`font.init()`, a
    real `pg.display.set_mode(CONFIG.resolution)` (required before any
    sprite's `convert_alpha()`), and a `pg.Surface` -- never a real
    window -- as the context's `screen`. Neither `pygame_init` (opens a
    real window; this project never may) nor `headless_init` (hardcodes
    `scale=1`) can be called as-is to get both properties at once.

    MUST BE THE FIRST DISPLAYCONTEXT-BUILDING CALL IN THE PROCESS (task 9
    review round 1, Important #1 -- the earlier version of this docstring
    named the wrong mechanism). Assigning to `tuxemon.prepare.
    DISPLAY_CONTEXT` below does NOT reach `tuxemon.map.view` or
    `tuxemon.graphics`: both do `from tuxemon.prepare import
    DISPLAY_CONTEXT` at their OWN module top level -- a name binding
    evaluated once, at each module's first import, into ITS OWN
    namespace. A later `tuxemon.prepare.DISPLAY_CONTEXT = ...`
    reassignment cannot reach an already-bound copy of that name sitting
    in a different module's namespace; only importing those modules
    fresh, AFTER this assignment, does. `tuxemon.graphics` compounds
    this: `load_and_scale`'s `scale: float = DISPLAY_CONTEXT.scale`
    default argument is evaluated at that same import time too,
    permanently baking whatever scale was live then into every future
    sprite load that does not pass `scale=` explicitly. If either module
    was already imported at a DIFFERENT scale before this call, the
    render measurably gets WORSE than the unscaled default it was meant
    to replace: measured a sparse grid of unscaled 16px tiles on an 80px
    pitch, player sprite absent entirely -- not merely wrong, worse. The
    guard below refuses instead of producing that silently.

    Practically: call this before ANYTHING else in the process boots a
    client -- `headless_context()`/`pygame_init()`/`build_client()`/
    `boot_from_save()` all transitively import both modules the first
    time any of them runs.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

    from tuxemon.platform.const.sizes import NATIVE_RESOLUTION
    from tuxemon.platform.const.sizes import TILE_SIZE as NATIVE_TILE_SIZE
    from tuxemon.scaling import make_default_scaling
    from tuxemon.user_config import CONFIG

    scaling = make_default_scaling(CONFIG, NATIVE_RESOLUTION)
    expected_scale = scaling._scale
    expected_tile_size = scaling.scale_point(NATIVE_TILE_SIZE)

    # See this function's own docstring: tuxemon.map.view and
    # tuxemon.graphics each bind their OWN copy of DISPLAY_CONTEXT at
    # import time, which a later reassignment to
    # tuxemon.prepare.DISPLAY_CONTEXT cannot reach. If either is already
    # imported at a scale/tile_size that disagrees with what this call is
    # about to install, refuse loudly rather than silently degrade.
    # (Importing the same scale twice -- e.g. calling scaled_context()
    # more than once in a process -- is fine: the stale copy still agrees
    # in VALUE even though it is a different object, and nothing here
    # depends on object identity.)
    stale = []
    for name in ("tuxemon.map.view", "tuxemon.graphics"):
        mod = sys.modules.get(name)
        bound = getattr(mod, "DISPLAY_CONTEXT", None) if mod is not None else None
        if bound is not None and (
            (bound.scale, bound.tile_size) != (expected_scale, expected_tile_size)
        ):
            stale.append((name, bound.scale, bound.tile_size))
    if stale:
        details = "; ".join(
            f"{name} bound at scale={scale!r}, tile_size={tile_size!r}"
            for name, scale, tile_size in stale
        )
        raise RuntimeError(
            "scaled_context() must be the FIRST DisplayContext-building "
            "call in this process -- it was not: "
            f"{details} (wanted scale={expected_scale!r}, "
            f"tile_size={expected_tile_size!r}). Reassigning "
            "tuxemon.prepare.DISPLAY_CONTEXT cannot reach a module that "
            "already imported its own copy of that name; the render "
            "would silently degrade instead of scaling (see this "
            "function's docstring). Call scaled_context() before "
            "anything else in this process boots a client -- e.g. as "
            "literally the first statement in a fresh interpreter/"
            "subprocess."
        )

    import pygame as pg
    from tuxemon.platform import platform

    from tuxemon import prepare

    platform.init()
    prepare.core_init()

    pg.init()
    pg.display.init()
    pg.font.init()

    # Required before any sprite's convert_alpha() -- see headless_init's
    # matching comment. The dummy driver opens no real window.
    pg.display.set_mode(CONFIG.resolution)

    screen = pg.Surface(CONFIG.resolution)
    rect = screen.get_rect()

    context = prepare.DisplayContext(
        screen=screen,
        rect=rect,
        resolution=CONFIG.resolution,
        tile_size=expected_tile_size,
        scale=expected_scale,
        scaling=scaling,
    )
    # Only reaches tuxemon.prepare's OWN namespace -- see this function's
    # docstring for why that is necessary but not sufficient, and the
    # guard above for what makes it safe anyway. Harmless, and matches
    # headless_init()'s own existing behaviour (it reassigns this same
    # global on every unscaled boot; containment of that mutation to a
    # single process-wide value has always been owed to headless_init's
    # own code, not to anything added here).
    prepare.DISPLAY_CONTEXT = context
    return context


def distinct_colors(
    surface: Any, stride: int = SAMPLE_STRIDE
) -> set[tuple[int, int, int]]:
    """Sampled distinct RGB values. A blank frame yields exactly one.

    Sampled rather than exhaustive because `numpy` is not a dependency of
    this project (`pygame.surfarray` is unavailable), and 1280x720
    `get_at` calls per assertion is slow enough to matter in a suite that
    already drives real game loops.
    """
    width, height = surface.get_size()
    return {
        tuple(surface.get_at((x, y))[:3])
        for x in range(0, width, stride)
        for y in range(0, height, stride)
    }


class FrameRenderer:
    """Installs a real map renderer, then renders frames on demand.

    Constructing this MUTATES the client: it swaps `NullRenderer` for a
    real `MapRenderer`. That is the point -- probed, `NullRenderer` draws
    exactly 1 distinct colour (pure black) where `MapRenderer` draws 48 on
    the same state -- but it means a caller cannot treat this as a
    read-only observer. Drawing itself is side-effect-free with respect to
    the digest (`tests/test_observe.py` pins that).
    """

    def __init__(self, client: Any, upscale: int = 4) -> None:
        import pygame as pg
        from tuxemon.map.view import DebugRenderer, MapRenderer
        from tuxemon.state.draw import StateDrawer
        from tuxemon.user_config import CONFIG

        if upscale < 1:
            raise ValueError(f"upscale must be >= 1, got {upscale!r}")

        # MapRenderer.draw() gates its DebugRenderer on the process-global
        # `tuxemon.user_config.CONFIG.collision_map` (tuxemon/map/
        # view.py:522), NOT on `client.config` -- a per-client deep copy
        # made at boot time (see `tuxghost/boot.py`'s `CONFIG.copy()`).
        # The two agree at boot by construction, but checking the copy
        # verifies a proxy: if `CONFIG` is ever mutated after boot (or a
        # future boot path stops copying it), this guard would read a
        # stale value while the real renderer keeps consulting the global
        # -- silently letting collision boxes and a red centre line into
        # every frame this class produces, contradicting the module
        # docstring's "no debug overlay" guarantee. Read the SAME global
        # `MapRenderer.draw()` actually reads (parked minor 1, whole-
        # branch review).
        if CONFIG.collision_map:
            raise ValueError(
                "FrameRenderer refuses to run with config.collision_map "
                "True: MapRenderer.draw() would draw upstream's debug "
                "overlay (collision boxes, a red centre line) into every "
                "frame, contradicting this module's no-debug-overlay "
                "guarantee. Set display.collision_map: false (the "
                "default) in ~/.tuxemon/tuxemon.yaml."
            )

        self._client = client
        self._upscale = upscale
        self._surface = pg.Surface(client.context.resolution)
        self._drawer = StateDrawer(
            self._surface, client.state_manager, client.config
        )
        self.last_frame_was_blank = False

        client.set_renderer(
            MapRenderer(
                client.camera_manager,
                client.npc_manager,
                DebugRenderer(
                    client.map_manager, client.npc_manager, client.context
                ),
                client.context,
            )
        )

    def surface(self) -> Any:
        """Draw the current active-state stack and return the full frame."""
        self._surface.fill(BACKGROUND)
        self._drawer.draw()
        return self._surface

    def png(self) -> bytes:
        """The current frame as PNG bytes: cropped to the drawn region and
        integer-upscaled.

        Cropped because at `headless_context()`'s unscaled `scale=1`
        (`tuxemon/prepare.py`'s `headless_init`), a small map renders as
        an island in a large `BACKGROUND` field -- sending a model a
        mostly-black 1280x720 image wastes tokens. At
        `tuxghost.observe.scaled_context()`'s scale (measured 5 on this
        repo's config), this crop is closer to a no-op: measured, a
        correctly-scaled frame's drawn region already fills the entire
        1280x720 canvas, so there is little or no background left to crop
        away. Upscaled with nearest-neighbour (`scale_by`, verified to
        preserve corner colour) because resampling pixel art into mush is
        worse than not scaling.

        A blank frame (nothing drawn at all -- legitimate between map
        loads) is encoded whole rather than cropped to nothing, and flagged
        on `last_frame_was_blank` so a caller can log it instead of
        silently treating black as a picture.
        """
        import pygame as pg

        full = self.surface()
        full.set_colorkey(BACKGROUND)
        rect = full.get_bounding_rect()
        full.set_colorkey(None)

        self.last_frame_was_blank = rect.width == 0 or rect.height == 0
        if self.last_frame_was_blank:
            out = full
        else:
            cropped = full.subsurface(rect).copy()
            out = (
                pg.transform.scale_by(cropped, self._upscale)
                if self._upscale > 1
                else cropped
            )

        buf = BytesIO()
        pg.image.save(out, buf, "PNG")
        return buf.getvalue()
