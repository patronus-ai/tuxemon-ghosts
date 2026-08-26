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

from io import BytesIO
from typing import Any

#: The colour a frame is cleared to before drawing. Doubles as the
#: "nothing was drawn here" key for cropping, via `set_colorkey`.
BACKGROUND = (0, 0, 0)

#: Pixel step used when sampling a surface's colours. 7 is coprime with
#: the tile size (16) and every power of two in the resolution, so the
#: sample does not land on a single repeating column of a tiled floor.
SAMPLE_STRIDE = 7


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

        if upscale < 1:
            raise ValueError(f"upscale must be >= 1, got {upscale!r}")

        # MapRenderer.draw() gates its DebugRenderer on exactly this flag
        # (tuxemon/map/view.py:522) -- wiring one in below is otherwise
        # inert only by accident of the default config. A True value would
        # silently burn collision boxes and a red centre line into every
        # frame this class produces: a frame that quietly differs from
        # what a human player sees, and the whole module docstring's claim
        # ("no debug overlay") would be false without anyone changing a
        # line in this file.
        if client.config.collision_map:
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

        Cropped because the headless display context is built at
        `scale=1` (`tuxemon/prepare.py`'s `headless_init`), so a small map
        renders as an island in a large `BACKGROUND` field -- sending a
        model a mostly-black 1280x720 image wastes tokens. Upscaled with
        nearest-neighbour (`scale_by`, verified to preserve corner colour)
        because resampling pixel art into mush is worse than not scaling.

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
