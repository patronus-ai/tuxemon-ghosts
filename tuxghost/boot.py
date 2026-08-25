"""Headless bootstrap for Tuxemon."""

from __future__ import annotations

import os
from typing import Any


def headless_context() -> Any:
    """Initialise pygame headlessly and return upstream's DisplayContext.

    Sets both SDL drivers to dummy before importing pygame so no window or
    audio device is ever opened.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

    from tuxemon.platform import platform

    platform.init()

    from tuxemon.prepare import headless_init

    return headless_init()
