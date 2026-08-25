"""Make the vendored Tuxemon importable and its assets findable.

Two separate problems, both measured:
  * From the repo root, `import tuxemon` resolves to the clone directory as an
    empty namespace package -- it imports fine, then every submodule raises
    ModuleNotFoundError. Putting tuxemon/ on sys.path makes the real package
    (which has __init__.py) win, since a regular package beats a namespace
    portion regardless of path order.
  * Asset loading is relative to the working directory: from the repo root the
    game dies with "Metadata file missing: 'mods/tuxemon/mod.yaml'".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

TUXEMON_DIR = Path(__file__).resolve().parent.parent / "tuxemon"

sys.path.insert(0, str(TUXEMON_DIR))
os.chdir(TUXEMON_DIR)
