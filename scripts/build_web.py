"""Packages the browser build into `build/web/`.

Zips are built from the VENDORED tree, so the page cannot drift from what
the native tests run against.

AUDIO IS STUBBED TO ZERO BYTES. Measured: the db's pydantic validators
check that music and sound records exist IN THE DB, not that the audio
files have content, so zero-byte placeholders satisfy them and 147MB of
music need not ship. Whether PLAYING one raises is UNTESTED -- if it
does, disable audio in the browser build (see the spec).

Filter on the audio DIRECTORY, never on `"music" in parts`: the db's own
records live under `mods/tuxemon/db/music/`, and an earlier attempt
zeroed those by mistake and failed loudly.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "web"
AUDIO_DIR = ROOT / "tuxemon" / "mods" / "tuxemon" / "music"


def _zip_python(target: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for base, prefix in (
            (ROOT / "tuxemon" / "tuxemon", "tuxemon"),
            (ROOT / "tuxghost", "tuxghost"),
        ):
            for f in base.rglob("*.py"):
                if "__pycache__" in f.parts:
                    continue
                z.write(f, f"{prefix}/{f.relative_to(base)}")


def _zip_mods(target: Path) -> None:
    mods = ROOT / "tuxemon" / "mods"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for f in mods.rglob("*"):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            arc = str(f.relative_to(ROOT / "tuxemon"))
            if AUDIO_DIR in f.parents:
                z.writestr(arc, b"")
            else:
                z.write(f, arc)


def _zip_fixtures(target: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(
            ROOT / "tests" / "fixtures" / "paper_town.save",
            "fixtures/paper_town.save",
        )
        z.write(
            ROOT / "tests" / "golden" / "scripted_town_1234.tuxghost",
            "fixtures/ghost.tuxghost",
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "web" / "index.html", OUT / "index.html")
    _zip_python(OUT / "code.zip")
    _zip_mods(OUT / "mods.zip")
    _zip_fixtures(OUT / "fixtures.zip")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:16s} {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
