"""Pins the one genuinely subtle line in `scripts/build_web.py`: the audio
filter must match the DIRECTORY `mods/tuxemon/music`, never the string
`"music"` anywhere in the path. The db's own records live under
`mods/tuxemon/db/music/`, and the feasibility spike zeroed those by
mistake -- a `"music" in f.parts` filter matches `db/music` too, and
`make check` failed loudly with a confusing "not existing in the db"
validation error.

Runs on a small synthetic tree under `tmp_path`, not the real 194MB
`tuxemon/mods/`, so this stays cheap. No SDL involved -- this module
never imports pygame or tuxemon.
"""

from __future__ import annotations

import importlib.util
import types
import zipfile
from pathlib import Path

BUILD_WEB_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "build_web.py"
)


def _load_build_web() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_web", BUILD_WEB_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_tree(root: Path) -> None:
    """Layout mirrors the real vendored tree closely enough to matter:
    `mods/tuxemon/music/` (audio, must be zeroed) sits alongside
    `mods/tuxemon/db/music/` (db records naming that audio, must NOT be
    zeroed) -- the exact adjacency that made the spike's bug easy to make.
    """
    music = root / "tuxemon" / "mods" / "tuxemon" / "music"
    db_music = root / "tuxemon" / "mods" / "tuxemon" / "db" / "music"
    other = root / "tuxemon" / "mods" / "tuxemon" / "maps"
    for d in (music, db_music, other):
        d.mkdir(parents=True)
    (music / "town.ogg").write_bytes(b"\x00" * 4096)  # fake audio bytes
    (db_music / "music.yaml").write_bytes(b"slug: town\nfile: town.ogg\n")
    (other / "town.tmx").write_bytes(b"<map/>")


def test_zip_mods_zeroes_audio_dir_but_keeps_db_music_entries(
    tmp_path: Path,
) -> None:
    module = _load_build_web()
    _make_tree(tmp_path)
    # `module` is loaded dynamically via importlib, so mypy --strict sees
    # it only as `types.ModuleType` and flags these as attr-defined --
    # both names genuinely exist on the loaded module (see build_web.py).
    module.ROOT = tmp_path  # type: ignore[attr-defined]
    module.AUDIO_DIR = (  # type: ignore[attr-defined]
        tmp_path / "tuxemon" / "mods" / "tuxemon" / "music"
    )

    target = tmp_path / "mods.zip"
    module._zip_mods(target)

    with zipfile.ZipFile(target) as z:
        sizes = {i.filename: i.file_size for i in z.infolist()}

    assert sizes["mods/tuxemon/music/town.ogg"] == 0
    assert sizes["mods/tuxemon/db/music/music.yaml"] > 0
    assert sizes["mods/tuxemon/maps/town.tmx"] > 0
