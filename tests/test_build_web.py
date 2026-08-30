"""Pins the one genuinely subtle line in `scripts/build_web.py`: the audio
filter must match the DIRECTORY `mods/tuxemon/music`, never the string
`"music"` anywhere in the path. The db's own records live under
`mods/tuxemon/db/music/`, and the feasibility spike zeroed those by
mistake -- a `"music" in f.parts` filter matches `db/music` too, and
`make check` failed loudly with a confusing "not existing in the db"
validation error.

Also pins the contract between this script's zip arcnames and
`web/index.html`'s shim: nothing else checks that `_zip_python`'s
prefixes match the page's `sys.path.insert(0, "/game")` (which needs
`tuxemon/` and `tuxghost/` to sit directly under `/game` once
`unpackArchive` extracts there), or that `_zip_fixtures`'s arcnames
match the literal `Path(...)` strings the page passes to `web.main`. A
silent drift on either side would only ever surface as a Pyodide
`ModuleNotFoundError`/`FileNotFoundError` in a real browser, well past
this project's native gate.

Runs on a small synthetic tree under `tmp_path`, not the real 194MB
`tuxemon/mods/`, so this stays cheap. No SDL involved -- this module
never imports pygame or tuxemon.
"""

from __future__ import annotations

import importlib.util
import re
import types
import zipfile
from pathlib import Path

BUILD_WEB_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "build_web.py"
)
INDEX_HTML_PATH = (
    Path(__file__).resolve().parent.parent / "web" / "index.html"
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


def test_zip_python_arcnames_match_the_shims_sys_path(
    tmp_path: Path,
) -> None:
    """`web/index.html` does `sys.path.insert(0, "/game")` then
    `import tuxghost.web as web` (which itself imports `tuxemon.*`).
    For those dotted imports to resolve, `unpackArchive`'s extraction
    into `/game` must land packages at `/game/tuxemon/...` and
    `/game/tuxghost/...` -- i.e. `_zip_python`'s arcnames must be
    prefixed `tuxemon/` and `tuxghost/`, not anything else, regardless
    of what the real vendored tree is named on disk.
    """
    module = _load_build_web()
    (tmp_path / "tuxemon" / "tuxemon").mkdir(parents=True)
    (tmp_path / "tuxemon" / "tuxemon" / "foo.py").write_text("# fake\n")
    (tmp_path / "tuxghost").mkdir()
    (tmp_path / "tuxghost" / "bar.py").write_text("# fake\n")
    module.ROOT = tmp_path  # type: ignore[attr-defined]

    target = tmp_path / "code.zip"
    module._zip_python(target)

    with zipfile.ZipFile(target) as z:
        names = set(z.namelist())

    assert names == {"tuxemon/foo.py", "tuxghost/bar.py"}, names


def test_zip_fixtures_arcnames_match_what_index_html_passes_to_main(
    tmp_path: Path,
) -> None:
    """`web/index.html`'s `runPythonAsync` script passes two literal
    `Path(...)` arguments to `web.main` -- extracted here from the real
    page text, not re-typed, so this test cannot drift out of sync with
    the page on its own. `_zip_fixtures`'s arcnames must equal those
    literals exactly, or the unpacked archive won't have a file where
    the page looks for one.

    `_zip_fixtures` reads the real, small, committed fixture files (via
    the module's real `ROOT`) but writes its output zip under `tmp_path`
    -- reading them is cheap and SDL-free; there is no reason to fake
    them, only to avoid littering the repo with the output.
    """
    literals = re.findall(
        r'Path\("([^"]+)"\)', INDEX_HTML_PATH.read_text()
    )
    assert len(literals) == 2, literals

    module = _load_build_web()
    target = tmp_path / "fixtures.zip"
    module._zip_fixtures(target)

    with zipfile.ZipFile(target) as z:
        names = set(z.namelist())

    assert names == set(literals), (names, literals)
