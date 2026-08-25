def test_package_imports() -> None:
    import tuxghost

    assert tuxghost.__version__ == "0.1.0"


def test_tuxemon_submodules_are_importable() -> None:
    """From the repo root `import tuxemon` yields an EMPTY namespace package
    (__file__ is None) and every submodule import fails. conftest.py fixes
    both the path and the working directory; this pins that it stays fixed."""
    from tuxemon.session import local_session

    import tuxemon

    assert tuxemon.__file__ is not None
    assert local_session is not None
