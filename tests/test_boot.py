def test_headless_context_can_convert_surfaces() -> None:
    """A converted surface is what sprite loading needs; without set_mode
    pygame raises 'No convert format has been set'."""
    import pygame as pg

    from tuxghost.boot import headless_context

    ctx = headless_context()
    assert ctx.resolution[0] > 0
    surface = pg.Surface((8, 8), pg.SRCALPHA)
    converted = surface.convert_alpha()
    assert converted.get_size() == (8, 8)
