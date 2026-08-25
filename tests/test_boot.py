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


def test_headless_context_leaves_pygame_globally_initialized() -> None:
    """pygame_menu asserts pygame.get_init() before pushing a menu state; if
    headless_init() only calls pg.display.init()/pg.font.init()/set_mode()
    without pg.init(), pygame.get_init() stays False and that assertion
    fails. Empirically, none of display.init(), font.init(), or set_mode()
    flip pygame.get_init() to True -- only pg.init() does."""
    import pygame as pg

    from tuxghost.boot import headless_context

    headless_context()
    assert pg.get_init() is True
