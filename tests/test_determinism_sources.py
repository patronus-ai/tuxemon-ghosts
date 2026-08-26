import random

from tuxghost.determinism import seed_all


def test_weather_rng_is_seeded_from_the_master_seed() -> None:
    from tuxghost.boot import build_client

    seed_all(1234)
    client_a, _ = build_client(seed=1234)
    draws_a = [client_a.weather_manager._rng.random() for _ in range(5)]

    seed_all(1234)
    client_b, _ = build_client(seed=1234)
    draws_b = [client_b.weather_manager._rng.random() for _ in range(5)]

    assert draws_a == draws_b


def test_master_seed_controls_the_global_generator() -> None:
    seed_all(7)
    first = [random.random() for _ in range(3)]
    seed_all(7)
    assert [random.random() for _ in range(3)] == first


def test_default_config_has_no_deterministic_seed() -> None:
    """`base_client.py` reads `config.deterministic_seed` unconditionally
    (`WorldWeatherManager(seed=config.deterministic_seed)`); a config that
    was never told to seed (e.g. the graphical entry point, which never
    calls `seed_all`) must still provide a value instead of raising
    `AttributeError`, and that value must be `None` so upstream's
    OS-entropy weather behaviour is preserved when nobody asked for
    determinism. `tuxghost.boot.build_client` always sets this attribute
    explicitly before it matters (see its own docstring/comments), so this
    test -- unlike `test_weather_rng_is_seeded_from_the_master_seed` --
    deliberately does NOT go through `build_client`: it is the only test in
    this file that would catch config.py's `__init__` default being
    dropped, since `build_client`'s own explicit override papers over that
    exact regression."""
    from tuxemon.config import TuxemonConfig

    assert TuxemonConfig().deterministic_seed is None


def test_copy_preserves_the_deterministic_seed() -> None:
    """`TuxemonConfig.copy()` deep-copies `config_model` but, before patch
    0002, silently dropped every other plain attribute it didn't know to
    re-list -- `deterministic_seed` included. `tuxghost.boot.build_client`
    always overrides the copy's seed explicitly (so this test does not
    exercise that call path; it hits `copy()` directly), but any other
    caller of `TuxemonConfig.copy()` -- present or future -- needs the copy
    to actually be a copy, not a reset to the class default."""
    from tuxemon.config import TuxemonConfig

    config = TuxemonConfig()
    config.deterministic_seed = 42
    assert config.copy().deterministic_seed == 42


def test_entity_ids_are_reproducible() -> None:
    from tuxghost.boot import build_client

    def ids(seed: int) -> list[str]:
        seed_all(seed)
        _client, session = build_client(seed=seed)
        session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
        return [str(m.instance_id) for m in session.player.monsters]

    assert ids(1234) == ids(1234)
    assert ids(1234) != ids(99)


def test_session_uuid_is_reproducible() -> None:
    """Pins `AbstractSession.reset()`'s `self._uuid = new_id()` redraw
    (`tuxemon/session.py`). `local_session = Session()` is a process-lifetime
    singleton: `__init__`'s own `new_id()` draw runs exactly once per
    process, on the first import of `tuxemon.session`. Without a redraw in
    `reset()` (called by every `build_client`/`boot_from_save`), `_uuid`
    would be drawn once, ever, and then frozen for the rest of the process
    -- every later build, regardless of seed, would silently keep that same
    first value. This does not go through a digest comparison because
    `tuxghost.digest.state_of()` deliberately never digests
    `session.session_state` (see the comment above `EXEMPTIONS` in
    `tuxghost/digest.py`); it asserts on `session._uuid` directly instead,
    which is the only way to observe this value at all today."""
    from tuxghost.boot import build_client

    def get(seed: int) -> str:
        _client, session = build_client(seed=seed)
        return str(session._uuid)

    first = get(1234)
    other_seed = get(99)
    second = get(1234)

    assert first == second
    assert first != other_seed
