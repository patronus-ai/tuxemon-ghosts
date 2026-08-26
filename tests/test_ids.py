"""Direct unit tests for `tuxemon.core.ids`, the seeded id factory patch
0003 added. Everything else in this suite only exercises it indirectly,
through game entities that happen to call `new_id()` -- these test the
factory's own two branches and its UUID-shape contract directly, so a
regression here does not depend on some entity's constructor still routing
through it correctly."""

from __future__ import annotations

import uuid


def test_new_id_returns_a_version_4_uuid() -> None:
    from tuxemon.core.ids import new_id, seed_ids

    seed_ids(1234)
    result = new_id()

    assert isinstance(result, uuid.UUID)
    assert result.version == 4


def test_seed_ids_none_restores_os_entropy() -> None:
    """`seed_ids(None)` is the only way to reach `new_id`'s `uuid.uuid4()`
    branch (`_rng is None`) -- every other test in this suite always seeds
    with an int first. Two draws from real OS entropy cannot be asserted
    equal to a fixed value, but they must still be valid version-4 UUIDs,
    and two draws must not coincide (a random.getrandbits(128) collision is
    astronomically unlikely; this at least confirms `uuid.uuid4()` is
    genuinely being called and not returning a constant)."""
    from tuxemon.core.ids import new_id, seed_ids

    seed_ids(1234)  # leave the module in a seeded state beforehand ...
    seed_ids(None)  # ... so this line is what actually flips the branch.

    a = new_id()
    b = new_id()

    assert isinstance(a, uuid.UUID)
    assert isinstance(b, uuid.UUID)
    assert a.version == 4
    assert b.version == 4
    assert a != b


def test_seed_ids_makes_new_id_reproducible() -> None:
    from tuxemon.core.ids import new_id, seed_ids

    seed_ids(1234)
    first = new_id()
    seed_ids(1234)
    second = new_id()

    assert first == second


def test_seed_ids_different_seeds_differ() -> None:
    from tuxemon.core.ids import new_id, seed_ids

    seed_ids(1234)
    a = new_id()
    seed_ids(99)
    b = new_id()

    assert a != b
