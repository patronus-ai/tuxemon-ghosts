"""Seed every entropy source from one master seed.

The spike found exactly three: the global `random` generator,
WorldWeatherManager's instance RNG, and uuid4. One master seed derives all
three; separate seeds would let a trace be 'seeded' while one source stayed
live, which is the state in which a probe reported nine identical hashes and
meant nothing.

All three are now wired up here. uuid4 draws from `os.urandom` and cannot
be seeded through `random.seed()` at all; patch 0003 adds
`tuxemon.core.ids.new_id()`, a factory with its own `random.Random`
instance seeded by `seed_ids()`, and routes every load-bearing
`instance_id` call site through it. `seed_all` seeds that factory from the
same master seed as everything else so that "seeded" continues to mean the
same thing everywhere.

Authority: `tuxghost.boot.build_client(seed)` (and `boot_from_save`) are
authoritative for the client they construct -- each calls `random.seed
(seed)`, sets `deterministic_seed = seed` directly on its own config copy,
and calls `seed_ids(seed)`, all from their own `seed` argument, so a client
built through them is fully seeded whether or not `seed_all` was ever
called first. This was not true of `seed_ids` at first: `build_client`
originally left id seeding entirely to whatever `seed_all` last did (or
didn't do), which broke `tests/test_digest.py` -- every test there builds
straight from `build_client` and never calls `seed_all`, so ids fell back
to OS entropy. `seed_all` remains useful for seeding entropy sources ahead
of / outside of any one client build; it also mirrors the seed onto the
process-wide `CONFIG` singleton's `deterministic_seed` so that any client
built by a path other than `build_client` (which does not override it)
still inherits a definite value instead of falling back to OS entropy.

`pin_clock` closes a fourth leak that is not an entropy source at all --
wall time. Patch 0004 routes `TimeHandler.get_current_time()` (and
therefore `get_ordinal()`/`get_time_variables()`), `Monster.capture_date`
(`today_month_day()`), `Battle.timestamp` (both where it is set --
`battle.py`'s `Battle.__init__` and, load-bearingly, `entity/battle.py`'s
`BattlesHandler.record_battle()`, which otherwise overwrites it via
`Battle.from_save_data()`), `SaveData.time`, and `AbstractSession`'s
`_start_time`/`_start_timestamp` bookkeeping through
`tuxemon.core.clock.now()`/`now_datetime()`. Unlike `seed_ids`, the pinned
epoch is not reset by `build_client`/`boot_from_save`: it is process-global
state on `tuxemon.core.clock`, so a caller must call `pin_clock` itself
(there is no per-build epoch argument to be authoritative over, the way
there is for the seed).
"""

from __future__ import annotations

import random


def seed_all(seed: int) -> None:
    random.seed(seed)

    from tuxemon.core.ids import seed_ids
    from tuxemon.user_config import CONFIG

    CONFIG.deterministic_seed = seed
    seed_ids(seed)


def pin_clock(epoch: int) -> None:
    """Pin all wall-clock reads to a fixed epoch (seconds since 1970)."""
    from tuxemon.core.clock import set_epoch

    set_epoch(float(epoch))
