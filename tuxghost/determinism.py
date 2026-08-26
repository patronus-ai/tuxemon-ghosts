"""Seed every entropy source from one master seed.

The spike found exactly three: the global `random` generator,
WorldWeatherManager's instance RNG, and uuid4. One master seed derives all
three; separate seeds would let a trace be 'seeded' while one source stayed
live, which is the state in which a probe reported nine identical hashes and
meant nothing.

Only the first two are wired up here. uuid4 draws from `os.urandom` and
cannot be seeded through `random.seed()` at all; it is closed by patch
0003's seeded uuid factory (see the `EXEMPTIONS` register in
`tuxghost/digest.py`), which is expected to consult the same master seed
recorded below so that "seeded" continues to mean the same thing everywhere.

Authority: `tuxghost.boot.build_client(seed)` is authoritative for the
client it constructs -- it always seeds the global generator and its own
config's `deterministic_seed` from its own `seed` argument, so a client
built through it is fully seeded whether or not `seed_all` was ever called.
`seed_all` is for seeding entropy sources ahead of / outside of any one
client build (right now, just the global generator); it also mirrors the
seed onto the process-wide `CONFIG` singleton's `deterministic_seed` so
that any client built by a path other than `build_client` (which does not
override it) still inherits a definite value instead of falling back to OS
entropy.
"""

from __future__ import annotations

import random


def seed_all(seed: int) -> None:
    random.seed(seed)

    from tuxemon.user_config import CONFIG

    CONFIG.deterministic_seed = seed
