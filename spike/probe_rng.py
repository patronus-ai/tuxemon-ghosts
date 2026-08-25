#!/usr/bin/env python3
"""THROWAWAY: does the probe route actually exercise any RNG?

Patches the random module BEFORE tuxemon imports, so that modules doing
`from random import choice` bind the counting wrapper.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from pathlib import Path

CALLS: list[str] = []

def _wrap(name):
    orig = getattr(random, name)
    def w(*a, **k):
        r = orig(*a, **k)
        CALLS.append(f"{name}{a!r}->{r!r}")
        return r
    return w

for _n in ("random", "randint", "choice", "choices", "shuffle", "uniform",
           "randrange", "sample", "gauss", "triangular", "getrandbits"):
    if hasattr(random, _n):
        setattr(random, _n, _wrap(_n))

# also catch instance RNGs: random.Random(...)
_OrigRandom = random.Random
class _CountingRandom(_OrigRandom):
    def random(self, *a, **k):
        r = super().random(*a, **k); CALLS.append(f"inst.random->{r!r}"); return r
    def randint(self, *a, **k):
        r = super().randint(*a, **k); CALLS.append(f"inst.randint{a!r}->{r!r}"); return r
    def choice(self, *a, **k):
        r = super().choice(*a, **k); CALLS.append(f"inst.choice->{r!r}"); return r
random.Random = _CountingRandom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_determinism import (build_client, install_input_schedule,  # noqa: E402
                               merge, mash, digest_state)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--label", default="run")
    ap.add_argument("--dump", type=Path, default=None)
    args = ap.parse_args()

    client, session = build_client(args.seed)
    boot_calls = len(CALLS)

    from tuxemon.platform.const import buttons
    schedule = merge(
        mash(buttons.A, start=30, count=40, period=12),
        mash(buttons.DOWN, start=560, count=30, period=8, hold=6),
        mash(buttons.RIGHT, start=820, count=30, period=8, hold=6),
        mash(buttons.A, start=1100, count=60, period=10),
    )
    install_input_schedule(client, schedule)
    for _ in range(args.steps):
        client.update(1 / 60.0)

    play_calls = len(CALLS) - boot_calls
    seq = hashlib.sha256("\n".join(CALLS).encode()).hexdigest()[:16]
    if args.dump:
        args.dump.write_text("\n".join(CALLS))
    st = digest_state(session)
    blob = json.dumps(st, sort_keys=True, separators=(",", ":"))
    print(json.dumps({
        "label": args.label,
        "seed": args.seed,
        "rng_calls_boot": boot_calls,
        "rng_calls_play": play_calls,
        "rng_seq_sha": seq,
        "state_sha": hashlib.sha256(blob.encode()).hexdigest()[:16],
        "tile_pos": st["tile_pos"],
        "map": st["map"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
