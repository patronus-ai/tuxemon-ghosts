#!/usr/bin/env python3
"""THROWAWAY: does state diverge once COMBAT rng is exercised?

Adds two controls the earlier probe lacked:
  --seed-weather  seed the one instance-RNG (world/weather.py:133)
  a forced random_battle, so combat RNG actually runs
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path

CALLS: list[str] = []

def _wrap(name):
    orig = getattr(random, name)
    def w(*a, **k):
        r = orig(*a, **k); CALLS.append(f"{name}->{r!r}"); return r
    return w

for _n in ("random", "randint", "choice", "choices", "shuffle", "uniform",
           "randrange", "sample", "gauss", "getrandbits"):
    if hasattr(random, _n):
        setattr(random, _n, _wrap(_n))

_OrigRandom = random.Random
class _CR(_OrigRandom):
    def random(self, *a, **k):
        r = super().random(*a, **k); CALLS.append(f"inst.random->{r!r}"); return r
    def randint(self, *a, **k):
        r = super().randint(*a, **k); CALLS.append(f"inst.randint->{r!r}"); return r
    def choice(self, *a, **k):
        r = super().choice(*a, **k); CALLS.append(f"inst.choice->{r!r}"); return r
random.Random = _CR

# uuid4() draws from OS entropy, bypassing random.seed() entirely.
# Route it through a seeded Random so instance_ids are reproducible.
import uuid as _uuid
_UUID_RNG = _OrigRandom(0)
def _seeded_uuid4():
    return _uuid.UUID(int=_UUID_RNG.getrandbits(128), version=4)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_determinism import (build_client, install_input_schedule,  # noqa: E402
                               merge, mash, digest_state)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--label", default="run")
    ap.add_argument("--seed-weather", action="store_true")
    ap.add_argument("--seed-uuid", action="store_true")
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--dump-state", type=Path, default=None)
    args = ap.parse_args()

    if args.seed_uuid:
        global _UUID_RNG
        _UUID_RNG = _OrigRandom(args.seed)
        _uuid.uuid4 = _seeded_uuid4
        import tuxemon.entity.entity as _e; _e.uuid4 = _seeded_uuid4
        import tuxemon.battle as _b; _b.uuid4 = _seeded_uuid4
        import tuxemon.technique.technique as _t; _t.uuid4 = _seeded_uuid4
        import tuxemon.status.status as _st; _st.uuid4 = _seeded_uuid4
        import tuxemon.session as _s; _s.uuid4 = _seeded_uuid4

    client, session = build_client(args.seed)

    if args.seed_weather:
        # the real fix is WorldWeatherManager(seed=...) in BaseClient
        client.weather_manager._rng = _OrigRandom(args.seed)

    from tuxemon.platform.const import buttons
    install_input_schedule(client, merge(
        mash(buttons.A, start=30, count=40, period=12),
    ))

    # clear the intro
    for _ in range(700):
        client.update(1 / 60.0)

    boot = len(CALLS)

    # give the player a party, then force combat
    ex = client.event_engine.execute_action
    ex("add_monster", ("rockitten", 12))
    ex("add_monster", ("budaye", 10))
    pre_battle = len(CALLS)
    ex("random_battle", (2, 8, 14))

    # mash A to drive the battle menus
    st = install_input_schedule(client, mash(buttons.A, start=0, count=400, period=6))
    for _ in range(args.steps):
        client.update(1 / 60.0)

    if args.dump:
        args.dump.write_text("\n".join(CALLS))

    d = digest_state(session)
    if args.dump_state:
        args.dump_state.write_text(json.dumps(d, indent=2, sort_keys=True))
    mons = d["npc_state"]["monsters"]
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
    print(json.dumps({
        "label": args.label,
        "seed": args.seed,
        "seed_weather": args.seed_weather,
        "rng_intro": boot,
        "rng_party": pre_battle - boot,
        "rng_battle": len(CALLS) - pre_battle,
        "rng_seq_sha": hashlib.sha256("\n".join(CALLS).encode()).hexdigest()[:16],
        "state_sha": hashlib.sha256(blob.encode()).hexdigest()[:16],
        "n_monsters": len(mons),
        "hp": [m.get("current_hp") for m in mons],
        "stack": d["state_stack"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
