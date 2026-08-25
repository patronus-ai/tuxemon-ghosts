#!/usr/bin/env python3
"""THROWAWAY: drive an ACTUAL CombatState and test determinism through it."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path

CALLS: list[str] = []
def _wrap(name):
    orig = getattr(random, name)
    def w(*a, **k):
        r = orig(*a, **k); CALLS.append(f"{name}->{r!r}"); return r
    return w
for _n in ("random","randint","choice","choices","shuffle","uniform",
           "randrange","sample","gauss","getrandbits"):
    if hasattr(random, _n): setattr(random, _n, _wrap(_n))
_OrigRandom = random.Random
class _CR(_OrigRandom):
    def random(self,*a,**k):
        r=super().random(*a,**k); CALLS.append(f"i.rand->{r!r}"); return r
    def randint(self,*a,**k):
        r=super().randint(*a,**k); CALLS.append(f"i.ri->{r!r}"); return r
    def choice(self,*a,**k):
        r=super().choice(*a,**k); CALLS.append(f"i.ch->{r!r}"); return r
random.Random = _CR

import uuid as _uuid
_UUID_RNG = _OrigRandom(0)
def _seeded_uuid4():
    return _uuid.UUID(int=_UUID_RNG.getrandbits(128), version=4)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_determinism import (build_client, install_input_schedule,  # noqa
                               merge, mash, digest_state)


def stack(client):
    return [s.name for s in client.state_manager.active_states]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--label", default="run")
    ap.add_argument("--battle-steps", type=int, default=6000)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--dump-state", type=Path, default=None)
    ap.add_argument("--checkpoint", type=int, default=0,
                    help="digest every N steps into --dump-state.ckpt")
    args = ap.parse_args()

    global _UUID_RNG
    _UUID_RNG = _OrigRandom(args.seed)
    _uuid.uuid4 = _seeded_uuid4
    for m in ("tuxemon.entity.entity","tuxemon.battle",
              "tuxemon.technique.technique","tuxemon.status.status",
              "tuxemon.session"):
        __import__(m); sys.modules[m].uuid4 = _seeded_uuid4

    import logging
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)

    client, session = build_client(args.seed)
    client.weather_manager._rng = _OrigRandom(args.seed)

    from tuxemon.platform.const import buttons
    install_input_schedule(client, mash(buttons.A, start=10, count=60, period=10))
    for _ in range(700):
        client.update(1/60.0)

    ex = client.event_engine.execute_action

    # The intro leaves a blocking InputMenu (name entry) that A-mashing cannot
    # dismiss. Remove ONLY the input-blocking UI states; SinkState must stay,
    # because the intro script later runs unlock_controls, which looks it up by
    # name and raises ValueError("Missing state SinkState") if it is gone.
    BLOCKING = {"DialogState", "InputMenu", "ChoiceState"}
    for st_name in [x.name for x in client.state_manager.active_states]:
        if st_name in BLOCKING:
            try:
                client.remove_state_by_name(st_name)
            except Exception as e:
                if args.trace:
                    print(f"  could not remove {st_name}: {e}",
                          file=sys.stderr, flush=True)
    if args.trace:
        print(f"stack after unwind: {stack(client)}", file=sys.stderr, flush=True)

    ex("add_monster", ("rockitten", 12))
    ex("add_monster", ("budaye", 10))
    # _start_battle refuses (silently, since logging is unconfigured) when no
    # environment is active.
    ex("set_environment", ("grass",))

    p = session.player
    techs = [len(m.moves.current_moves) for m in p.monsters]
    if args.trace:
        print(f"party={[m.slug for m in p.monsters]} techs={techs} "
              f"fainted={p.party.is_fainted} no_tech={p.party.no_tech}",
              file=sys.stderr, flush=True)
        print(f"stack before battle: {stack(client)}", file=sys.stderr, flush=True)

    pre = len(CALLS)
    import time as _t
    _t0 = _t.time()
    ex("random_battle", (2, 8, 14), True)  # skip=True: run start(), do not spin
    if args.trace:
        print(f"random_battle returned in {_t.time()-_t0:.1f}s "
              f"stack={stack(client)}", file=sys.stderr, flush=True)

    # drive the combat menus
    install_input_schedule(client, mash(buttons.A, start=0,
                                   count=args.battle_steps // 5 + 2, period=5))
    seen, combat_steps, ended_at = set(), 0, None
    was_in = False
    ckpts = []
    for i in range(args.battle_steps):
        client.update(1/60.0)
        st = stack(client)
        seen.update(st)
        if "CombatState" in st:
            combat_steps += 1
            was_in = True
        elif was_in and ended_at is None:
            ended_at = i
            if args.trace:
                print(f"  COMBAT ENDED at step {i}: {st}",
                      file=sys.stderr, flush=True)
            break
        if args.checkpoint and i and i % args.checkpoint == 0:
            _d = digest_state(session)
            _b = json.dumps(_d, sort_keys=True, separators=(",", ":"))
            ckpts.append((i, hashlib.sha256(_b.encode()).hexdigest()[:16]))
        if args.trace and i % 200 == 0:
            print(f"  step {i:5d} t={_t.time()-_t0:6.1f}s: {st}",
                  file=sys.stderr, flush=True)

    d = digest_state(session)
    if args.dump_state:
        args.dump_state.write_text(json.dumps(d, indent=2, sort_keys=True))
        if ckpts:
            Path(str(args.dump_state) + ".ckpt").write_text(
                "\n".join(f"{i}\t{h}" for i, h in ckpts))
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
    print(json.dumps({
        "label": args.label, "seed": args.seed,
        "techs": techs,
        "combat_entered": "CombatState" in seen,
        "combat_ended_at": ended_at,
        "levels": [m.get("level") for m in d["npc_state"]["monsters"]],
        "exp": [m.get("total_experience") for m in d["npc_state"]["monsters"]],
        "money": d["npc_state"]["money"]["money"],
        "combat_steps": combat_steps,
        "rng_battle": len(CALLS) - pre,
        "rng_seq_sha": hashlib.sha256("\n".join(CALLS).encode()).hexdigest()[:16],
        "state_sha": hashlib.sha256(blob.encode()).hexdigest()[:16],
        "hp": [m.get("current_hp") for m in d["npc_state"]["monsters"]],
        "final_stack": stack(client),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
