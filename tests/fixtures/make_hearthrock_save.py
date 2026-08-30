"""Generates tests/fixtures/hearthrock_city.save -- a starting state in
the `classic_*` campaign, where the gym goal is actually reachable.

Run from the repo root:

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYTHONHASHSEED=0 \
      ./.venv/bin/python tests/fixtures/make_hearthrock_save.py

WHY A SECOND SAVE. `paper_town.save` sits in the `spyder_*` campaign,
and the two campaigns are topologically DISCONNECTED: all 13
`classic_gym_*.tmx` maps live in a 22-map component containing no
`spyder_*` map at all (measured; see `tests/test_rules_mapgraph.py`).
So `tuxghost.rules.TuxemonGymRules` -- the Boulder-Badge analogue --
could never fire from `paper_town.save`, no matter how long a search ran.
This save fixes that by starting inside the other campaign.

WHY HEARTHROCK CITY. It is the only city in the component with TWO gyms
on it (Granite at tile (23,14), Mila at (18,14), both entered through
the same door tile), and it links to `classic_route_1`, so it is the
campaign's first city. One map transition from the start reaches two
different gym leaders.

WHY LEVEL 25. Both leaders field `hydrone,20` and `nudikill,20` -- read
from the map data, not guessed. A level-5 party like `paper_town.save`'s
would make the goal reachable in topology but unwinnable in practice,
which moves the problem rather than solving it. Two at 25 against two at
20 is a fair fight, not a walkover.

Teleport rather than playing the intro, for the same reason
`make_paper_town_save.py` gives: mashing A from a cold boot wedges on
the name-entry keyboard. `boot_from_save` restores no state stack
(SaveData has none), so the cold-boot intro states present while this
script runs do not enter the artifact -- verified by
`tests/test_hearthrock_save.py`, which boots the output and asserts a
bare WorldState.

The output is COMMITTED. This script exists so the artifact is
reproducible and reviewable, not because the gate re-runs it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tuxemon"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT / "tuxemon")

from tuxghost.boot import build_client, snapshot_save
from tuxghost.determinism import pin_clock
from tuxghost.loop import install_schedule, run_steps

EPOCH = 1787659200  # same pinned instant as every other fixture here
SEED = 1234
MAP = "classic_hearthrock_city.tmx"
#: Granite's doorstep. Probed: the player lands here and STAYS -- the
#: tile is walkable and does not re-trigger the gym door.
TILE = (23, 14)
OUT = ROOT / "tests" / "fixtures" / "hearthrock_city.save"


def main() -> None:
    pin_clock(EPOCH)
    client, session = build_client(seed=SEED, clock_epoch=EPOCH)
    install_schedule(client, {})
    run_steps(client, 60)  # let the launcher map settle

    action = client.event_engine.execute_action
    action("add_monster", ("rockitten", 25))
    action("add_monster", ("budaye", 25))
    action("add_item", ("potion",))
    action("add_item", ("potion",))
    action("teleport", ("npc_red", MAP, TILE[0], TILE[1]))
    run_steps(client, 90)  # teleports are queued and resolve over frames

    save = snapshot_save(session)
    assert save.npc_state is not None
    assert save.npc_state.current_map == MAP, save.npc_state.current_map
    print(f"current_map={save.npc_state.current_map!r}")
    print(f"tile={tuple(int(v) for v in session.player.tile_pos)}")
    print(f"party={[(m.slug, m.level) for m in session.player.monsters]}")
    OUT.write_text(save.model_dump_json() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
