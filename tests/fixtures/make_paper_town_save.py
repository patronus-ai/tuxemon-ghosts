"""Generates tests/fixtures/paper_town.save -- the agent harness's start
state. Run from the repo root:

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYTHONHASHSEED=0 \
      ./.venv/bin/python tests/fixtures/make_paper_town_save.py

Why teleport rather than play the intro: probed, mashing A from a cold
boot does not reach a bare WorldState within 6000 steps -- it wedges on
`InputMenu`, the name-entry keyboard, by step 1140 and the state stack
stops changing. Teleport is the engine's own event action, and
`boot_from_save` restores no state stack (SaveData has none), so the
resulting save boots to a bare WorldState in a real town.

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

EPOCH = 1787659200  # 2026-08-25T12:00:00 UTC -- daytime, as elsewhere
SEED = 1234
MAP = "spyder_paper_town.tmx"
TILE = (12, 12)
OUT = ROOT / "tests" / "fixtures" / "paper_town.save"


def main() -> None:
    pin_clock(EPOCH)
    client, session = build_client(seed=SEED, clock_epoch=EPOCH)
    install_schedule(client, {})
    run_steps(client, 60)  # let the launcher map settle

    action = client.event_engine.execute_action
    action("add_monster", ("rockitten", 5))
    action("add_monster", ("budaye", 5))
    action("add_item", ("potion",))
    action("teleport", ("npc_red", MAP, TILE[0], TILE[1]))
    run_steps(client, 90)  # teleports are queued and resolve over frames

    save = snapshot_save(session)
    assert save.npc_state is not None
    print(f"current_map={save.npc_state.current_map!r}")
    OUT.write_text(save.model_dump_json())
    print(f"wrote {OUT}")


main()
