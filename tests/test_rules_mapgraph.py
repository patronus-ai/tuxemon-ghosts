"""Pins `tuxghost.rules.CRITICAL_PATH` against the SHIPPED MAP DATA.

Both facts below were assumptions carried since S5 and are now measured.
Neither is checkable by reading `rules.py`: they are properties of 263
`.tmx` files under `tuxemon/mods/tuxemon/maps/`, which a vendor bump can
change without touching a line of our code. That is exactly why they get
a test rather than a comment.

Parsing all 263 maps takes ~0.04s, so this stays in the fast suite.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

from tuxghost.rules import CRITICAL_PATH

MAPS = Path(__file__).resolve().parent.parent / "tuxemon/mods/tuxemon/maps"

#: The engine's own transition verb, as it appears in the map data:
#: `transition_teleport player,<map>.tmx,<x>,<y>,<duration>`.
_TELEPORT = re.compile(r"transition_teleport\s+player,([A-Za-z0-9_\-]+\.tmx)")

#: The map the committed save (`tests/fixtures/paper_town.save`) boots on.
START = "spyder_paper_town.tmx"


def _map_graph() -> dict[str, set[str]]:
    """Directed map graph, derived from the shipped `.tmx` files."""
    graph: dict[str, set[str]] = collections.defaultdict(set)
    for path in MAPS.glob("*.tmx"):
        for target in _TELEPORT.findall(path.read_text(errors="replace")):
            graph[path.name].add(target)
    return graph


def _hops(graph: dict[str, set[str]], start: str) -> dict[str, int]:
    """Breadth-first hop count from `start`, over `graph`."""
    dist = {start: 0}
    queue = collections.deque([start])
    while queue:
        here = queue.popleft()
        for nxt in sorted(graph.get(here, ())):
            if nxt not in dist:
                dist[nxt] = dist[here] + 1
                queue.append(nxt)
    return dist


@pytest.fixture(scope="module")
def graph() -> dict[str, set[str]]:
    g = _map_graph()
    assert g, f"no map data found under {MAPS}"
    return g


def test_critical_path_is_ordered_by_real_map_distance(
    graph: dict[str, set[str]],
) -> None:
    """`CRITICAL_PATH`'s ORDER carried a VERIFY marker from S5 until now.

    Adjacency alone never settled it -- `spyder_route1` is a hub touching
    all three towns, so "which town is further along" is not visible from
    the edges. Hop count from the starting save's own map settles it, and
    it must be STRICTLY increasing or the map term of `progress` rewards
    the wrong direction.
    """
    dist = _hops(graph, START)
    hops = []
    for name in CRITICAL_PATH:
        assert name in dist, f"{name} is unreachable from {START}"
        hops.append(dist[name])
    assert hops == sorted(hops), dict(zip(CRITICAL_PATH, hops))
    assert len(set(hops)) == len(hops), (
        f"two CRITICAL_PATH maps tie at the same distance: "
        f"{dict(zip(CRITICAL_PATH, hops))}"
    )
    assert hops[0] == 0, "the path must begin at the starting map"


def test_no_gym_is_reachable_from_the_starting_map(
    graph: dict[str, set[str]],
) -> None:
    """`TuxemonGymRules` is aspirational, and this pins WHY.

    Not "the search horizon is long" -- there is no path at all. The
    `classic_*` campaign is a separate connected component from the
    `spyder_*` one the shipped save lives in. If a vendor bump ever
    connects them, the gym goal becomes reachable and this test fails,
    which is the moment someone should revisit `TuxemonGymRules`.

    Checked UNDIRECTED, deliberately: a one-way link would still make the
    goal reachable, and testing only the directed graph would miss it.
    """
    undirected: dict[str, set[str]] = collections.defaultdict(set)
    for src, dsts in graph.items():
        for dst in dsts:
            undirected[src].add(dst)
            undirected[dst].add(src)

    reachable = set(_hops(undirected, START))
    gyms = {p.name for p in MAPS.glob("classic_gym_*.tmx")}
    assert gyms, "no classic_gym_* maps on disk -- has the vendor tree moved?"
    assert not (reachable & gyms), sorted(reachable & gyms)


def test_the_starting_map_is_the_one_the_committed_save_boots_on() -> None:
    """The two tests above are only meaningful if `START` is really where
    a run begins. Read from the save itself, never trusted to a literal.
    """
    import json

    save = Path(__file__).parent / "fixtures" / "paper_town.save"
    npc_state = json.loads(save.read_text())["npc_state"]
    assert npc_state["current_map"] == START
