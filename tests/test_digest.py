import json
import re
from typing import Any

from tuxemon.platform.const import buttons

from tuxghost.boot import build_client
from tuxghost.determinism import pin_clock
from tuxghost.digest import EXEMPTIONS, digest_of, state_of
from tuxghost.loop import InputSchedule, install_schedule, run_steps

# Fix round 1 (patch 0004): this file predates `pin_clock` (Task 5/6, before
# Task 9) and never pinned the clock, so `_run`'s 900-step mash was reading
# real wall time. `mods/tuxemon/maps/spyder.yaml`'s ambient "Night Day Cycle
# Outside/Inside" world events call `set_layer` whenever `stage_of_day`
# reaches "night", and headless `NullRenderer` (`tuxemon/map/view.py`) had
# no `.layer` attribute -- `AttributeError: 'NullRenderer' object has no
# attribute 'layer'`. Left unpinned, this suite silently depended on the
# real time of day it happened to run at -- it passed every earlier run in
# this project purely because those all fell in real daytime hours, and
# started failing the moment a session ran past real dusk.
#
# Fix round 2 (patch 0001): the `NullRenderer` gap itself -- flagged as
# "parked" in fix round 1 -- turned out to block a third of `clock_epoch`'s
# domain (`stage_of_day == "night"` is `hour < 4 or hour >= 20`, 8 of 24
# UTC hours) on frame one of *any* route, not just this file's, so it was
# promoted from parked to fixed: `NullRenderer.__init__` now sets a real
# `layer` Surface (plus `layer_color`/`layer_image`), mirroring
# `MapRenderer`, so `set_layer`'s calls are harmless no-ops instead of
# crashes. `DIGEST_EPOCH` stays a daytime epoch (no reason to move a
# passing route back to the region that used to crash); `NIGHT_EPOCH` below
# adds the coverage that region never had, rather than trading one for the
# other.
DIGEST_EPOCH = 1787659200  # 2026-08-25T12:00:00 UTC -- safely inside `daytime`
NIGHT_EPOCH = 1787774400  # 2026-08-26T20:00:00 UTC -- stage_of_day == "night"


def _run(seed: int, steps: int = 900, epoch: int = DIGEST_EPOCH) -> str:
    # `build_client` resets the `local_session` singleton internally before
    # building, so two in-process calls here do not contaminate each other
    # (see `tuxghost/boot.py`; previously that reset had to be done here by
    # hand, three times over, which is exactly the kind of workaround a
    # forgotten call turns into silent contamination rather than an error).
    pin_clock(epoch)
    client, session = build_client(seed=seed)
    schedule: InputSchedule = {}
    for i in range(40):
        schedule[30 + i * 12] = [(buttons.A, 1.0)]
        schedule[32 + i * 12] = [(buttons.A, 0.0)]
    install_schedule(client, schedule)
    run_steps(client, steps)
    return digest_of(session)


def test_digest_is_stable_for_one_seed() -> None:
    assert _run(1234) == _run(1234)


def test_digest_is_stable_for_one_seed_at_a_night_epoch() -> None:
    """Fix round 2 (patch 0001): pins the `NullRenderer.layer` fix and
    closes the coverage hole fix round 1 opened -- `DIGEST_EPOCH` moved
    every digest test in this file to daytime, so nothing exercised the
    night third of `clock_epoch`'s domain at all until this test. Per the
    project's process rule, a route that only ever ran once could not
    distinguish "fixed" from "coincidentally didn't crash this time" --
    same-seed equality on the same night route is what actually proves it.
    Before the `NullRenderer` fix this raised `AttributeError:
    'NullRenderer' object has no attribute 'layer'` inside `run_steps`,
    not merely a digest mismatch; see the task-9 fix-round-2 report for
    the real traceback."""
    assert _run(1234, epoch=NIGHT_EPOCH) == _run(1234, epoch=NIGHT_EPOCH)


def test_digest_is_stable_for_one_seed_on_a_route_that_adds_a_monster() -> None:
    """Companion control the brief's own suite lacked: `_run`'s A-mash route
    never calls `add_monster`, so it never exercises `game_variables`
    (where `add_monster` stashes a uuid4 -- see EXEMPTIONS). Without this
    test, nothing in the suite ever asked "is the discriminating route
    itself stable across two runs of the SAME seed?" -- which is exactly
    how a fix-round-1 defect (a raw, unexempted uuid4 nonce making the
    digest unstable even for one seed) went undetected: the stability test
    used a route that couldn't see it, and the discrimination test used a
    route where two different seeds happening to look identical would
    still pass."""

    def run_with_monster() -> str:
        _client, session = build_client(seed=1234)
        session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
        return digest_of(session)

    assert run_with_monster() == run_with_monster()


def test_digest_is_stable_for_one_seed_on_a_route_that_adds_an_item() -> None:
    """Companion control for `Item.instance_id`: `tuxemon/item/item.py`
    called raw `uuid4()` even after patch 0003's sweep of the five (then
    six, once `monster.py` was found) load-bearing entity-id sites --
    `items` is a real `NPCState` field, and this route reaches it, so
    fix-round-1 review caught a genuine, digested divergence: two same-seed
    runs disagreed on `npc_state.items[0].instance_id`. Route through
    `add_item` (companion to `add_monster`'s own test above) so this stays
    reachable and `test_exemptions_are_all_reachable_in_the_digested_tree`
    would catch it if it ever stopped being so."""

    def run_with_item() -> str:
        _client, session = build_client(seed=1234)
        session.client.event_engine.execute_action("add_item", ("potion", 2))
        return digest_of(session)

    assert run_with_item() == run_with_item()


def test_digest_discriminates_between_seeds() -> None:
    """A digest that cannot tell seeds apart makes every determinism test
    vacuous. This is the control that caught a worthless probe in the spike.

    `tuxemon.session.local_session` is a module-level singleton:
    `build_client()` returns that same object every call, so `session_a`
    and `session_b` below are literally the same object once the second
    `build_client` call has run -- `GameLauncher.launch` re-creates the
    player *in place* on that shared session. Computing `digest_of(session_a)`
    only after `session_b` exists would silently digest the same
    already-mutated session twice (an assertion of x != x on identical
    state, which reads as "discrimination" but proves nothing). Digest A is
    therefore captured as a string BEFORE the second client is built.
    """
    _client_a, session_a = build_client(seed=1234)
    session_a.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    digest_a = digest_of(session_a)

    _client_b, session_b = build_client(seed=99)
    session_b.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    digest_b = digest_of(session_b)

    assert digest_a != digest_b


def test_digest_sees_gameplay_state() -> None:
    _client, session = build_client(seed=1234)
    before = digest_of(session)
    session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    assert digest_of(session) != before


def test_exemptions_each_name_a_defect() -> None:
    for field, reason in EXEMPTIONS.items():
        assert reason.strip(), f"exemption {field!r} has no stated reason"
        # "because" or similar hand-waving must not pass: every reason has
        # to name the specific patch that closes the defect, so the
        # register stays auditable and each entry is deletable on its own
        # schedule rather than by vibes.
        assert re.search(r"patch \d{4}", reason), (
            f"exemption {field!r} reason {reason!r} does not name a "
            "specific patch that closes it"
        )
    assert "timestamp" not in EXEMPTIONS, (
        "Battle.timestamp becomes deterministic under patch 0004; exempting "
        "it would re-hide the bug the spike spent a bisect finding"
    )
    assert not any(
        field == "timestamp" or field.endswith(".timestamp") for field in EXEMPTIONS
    ), "no path may exempt a timestamp field, under any nesting"


def test_exemptions_are_all_reachable_in_the_digested_tree() -> None:
    """An EXEMPTIONS entry that never appears anywhere `state_of()` looks is
    a dead entry: it cannot be masking any defect, and because paths are
    matched exactly, it also cannot accidentally swallow an unrelated
    future field. Walk the real (pre-filter) `npc_state`/`world_state`
    trees for a representative session and require every registered path
    to actually occur in them."""
    _client, session = build_client(seed=1234)
    session.client.event_engine.execute_action("add_monster", ("rockitten", 12))
    player = session.player
    npc_raw = json.loads(player.get_state(session).model_dump_json())
    world_raw = json.loads(session.world.get_state(session).model_dump_json())

    paths: set[str] = set()

    def collect(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                child_path = f"{path}.{k}" if path else k
                paths.add(child_path)
                collect(v, child_path)
        elif isinstance(obj, list):
            for item in obj:
                collect(item, path)

    collect(npc_raw, "npc_state")
    collect(world_raw, "world_state")

    for field in EXEMPTIONS:
        assert field in paths, (
            f"exemption {field!r} is unreachable (dead) in the tree "
            "state_of() actually digests -- delete it or fix its path"
        )


def test_state_of_is_not_blind_to_the_state_stack() -> None:
    """The spike's worthless digest never looked at `state_stack`, so it
    could not see the one thing that was actually changing. Pin its
    presence directly rather than only through an end-to-end hash
    comparison."""
    _client, session = build_client(seed=1234)
    state = state_of(session)
    assert state["state_stack"], "state_stack must be present and non-empty"
