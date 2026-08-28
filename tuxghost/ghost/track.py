"""Deriving a ghost's path from a trace.

Everything with a decision in it lives in this package rather than in
`tuxghost.play`, because `make check` runs headless and `play` cannot be
gated. A reviewer should be able to satisfy themselves about ghost
behaviour without ever opening a window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tuxghost.boot import boot_from_save
from tuxghost.determinism import pin_clock, seed_all
from tuxghost.execute import _schedule_of
from tuxghost.loop import install_schedule, run_steps
from tuxghost.trace import Trace


@dataclass(frozen=True)
class GhostFrame:
    map_name: str
    tile: tuple[int, int]
    facing: str


@dataclass(frozen=True)
class GhostTrack:
    frames: tuple[GhostFrame, ...]
    source_digest: str
    recorder: str
    model: str | None

    def at(self, step: int) -> GhostFrame | None:
        """The ghost's frame entering `step`, or None past the end."""
        if step < 0 or step >= len(self.frames):
            return None
        return self.frames[step]


def _sample(session: Any) -> GhostFrame:
    player = session.player
    return GhostFrame(
        map_name=session.client.get_map_name(),
        tile=(int(player.tile_pos[0]), int(player.tile_pos[1])),
        facing=str(player.facing),
    )


def build_track(trace: Trace) -> GhostTrack:
    """Replay `trace` once, keeping only what a ghost needs to be drawn.

    Deliberately NOT `execute(checkpoint=1, capture_states=True)`: that
    hook runs `digest_of` AND `state_of` on every step -- a full hash
    plus a full save serialisation (party, items, monsters) -- when three
    attribute reads suffice. On a 900-step trace that is 900 of each, and
    the digest half is entirely wasted work for a ghost.

    Mirrors `tuxghost.execute.execute`'s own preamble (seed/clock pin,
    then validate `initial_state` as a `SaveData`, then boot) but
    deliberately skips its map-resolution refusal checks: those are the
    CLI's refusal boundary's job, not this function's -- a caller here is
    expected to have already read the trace through that boundary.
    """
    from tuxemon.save_system.save_state import SaveData

    seed_all(trace.header.seed)
    pin_clock(trace.header.clock_epoch)

    save_data = SaveData.model_validate(trace.initial_state)

    _client, session = boot_from_save(
        save_data, seed=trace.header.seed, clock_epoch=trace.header.clock_epoch
    )
    client = session.client
    install_schedule(client, _schedule_of(trace))

    frames: list[GhostFrame] = []

    def hook(i: int) -> None:
        del i
        frames.append(_sample(session))

    run_steps(client, trace.header.step_count, hook=hook)
    # The final sample the hook can never take: `run_steps` calls
    # `hook(i)` BEFORE `update`, so the state after the last update is
    # unobserved. See this module's test for why this is asserted
    # arithmetically rather than trusted to a digest.
    frames.append(_sample(session))

    return GhostTrack(
        frames=tuple(frames),
        source_digest=trace.header.final_digest,
        recorder=trace.provenance.recorder,
        model=trace.provenance.model,
    )
