"""Records a step-indexed trace from a live session.

`Recorder` is the write side of the format `tuxghost.trace` defines. It
mirrors `read()`'s environment fingerprint back at it: `patch_series_id`,
`THIS_UPSTREAM_COMMIT`, `THIS_MOD_VERSION`, and `THIS_PLATFORM` all come
from `tuxghost.trace` rather than being re-derived here, so a trace
recorded and immediately read back on the same build can never spuriously
warn or refuse from the writer and reader computing the "current build"
fingerprint two different ways.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from tuxghost.boot import snapshot_save
from tuxghost.determinism import pin_clock
from tuxghost.digest import digest_of
from tuxghost.loop import STEP_RATE
from tuxghost.trace import (
    FORMAT_VERSION,
    THIS_MOD_VERSION,
    THIS_PLATFORM,
    THIS_UPSTREAM_COMMIT,
    Provenance,
    Trace,
    TraceHeader,
    digest_of_initial_state,
    patch_series_id,
)


class Recorder:
    """Turns a live session into a `Trace`.

    Inputs are step-indexed, never wall-clock-indexed: `observe()` takes
    the step index the caller scheduled the input for, and `finish()`
    sorts by that index before sealing the trace -- the order `observe`
    happens to be called in does not matter. See the module docstring on
    `tuxghost.trace` for why `PlayerInput.timestamp` must never enter this
    format.
    """

    def __init__(
        self,
        session: Any,
        seed: int,
        clock_epoch: int,
        recorder: Literal["cu-agent", "offline-agent", "human"],
        model: str | None = None,
    ) -> None:
        self._session = session
        self._seed = seed
        self._clock_epoch = clock_epoch
        self._recorder = recorder
        self._model = model
        self._inputs: list[tuple[int, int, float]] = []

        # `AbstractSession._start_timestamp`/`_start_time` (upstream
        # `tuxemon/session.py`) are set ONCE, at process start, when the
        # module-level `local_session` singleton is constructed -- before
        # any caller has had a chance to call `pin_clock`. `build_client`'s
        # `reset()` does not touch them (only `reset_time()` does, and
        # nothing calls it). `get_state()` (what `snapshot_save` drives)
        # then computes `SessionSave.duration` from that frozen,
        # real-wall-clock timestamp and, as a side effect, overwrites it
        # with `now()` -- so a second recording in the same process
        # measures a completely different (near-zero) `duration` than the
        # first, even for the identical seed/schedule. Caught by a
        # same-seed stability check recording twice in one process: the
        # first `initial_state_digest` came out as `duration=-26762.9...`,
        # the second as `duration=0.0`, for byte-identical gameplay.
        # `duration`/`total_playtime`/`start_time` are playtime telemetry,
        # not comparable game state -- `tuxghost.digest.state_of` never
        # reads `session_state` at all, for the same reason. Pinning the
        # clock (idempotent if the caller already did it) and resetting
        # session time bookkeeping right before the snapshot makes
        # `initial_state` a pure function of `(seed, clock_epoch)` again,
        # matching every other reproducibility guarantee in this project.
        pin_clock(clock_epoch)
        session.reset_time()

        # Captured now, via the game's own save serialisation, rather than
        # re-derived at `finish()` time: `finish()` may be called long
        # after boot, and this is meant to be the state the run *started*
        # from.
        self._initial: dict[str, Any] = json.loads(
            snapshot_save(session).model_dump_json()
        )

    def observe(self, step: int, button: int, value: float) -> None:
        """Record an input scheduled for `step`."""
        self._inputs.append((step, button, value))

    def finish(self, step_count: int) -> Trace:
        """Seal the trace, recording the state the run actually reached.

        `final_digest` is read from the LIVE session, not recomputed by
        replaying anything: without a recorded outcome, `verify` (a later
        task) could only prove a trace is reproducible by executing it
        twice and comparing the two runs -- not that it still reaches what
        it originally reached, which is what verification means. `verify`
        instead executes ONCE and compares against this field.
        """
        return Trace(
            format_version=FORMAT_VERSION,
            header=TraceHeader(
                upstream_commit=THIS_UPSTREAM_COMMIT,
                patch_series_id=patch_series_id(),
                platform=THIS_PLATFORM,
                mod_id="tuxemon",
                mod_version=THIS_MOD_VERSION,
                seed=self._seed,
                step_rate=STEP_RATE,
                clock_epoch=self._clock_epoch,
                initial_state_digest=digest_of_initial_state(self._initial),
                step_count=step_count,
                final_digest=digest_of(self._session),
            ),
            initial_state=self._initial,
            inputs=sorted(self._inputs),
            provenance=Provenance(recorder=self._recorder, model=self._model),
        )
