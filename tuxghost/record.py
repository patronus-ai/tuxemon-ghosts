"""Records a step-indexed trace from a live session.

`Recorder` is the write side of the format `tuxghost.trace` defines. It
mirrors `read()`'s environment fingerprint back at it: `patch_series_id`,
`THIS_UPSTREAM_COMMIT`, `THIS_MOD_VERSION`, and `THIS_PLATFORM` all come
from `tuxghost.trace` rather than being re-derived here, so a trace
recorded and immediately read back on the same build can never spuriously
warn or refuse from the writer and reader computing the "current build"
fingerprint two different ways.

`Recorder` does NOT pin the clock or reset the session's own elapsed-time
bookkeeping itself -- constructing a `Recorder` around an already-live
session must not silently mutate that session's clock/playtime state as a
side effect the caller didn't ask for. Instead, build the session with
`clock_epoch` passed through `tuxghost.boot.build_client`/`boot_from_save`
*before* constructing `Recorder`; see those functions' docstrings for why
(`SessionSave.duration`/`total_playtime`/`start_time` otherwise leak real
wall-clock time into `initial_state`, non-reproducibly). This also covers
the executor (a later task), which calls `boot_from_save` directly and
never constructs a `Recorder` at all.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from tuxghost.boot import snapshot_save
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

    `session` should already have been built with this same `clock_epoch`
    (`tuxghost.boot.build_client(seed, clock_epoch=...)` or
    `boot_from_save(save, seed, clock_epoch=...)`) -- see the module
    docstring. `Recorder` trusts the caller here rather than resetting
    session time itself.
    """

    def __init__(
        self,
        session: Any,
        seed: int,
        clock_epoch: int,
        recorder: Literal["cu-agent", "offline-agent", "human"],
        model: str | None = None,
        taints: Sequence[str] = (),
    ) -> None:
        """`taints` is for a recording whose decisions did not come from the
        thing `recorder`/`model` name -- specifically a `ReplayPolicy`
        run, whose actions are replayed from a captured transcript rather
        than taken live by the model. `tuxghost.cli`'s `compare` already
        surfaces taints as findings, so recording it here costs no new
        reader.
        """
        self._session = session
        self._seed = seed
        self._clock_epoch = clock_epoch
        self._recorder = recorder
        self._model = model
        self._taints = list(taints)
        self._inputs: list[tuple[int, int, float]] = []

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

    def finish(
        self, step_count: int, claimed_outcome: str | None = None
    ) -> Trace:
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
            provenance=Provenance(
                recorder=self._recorder,
                model=self._model,
                taints=list(self._taints),
                claimed_outcome=claimed_outcome,
            ),
        )
