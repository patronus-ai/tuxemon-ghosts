"""Trace format version 1.

Inputs are ground truth and are step-indexed. PlayerInput.timestamp, which
upstream defaults to time.time(), deliberately does not enter this format.

Refuse/warn matrix (see `read`):
  * `format_version` mismatch -- always refuse. Not downgradable by
    `allow_mismatch`: an unknown layout cannot be guessed at safely.
  * `header.seed` / `header.clock_epoch` missing -- always refuse. Not
    downgradable by `allow_mismatch`: an unpinned clock or seed produces a
    silently wrong comparison, not a version quibble (see the docstring on
    `TraceHeader.clock_epoch`).
  * `header.mod_version` / `header.step_rate` mismatch -- refuse unless
    `allow_mismatch` is set, in which case the trace is read and
    `"trace_mismatch"` is appended to `Provenance.taints`.
  * `header.upstream_commit` mismatch -- never refuses. Always emits a
    `UserWarning` naming both commits and reads the trace regardless of
    `allow_mismatch`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

FORMAT_VERSION = 1

#: What this build is. A trace recorded elsewhere is still runnable, but the
#: reader says so rather than pretending the environments match.
THIS_UPSTREAM_COMMIT = "59a34164f442ddecbaee4c436e3f1a5ba9474e29"
THIS_MOD_VERSION = "0.4.35"
THIS_STEP_RATE = 60

#: Refused outright: reading on would produce a silently wrong comparison.
_REFUSE_IF_MISSING = ("seed", "clock_epoch")


class Refused(Exception):
    """A precondition failed; the trace will not be executed here."""


class TraceHeader(BaseModel):
    upstream_commit: str
    patch_series_id: str
    mod_id: str
    mod_version: str
    seed: int
    step_rate: int
    #: Seconds since the Unix epoch, interpreted in UTC -- see
    #: `tuxemon.core.clock.now_datetime` and `tuxghost.determinism.pin_clock`.
    #: Measurement (task 9) showed the SAME epoch producing a different
    #: `(hour, day_of_year)`, and therefore a different `Monster
    #: .capture_date` and combat digest, depending on the host's `TZ` when
    #: interpreted in local time. A trace recorded on one machine must
    #: replay byte-identically on another regardless of the executing
    #: host's timezone, so `clock_epoch` is UTC by definition here, not
    #: host-local.
    clock_epoch: int = Field(
        description="Seconds since the Unix epoch, interpreted in UTC."
    )
    initial_state_digest: str
    step_count: int
    #: Set by the recorder from the live session when the trace is sealed.
    #: `verify` (a later task) executes the trace ONCE and compares the
    #: digest it reaches against this field. Executing twice and comparing
    #: the two runs against each other would only prove the engine is
    #: reproducible; it would not prove the trace still reaches what it
    #: originally reached, which is what verification means.
    final_digest: str


class Provenance(BaseModel):
    recorder: Literal["cu-agent", "offline-agent", "human"]
    model: str | None = None
    claimed_outcome: str | None = None
    taints: list[str] = Field(default_factory=list)
    recorded_at: str | None = None


class Trace(BaseModel):
    format_version: int
    header: TraceHeader
    initial_state: dict[str, Any]
    inputs: list[tuple[int, int, float]]
    provenance: Provenance


def read(path: Path, allow_mismatch: bool = False) -> Trace:
    raw: dict[str, Any] = json.loads(Path(path).read_text())

    version = raw.get("format_version")
    if version != FORMAT_VERSION:
        raise Refused(
            f"format_version {version!r} is not {FORMAT_VERSION}; refusing "
            "rather than guessing at an unknown layout"
        )

    header: dict[str, Any] = raw.get("header", {})
    for field in _REFUSE_IF_MISSING:
        if header.get(field) is None:
            raise Refused(
                f"header.{field} is required. allow_mismatch does not "
                f"downgrade this: silencing a version quibble must not also "
                f"silence the world changing underneath the trace."
            )

    # Refusals that allow_mismatch may downgrade -- a version quibble, not a
    # changed world.
    downgradable: dict[str, tuple[Any, Any]] = {
        "mod_version": (header.get("mod_version"), THIS_MOD_VERSION),
        "step_rate": (header.get("step_rate"), THIS_STEP_RATE),
    }
    tainted = False
    for field, (got, want) in downgradable.items():
        if got != want:
            if not allow_mismatch:
                raise Refused(
                    f"header.{field} is {got!r}, this build is {want!r}. "
                    f"Pass allow_mismatch to run anyway; the trace will be "
                    f"tainted."
                )
            tainted = True

    # Environment differences that are reported, never fatal.
    if header.get("upstream_commit") != THIS_UPSTREAM_COMMIT:
        warnings.warn(
            f"upstream_commit {header.get('upstream_commit')!r} differs from "
            f"this build {THIS_UPSTREAM_COMMIT!r}",
            stacklevel=2,
        )

    trace = Trace.model_validate(raw)
    if tainted:
        trace.provenance.taints.append("trace_mismatch")
    return trace


def write(trace: Trace, path: Path) -> None:
    Path(path).write_text(trace.model_dump_json(indent=2))
