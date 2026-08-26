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
  * `header.initial_state_digest` mismatch (recomputed from `initial_state`
    itself) -- always refuse. Not downgradable by `allow_mismatch`, for the
    same reason as seed/clock_epoch: this is the recorded state having
    changed underneath the trace (hand-edited or corrupted), not a version
    quibble.
  * `header.mod_version` / `header.step_rate` mismatch -- refuse unless
    `allow_mismatch` is set, in which case the trace is read and
    `"trace_mismatch"` is appended to `Provenance.taints`.
  * `header.upstream_commit` mismatch -- never refuses. Always emits a
    `UserWarning` naming both commits and reads the trace regardless of
    `allow_mismatch`.
  * `header.patch_series_id` mismatch -- never refuses. Always emits a
    `UserWarning` naming both ids and reads the trace regardless of
    `allow_mismatch`.
  * `header.platform` mismatch -- never refuses. Always emits a
    `UserWarning` naming both platforms and reads the trace regardless of
    `allow_mismatch`. Cross-platform replay is unmeasured (one machine, one
    OS, one Python were ever exercised) so this is informational, not a
    correctness claim either way.
"""

from __future__ import annotations

import hashlib
import json
import platform
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
#: OS/arch/Python this reader is running on (`platform.platform()`).
#: Informational only -- see the module docstring's `platform` row.
THIS_PLATFORM = platform.platform()

#: The directory holding the applied patch series (0001..0006 as of this
#: writing). `patch_series_id()` digests it so a trace records which
#: engine build it was produced against.
PATCHES_DIR = Path(__file__).resolve().parent.parent / "patches"

#: Refused outright: reading on would produce a silently wrong comparison.
_REFUSE_IF_MISSING = ("seed", "clock_epoch")


def patch_series_id() -> str:
    """Digest of the currently applied patch series (the files in
    `patches/`), so a trace records which engine build it was produced
    against. Lives here, not in `tuxghost.record`, so the recorder (which
    writes it) and `read()` (which compares against it) can never drift
    apart by computing it two different ways."""
    digest = hashlib.sha256()
    for patch in sorted(PATCHES_DIR.glob("*.patch")):
        digest.update(patch.name.encode())
        digest.update(patch.read_bytes())
    return "sha256:" + digest.hexdigest()


def digest_of_initial_state(initial_state: dict[str, Any]) -> str:
    """Digest over a trace's `initial_state`, written by the recorder when a
    trace is sealed and reverified by `read()` on every load. This proves
    the trace being read is the same bytes the recorder produced -- not
    downgradable by `allow_mismatch` (see the module docstring)."""
    blob = json.dumps(initial_state, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


class Refused(Exception):
    """A precondition failed; the trace will not be executed here."""


class TraceHeader(BaseModel):
    upstream_commit: str
    patch_series_id: str
    #: The OS/arch/Python build the trace was recorded on
    #: (`platform.platform()`). Mirrors `upstream_commit`: informational
    #: only, `read()` warns on a mismatch and never refuses. Never affects
    #: execution.
    platform: str
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
    #: Free-form provenance only -- e.g. a human-readable timestamp for a
    #: UI to display. Never read by `read()`/`write()` and never affects
    #: execution or reproducibility, unlike `TraceHeader.clock_epoch`
    #: (which does): traces are step-indexed, not wall-clock-indexed (see
    #: the module docstring).
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

    # Not downgradable, same reasoning as seed/clock_epoch above: this is
    # the recorded state having changed underneath the trace (hand-edited
    # or corrupted), not a version quibble allow_mismatch is meant to paper
    # over.
    initial_state = raw.get("initial_state", {})
    recorded_digest = header.get("initial_state_digest")
    actual_digest = digest_of_initial_state(initial_state)
    if recorded_digest != actual_digest:
        raise Refused(
            f"header.initial_state_digest is {recorded_digest!r}, but "
            f"initial_state recomputes to {actual_digest!r}. allow_mismatch "
            f"does not downgrade this: silencing a version quibble must not "
            f"also silence the recorded state having changed underneath the "
            f"trace."
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

    current_patch_series_id = patch_series_id()
    if header.get("patch_series_id") != current_patch_series_id:
        warnings.warn(
            f"patch_series_id {header.get('patch_series_id')!r} differs "
            f"from this build {current_patch_series_id!r}",
            stacklevel=2,
        )

    if header.get("platform") != THIS_PLATFORM:
        warnings.warn(
            f"platform {header.get('platform')!r} differs from this build "
            f"{THIS_PLATFORM!r}",
            stacklevel=2,
        )

    trace = Trace.model_validate(raw)
    if tainted:
        trace.provenance.taints.append("trace_mismatch")
    return trace


def write(trace: Trace, path: Path) -> None:
    Path(path).write_text(trace.model_dump_json(indent=2))
