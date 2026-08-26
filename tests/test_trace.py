"""Pins the trace format's refuse/warn preconditions.

Refuse/warn matrix (see the module docstring on `tuxghost.trace`):
  * `format_version` mismatch -- refuse, not downgradable.
  * `header.seed` / `header.clock_epoch` missing -- refuse, not
    downgradable by `allow_mismatch`.
  * `header.initial_state_digest` mismatch -- refuse, not downgradable by
    `allow_mismatch`.
  * `header.mod_version` / `header.step_rate` mismatch -- refuse, but
    `allow_mismatch` downgrades it to a taint.
  * `header.upstream_commit` mismatch -- warn, never refuse.
  * `header.patch_series_id` mismatch -- warn, never refuse.
  * `header.platform` mismatch -- warn, never refuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tuxghost.trace import (
    FORMAT_VERSION,
    THIS_PLATFORM,
    Refused,
    digest_of_initial_state,
    patch_series_id,
    read,
    write,
)

#: `_minimal`'s default `initial_state` is `{}`; its default
#: `initial_state_digest` must be the REAL digest of that value, not a
#: placeholder, or every test that doesn't touch initial_state would
#: spuriously refuse now that `read()` verifies it.
_EMPTY_STATE_DIGEST = digest_of_initial_state({})


def _minimal_header() -> dict[str, Any]:
    """The header `_minimal` below embeds -- pulled out so tests that need
    to override a single header field (e.g. `seed`) don't have to repeat
    the other ten."""
    return {
        "upstream_commit": "59a34164f442ddecbaee4c436e3f1a5ba9474e29",
        # Real, current values -- not placeholders -- so that tests
        # which don't target patch_series_id/platform don't spuriously
        # warn now that `read()` compares both against this build.
        "patch_series_id": patch_series_id(),
        "platform": THIS_PLATFORM,
        "mod_id": "tuxemon",
        "mod_version": "0.4.35",
        "seed": 1234,
        "step_rate": 60,
        "clock_epoch": 1787694000,
        "initial_state_digest": _EMPTY_STATE_DIGEST,
        "step_count": 10,
        "final_digest": "sha256:000",
    }


def _minimal(tmp_path: Path, **overrides: Any) -> Path:
    data: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "header": _minimal_header(),
        "initial_state": {},
        "inputs": [[5, 64, 1.0]],
        "provenance": {"recorder": "offline-agent"},
    }
    data.update(overrides)
    path = tmp_path / "t.tuxghost"
    path.write_text(json.dumps(data))
    return path


def test_reads_a_valid_trace(tmp_path: Path) -> None:
    trace = read(_minimal(tmp_path))
    assert trace.header.seed == 1234
    assert trace.inputs == [(5, 64, 1.0)]


def test_reads_final_digest_from_the_header(tmp_path: Path) -> None:
    """`final_digest` is a required header field (ruling: `verify`, a later
    task, executes ONCE and compares against it -- running twice and
    comparing the runs to each other would only prove the engine is
    reproducible, not that the trace still reaches what it originally
    reached). Pin that it actually round-trips through `read`, not just
    that the schema accepts the key."""
    trace = read(_minimal(tmp_path))
    assert trace.header.final_digest == "sha256:000"


def test_rejects_an_unknown_format_version(tmp_path: Path) -> None:
    with pytest.raises(Refused, match="format_version"):
        read(_minimal(tmp_path, format_version=2))


def test_allow_mismatch_does_not_downgrade_unknown_format_version(
    tmp_path: Path,
) -> None:
    """format_version is not in the downgradable set at all -- allow_mismatch
    must not let a reader guess at an unknown trace layout."""
    with pytest.raises(Refused, match="format_version"):
        read(_minimal(tmp_path, format_version=2), allow_mismatch=True)


def test_refuses_a_trace_with_no_clock_epoch(tmp_path: Path) -> None:
    """An unpinned clock silently stops reproducing at midnight. It corrupted
    a comparison twice during the spike while every outcome looked correct,
    so this is a refusal, not a warning."""
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    del data["header"]["clock_epoch"]
    path.write_text(json.dumps(data))
    with pytest.raises(Refused, match="clock_epoch"):
        read(path)


def test_allow_mismatch_does_not_downgrade_clock_epoch(tmp_path: Path) -> None:
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    del data["header"]["clock_epoch"]
    path.write_text(json.dumps(data))
    with pytest.raises(Refused, match="clock_epoch"):
        read(path, allow_mismatch=True)


def test_refuses_a_trace_with_no_seed(tmp_path: Path) -> None:
    """Same rule as clock_epoch: an unseeded RNG/id-factory/weather-RNG
    silently diverges rather than raising, so this must refuse."""
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    del data["header"]["seed"]
    path.write_text(json.dumps(data))
    with pytest.raises(Refused, match="seed"):
        read(path)


def test_allow_mismatch_does_not_downgrade_missing_seed(tmp_path: Path) -> None:
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    del data["header"]["seed"]
    path.write_text(json.dumps(data))
    with pytest.raises(Refused, match="seed"):
        read(path, allow_mismatch=True)


def test_refuses_a_non_integer_seed(tmp_path: Path) -> None:
    """The spec's refuse/warn matrix reads 'seed missing OR non-integer'
    -- the missing half was covered above, but a present, wrong-typed
    seed used to fall through `_REFUSE_IF_MISSING`'s `is None` check
    straight into `Trace.model_validate`, which raises
    `pydantic.ValidationError` (not `Refused`) and would exit 1
    ("diverged") rather than 2 ("refused"). Must raise `Refused`, named
    specifically as `seed`, before validation ever runs."""
    path = _minimal(tmp_path, header={**_minimal_header(), "seed": "not-an-int"})
    with pytest.raises(Refused, match="seed"):
        read(path)


def test_allow_mismatch_does_not_downgrade_a_non_integer_seed(tmp_path: Path) -> None:
    path = _minimal(tmp_path, header={**_minimal_header(), "seed": "not-an-int"})
    with pytest.raises(Refused, match="seed"):
        read(path, allow_mismatch=True)


def test_refuses_a_boolean_seed(tmp_path: Path) -> None:
    """`bool` is a Python `int` subclass, so a naive `isinstance(v, int)`
    check would accept a JSON `true`/`false` as a seed. It is not an
    integer; must still refuse."""
    path = _minimal(tmp_path, header={**_minimal_header(), "seed": True})
    with pytest.raises(Refused, match="seed"):
        read(path)


def test_a_real_integer_seed_does_not_refuse(tmp_path: Path) -> None:
    """Positive control for the three tests above: a correctly-typed seed
    must read fine, otherwise they could pass even if `read` refused
    every seed unconditionally."""
    trace = read(_minimal(tmp_path))
    assert trace.header.seed == 1234


def test_refuses_a_non_dict_json_root(tmp_path: Path) -> None:
    """A trace file whose JSON root is not an object (e.g. `[]`) used to
    crash with `AttributeError: 'list' object has no attribute 'get'` on
    `raw.get("format_version")` -- an uncaught exception, which exits 1
    ("diverged") rather than 2 ("refused"). A malformed trace file must
    never read as a divergence."""
    path = tmp_path / "list_root.tuxghost"
    path.write_text("[]")
    with pytest.raises(Refused):
        read(path)


def test_refuses_a_non_dict_header(tmp_path: Path) -> None:
    """Same class of bug as the non-dict root above, one level down: a
    `header` that isn't itself a JSON object would crash the exact same
    way on `header.get(field)`."""
    path = tmp_path / "list_header.tuxghost"
    path.write_text(json.dumps({"format_version": FORMAT_VERSION, "header": []}))
    with pytest.raises(Refused):
        read(path)


def test_mod_version_mismatch_refuses_but_allow_mismatch_taints(
    tmp_path: Path,
) -> None:
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    data["header"]["mod_version"] = "0.4.99"
    path.write_text(json.dumps(data))

    with pytest.raises(Refused, match="mod_version"):
        read(path)

    trace = read(path, allow_mismatch=True)
    assert "trace_mismatch" in trace.provenance.taints


def test_step_rate_mismatch_refuses_but_allow_mismatch_taints(
    tmp_path: Path,
) -> None:
    """The same downgrade rule applies to step_rate as to mod_version --
    pinned separately since the matrix has two independently-downgradable
    fields, not one, and a fix to one must not silently cover the other."""
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    data["header"]["step_rate"] = 30
    path.write_text(json.dumps(data))

    with pytest.raises(Refused, match="step_rate"):
        read(path)

    trace = read(path, allow_mismatch=True)
    assert "trace_mismatch" in trace.provenance.taints


def test_matching_mod_version_and_step_rate_leave_taints_empty(
    tmp_path: Path,
) -> None:
    """Negative control for the two taint tests above: without a mismatch,
    reading (with or without allow_mismatch) must not fabricate a taint."""
    trace = read(_minimal(tmp_path), allow_mismatch=True)
    assert trace.provenance.taints == []


def test_differing_upstream_commit_warns_rather_than_refuses(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    data["header"]["upstream_commit"] = "0" * 40
    path.write_text(json.dumps(data))

    trace = read(path)
    assert trace.header.upstream_commit == "0" * 40
    assert any("upstream_commit" in str(w.message) for w in recwarn)


def test_matching_upstream_commit_does_not_warn(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """Positive control for the warn test above: a matching commit must not
    itself trigger the warning path -- otherwise the prior test would pass
    even if `read` warned unconditionally on every trace."""
    read(_minimal(tmp_path))
    assert not any("upstream_commit" in str(w.message) for w in recwarn)


def test_round_trips(tmp_path: Path) -> None:
    trace = read(_minimal(tmp_path))
    out = tmp_path / "out.tuxghost"
    write(trace, out)
    assert read(out).header == trace.header


# --- initial_state_digest: REFUSE, not downgradable (task 11) --------------
#
# Task 10 could not implement this row: it needs a real `initial_state` and
# a real digest of it, which only the recorder (task 11) produces. Flagged
# explicitly by task 10's reviewer so it would not fall through the gap.


def test_initial_state_digest_mismatch_refuses(tmp_path: Path) -> None:
    """The world changed underneath the trace (hand-edited or corrupted
    initial_state) -- not a version quibble, so this refuses."""
    path = _minimal(tmp_path, initial_state={"player": {"tile_pos": [1, 1]}})
    with pytest.raises(Refused, match="initial_state_digest"):
        read(path)


def test_allow_mismatch_does_not_downgrade_initial_state_digest(
    tmp_path: Path,
) -> None:
    """Same rule as seed/clock_epoch: `allow_mismatch` silences version
    quibbles, not the recorded state having changed underneath the trace.
    Must STAY refused with the flag set."""
    path = _minimal(tmp_path, initial_state={"player": {"tile_pos": [1, 1]}})
    with pytest.raises(Refused, match="initial_state_digest"):
        read(path, allow_mismatch=True)


def test_matching_initial_state_digest_does_not_refuse(
    tmp_path: Path,
) -> None:
    """Positive control: a correctly-digested initial_state must read fine,
    otherwise the two refusal tests above could pass even if `read` refused
    unconditionally."""
    state = {"player": {"tile_pos": [1, 1]}}
    path = _minimal(
        tmp_path,
        initial_state=state,
        header={
            "upstream_commit": "59a34164f442ddecbaee4c436e3f1a5ba9474e29",
            "patch_series_id": patch_series_id(),
            "platform": THIS_PLATFORM,
            "mod_id": "tuxemon",
            "mod_version": "0.4.35",
            "seed": 1234,
            "step_rate": 60,
            "clock_epoch": 1787694000,
            "initial_state_digest": digest_of_initial_state(state),
            "step_count": 10,
            "final_digest": "sha256:000",
        },
    )
    trace = read(path)
    assert trace.initial_state == state


# --- patch_series_id / platform: WARN, never refuse (task 11) --------------
#
# Same gap as above: task 10's brief said "Consumes: nothing", so it could
# not compute "this build's" patch_series_id or platform to compare
# against. Both rows must be OBSERVABLE by a test (pytest.warns / recwarn),
# not merely non-fatal -- an earlier draft of this plan shipped refusals
# with the warn half silently accepting.


def test_differing_patch_series_id_warns_rather_than_refuses(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    data["header"]["patch_series_id"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(data))

    trace = read(path)
    assert trace.header.patch_series_id == "sha256:" + "0" * 64
    assert any("patch_series_id" in str(w.message) for w in recwarn)


def test_matching_patch_series_id_does_not_warn(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """Positive control for the warn test above."""
    read(_minimal(tmp_path))
    assert not any("patch_series_id" in str(w.message) for w in recwarn)


def test_differing_platform_warns_rather_than_refuses(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    data["header"]["platform"] = "some-other-platform"
    path.write_text(json.dumps(data))

    trace = read(path)
    assert trace.header.platform == "some-other-platform"
    assert any("platform" in str(w.message) for w in recwarn)


def test_matching_platform_does_not_warn(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """Positive control for the warn test above."""
    read(_minimal(tmp_path))
    assert not any("platform" in str(w.message) for w in recwarn)


def test_patch_series_id_and_platform_mismatches_do_not_taint(
    tmp_path: Path,
) -> None:
    """Warn-only rows must not join the downgradable set: a mismatch here
    must never add "trace_mismatch" to taints, since nothing was
    downgraded -- there was never a refusal to downgrade."""
    path = _minimal(tmp_path)
    data = json.loads(path.read_text())
    data["header"]["patch_series_id"] = "sha256:" + "0" * 64
    data["header"]["platform"] = "some-other-platform"
    path.write_text(json.dumps(data))

    trace = read(path, allow_mismatch=True)
    assert trace.provenance.taints == []
