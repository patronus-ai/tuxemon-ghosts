"""Pins the trace format's refuse/warn preconditions.

Refuse/warn matrix (see the module docstring on `tuxghost.trace`):
  * `format_version` mismatch -- refuse, not downgradable.
  * `header.seed` / `header.clock_epoch` missing -- refuse, not
    downgradable by `allow_mismatch`.
  * `header.mod_version` / `header.step_rate` mismatch -- refuse, but
    `allow_mismatch` downgrades it to a taint.
  * `header.upstream_commit` mismatch -- warn, never refuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tuxghost.trace import FORMAT_VERSION, Refused, read, write


def _minimal(tmp_path: Path, **overrides: Any) -> Path:
    data: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "header": {
            "upstream_commit": "59a34164f442ddecbaee4c436e3f1a5ba9474e29",
            "patch_series_id": "sha256:abc",
            "mod_id": "tuxemon",
            "mod_version": "0.4.35",
            "seed": 1234,
            "step_rate": 60,
            "clock_epoch": 1787694000,
            "initial_state_digest": "sha256:def",
            "step_count": 10,
            "final_digest": "sha256:000",
        },
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
