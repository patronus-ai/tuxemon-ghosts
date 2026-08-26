"""Field-level divergence reporting.

Two capabilities, meant to compose:

  * `first_difference` names the exact dotted field path two already-
    compared state dicts disagree on.
  * `first_divergent_step` narrows a pair of checkpointed runs down to the
    earliest step whose digests disagree.

Naming a hash instead of a field is what previously turned "the digest
differs" into a hand-run bisect over 120,000 steps to locate one field
(`Battle.timestamp`, per the task-12 brief). `describe_divergence` below
chains the two: once `first_divergent_step` has narrowed a divergence down
to one checkpoint, `first_difference` explains what actually differs
*there*, and `_value_at` recovers the two disagreeing values so the report
does not just name the field but shows what it held on each side. See
`tuxghost.execute.bisect_traces` for the end-to-end orchestration that
drives full states into this at the located step.
"""

from __future__ import annotations

import re
from typing import Any

#: Matches one path segment as `first_difference` emits it: a bare key
#: name, optionally followed by any number of `[<int>]` list-index
#: suffixes (e.g. "battles[0]", or "battles" with no suffix at all).
_SEGMENT = re.compile(r"^([^\[]*)((?:\[\d+\])*)$")


def first_difference(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """Return the dotted path of the first field on which `a` and `b`
    differ, or `None` if they are equal. Paths look like
    `npc_state.battles[0].timestamp`: dict keys are joined with `.`, list
    elements share their parent's path with a `[<index>]` suffix appended
    (matching `tuxghost.digest._canonical`'s own path convention -- a
    missing list element still needs to name *which* list diverged)."""
    return _first_difference(a, b, "")


def _first_difference(a: Any, b: Any, path: str) -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else key
            if key not in a or key not in b:
                return sub
            found = _first_difference(a[key], b[key], sub)
            if found is not None:
                return found
        return None

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}[len]"
        for i, (x, y) in enumerate(zip(a, b)):
            found = _first_difference(x, y, f"{path}[{i}]")
            if found is not None:
                return found
        return None

    return None if a == b else path


def first_divergent_step(
    a: list[tuple[int, str]], b: list[tuple[int, str]]
) -> int | None:
    """Earliest checkpoint step at which two runs' digests differ, or
    `None` if every checkpoint they share matches. `a`/`b` come from
    `tuxghost.execute.ExecutionResult.checkpoints`; both must have been
    produced with the same checkpoint interval, or the zip below silently
    compares mismatched steps against each other -- the assert catches
    that rather than returning a wrong step number."""
    for (step_a, digest_a), (step_b, digest_b) in zip(a, b):
        assert step_a == step_b, "checkpoint cadences must match"
        if digest_a != digest_b:
            return step_a
    return None


def _value_at(obj: Any, path: str) -> Any:
    """Resolve a dotted/bracketed path from `first_difference` back to the
    value it names in `obj`, e.g. `npc_state.battles[0].timestamp`."""
    node = obj
    for part in path.split("."):
        match = _SEGMENT.match(part)
        assert match is not None, f"malformed path segment: {part!r}"
        name, brackets = match.group(1), match.group(2)
        if name:
            node = node[name]
        for index in re.findall(r"\[(\d+)\]", brackets):
            node = node[int(index)]
    return node


def describe_divergence(step: int, a: dict[str, Any], b: dict[str, Any]) -> str:
    """Format the divergence report: which step, which field, which two
    values -- e.g.::

        diverged at step 10000
          npc_state.battles[0].timestamp: 1787694198.869361 != 1787694387.998919

    Callers are expected to reach this only once they already know `a` and
    `b` differ (typically because `first_divergent_step` found `step` by
    comparing digests) -- `first_difference` returning `None` here would
    mean the two full states are equal despite their digests disagreeing,
    which would point at a bug in `digest_of`/`state_of` (something the
    digest hashes but `state_of`'s returned tree does not expose), not a
    normal "nothing differs" outcome, so it is reported rather than
    silently treated as `path is None -> no divergence`."""
    path = first_difference(a, b)
    if path is None:
        return (
            f"diverged at step {step}\n"
            "  digests differ but no field differs in the compared state "
            "trees -- digest_of() is hashing something state_of() does not "
            "expose"
        )
    value_a = _value_at(a, path)
    value_b = _value_at(b, path)
    return f"diverged at step {step}\n  {path}: {value_a} != {value_b}"
