"""tests/test_rules.py -- S5 progress rules."""

import json
from pathlib import Path
from typing import Any

from tuxghost.rules import TuxemonRules

SAVE = Path(__file__).parent / "fixtures" / "paper_town.save"
#: Read BEFORE bootstrapping: `_bootstrap_vendored_tuxemon` chdirs into
#: the vendored tuxemon/ directory, so a relative path resolves wrong
#: afterwards. This is the same hazard `_PATH_ARGS` exists for in the CLI.
_RAW = json.loads(SAVE.read_text())


def booted() -> Any:
    """A fresh session on the committed fixture. Returns the session.

    `boot_from_save` returns a (client, session) TUPLE and takes a
    validated `SaveData`, not a raw dict -- both verified against the
    tree. It also RESETS the module-level `local_session` singleton, so
    anything captured from a previous session goes stale: build what you
    need from THIS return value, never from an earlier one. That aliasing
    is a documented cause of vacuous tests in this project.
    """
    from tuxghost.execute import _bootstrap_vendored_tuxemon

    _bootstrap_vendored_tuxemon()
    from tuxemon.save_system.save_state import SaveData

    from tuxghost.boot import boot_from_save

    _client, session = boot_from_save(
        SaveData.model_validate(_RAW), seed=1234, clock_epoch=1787659200
    )
    return session


def test_progress_never_decreases_across_a_run() -> None:
    """`progress` is the search's gradient, so a decrease would tell the
    optimizer a candidate got WORSE when it merely wandered. Their
    docstring calls the equivalent 'monotone-as-possible'; we assert it
    outright because every term we use is monotone by construction
    (battles only append, the map index is max-tracked, party count only
    grows in normal play)."""
    from tuxghost.loop import run_steps

    session = booted()
    rules = TuxemonRules()
    seen: list[int] = []
    run_steps(
        session.client, 120,
        hook=lambda i: seen.append(rules.progress(session)),
    )
    assert seen, "no observations recorded"
    assert seen == sorted(seen), f"progress went backwards: {seen}"


def test_progress_discriminates_a_further_run() -> None:
    """THE ANTI-VACUITY TEST. This project's original state digest was
    'deterministic' across nine runs for the worst possible reason: it
    captured nothing that changed. A progress function that returns a
    constant would pass the monotonicity test above and be useless.

    A party gaining a monster must raise progress, because `party_count`
    is a real term.
    """
    session = booted()
    rules = TuxemonRules()
    before = rules.progress(session)

    party = session.player.monsters
    party.append(party[0])                     # crude but real: party_count += 1
    after = rules.progress(session)

    assert after > before, (before, after)
