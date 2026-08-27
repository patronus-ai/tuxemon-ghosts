"""Task 2: `add_edge`'s contract, tested directly rather than through
`run_agent`.

Both halves of that contract have a history. It APPENDS rather than
assigns because a `settle=0` action schedules its release on exactly the
step the next action's press lands on, and `schedule[step] = [...]`
silently dropped one of them. It RE-SORTS on every insert because
`install_schedule` delivers a step's edges in list order, so insertion
order could match `_schedule_of`'s output by luck on one route and
deliver a press/release pair to the engine in the opposite order on
another. Neither was caught directly before: both were only reachable
through a full `run_agent` route.
"""

from __future__ import annotations

from tuxghost.loop import PRESSED, RELEASED, InputSchedule, add_edge


def test_two_edges_on_one_step_are_both_kept() -> None:
    schedule: InputSchedule = {}
    add_edge(schedule, 16, 2, RELEASED)
    add_edge(schedule, 16, 8, PRESSED)
    assert schedule == {16: [(2, RELEASED), (8, PRESSED)]}


def test_edges_are_sorted_canonically_regardless_of_insert_order() -> None:
    """Insert in DESCENDING button order; the stored list must still be
    ascending. Descending on purpose: an ascending insert order would
    leave the list already sorted and the missing-sort defect invisible,
    which is exactly how that defect once survived a test."""
    schedule: InputSchedule = {}
    add_edge(schedule, 4, 8, PRESSED)
    add_edge(schedule, 4, 2, PRESSED)
    add_edge(schedule, 4, 1, RELEASED)
    assert schedule[4] == [(1, RELEASED), (2, PRESSED), (8, PRESSED)]


def test_the_same_button_and_value_twice_is_kept_twice() -> None:
    """`add_edge` is not a set: it records what it was told. A duplicate
    is a caller bug, and silently swallowing one would hide it."""
    schedule: InputSchedule = {}
    add_edge(schedule, 0, 64, PRESSED)
    add_edge(schedule, 0, 64, PRESSED)
    assert schedule[0] == [(64, PRESSED), (64, PRESSED)]


def test_constants_are_the_values_the_engine_expects() -> None:
    assert (PRESSED, RELEASED) == (1.0, 0.0)
