"""The real-time half of S4: a fixed-timestep accumulator for a windowed,
real-time render loop.

`tuxghost.loop.run_steps` takes exactly N steps with no wall clock
involved at all -- exactly right for offline replay/record, but it says
nothing about how many steps a REAL frame, arriving after some measured
`elapsed` wall-clock time, owes the simulation. `steps_owed` is that
accumulator maths.
"""

from __future__ import annotations


def steps_owed(accumulator: float, elapsed: float, cap: int) -> tuple[int, float]:
    """How many fixed steps a frame owes, and the time left over.

    Past `cap`, leftover time is DISCARDED rather than banked. Banking it
    is the spiral of death: a slow frame owes more steps, which take
    longer, which owes more still. Discarding makes the game run slow
    under load -- but every step index stays exact, so the recorded trace
    and the ghost, both indexed by step rather than by time, stay
    correct. Frame drops degrade smoothness, never correctness.
    """
    from tuxghost.loop import FIXED_DT

    total = accumulator + elapsed
    steps = int(total // FIXED_DT)
    if steps > cap:
        return cap, 0.0
    return steps, total - steps * FIXED_DT
