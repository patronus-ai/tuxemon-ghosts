"""The agent's vocabulary: what it sees, what it may answer.

An action is a DURATION, not an instant. At 60 steps/s one step of UP does
not move the player a tile, a dialog needs A pressed AND released, and a
combat menu needs several presses -- so `(button, hold, settle)` is the
smallest unit that does anything. The harness owns durations and cadence
becomes a property of the policy: a fixed-window RL policy (S3) is simply
a policy that always returns the same numbers, needing no harness change.

Prior art, for why this shape and not a per-frame button mask:
`ClaudePlaysPokemonStarter`'s `press_buttons` holds 10 frames then settles
120; `PokemonRedExperiments` presses at tick 0 and releases at tick 8 of a
fixed window. Those frame counts are Game Boy-specific and deliberately
NOT inherited -- `WALK_ONE_TILE` below is measured on this game.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tuxemon.platform.const import buttons as _buttons

#: Steps a held direction takes to cross one tile. MEASURED on the
#: `paper_town` fixture, not inherited from any Game Boy harness: raw
#: per-tile deltas were `[2, 16, 16, 16, 16]` over five consecutive tiles
#: (the leading 2 is a start-position edge effect, not the real cost).
#: Corroborated independently by the engine's own numbers --
#: `STEP_RATE` 60 / `CONFIG.player_walkrate` 3.75 = 16.
WALK_ONE_TILE = 16

#: A single decision may not hold a button, or wait, for longer than 10
#: seconds of game time. A cap is not a style preference: an LLM that
#: emits `hold: 1000000` would otherwise run the game for four and a half
#: hours of game time inside one `run_steps` call, unobservable and
#: uninterruptible.
HOLD_CAP = 600
SETTLE_CAP = 600

#: Every virtual button the platform layer defines. Built from the module
#: rather than hardcoded so a new upstream button does not silently become
#: "invalid input" here.
VALID_BUTTONS = frozenset(
    value
    for name, value in vars(_buttons).items()
    if not name.startswith("_") and isinstance(value, int)
)


@dataclass(frozen=True)
class Action:
    """Press `button`, hold it `hold` steps, release, then run `settle`
    more steps before the next observation."""

    button: int
    hold: int
    settle: int


@dataclass(frozen=True)
class Observation:
    """What the policy is given. `frame_png` is the observation; everything
    else is metadata for logs and tests.

    A policy that decides from `state_stack` rather than the frame is a
    state-machine agent wearing a computer-use agent's clothes -- the
    metadata is here so a failing test can say WHERE the agent was, and so
    a run directory is auditable after the fact.
    """

    step: int
    frame_png: bytes
    frame_is_blank: bool
    state_stack: tuple[str, ...]
    map_name: str
    tile_pos: tuple[int, int]


@runtime_checkable
class Policy(Protocol):
    def decide(self, obs: Observation) -> Sequence[Action]:
        """Return the actions to take now. An EMPTY sequence means STOP --
        it is a valid answer, not an error (see `STOP`)."""
        ...


#: The documented way for a policy to end a run.
STOP: tuple[Action, ...] = ()


def validate_actions(actions: Sequence[Action]) -> tuple[Action, ...]:
    """Check a policy's answer before any of it reaches the engine.

    A policy is untrusted input: `ClaudePolicy` parses a language model's
    JSON, and a malformed action that slipped through would corrupt a
    trace (an out-of-range `hold` silently rescales it) rather than fail
    loudly.
    """
    checked = []
    for index, action in enumerate(actions):
        if not isinstance(action, Action):
            raise TypeError(
                f"actions[{index}] must be an Action, got "
                f"{type(action).__name__}"
            )
        if action.button not in VALID_BUTTONS:
            raise ValueError(
                f"actions[{index}].button={action.button!r} is not a known "
                "virtual button (tuxemon.platform.const.buttons)"
            )
        if not 1 <= action.hold <= HOLD_CAP:
            raise ValueError(
                f"actions[{index}].hold={action.hold!r} must be between 1 "
                f"and HOLD_CAP={HOLD_CAP}"
            )
        if not 0 <= action.settle <= SETTLE_CAP:
            raise ValueError(
                f"actions[{index}].settle={action.settle!r} must be between "
                f"0 and SETTLE_CAP={SETTLE_CAP}"
            )
        checked.append(action)
    return tuple(checked)
