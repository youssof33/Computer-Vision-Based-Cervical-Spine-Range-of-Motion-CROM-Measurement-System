"""Guided six-movement protocol.

The protocol only tracks *which* movement is being measured. It never changes an
angle; it advances when the current channel has been held near its maximum for
`hold_s` seconds, or when the operator skips with Space.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Step:
    movement: str
    instruction: str


STEPS: tuple[Step, ...] = (
    Step("Flexion", "Chin down toward the chest"),
    Step("Extension", "Look up as far as comfortable"),
    Step("Right lateral flexion", "Right ear toward right shoulder"),
    Step("Left lateral flexion", "Left ear toward left shoulder"),
    Step("Right rotation", "Turn the head to your right"),
    Step("Left rotation", "Turn the head to your left"),
)


class Protocol:
    """Current step of the guided sequence."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self._best = 0.0
        self._held_since: float | None = None

    @property
    def total(self) -> int:
        return len(STEPS)

    @property
    def done(self) -> bool:
        return self.index >= self.total

    @property
    def step(self) -> Step | None:
        return None if self.done else STEPS[self.index]

    def next(self) -> None:
        self.index += 1
        self._best = 0.0
        self._held_since = None

    def update(
        self,
        channels: dict[str, float],
        now: float,
        hold_s: float | None = None,
        min_deg: float | None = None,
    ) -> bool:
        """Feed one frame. Returns True when the step just completed."""
        if self.done:
            return False
        hold_s = config.PROTOCOL_HOLD_S if hold_s is None else hold_s
        min_deg = config.PROTOCOL_MIN_DEG if min_deg is None else min_deg

        value = channels[self.step.movement]
        if value > self._best + 0.5:  # still reaching end range
            self._best = value
            self._held_since = now
            return False

        if (
            self._best >= min_deg
            and self._held_since is not None
            and now - self._held_since >= hold_s
        ):
            self.next()
            return True
        return False
